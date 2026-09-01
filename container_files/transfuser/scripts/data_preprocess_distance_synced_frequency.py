import argparse
import glob
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import yaml

import numpy as np

#Image imports
try:
    import cv2
except ImportError:
    sys.exit("cv2 (opencv-python) is required. Run this inside the SIL container.")
# Optional: cv_bridge gives us robust encoding handling; fall back to numpy if
# it isn't in the image.
try:
    from cv_bridge import CvBridge
    _BRIDGE = CvBridge()
except ImportError:
    _BRIDGE = None

#Packages to read rosbag2 files
import rosbag2_py
import tf2_ros
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rclpy.time import Time

#Point cloud imports
from sensor_msgs_py import point_cloud2

# Segment images into occupied and free space
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch


DEFAULT_BAG_DATA = '/workspace/bag_data'

#Topic details
CAMERA_TOPIC_CANDIDATES = ('/front_camera/color/image_view','/front_camera/color/image_view_throttled')

DEPTH_TOPIC_CANDIDATES = ('/front_camera/aligned_depth_to_color/image_rect_raw', '/front_camera/aligned_depth_to_color/image_rect_raw_throttled',)

LIDAR_TOPIC_CANDIDATES = ('/hesai/pandar_points_isaac', '/hesai/pandar_points')
COSTMAP_TOPIC = '/local_costmap'
CMD_VEL_TOPIC = '/cmd_vel/isaac'
TF_TOPIC, TF_STATIC_TOPIC = '/tf', '/tf_static'
MAX_REF_FRAMES = 200

# Utility functions for reading rosbag2 files. 
def _open_reader(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''),
    )
    return reader


def _resolve_topic(type_map, requested):
    """Match the requested topic to a stored one, tolerating a leading '/'."""
    if requested in type_map:
        return requested
    for name in type_map:
        if name.lstrip('/') == requested.lstrip('/'):
            return name
    return None

def _raw_image_to_bgr(msg):
    """Decode a sensor_msgs/Image into a BGR uint8 frame without cv_bridge."""
    enc = msg.encoding.lower()
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ('rgb8', 'bgr8'):
        img = buf.reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == 'rgb8' else img
    if enc in ('rgba8', 'bgra8'):
        img = buf.reshape(h, w, 4)
        code = cv2.COLOR_RGBA2BGR if enc == 'rgba8' else cv2.COLOR_BGRA2BGR
        return cv2.cvtColor(img, code)
    if enc == 'mono8':
        return cv2.cvtColor(buf.reshape(h, w), cv2.COLOR_GRAY2BGR)
    if enc in ('mono16', '16uc1'):
        img = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
        img8 = cv2.convertScaleAbs(img, alpha=255.0 / max(int(img.max()), 1))
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported image encoding: {msg.encoding!r}")

def _msg_to_bgr(msg, msg_type_name):
    """Convert an Image / CompressedImage message to a BGR uint8 frame."""
    if 'CompressedImage' in msg_type_name:
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2 failed to decode a CompressedImage frame")
        return frame
    if _BRIDGE is not None:
        return _BRIDGE.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    return _raw_image_to_bgr(msg)

def _raw_depth_to_array(msg):
    """Decode a depth Image message into a raw numpy array, preserving units.
 
    Returns uint16 (typically millimetres) for 16UC1/mono16 encodings, or
    float32 (typically metres) for 32FC1. No rescaling/normalisation is
    applied so downstream consumers get true depth values.
    """
    enc = msg.encoding.lower()
    # print(enc)
    h, w = msg.height, msg.width
 
    if enc in ('16uc1', 'mono16'):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
    if enc == '32fc1':
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
        # print(depth.max())
        return depth
    raise ValueError(f"unsupported depth encoding: {msg.encoding!r}")

def _depth_array_to_png_uint16(depth_arr):
    """Coerce a decoded depth array into a uint16 image suitable for cv2.imwrite.
 
    float32 (metres) is converted to millimetres and clipped to uint16 range;
    uint16 arrays are passed through unchanged.
    """
    if depth_arr.dtype == np.uint16:
        return depth_arr
    if depth_arr.dtype == np.float32:
        mm = depth_arr * 1000.0
        mm = np.nan_to_num(mm, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    raise ValueError(f"unsupported depth array dtype: {depth_arr.dtype!r}")

def _bag_topic_counts(bag_path):
    with open(os.path.join(bag_path, 'metadata.yaml')) as fh:
        meta = yaml.safe_load(fh)
    return {
        e['topic_metadata']['name']: e['message_count']
        for e in meta['rosbag2_bagfile_information']['topics_with_message_count']
    }

def _grid_to_rgb(msg):
    """nav_msgs/OccupancyGrid -> (HxWx3 float image in [0,1], imshow extent).

    White = free, black = occupied, light gray = unknown (-1).
    """
    w, h = msg.info.width, msg.info.height
    data = np.array(msg.data, dtype=np.float32).reshape(h, w)
    free = (100.0 - np.clip(data, 0, 100)) / 100.0
    rgb = np.stack([free, free, free], axis=-1)
    rgb[data < 0] = (0.8, 0.8, 0.8)
    ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
    res = msg.info.resolution
    extent = (ox, ox + w * res, oy, oy + h * res)
    return rgb, extent

def _type_map(reader):
    return {t.name: t.type for t in reader.get_all_topics_and_types()}

def _default_output_path(bag_path):
    bag_path = os.path.abspath(bag_path.rstrip('/'))
    run_name = os.path.basename(bag_path)
    date_dir = os.path.basename(os.path.dirname(bag_path))       # .../bag_data/<date>/<run>
    bag_data_dir = os.path.dirname(os.path.dirname(bag_path))    # .../bag_data
    root = os.path.dirname(bag_data_dir)
    return os.path.join(root, 'data', date_dir, run_name)

#Calculate bag frequency
def _bag_duration_s(bag_path):
    with open(os.path.join(bag_path, 'metadata.yaml')) as fh:
        meta = yaml.safe_load(fh)
    info = meta['rosbag2_bagfile_information']
    return info['duration']['nanoseconds'] / 1e9

def average_topic_hz(bag_path):
    counts = _bag_topic_counts(bag_path)
    duration_s = _bag_duration_s(bag_path)
    return {topic: n / duration_s for topic, n in counts.items()}

def world_to_pixel(x, y, msg):
    res = msg.info.resolution
    ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
    col = int((x - ox) / res)
    row = int((y - oy) / res)
    return col, row  # (x_pixel, y_pixel) for cv2

def closest_command_point(waypoints, x, y, yaw):
    """Return the waypoint (dict with x, y, yaw) whose (x, y) is nearest to (x, y)
    in the map frame, restricted to waypoints ahead of the robot's heading (yaw,
    radians, map frame). Rotation-only waypoints (x or y omitted) are skipped.
    Falls back to the closest waypoint overall if none lie ahead.
    """
    candidates = [wp for wp in waypoints if wp['x'] is not None and wp['y'] is not None]
    forward_x, forward_y = math.cos(yaw), math.sin(yaw)
    ahead = [wp for wp in candidates
             if (wp['x'] - x) * forward_x + (wp['y'] - y) * forward_y > 0]
    pool = ahead if ahead else candidates
    return min(pool, key=lambda wp: (wp['x'] - x) ** 2 + (wp['y'] - y) ** 2)

def save_synced_frames(bag_path, type_map, counts, output_path,
                        ref_side='front', max_dt_ns=1e7, min_dist_m=0.5):
    """Single pass over the bag. Each /tf tick is a candidate reference tick,
    accepted only once the robot has moved at least min_dist_m (map frame,
    base_link) since the last saved tick. Camera RGB, depth, lidar, costmap
    and cmd_vel are only written when a sample within max_dt_ns of the
    reference timestamp exists. Nothing is written for reference frames
    without full coverage.
    """

    os.makedirs(os.path.join(output_path, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'lidar'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'costmap'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'trajectory'), exist_ok=True)

    # ref_topic = CAMERA_TOPIC_CANDIDATES[0]
    ref_topic = TF_TOPIC

    val = _bag_topic_counts(bag_path).get(ref_topic, None)

    # if val is not None and val>200:
    #     stride = math.ceil(val / MAX_REF_FRAMES)
    # else:
    #     stride = 1

    # stride = 1

     
    #calculate the average frequency of the reference topic
    avg_freq = average_topic_hz(bag_path).get(ref_topic, 0)
    print(f"[info] average frequency of {ref_topic}: {avg_freq:.2f} Hz")

    depth_topics = next((t for t in DEPTH_TOPIC_CANDIDATES if t in type_map), None)
    camera_topic = next((t for t in CAMERA_TOPIC_CANDIDATES if t in type_map), None)

    print(depth_topics, camera_topic)

    lidar_topic = next((t for t in LIDAR_TOPIC_CANDIDATES if t in type_map), None)

    """
    {'/hesai/pandar_points_isaac': 'sensor_msgs/msg/PointCloud2',
    '/cmd_vel/isaac': 'geometry_msgs/msg/Twist',
    '/front_camera/color/image_view': 'sensor_msgs/msg/Image',
    '/front_camera/aligned_depth_to_color/image_rect_raw': 'sensor_msgs/msg/Image',
    '/tf_static': 'tf2_msgs/msg/TFMessage',
    '/tf': 'tf2_msgs/msg/TFMessage',
    '/local_costmap': 'nav_msgs/msg/OccupancyGrid'}

    """

    topics = (
        [ref_topic, TF_STATIC_TOPIC, COSTMAP_TOPIC, CMD_VEL_TOPIC]
        + ([depth_topics] if depth_topics else [])
        + ([camera_topic] if camera_topic else [])
        + ([lidar_topic] if lidar_topic else [])
    )

    tf_buffer = tf2_ros.Buffer()
    latest = {}  # topic_name -> (t, msg, msg_type_name)

    n_saved, n_skipped = 0, 0
    ref_seen = 0

    command_pts = find_command_point(bag_path)
    
    print(command_pts)

    # SAMPLE_PERIOD_NS = int(0.5 * 1e9)   # 2 Hz
    # next_t = None
    

    last_saved_xy = None

    for t, msg, msg_type_name, topic_name in _iter_messages_multi(bag_path, topics):
        if topic_name == TF_STATIC_TOPIC:
            for tf_msg in msg.transforms:
                tf_buffer.set_transform_static(tf_msg, 'bag')
            continue

        if topic_name != ref_topic:
            latest[topic_name] = (t, msg, msg_type_name)
            continue

        # topic_name == ref_topic == TF_TOPIC: update the buffer, then gate on
        # distance traveled since the last saved tick before treating this as
        # a reference tick.
        for tf_msg in msg.transforms:
            tf_buffer.set_transform(tf_msg, 'bag')

        try:
            transform = tf_buffer.lookup_transform('map', 'base_link', Time())
        except tf2_ros.TransformException:
            continue  # no pose yet -> can't evaluate distance, not a candidate tick

        robot_x = transform.transform.translation.x
        robot_y = transform.transform.translation.y

        if last_saved_xy is not None:
            dist = math.hypot(robot_x - last_saved_xy[0], robot_y - last_saved_xy[1])
            if dist < min_dist_m:
                continue  # hasn't moved far enough since the last saved tick

        ref_seen += 1

        # get reference values for the topics
        def closest_val(topic, max_dt):
            entry = latest.get(topic)
            if entry is None or abs(entry[0] - t) > max_dt:
                return None
            return entry

        camera_entry = closest_val(camera_topic, max_dt_ns)
        depth_entry = closest_val(depth_topics, max_dt_ns)
        lidar_entry = closest_val(lidar_topic, max_dt_ns) if lidar_topic else True  # optional
        # No freshness window: reuse whatever costmap has most recently arrived,
        # however stale, so long as one has been seen at all.
        costmap_entry = latest.get(COSTMAP_TOPIC)
        cmd_vel_entry = closest_val(CMD_VEL_TOPIC, max_dt_ns)


        if (camera_entry is None or depth_entry is None or lidar_entry is None
                or cmd_vel_entry is None):
            n_skipped += 1
            continue  # incomplete -> write nothing for this tick

        # only now do we touch disk
        c_t, c_msg, c_type = camera_entry
        rgb_bgr = _msg_to_bgr(c_msg, c_type)
        cv2.imwrite(os.path.join(output_path, 'rgb', ref_side, f'{t}.png'), rgb_bgr)
        print(f"[info] saved rgb frame at t={t} to {os.path.join(output_path, 'rgb', ref_side, f'{t}.png')}")

        d_t, d_msg, d_type = depth_entry
        depth_png = _depth_array_to_png_uint16(_raw_depth_to_array(d_msg))
        cv2.imwrite(os.path.join(output_path, 'depth', ref_side, f'{t}.png'), depth_png)

        if lidar_topic:
            l_t, l_msg, l_type = lidar_entry
            # print(l_t, l_type)
            points = point_cloud2.read_points(l_msg, field_names=("x", "y", "z"), skip_nans=True)
            np.save(os.path.join(output_path, 'lidar', f'{t}.npy'), np.array(list(points)))

        if costmap_entry is not None:
            cm_t, cm_msg, cm_type = costmap_entry
            rgb_grid, _ = _grid_to_rgb(cm_msg)
            rgb_grid = (np.clip(rgb_grid, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_path, 'costmap', f'{t}.png'), rgb_grid)
        cv_t, cv_msg, cv_type = cmd_vel_entry
        cv_linear, cv_angular = _twist_components(cv_msg, cv_type)

        #find the closest command point
        robot_z = transform.transform.translation.z
        robot_yaw = _yaw_from_quat(transform.transform.rotation)

        cmd_pt = closest_command_point(command_pts, robot_x, robot_y, robot_yaw)

        with open(os.path.join(output_path, 'trajectory', f'{t}.json'), 'w') as f:
            json.dump({
                'translation': {'x': robot_x,
                                 'y': robot_y,
                                 'z': robot_z},
                'rotation': {'x': transform.transform.rotation.x,
                             'y': transform.transform.rotation.y,
                             'z': transform.transform.rotation.z,
                             'w': transform.transform.rotation.w},
                'velocity': {'linear': {'x': cv_linear.x, 'y': cv_linear.y, 'z': cv_linear.z},
                                'angular': {'x': cv_angular.x, 'y': cv_angular.y, 'z': cv_angular.z}},
                'x_command': cmd_pt['x'],
                'y_command': cmd_pt['y'],
            }, f)

        last_saved_xy = (robot_x, robot_y)
        n_saved += 1


    print(f"[info] saved {n_saved} synced frames, skipped {n_skipped} incomplete ticks "
          f"(max_dt={max_dt_ns / 1e9:.1f}s, min_dist={min_dist_m:.2f}m)")


def find_command_point(bag_path):
    """
    Given a bag run directory (e.g. .../bag_data/scenario_1/corridor1_w1_route1_225231),
    look up its route definition in /workspace/routes/<scenario>.xml and return the
    ordered list of waypoints ({x, y, yaw}, None for any omitted attribute) for that
    route - the <start> spawn pose is not included, only the <waypoint> entries.
    """
    scenario = os.path.split(bag_path)[0].split('/')[-1]
    route_dir_name = os.path.split(bag_path)[-1]

    # route_dir_name looks like "corridor1_w1_route1_225231" - pull the route number
    # out of the "routeN" token, ignoring the corridor/world prefix and the trailing
    # capture timestamp.
    route_token = next(tok for tok in route_dir_name.split('_') if tok.startswith('route'))
    route_num = route_token[len('route'):]

    scenario_path = os.path.join("/workspace", "routes", scenario + ".xml")

    if "scenario" not in scenario_path: 
        scenario_path = os.path.join("/workspace", "routes", "scenario_1" + ".xml")

    root = ET.parse(scenario_path).getroot()

    route_elem = root.find(f"./route[@id='{route_num}']")
    if route_elem is None:
        raise ValueError(f"No route id={route_num!r} found in {scenario_path}")

    def _get_float(elem, attr):
        val = elem.get(attr)
        return float(val) if val is not None else None

    return [
        {'x': _get_float(wp, 'x'), 'y': _get_float(wp, 'y'), 'yaw': _get_float(wp, 'yaw')}
        for wp in route_elem.findall('waypoint')
    ]


def _iter_messages(bag_path, topic):
    """Return the (deserialized `index`-th message, msg_type_name) on `topic`.

    Clamped to the last message if the topic has fewer than `index + 1` of them.
    """
    reader = _open_reader(bag_path)
    type_map = _type_map(reader)
    msg_type_name = type_map[topic]
    msg_class = get_message(msg_type_name)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    while reader.has_next():
        _, data, t = reader.read_next()
        yield t, deserialize_message(data, msg_class), msg_type_name


def _twist_components(msg, msg_type_name):
    """Return (linear, angular) Vector3 fields, from either a Twist or a TwistStamped."""
    twist = msg.twist if 'TwistStamped' in msg_type_name else msg
    return twist.linear, twist.angular

def _yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    
def _project_xy(x, y, transform):
    """Apply a planar (yaw-only) rigid TransformStamped to a batch of xy points."""
    tx = transform.transform.translation.x
    ty = transform.transform.translation.y
    yaw = _yaw_from_quat(transform.transform.rotation)
    c, s = math.cos(yaw), math.sin(yaw)
    return tx + c * x - s * y, ty + s * x + c * y


def _iter_messages_multi(bag_path, topics):
    """Yield (timestamp_ns, msg, msg_type_name, requested_topic) across several
    topics, in the order they were recorded in the bag.
 
    This is what lets us feed a tf2 buffer transforms and process another
    topic (e.g. a costmap) using whatever tf state existed at that point in
    time, rather than re-reading the whole bag once per topic. Topics not
    present in the bag are silently skipped.
    """
    reader = _open_reader(bag_path)
    type_map = _type_map(reader)
 
    resolved_to_requested = {}
    for topic in topics:
        resolved = _resolve_topic(type_map, topic)
        if resolved is not None:
            resolved_to_requested[resolved] = topic
 
    if not resolved_to_requested:
        return
 
    msg_classes = {
        resolved: get_message(type_map[resolved]) for resolved in resolved_to_requested
    }
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(resolved_to_requested.keys())))
 
    while reader.has_next():
        topic_name, data, t = reader.read_next()
        msg_class = msg_classes[topic_name]
        msg = deserialize_message(data, msg_class)
        yield t, msg, type_map[topic_name], resolved_to_requested[topic_name]


def save_camera_data(bag_path, type_map, counts, output_path):
    """Save camera data from a rosbag2 reader to disk."""

    for side in CAMERA_SIDES:
        #make RGB directory
        rgb_dir = os.path.join(output_path, 'rgb', side)
        os.makedirs(rgb_dir, exist_ok=True)

        #make depth directory
        depth_dir = os.path.join(output_path, 'depth', side)
        os.makedirs(depth_dir, exist_ok=True)

        cam_topic = CAMERA_TOPIC_CANDIDATES[side]
        depth_topic = DEPTH_TOPIC_CANDIDATES[side]

        cam_count = counts.get(cam_topic, 0)
        depth_count = counts.get(depth_topic, 0)

        # print(cam_count, depth_count)

        if cam_count !=0:
            n_saved = 0
            for t, msg, msg_type_name in _iter_messages(bag_path, cam_topic):
                try:
                    frame_bgr = _msg_to_bgr(msg, msg_type_name)
                except ValueError as exc:
                    print(f"[warn] {side} rgb frame at t={t} skipped: {exc}")
                    continue
                cv2.imwrite(os.path.join(rgb_dir, f'{t}.png'), frame_bgr)
                n_saved += 1
            expected = counts.get(cam_topic, 0)
            print(f"[info] {side}: saved {n_saved}/{expected} rgb frames -> {rgb_dir}")

        if depth_count != 0:
            n_saved = 0
            for t, msg, msg_type_name in _iter_messages(bag_path, depth_topic):
                try:
                    depth_arr = _raw_depth_to_array(msg)
                    depth_png = _depth_array_to_png_uint16(depth_arr)
                except ValueError as exc:
                    print(f"[warn] {side} depth frame at t={t} skipped: {exc}")
                    continue
                cv2.imwrite(os.path.join(depth_dir, f'{t}.png'), depth_png)
                n_saved += 1
            expected = counts.get(depth_topic, 0)
            print(f"[info] {side}: saved {n_saved}/{expected} depth frames -> {depth_dir}")
            
def save_lidar_data(bag_path, type_map, counts, output_path):
    """Save lidar data from a rosbag2 reader to disk."""
    lidar_dir = os.path.join(output_path, 'lidar')
    os.makedirs(lidar_dir, exist_ok=True)

    for topic in LIDAR_TOPIC_CANDIDATES:
        if topic in type_map:
            n_saved = 0
            for t, msg, msg_type_name in _iter_messages(bag_path, topic):
                points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
                points_array = np.array(list(points))
                np.save(os.path.join(lidar_dir, f'{t}.npy'), points_array)
                n_saved += 1
            expected = counts.get(topic, 0)
            print(f"[info] saved {n_saved}/{expected} lidar frames -> {lidar_dir}")
            break
    else:
        print("[warn] No lidar topic found in bag.")

def save_cost_map_bev(bag_path, type_map, counts, output_path, base_frame='base_link', map_frame='map'):
    """Save the cost map data from a rosbag2 reader to disk."""
    costmap_dir = os.path.join(output_path, 'costmap')
    os.makedirs(costmap_dir, exist_ok=True)

    if COSTMAP_TOPIC not in type_map:
        print("[warn] No cost map topic found in bag.")
        return
    
    n_saved = 0
    n_skipped = 0
    buffer = tf2_ros.Buffer()
    for t, msg, msg_type_name, topic_name in _iter_messages_multi(bag_path, [TF_STATIC_TOPIC, TF_TOPIC, COSTMAP_TOPIC]):
        if topic_name == TF_STATIC_TOPIC:
            for tf_msg in msg.transforms:
                buffer.set_transform_static(tf_msg, 'bag')
            continue
        if topic_name == TF_TOPIC:
            for tf_msg in msg.transforms:
                buffer.set_transform(tf_msg, 'bag')
            continue
 
        # topic_name == COSTMAP_TOPIC
        try:
            grid_to_map = buffer.lookup_transform(map_frame, msg.header.frame_id, Time())
        except tf2_ros.TransformException as exc:
            print(f"[warn] costmap frame at t={t} skipped (no tf {msg.header.frame_id}->{map_frame}: {exc})")
            n_skipped += 1
            continue
        chosen_grid = (msg, grid_to_map)
 
        if chosen_grid is not None:
            grid_msg, grid_from_map = chosen_grid
            rgb, extent = _grid_to_rgb(grid_msg)
        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(costmap_dir, f'{t}.png'), rgb_u8)
        n_saved += 1
      
    expected = counts.get(COSTMAP_TOPIC, 0)
    print(f"[info] saved {n_saved}/{expected} cost map frames -> {costmap_dir}")

def save_trajectory(bag_path, type_map, counts, output_path, base_frame='base_link', map_frame='map'):
    """Save the robot trajectory from a rosbag2 reader to disk."""
    trajectory_dir = os.path.join(output_path, 'trajectory')
    os.makedirs(trajectory_dir, exist_ok=True)

    if TF_TOPIC not in type_map:
        print("[warn] No tf topic found in bag.")
        return

    n_saved = 0
    n_skipped = 0
    buffer = tf2_ros.Buffer()
    for t, msg, msg_type_name, topic_name in _iter_messages_multi(bag_path, [TF_STATIC_TOPIC, TF_TOPIC]):
        if topic_name == TF_STATIC_TOPIC:
            for tf_msg in msg.transforms:
                buffer.set_transform_static(tf_msg, 'bag')
            
        if topic_name == TF_TOPIC:
            for tf_msg in msg.transforms:
                buffer.set_transform(tf_msg, 'bag')

        try:
            transform = buffer.lookup_transform(map_frame, base_frame, Time())
        except tf2_ros.TransformException as exc:
            print(f"[warn] trajectory frame at t={t} skipped (no tf {base_frame}->{map_frame}: {exc})")
            n_skipped += 1
            continue

        with open(os.path.join(trajectory_dir, f'{t}.json'), 'w') as f:
            json.dump({
                'translation': {
                    'x': transform.transform.translation.x,
                    'y': transform.transform.translation.y,
                    'z': transform.transform.translation.z,
                },
                'rotation': {
                    'x': transform.transform.rotation.x,
                    'y': transform.transform.rotation.y,
                    'z': transform.transform.rotation.z,
                    'w': transform.transform.rotation.w,
                },
            }, f)
        n_saved += 1

    print(f"[info] saved {n_saved} trajectory frames -> {trajectory_dir}")
    
def save_segmented_images(bag_path, type_map, counts, output_path):
    """Segment images into occupied and free space using a pre-trained model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", use_fast=True)
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(device)
    

    # ADE20K class 3 = "floor" (confirmed via model.config.id2label; this checkpoint
    # was trained on the 150-class ADE20K label set).
    FLOOR_CLASS_ID = 3
    
    rgb_dir = os.path.join(output_path, 'rgb', 'front')  # Assuming you want to segment the front camera images
    segmented_dir = os.path.join(output_path, 'segmented', 'front')
    os.makedirs(segmented_dir, exist_ok=True)

    n_saved = 0
    for img_file in glob.glob(os.path.join(rgb_dir, '*.png')):
        img = Image.open(img_file).convert("RGB")
        inputs = image_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predicted = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[img.size[::-1]]  # (H, W); PIL .size is (W, H)
        )[0]
        floor_mask = (predicted == FLOOR_CLASS_ID).cpu().numpy()

        image_np = np.array(img)
        overlay = image_np.copy()
        overlay[floor_mask] = (0.5 * overlay[floor_mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)

        # Convert segmentation to an RGB image for visualization
        segmented_img = Image.fromarray((floor_mask.astype(np.uint8) * 255), mode='L')
        segmented_img.save(os.path.join(segmented_dir, os.path.basename(img_file)))
        n_saved += 1

    expected = len(glob.glob(os.path.join(rgb_dir, '*.png')))
    print(f"[info] saved {n_saved}/{expected} segmented images -> {segmented_dir}")



def process_data(bag_path, output_path, base_frame, map_frame, max_sync_dt_s=10.0, min_dist_m=0.05):
    bag_path = os.path.abspath(bag_path.rstrip('/'))
    run_name = os.path.basename(bag_path)
    counts = _bag_topic_counts(bag_path)
    type_map = _type_map(_open_reader(bag_path))
    save_synced_frames(bag_path, type_map, counts, output_path,
                        max_dt_ns=int(max_sync_dt_s * 1e9), min_dist_m=min_dist_m)
    # save_camera_data(bag_path, type_map, counts, output_path)
    # save_lidar_data(bag_path, type_map, counts, output_path)
    # save_cost_map_bev(bag_path, type_map, counts, output_path)
    # save_trajectory(bag_path, type_map, counts, output_path, base_frame, map_frame)
    save_segmented_images(bag_path, type_map, counts, output_path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', help='Path to a bag run directory (contains metadata.yaml + .mcap).')
    parser.add_argument('--output', default=None,
                         help="Output PNG path (default: visualizations/<date>/<run>.png "
                              "next to bag_data).")
    
    parser.add_argument('--base-frame', default='base_link',
                         help="Robot base frame for the trajectory (default: %(default)s).")
    parser.add_argument('--map-frame', default='map',
                         help="Fixed frame the trajectory is expressed in (default: %(default)s).")
    parser.add_argument('--max-sync-dt', type=float, default=0.1,
                         help="Max allowed time gap, in seconds, between the reference "
                              "front-camera tick and every other synced stream "
                              "(depth/lidar/costmap/left+right camera). The throttled "
                              "topics in these bags fire independently at ~0.05-0.3 Hz, "
                              "so a sub-second window rarely finds a full match; widen "
                              "this (and rerun) if you still see 0 synced frames, or "
                              "tighten it if the matches look too stale (default: %(default)s).")
    parser.add_argument('--min-dist', type=float, default=0.05,
                         help="Minimum distance, in meters, the robot must travel "
                              "(map frame, base_link) between saved reference ticks "
                              "(default: %(default)s).")
    args = parser.parse_args()

    output_path = args.output or _default_output_path(args.bag)
    print(output_path)
    os.makedirs(output_path, exist_ok=True)
    process_data(args.bag, output_path, args.base_frame, args.map_frame,
                 args.max_sync_dt, args.min_dist)

if __name__ == '__main__':
    main()