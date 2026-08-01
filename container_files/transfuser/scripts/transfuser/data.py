import ujson
from skimage.transform import rotate
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
from pathlib import Path
import cv2
import random
from copy import deepcopy
import io
from utils import get_base_to_lidar_transform, get_lidar_to_base_transform
from config import GlobalConfig

def is_valid_lidar_frame(path):
    """Isaac Sim lidar frames should load as a structured (x, y, z) array. A few
    captures produced malformed frames instead - an object-dtype array (numpy
    fell back to boxing per-point records) or a file that fails to load at all."""
    try:
        arr = np.load(path, allow_pickle=True)
    except Exception:
        return False
    return arr.dtype.names is not None

class IsaacSimData(Dataset):

    def __init__(self, root, config, shared_dict=None):
        self.seq_len = np.array(config.seq_len)
        assert (config.img_seq_len == 1)
        self.pred_len = np.array(config.pred_len)

        self.img_resolution = np.array(config.img_resolution)
        self.img_crop_shift_y = np.array(config.img_crop_shift_y)
        self.img_width = np.array(config.img_width)
        self.scale = np.array(config.scale)
        self.multitask = np.array(config.multitask)
        self.data_cache = shared_dict
        self.augment = np.array(config.augment) #Introducing a 20% yaw jitter
        self.aug_max_rotation = np.array(config.aug_max_rotation)
        self.use_point_pillars = np.array(config.use_point_pillars)
        self.max_lidar_points = np.array(config.max_lidar_points)
        self.backbone = np.array(config.backbone).astype(np.string_)
        self.inv_augment_prob = np.array(config.inv_augment_prob)
        self.lidar_pos = config.lidar_pos
        self.camera_pos = config.camera_pos

        
        self.converter = np.uint8(config.converter)
    
        self.images = []
        self.bevs = []
        self.depths = []
        self.semantics = []
        self.lidars = []
        self.measurements = []

        # root entries are already leaf route directories (e.g. GlobalConfig.train_data
        # gives <root_dir>/<scenario>/<route>), each directly containing rgb/lidar/depth/
        # segmented/costmap/trajectory - no further "route" subfolder level underneath.
        for route_dir in tqdm(root, file=sys.stdout):
            route_dir = Path(route_dir)

            # Frames are named by capture timestamp, not a sequential %04d index, so
            # enumerate them from the lidar folder and index into that ordered list.
            frame_stems = sorted(p.stem for p in (route_dir / "lidar").glob("*.npy"))
            num_seq = len(frame_stems)

            # A handful of captures have malformed lidar frames (object-dtype arrays
            # instead of the expected structured (x,y,z) dtype, or files that fail to
            # load at all). Precompute which stems are usable so any window touching
            # a bad frame can be skipped below instead of crashing __getitem__ later.
            # valid_lidar_stem = {
            #     stem: is_valid_lidar_frame(route_dir / "lidar" / (stem + ".npy"))
            #     for stem in frame_stems
            # }

            # ignore the first two and last two frame
            for seq in range(2, num_seq - self.pred_len - self.seq_len - 2):
                # window_stems = [frame_stems[seq + idx] for idx in range(self.seq_len)]
                # if not all(valid_lidar_stem[stem] for stem in window_stems):
                #     continue

                # load input seq and pred seq jointly
                image = []
                bev = []
                depth = []
                semantic = []
                lidar = []
                measurement= []
                # Loads the current (and past) frames (if seq_len > 1)
                for idx in range(self.seq_len):
                    stem = frame_stems[seq + idx]
                    image.append(route_dir / "rgb" / "front" / (stem + ".png"))
                    bev.append(route_dir / "costmap" / (stem + ".png"))
                    depth.append(route_dir / "depth" / "front" / (stem + ".png"))
                    semantic.append(route_dir / "segmented" / "front" / (stem + ".png"))
                    lidar.append(route_dir / "lidar" / (stem + ".npy"))

                # Additionally keep the future frames' trajectory so ego waypoints can be
                # computed directly from our own recorded pose, with no bounding-box labels.
                for idx in range(self.seq_len + self.pred_len):
                    stem = frame_stems[seq + idx]
                    measurement.append(route_dir / "trajectory" / (stem + ".json"))

                self.images.append(image)
                self.bevs.append(bev)
                self.depths.append(depth)
                self.semantics.append(semantic)
                self.lidars.append(lidar)
                self.measurements.append(measurement)

        # There is a complex "memory leak"/performance issue when using Python objects like lists in a Dataloader that is loaded with multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects because they only have 1 refcount.
        self.images       = np.array(self.images      ).astype(np.string_)
        self.bevs         = np.array(self.bevs        ).astype(np.string_)
        self.depths       = np.array(self.depths      ).astype(np.string_)
        self.semantics    = np.array(self.semantics   ).astype(np.string_)
        self.lidars       = np.array(self.lidars      ).astype(np.string_)
        self.measurements = np.array(self.measurements).astype(np.string_)
        print(self.images.shape)
        print("Loading %d lidars from %d folders"%(len(self.lidars), len(root)))

    def __len__(self):
        """Returns the length of the dataset. """
        return self.lidars.shape[0]
    
    def __getitem__(self, index):
        "Return item at index idx"
        cv2.setNumThreads(0) # Disable threading because the data loader will already split in threads.

        data = dict()
        backbone = str(self.backbone, encoding='utf-8')

        images = self.images[index]
        bevs = self.bevs[index]
        depths = self.depths[index]
        semantics = self.semantics[index]
        lidars = self.lidars[index]
        measurements = self.measurements[index]

        

        # load measurements
        loaded_images = []
        loaded_bevs = []
        loaded_depths = []
        loaded_semantics = []
        loaded_lidars = []
        loaded_measurements = []
        
        if(backbone == 'geometric_fusion'):
            loaded_lidars_raw = []

        for i in range(self.seq_len):
            if not self.data_cache is None and str(measurements[i], encoding='utf-8') in self.data_cache:
                    measurements_i, images_i, lidars_i, lidars_raw_i, bevs_i, depths_i, semantics_i = self.data_cache[str(measurements[i], encoding='utf-8')]
                    images_i = cv2.imdecode(images_i, cv2.IMREAD_UNCHANGED)
                    depths_i = cv2.imdecode(depths_i, cv2.IMREAD_UNCHANGED)
                    semantics_i = cv2.imdecode(semantics_i, cv2.IMREAD_UNCHANGED)
                    bevs_i.seek(0) # Set the point to the start of the file like object
                    bevs_i = np.load(bevs_i)['arr_0']
            else:
                with open(str(measurements[i], encoding='utf-8'), 'r') as f1:
                    measurements_i = ujson.load(f1)

                # Isaac Sim saves lidar as a structured (x, y, z) array (no intensity),
                # not the [metadata, points] pickle CARLA used - load the fields directly
                # and pad a constant 4th column so align()'s homogeneous transform still works.
                lidar_points = np.load(str(lidars[i], encoding='utf-8'), allow_pickle=True)
                lidars_i = np.stack([lidar_points['x'], lidar_points['y'], lidar_points['z'],
                                     np.ones(len(lidar_points), dtype=np.float32)], axis=-1).astype(np.float32)
                if (backbone == 'geometric_fusion'):
                    lidars_raw_i = lidars_i[..., :3]
                else:
                    lidars_raw_i = None
                    
                images_i = cv2.imread(str(images[i], encoding='utf-8'), cv2.IMREAD_COLOR)
                if(images_i is None):
                    print("Error loading file: ", str(images[i], encoding='utf-8'))
                images_i = scale_image_cv2(cv2.cvtColor(images_i, cv2.COLOR_BGR2RGB), self.scale)

                # costmap is already a plain black/white occupancy image (no CARLA-style
                # bit-packed multi-class encoding), so just threshold it into a single
                # binary occupancy channel: 1 = occupied, 0 = free.
                bev_array = cv2.imread(str(bevs[i], encoding='utf-8'), cv2.IMREAD_GRAYSCALE)
                if (bev_array is None):
                    print("Error loading file: ", str(bevs[i], encoding='utf-8'))
                bevs_i = (bev_array > 127).astype(np.uint8)
                # print(bevs_i.shape)
                if self.multitask:
                    depths_i = cv2.imread(str(depths[i], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
                    if (depths_i is None):
                        print("Error loading file: ", str(depths[i], encoding='utf-8'))
                    depths_i = scale_image_cv2(depths_i, self.scale)

                    semantics_i = cv2.imread(str(semantics[i], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
                    if (semantics_i is None):
                        print("Error loading file: ", str(semantics[i], encoding='utf-8'))
                    semantics_i = scale_seg(semantics_i, self.scale)
                else:
                    depths_i = None
                    semantics_i = None

                if not self.data_cache is None:
                    # We want to cache the images in png format instead of uncompressed, to reduce memory usage
                    result, compressed_imgage = cv2.imencode('.png', images_i)
                    result, compressed_depths = cv2.imencode('.png', depths_i)
                    result, compressed_semantics = cv2.imencode('.png', semantics_i)
                    compressed_bevs = io.BytesIO()  # bev has 2 channels which does not work with png compression so we use generic numpy in memory compression
                    np.savez_compressed(compressed_bevs, bevs_i)
                    self.data_cache[str(measurements[i], encoding='utf-8')] = (measurements_i, compressed_imgage, lidars_i, lidars_raw_i, compressed_bevs, compressed_depths, compressed_semantics)

            loaded_images.append(images_i)
            loaded_bevs.append(bevs_i)
            loaded_depths.append(depths_i)
            loaded_semantics.append(semantics_i)
            loaded_lidars.append(lidars_i)
            loaded_measurements.append(measurements_i)
            if (backbone == 'geometric_fusion'):
                loaded_lidars_raw.append(lidars_raw_i)

        # measurements holds seq_len+pred_len trajectory paths; grab the future ones
        # (beyond the seq_len used above for lidar alignment) before measurements gets
        # reassigned to the loaded current/past frames below.
        future_measurements = []
        for i in range(self.seq_len, self.seq_len + self.pred_len):
            with open(str(measurements[i], encoding='utf-8'), 'r') as f2:
                future_measurements.append(ujson.load(f2))

        measurements = loaded_measurements

        # print(measurements)

        # load image, only use current frame
        # augment here
        crop_shift = 0
        degree = 0
        rad = np.deg2rad(degree)
        do_augment = self.augment and random.random() > self.inv_augment_prob
        if do_augment:
            degree = (random.random() * 2. - 1.) * self.aug_max_rotation
            rad = np.deg2rad(degree)
            crop_shift = degree / 60 * self.img_width / self.scale # we scale first

        images_i = loaded_images[self.seq_len-1]
        images_i = crop_image_cv2(images_i, crop=self.img_resolution, crop_shift=crop_shift, crop_shift_y=self.img_crop_shift_y)
        # Use the full captured image instead of center-cropping to config.img_resolution -
        # crop_image_cv2 would also transpose HWC -> CHW, so do that directly here since
        # there's no crop window (and therefore no crop_shift) to apply.
        # images_i = np.transpose(images_i, (2, 0, 1))

        bevs_i = load_crop_bev_npy(loaded_bevs[self.seq_len-1], degree)
        
        data['rgb'] = images_i
        data['bev'] = bevs_i

        if self.multitask:
            depths_i = loaded_depths[self.seq_len-1]
            depths_i = get_depth(crop_image_cv2(depths_i[..., None], crop=self.img_resolution, crop_shift=crop_shift, crop_shift_y=self.img_crop_shift_y))

            semantics_i = loaded_semantics[self.seq_len-1]
            semantics_i = self.converter[crop_seg(semantics_i, crop=self.img_resolution, crop_shift=crop_shift, crop_shift_y=self.img_crop_shift_y)]

            data['depth'] = depths_i
            data['semantic'] = semantics_i

        # need to concatenate seq data here and align to the same coordinate
        lidars = []
        if (backbone == 'geometric_fusion'):
            lidars_raw = []
        if (self.use_point_pillars == True):
            lidars_pillar = []

        for i in range(self.seq_len):
            lidar = loaded_lidars[i]
            # transform lidar to lidar seq-1
            lidar = align(lidar, measurements[i], measurements[self.seq_len-1], degree=degree)
            lidar_bev = lidar_to_histogram_features(lidar)
            lidars.append(lidar_bev)

            if (backbone == 'geometric_fusion'):
                # We don't align the raw LiDARs for now
                lidar_raw = loaded_lidars_raw[i]
                lidars_raw.append(lidar_raw)

            if (self.use_point_pillars == True):
                # We want to align the LiDAR for the point pillars, but not voxelize them
                lidar_pillar = deepcopy(loaded_lidars[i])
                lidar_pillar = align(lidar_pillar, measurements[i], measurements[self.seq_len-1], degree=degree)
                lidars_pillar.append(lidar_pillar)

        # NOTE: This flips the ordering of the LiDARs since we only use 1 it does nothing. Can potentially be removed.
        lidar_bev = np.concatenate(lidars[::-1], axis=0)
        if (backbone == 'geometric_fusion'):
            lidars_raw = np.concatenate(lidars_raw[::-1], axis=0)
        if (self.use_point_pillars == True):
            lidars_pillar = np.concatenate(lidars_pillar[::-1], axis=0)

        if (backbone == 'geometric_fusion'):
            curr_bev_points, curr_cam_points = lidar_bev_cam_correspondences(deepcopy(lidars_raw), lidar_pos=self.lidar_pos, camera_pos=self.camera_pos, lidar_vis=lidar_bev, image_vis=images_i, step=index, debug=True)

        # No per-actor bounding-box labels are available for this dataset, so waypoints
        # come directly from our own recorded ego trajectory instead of matching actor
        # IDs across future label frames. Future poses relative to the current frame,
        # in meters, mirror what get_waypoints()+transform_waypoints() did for the ego
        # car previously.
        current_matrix = pose_to_matrix(measurements[self.seq_len-1])
        current_matrix_inv = np.linalg.inv(current_matrix)

        ego_waypoint = np.array([
            (current_matrix_inv @ pose_to_matrix(p))[:2, 3]
            for p in future_measurements
        ])

        # for the augmentation we only need to transform the waypoints for ego car
        degree_matrix = np.array([[np.cos(rad), np.sin(rad)],
                              [-np.sin(rad), np.cos(rad)]])
        ego_waypoint = (degree_matrix @ ego_waypoint.T).T

        # No source of per-actor bounding-box detections for this dataset.
        label_pad = np.zeros((20, 7), dtype=np.float32)

        if(self.use_point_pillars == True):
            # We need to have a fixed number of LiDAR points for the batching to work, so we pad them and save to total amound of real LiDAR points.
            fixed_lidar_raw = np.empty((self.max_lidar_points, 4), dtype=np.float32)
            num_points = min(self.max_lidar_points, lidars_pillar.shape[0])
            fixed_lidar_raw[:num_points, :4] = lidars_pillar
            data['lidar_raw'] = fixed_lidar_raw
            data['num_points'] = num_points

        if (backbone == 'geometric_fusion'):
            data['bev_points'] = curr_bev_points
            data['cam_points'] = curr_cam_points

        data['lidar'] = lidar_bev
        data['label'] = label_pad
        data['ego_waypoint'] = ego_waypoint
        data['theta'] = compute_yaw(measurements[self.seq_len-1])
        data['velocity'] = measurements[self.seq_len-1]["velocity"]["linear"]["x"]
        
        # data['x_command'] = measurements[self.seq_len-1]['x_command']
        # data['y_command'] = measurements[self.seq_len-1]['y_command']

        # target points
        # convert x_command, y_command to local coordinates
        # taken from LBC code (uses 90+theta instead of theta)
        ego_theta = compute_yaw(measurements[self.seq_len-1]) + rad # + rad for augmentation
        ego_x = measurements[self.seq_len-1]["translation"]['x']
        ego_y = measurements[self.seq_len-1]["translation"]['y']

        x_command = measurements[self.seq_len-1]['x_command']
        y_command = measurements[self.seq_len-1]['y_command']
        R = np.array([
                    [np.cos(ego_theta), -np.sin(ego_theta)],
                    [np.sin(ego_theta),  np.cos(ego_theta)]
                    ])
        local_command_point = np.array([x_command-ego_x, y_command-ego_y])
        local_command_point = R.T.dot(local_command_point)
        # data['target_point'] = ego_waypoint[-1]
        data['target_point'] = local_command_point

        # try:
        #     x_command = measurements[self.seq_len-1]['x_command']
        #     y_command = measurements[self.seq_len-1]['y_command']
        #     R = np.array([
        #         [np.cos(ego_theta), -np.sin(ego_theta)],
        #         [np.sin(ego_theta),  np.cos(ego_theta)]
        #         ])
        #     local_command_point = np.array([x_command-ego_x, y_command-ego_y])
        #     local_command_point = R.T.dot(local_command_point)
        #     data['target_point'] = ego_waypoint[-1]
        #     print(local_command_point, data['target_point'])
        # except:
        #     pass
        # finally:
        #     data['target_point'] = ego_waypoint[-1]
            

        # print(data['target_point'])
        
        data['target_point_image'] = draw_target_point(data['target_point'])
        return data

    

def get_depth(data):
    """Compute normalized depth from a (1, H, W) single-channel depth crop, in millimetres."""
    depth_m = data[0].astype(np.float32) / 1000.0
    normalized = np.clip(depth_m, 0.0, 50.0) / 50.0 #Clipping max dpeth to 50.0
    return normalized


def load_crop_bev_npy(bev_array, degree):
    """
    Rotate (augmentation) the BEV occupancy mask.
    Returns a single-channel class-index map: 0 = free, 1 = occupied.

    The local costmap is a rolling-window grid already centered on base_link
    (isaac_sim_local_costmap.yaml: 300x300 px @ 0.05 m/px), so unlike the
    original CARLA BEV target this needs no lidar-offset row shift or crop
    to a smaller window - it's used at its native size (config.bev_resolution).
    """
    bev_array = bev_array.astype(np.float32)
    bev_rotated = rotate(bev_array, degree)

    return (bev_rotated > 0.5).astype(np.int64)



def pose_to_matrix(pose):
    """Convert a {translation: {x,y,z}, rotation: {x,y,z,w}} trajectory record
    (Isaac Sim's ego pose format) into a 4x4 homogeneous ego_matrix, the format
    align() and the waypoint computation expect.
    """
    t = pose['translation']
    q = pose['rotation']
    x, y, z, w = q['x'], q['y'], q['z'], q['w']

    rotation = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)],
    ])

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [t['x'], t['y'], t['z']]
    return matrix

#LiDAR transforms
def align(lidar_0, measurements_0, measurements_1, degree=0):
    """Transform lidar_0 (captured at measurements_0's ego pose) into measurements_1's
    ego frame.
    """
    matrix_0 = pose_to_matrix(measurements_0)
    matrix_1 = pose_to_matrix(measurements_1)

    Tr_lidar_to_base = get_lidar_to_base_transform()
    Tr_base_to_lidar = get_base_to_lidar_transform()

    transform_0_to_1 = Tr_base_to_lidar @ np.linalg.inv(matrix_1) @ matrix_0 @ Tr_lidar_to_base

    # augmentation
    rad = np.deg2rad(degree)
    degree_matrix = np.array([[np.cos(rad), np.sin(rad), 0, 0],
                              [-np.sin(rad), np.cos(rad), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
    transform_0_to_1 = degree_matrix @ transform_0_to_1

    lidar = lidar_0.copy()

    #for intensity
    lidar[:, -1] = 1.

    lidar = transform_0_to_1 @ lidar.T
    lidar = lidar.T
    lidar[:, -1] = lidar_0[:, -1]

    return lidar

def lidar_to_histogram_features(lidar):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 32
        y_meters_max = 16
        xbins = np.linspace(0, x_meters_max, 32*pixels_per_meter+1)
        ybins = np.linspace(-y_meters_max, y_meters_max, 32*pixels_per_meter+1)
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat #in LiDAR frame

    # print("shape of lidar:", lidar.shape)
    below = lidar[lidar[...,2]<=-1.71]
    above = lidar[lidar[...,2]>-1.71]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([above_features, below_features], axis=-1)

    # print("shape of features:", features.shape)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    # print(features.shape)
    features = np.rot90(features, 1, axes=(1,2)).copy() #rotate 90 degrees counter-clockwise
    return features

def scale_image(image, scale):
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    return im_resized

def scale_image_cv2(image, scale):
    (width, height) = (int(image.shape[1] // scale), int(image.shape[0] // scale))
    im_resized = cv2.resize(image, (width, height))
    return im_resized

def crop_image(image, crop=(128, 640), crop_shift=0):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    width = image.width
    height = image.height
    crop_h, crop_w = crop
    start_y = height//2 - crop_h//2
    start_x = width//2 - crop_w//2
    
    # only shift for x direction
    start_x += int(crop_shift)

    image = np.asarray(image)
    cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image

def crop_image_cv2(image, crop=(128, 640), crop_shift=0, crop_shift_y=0):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    width = image.shape[1]
    height = image.shape[0]
    crop_h, crop_w = crop
    start_y = height // 2 - crop_h // 2 + int(crop_shift_y)
    start_x = width // 2 - crop_w // 2

    # only shift for x direction
    start_x += int(crop_shift)

    cropped_image = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    cropped_image = np.transpose(cropped_image, (2, 0, 1))
    return cropped_image


def crop_image_cv2_centered(image, crop=(128, 640), crop_shift=0):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    width = image.shape[1]
    height = image.shape[0]
    crop_h, crop_w = crop
    start_y = height // 2 - crop_h // 2
    start_x = width // 2 - crop_w // 2

    # only shift for x direction
    start_x += int(crop_shift)

    cropped_image = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    cropped_image = np.transpose(cropped_image, (2, 0, 1))
    return cropped_image

def scale_seg(image, scale):
    (width, height) = (int(image.shape[1] / scale), int(image.shape[0] / scale))
    if scale != 1:
        im_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        im_resized = image
    return im_resized

def crop_seg(image, crop=(128, 640), crop_shift=0, crop_shift_y=0):
    """
    Scale and crop a seg image, returning a channels-first numpy array.
    """
    width = image.shape[1]
    height = image.shape[0]
    crop_h, crop_w = crop

    start_y = height//2 - crop_h//2 + int(crop_shift_y)
    start_x = width//2 - crop_w//2
    # only shift for x direction
    start_x += int(crop_shift)

    cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    return cropped_image

def correspondences_at_one_scale(valid_bev_points, valid_cam_points, lidar_x, lidar_y, camera_x, camera_y, scale):
    """
    Compute projections between LiDAR BEV and image space
    """
    cam_to_bev_proj_locs = np.zeros((lidar_x, lidar_y, 5, 2))
    bev_to_cam_proj_locs = np.zeros((camera_x, camera_y, 5, 2))

    tmp_bev = np.empty((lidar_x, lidar_y, ), dtype=object)
    tmp_cam = np.empty((camera_x, camera_y, ), dtype=object)
    for i in range(lidar_x):
        for j in range(lidar_y):
            tmp_bev[i,j] = []

    for i in range(camera_x):
        for j in range(camera_y):
            tmp_cam[i, j] = []

    for i in range(valid_bev_points.shape[0]):
        # print(tmp_bev.shape)
        # print(valid_bev_points[i][0]//scale)
        tmp_bev[valid_bev_points[i][0]//scale, valid_bev_points[i][1]//scale].append(valid_cam_points[i]//scale)
        tmp_cam[valid_cam_points[i][0]//scale, valid_cam_points[i][1]//scale].append(valid_bev_points[i]//scale)

    for i in range(lidar_x):
        for j in range(lidar_y):
            cam_to_bev_points = tmp_bev[i,j]

            if len(cam_to_bev_points) > 5:
                cam_to_bev_proj_locs[i,j] = np.array(random.sample(cam_to_bev_points, 5))
            elif len(cam_to_bev_points) > 0:
                num_points = len(cam_to_bev_points)
                cam_to_bev_proj_locs[i,j,:num_points] = np.array(cam_to_bev_points)

    for i in range(camera_x):
        for j in range(camera_y):
            bev_to_cam_points = tmp_cam[i,j]

            if len(bev_to_cam_points) > 5:
                bev_to_cam_proj_locs[i,j] = np.array(random.sample(bev_to_cam_points, 5))
            elif len(bev_to_cam_points) > 0:
                num_points = len(bev_to_cam_points)
                bev_to_cam_proj_locs[i,j,:num_points] = np.array(bev_to_cam_points)

    return cam_to_bev_proj_locs, bev_to_cam_proj_locs

def lidar_bev_cam_correspondences(world, lidar_pos, camera_pos, lidar_vis=None, image_vis=None, step=None, debug=True):
    """
    Convert LiDAR point cloud to camera co-ordinates

    world: Expects the point cloud from Isaac Sim in the world frame
    lidar_pos: config.lidar_pos, LiDAR mounting position (x, y, z) relative to base_link
    camera_pos: config.camera_pos, camera mounting position (x, y, z) relative to base_link
    lidar_vis: lidar prjected to BEV
    image_vis: RGB input image to the network (CHW), required - its shape defines the
        projection geometry below, not just the debug overlay
    step: current timestep
    debug: Whether to save the debug images. If false only world is required
    """

    #LiDAR
    pixels_per_meter = 8
    lidar_width = 256
    lidar_height = 256
    lidar_meters_x = (lidar_width/pixels_per_meter)  #forward depth (col 0), 0 to lidar_meters_x only
    lidar_meters_y = (lidar_height/pixels_per_meter)/2   # lateral -lidar_y to + lidar_y

    # To be consistent with the network
    downscale_factor = 32
    # Use the real captured image's resolution instead of a hardcoded size - data.py no
    # longer crops rgb to a fixed config.img_resolution, so that size no longer matches.
    img_height, img_width = config.img_resolution


    left_camera_rotation = 90.0
    right_camera_rotation = -90.0

    # calculating fov
    # tan(θ/2) = (aperture/2) / focalLength
    # θ/2 = atan(aperture / (2·focalLength))
    # θ = 2·atan(aperture / (2·focalLength))

    vertical_aperture = 2.459
    horizontal_aperture = 3.86
    focal_length = 1.93*2*2*2*2

    fov_height = 2* np.arctan(vertical_aperture/ 2* focal_length)
    fov_width = 2* np.arctan(horizontal_aperture/ 2*focal_length)

    fov_height = np.rad2deg(fov_height)
    fov_width = np.rad2deg(fov_width)

    # LiDAR and camera are mounted at different positions on base_link (config.lidar_pos
    # vs config.camera_pos, both x-forward/y-left/z-up, no rotation between them) - shift
    # points from the LiDAR's frame into the camera's frame before projecting, or every
    # point ends up offset by the difference between the two mounting positions.
    world[:, :3] = world[:, :3] + (np.array(lidar_pos) - np.array(camera_pos))

    # base_link/ROS convention has +y = left, but the pinhole model below needs
    # +x (image column) = right, so flip the lateral axis before filtering/projecting.
    # world[:, 1] *= -1

    # get valid points in 64x64 grid
    lidar = world[abs(world[:,1])<lidar_meters_y] # 32m to the sides
    lidar = lidar[lidar[:,0]<lidar_meters_x] # 64m to the front
    lidar = lidar[lidar[:,0]>0] # 0m to the back

    # Translate Lidar cloud to the same coordinate system as the cameras (They only differ in height)
    # lidar_z = lidar_pos[2]
    # cam_z = lidar_pos[2]
        
    # lidar[..., 2] = lidar[..., 2] + (lidar_z - cam_z)

    # Make copies because we will rotate the new pointclouds
    lidar_for_left_camera  = deepcopy(lidar)
    lidar_for_right_camera = deepcopy(lidar)

    print(lidar.shape)

    c_x = (img_width/2)
    c_y = (img_height/2) 

    lidar_indices = np.arange(0, lidar.shape[0], 1)
    # Use a pinhole camera model to project the LiDAR points onto the camera image
    z = lidar[..., 0]
    x = ((c_x * lidar[..., 1]) / z) + c_x
    y = ((c_x * lidar[..., 2]) / z) + c_y
    result_center = np.stack([x, y, lidar_indices], 1)

    # Remove points that are outside of the image
    result_center = result_center[np.logical_and(result_center[...,0] > 0, result_center[...,0] < img_width)]
    result_center = result_center[np.logical_and(result_center[...,1] > 0, result_center[...,1] < img_height)]

    result_center_shifted = result_center
    result_center_shifted[..., 0] = result_center_shifted[..., 0] + (img_width / 2.0)

    # Rotate the left camera to align with the axis for projection with a pinhole camera model
    theta = np.radians(left_camera_rotation)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])
    lidar_for_left_camera = R.dot(lidar_for_left_camera.T).T

    # Use a pinhole camera model to project the LiDAR points onto the camera image
    z = lidar_for_left_camera[..., 0]
    x = ((c_x * lidar_for_left_camera[..., 1]) / z) + c_x
    y = ((c_x * lidar_for_left_camera[..., 2]) / z) + c_y
    result_left = np.stack([x, y, lidar_indices], 1)

    # Remove points that are outside of the image
    result_left = result_left[np.logical_and(result_left[...,0] > 0, result_left[...,0] < img_width)]
    result_left = result_left[np.logical_and(result_left[...,1] > 0, result_left[...,1] < img_height)]

    # # We only use half of the left image, so we cut the unneccessary points
    result_left_shifted        = result_left[result_left[...,0] >= (img_width/2.0)]
    result_left_shifted[...,0] = result_left_shifted[...,0] - (img_width/2.0)

    # Do the same for the right image
    theta = np.radians(right_camera_rotation)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])
    lidar_for_right_camera = R.dot(lidar_for_right_camera.T).T

    # Use a pinhole camera model to project the LiDAR points onto the camera image
    z = lidar_for_right_camera[..., 0]
    x = ((c_x * lidar_for_right_camera[..., 1]) / z) + c_x
    y = ((c_x * lidar_for_right_camera[..., 2]) / z) + c_y
    result_right = np.stack([x, y, lidar_indices], 1)

    # Remove points that are outside of the image
    result_right = result_right[np.logical_and(result_right[..., 0] > 0, result_right[..., 0] < img_width)]
    result_right = result_right[np.logical_and(result_right[..., 1] > 0, result_right[..., 1] < img_height)]

    # # We only use half of the left image, so we cut the unneccessary points
    result_right_shifted = result_right[result_right[...,0] < (img_width/2.0)] # Cut of right part, it's not used.
    result_right_shifted[...,0] = result_right_shifted[...,0] + (img_width/2.0) + img_width

    # Combine the three images into one
    results_total = np.concatenate((result_left_shifted, result_center_shifted, result_right_shifted), axis=0)

    if(debug == True):
        # Visualize LiDAR hits in image
        vis = np.zeros([img_height, 2 * img_width])
        vis_bev = np.zeros([lidar_height, lidar_width])
        # image_vis is a single un-batched HWC front-camera image; only the "center"
        # slot of the stitched canvas has a real image, left/right stay blank.
        vis_original_image = np.zeros([img_height, 2 * img_width, 3])
        center_x_offset = int(img_width / 2)
        image_vis_hwc = np.transpose(image_vis, (1, 2, 0))
        vis_original_image[:, center_x_offset:center_x_offset + image_vis_hwc.shape[1]] = image_vis_hwc / 255.0
        vis_original_lidar = np.zeros([lidar_height, lidar_width])
        vis_original_lidar[np.greater(lidar_vis[0], 0)] = 255
        vis_original_lidar[np.greater(lidar_vis[1], 0)] = 255

    valid_bev_points = []
    valid_cam_points = []
    for i in range(results_total.shape[0]):
        # Project the LiDAR point to BEV and save index of the BEV image pixel.
        lidar_index = int(results_total[i, 2])
        bev_y = int((lidar[lidar_index][1] + lidar_meters_y) * pixels_per_meter) *-1
        bev_x = (int(lidar[lidar_index][0] * pixels_per_meter) )
        # bev_y = (int(lidar[lidar_index][0] * pixels_per_meter) - (lidar_height-1)) * -1

        valid_bev_points.append([bev_x, bev_y])
        # Calculate index in the final image by rounding down
        img_x = int(results_total[i][0])
        # The network input images use a top left coordinate system, we need to convert the bottom left coordinates by inverting the y axis
        img_y = (int(results_total[i][1]) )*-1
        valid_cam_points.append([img_x, img_y])


        if (debug == True):
            vis_original_image[img_y, img_x] = np.array([1.0,0.0,0.0])
            vis_bev[bev_y, bev_x] = 255 #Debug visualization
            vis[img_y, img_x] = 255

    if (debug == True):
        # NOTE add the paths you want the images to land in here before debugging
        from matplotlib import pyplot as plt
        plt.ion()
        plt.imshow(vis_bev)
        plt.savefig(r'/workspace/viz/bev_lidar_{}.png'.format(step), bbox_inches='tight')
        plt.close()
        plt.imshow(vis_original_image)
        plt.savefig(r'/workspace/viz/image_with_lidar_{}.png'.format(step), bbox_inches='tight')
        plt.close()
        plt.ioff()


    valid_bev_points = np.array(valid_bev_points)
    valid_cam_points = np.array(valid_cam_points)

    bev_points, cam_points = correspondences_at_one_scale(valid_bev_points, valid_cam_points,  (lidar_width // downscale_factor),
                                                          (lidar_height // downscale_factor), (img_width // downscale_factor) * 2,
                                                          (img_height // downscale_factor), downscale_factor)


    return bev_points, cam_points

def draw_target_point(target_point, color = (255, 255, 255)):
    image = np.zeros((256, 256), dtype=np.uint8)
    target_point = target_point.copy()

    # convert to lidar coordinate
    target_point[0] += 0.115
    point = target_point * 8.
    # point[1] *= -1
    # point[1] = 256 - point[1] 
    point[1] += 128 
    point = point.astype(np.int32)
    point = np.clip(point, 0, 256)
    cv2.circle(image, tuple(point), radius=5, color=color, thickness=3)
    cv2.imwrite("/workspace/viz/check.png", image)
    image = image.reshape(1, 256, 256)
    return image.astype(float) / 255.

def compute_yaw(measurement):
    """Extract yaw (rotation about z, in radians) from a trajectory record's quaternion."""
    q = measurement['rotation']
    x, y, z, w = q['x'], q['y'], q['z'], q['w']
    return np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))

# def plot_lidar_bev(lidar_points, output_path, title=None):
#     """Human-readable BEV plot: above/below split as two labeled heatmaps in real
#     map-frame meters, with a colorbar. render_histogram_bev()'s raw grayscale image
#     is oriented for feeding a model (rot90 + concatenated), not for reading by eye --
#     this uses the same bins/threshold but keeps the natural x/y orientation instead.
#     """
#     pixels_per_meter = 8
#     hist_max_per_pixel = 5
#     x_meters_max = 16
#     y_meters_max = 16
#     xbins = np.linspace(-x_meters_max, x_meters_max, 32 * pixels_per_meter + 1)
#     ybins = np.linspace(-y_meters_max, y_meters_max, 32 * pixels_per_meter + 1)

#     def splat(points):
#         hist = np.histogramdd(points[:, :2], bins=(xbins, ybins))[0]
#         hist = np.clip(hist, 0, hist_max_per_pixel) / hist_max_per_pixel
#         return hist.T  # rows=y, cols=x, matching imshow's (row, col) convention

#     above = lidar_points[lidar_points[:, 2] > -1.71]
#     below = lidar_points[lidar_points[:, 2] <= -1.71]
#     extent = (xbins[0], xbins[-1], ybins[0], ybins[-1])

#     fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
#     im = None
#     for ax, points, label in ((axes[0], above, 'above -1.71m (obstacles)'),
#                                (axes[1], below, 'below -1.71m (ground-level)')):
#         im = ax.imshow(splat(points), origin='lower', extent=extent,
#                         cmap='viridis', vmin=0, vmax=1, aspect='equal')
#         ax.set_title(label)
#         ax.set_xlabel('x (m)')
#     axes[0].set_ylabel('y (m)')
#     fig.colorbar(im, ax=axes, shrink=0.8, label='occupancy (clipped, normalized)')
#     if title:
#         fig.suptitle(title)
#     fig.savefig(output_path, dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     return output_path


if __name__ == "__main__":
    root = "/workspace/data"
    config = GlobalConfig(root_dir=root, setting="all")
    print(config.train_data)
    test_datset = IsaacSimData(root=config.train_data, config=config)
    for i in range(len(test_datset)):
        test_datset.__getitem__(i)

