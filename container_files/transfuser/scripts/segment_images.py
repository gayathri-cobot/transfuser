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
import torch.nn.functional as F

def _largest_reachable_component(floor):
    """Keep only the floor component the robot can actually drive to.

    Seeded from the strip of ground directly in front of the camera (bottom of the
    frame, horizontally centred), which is floor whenever the robot is on floor.
    Returns an all-False mask if that strip contains no floor, so the caller can
    count and skip those frames rather than emit a blank label.
    """
    n_labels, labels = cv2.connectedComponents(floor.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(floor)

    h, w = floor.shape
    seed = labels[int(0.92 * h):, int(0.35 * w):int(0.65 * w)]
    seed = seed[seed > 0]
    if seed.size == 0:
        return np.zeros_like(floor)

    return labels == np.bincount(seed).argmax()

def _run_yaw_deg(output_path):
    """Yaw of the run in degrees from the first trajectory sample, or None.

    """
    traj_dir = os.path.join(output_path, 'trajectory')
    files = sorted(glob.glob(os.path.join(traj_dir, '*.json')))
    if not files:
        return None

    with open(files[0]) as fh:
        sample = json.load(fh)
    q = sample['rotation']

    # Standard quaternion -> yaw. Only the z/w terms matter for a robot on the
    # floor, but the full form costs nothing and survives a bit of roll/pitch.
    siny_cosp = 2.0 * (q['w'] * q['z'] + q['x'] * q['y'])
    cosy_cosp = 1.0 - 2.0 * (q['y'] ** 2 + q['z'] ** 2)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp)) % 360.0


def _snap_yaw_deg(yaw_deg):
    """Nearest multiple of 90, in [0, 360).
    """
    return int(round(yaw_deg / 90.0) * 90) % 360


def save_segmented_images(output_path):
    """Segment images into occupied and free space using a pre-trained model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640", use_fast=True)
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640").to(device)
    

    # Resolve ADE20K ids by name from the checkpoint's own label set, so swapping in a
    # different model can't silently shift them out from under us.
    name2id = {v: k for k, v in model.config.id2label.items()}
    # FLOOR_CLASSES = 'floor'
    FLOOR_CLASSES = [name2id['floor']]

    LEFT_CROP_FRAC = 0.25

    # Writes a barrier overlay next to each mask for eyeballing what the cut is using.
    SAVE_DEBUG_OVERLAY = True

    rgb_dir = os.path.join(output_path, 'rgb', 'front')  # Assuming you want to segment the front camera images
    depth_dir = os.path.join(output_path, 'depth', 'front')  # Assuming you want to segment the front camera images
    segmented_dir = os.path.join(output_path, 'segmented', 'front')
    os.makedirs(segmented_dir, exist_ok=True)

    # Read once, outside the loop: the heading does not change which crop applies
    # part way through a run.
    raw_yaw = _run_yaw_deg(output_path)
    if raw_yaw is None:
        yaw = None
        print(f"[warn] no trajectory json under "
              f"{os.path.join(output_path, 'trajectory')} - falling back to the "
              f"diagonal-only filter")
    else:
        yaw = _snap_yaw_deg(raw_yaw)
        print(f"[info] run yaw {raw_yaw:.1f} deg -> snapped to {yaw} deg")

    n_saved = 0
    n_empty = 0

    for img_file in glob.glob(os.path.join(rgb_dir, '*.png')):
        img = Image.open(img_file).convert("RGB")
        inputs = image_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        FLOOR_CONF = 0.85

        # Argmax version, replaced by the thresholded one below:
        # predicted = image_processor.post_process_semantic_segmentation(
        # outputs, target_sizes=[img.size[::-1]]  # (H, W); PIL .size is (W, H)
        # )[0]
        # pred = predicted.cpu().numpy()
        # floor = np.isin(pred, FLOOR_CLASSES)
        # barrier = np.isin(pred, BARRIER_CLASSES)

        probs = outputs.logits.softmax(dim=1)[0]
  
        groups = probs[FLOOR_CLASSES].sum(0)[None, None]
        groups = F.interpolate(groups, size=img.size[::-1],  # (H, W); PIL .size is (W, H)
                                       mode='bilinear', align_corners=False)
        floor_prob = groups[0, 0].cpu().numpy()
        floor = floor_prob > FLOOR_CONF

        floor_mask = _largest_reachable_component(floor)

        if yaw in (0, 90):
            #define the filter
            fh, fw = floor_mask.shape
            xs = np.arange(fw)[None, :]
            ys = np.arange(fh)[:, None]

            # Left of the vertical line.
            left_of_line = xs < int(LEFT_CROP_FRAC * fw)
            above_diag = ys < (fh - 1) * (1.0 - xs / max(fw - 1, 1))

            filter = left_of_line | above_diag

            floor_mask = floor_mask & ~filter

        else:

            # Alternate filter for the other set of images, matching the single line
            # the overlay draws below. That diagonal runs (0, 0) -> (fw-1, fh-1), so
            # the row on the line at column x is (fh-1) * x/(fw-1). 'Above' is the
            # smaller row index, which for this diagonal is the upper right triangle.
            fh, fw = floor_mask.shape
            xs = np.arange(fw)[None, :]
            ys = np.arange(fh)[:, None]

            above_diag = ys < (fh - 1) * (xs / max(fw - 1, 1))

            filter = above_diag

            floor_mask = floor_mask & ~filter


        if not floor_mask.any():
            n_empty += 1

        if SAVE_DEBUG_OVERLAY:
            overlay = np.array(img)
            # opaque = barrier & ~glass
            overlay[floor_mask] = (0.5 * overlay[floor_mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
            oh, ow = overlay.shape[:2]
            if yaw in (0, 90):
                cv2.line(overlay, (0, oh - 1),
                     (ow-1, 0), (255, 0, 255), 2)
            else:
                cv2.line(overlay, (0, 0),
                     (ow-1, oh - 1), (255, 0, 255), 2)
            Image.fromarray(overlay).save(
                os.path.join(segmented_dir, 'overlay_' + os.path.basename(img_file)))

        # Convert segmentation to an RGB image for visualization
        segmented_img = Image.fromarray((floor_mask.astype(np.uint8) * 255), mode='L')
        segmented_img.save(os.path.join(segmented_dir, os.path.basename(img_file)))
        n_saved += 1

    expected = len(glob.glob(os.path.join(rgb_dir, '*.png')))
    print(f"[info] saved {n_saved}/{expected} segmented images -> {segmented_dir}")
    if n_empty:
        print(f"[warn] {n_empty}/{n_saved} masks are empty (no floor found in front of "
              f"the robot) - check camera tilt or the seed window in "
              f"_largest_reachable_component")

def main():
    parser = argparse.ArgumentParser(description="Segment images into occupied and free space using a pre-trained model.")
    parser.add_argument("data_path", type=str, help="Path to the data directory containing the bag files.")

    args = parser.parse_args()
    save_segmented_images(args.data_path)

if __name__ == "__main__":
    main()
