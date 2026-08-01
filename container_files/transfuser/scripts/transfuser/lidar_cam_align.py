"""
Standalone LiDAR-to-camera projection check for a single recorded frame.

Loads one timestamp's LiDAR + front-camera image directly from a route folder
(bypassing IsaacSimData - no seq_len alignment, augmentation, bev/depth/semantic
loading needed for a single frame) and reuses the actual projection logic from
data.py's lidar_bev_cam_correspondences (the extrinsics/sign-convention fixes we
made there apply here automatically).

Usage:
    python lidar_cam_align.py
    python lidar_cam_align.py --route /path/to/<scenario>/<route> --timestamp 1234567890
"""
import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import cv2
import numpy as np

from config import GlobalConfig
from data import lidar_bev_cam_correspondences, lidar_to_histogram_features


def find_default_route(data_root):
    """Pick the first <scenario>/<route> folder under data_root that has lidar frames."""
    data_root = Path(data_root)
    for scenario_dir in sorted(data_root.iterdir()):
        if not scenario_dir.is_dir():
            continue
        for route_dir in sorted(scenario_dir.iterdir()):
            if (route_dir / "lidar").is_dir() and any((route_dir / "lidar").glob("*.npy")):
                return route_dir
    raise SystemExit(f"No route with a lidar/ folder found under {data_root}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', type=str,
                         default=os.path.join(SCRIPT_DIR, '..', '..', 'data'),
                         help='Root dir containing <scenario>/<route> folders, used only '
                              'if --route is not given')
    parser.add_argument('--route', type=str, default=None,
                         help='Path to a specific <scenario>/<route> folder '
                              '(default: first one found under --data_root)')
    parser.add_argument('--timestamp', type=str, default=None,
                         help='Frame stem (filename without extension) to load '
                              '(default: the frame in the middle of the route)')
    parser.add_argument('--viz_dir', type=str, default='/workspace/viz',
                         help='Hardcoded debug-image output dir used by '
                              'lidar_bev_cam_correspondences')
    return parser.parse_args()


def main():
    args = parse_args()

    route_dir = Path(args.route) if args.route else find_default_route(args.data_root)
    print(f"Route: {route_dir}")

    frame_stems = sorted(p.stem for p in (route_dir / "lidar").glob("*.npy"))
    if not frame_stems:
        raise SystemExit(f"No lidar frames found under {route_dir / 'lidar'}")

    stem = args.timestamp if args.timestamp else frame_stems[80]
    if stem not in frame_stems:
        raise SystemExit(f"Timestamp {stem} not found under {route_dir / 'lidar'}")
    print(f"Timestamp: {stem}")

    # Load the raw LiDAR point cloud (x, y, z only - no need for align()'s homogeneous
    # 4th column since we're only projecting a single frame, not aligning across frames).
    lidar_points = np.load(str(route_dir / "lidar" / f"{stem}.npy"), allow_pickle=True)
    lidar_xyz = np.stack(
        [lidar_points['x'], lidar_points['y'], lidar_points['z']], axis=-1
    ).astype(np.float32)
    print(f"Loaded {lidar_xyz.shape[0]} lidar points")

    # Load the matching front-camera image. cv2 loads BGR; lidar_bev_cam_correspondences
    # expects CHW (see data.py's own image_vis handling).
    image_path = route_dir / "rgb" / "front" / f"{stem}.png"
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit(f"Could not load image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_chw = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32)
    print(f"Loaded image: shape={image_chw.shape}")

    # lidar_bev_cam_correspondences also needs a BEV histogram for its (unused-by-us but
    # unconditionally accessed) debug overlay - build it the same way data.py does.
    lidar_bev = lidar_to_histogram_features(lidar_xyz)

    config = GlobalConfig(root_dir=str(route_dir), setting='eval')

    if not os.path.isdir(args.viz_dir):
        os.makedirs(args.viz_dir, exist_ok=True)

    lidar_bev_cam_correspondences(
        lidar_xyz.copy(),
        lidar_pos=config.lidar_pos,
        camera_pos=config.camera_pos,
        lidar_vis=lidar_bev,
        image_vis=image_chw,
        step=stem,
        debug=True,
    )

    print(f"Wrote {args.viz_dir}/image_with_lidar_{stem}.png and "
          f"{args.viz_dir}/bev_lidar_{stem}.png")


if __name__ == '__main__':
    main()
