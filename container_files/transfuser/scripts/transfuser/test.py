"""
Smoke test wiring IsaacSimData (data.py) into LidarCenterNet (model.py): builds one
batch from real recorded routes and runs a single inference forward pass.

Known gaps this test surfaces:
- IsaacSimData.__getitem__ never sets 'target_point', 'target_point_image' or 'speed'
  (data.py:304-305, 320-325 are commented out, and draw_target_point() no longer
  exists in this file). LidarCenterNet.forward_ego requires all three, so this script
  fills them with zero placeholders and prints a warning - real values need to be
  wired up in data.py before this is a meaningful test of those inputs.
- lidar_bev_cam_correspondences (used for backbone == 'geometric_fusion') is called
  with debug=True unconditionally from __getitem__ (data.py:264) and hardcodes its
  output path to /workspace/viz/ (data.py:720-723). We create that directory up front
  so the first dataset access doesn't fail on plt.savefig.
- The default checkpoint (model_ckpt/models_2022/*) is the original CARLA-trained
  TransFuser release, not anything trained on this real-robot/Isaac Sim data - its
  predicted waypoints have no reason to resemble gt_waypoints from this dataset. This
  only verifies that a real checkpoint's weights load and run through the pipeline,
  not that its predictions are meaningful here.
"""
import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
from torch.utils.data import DataLoader

from config import GlobalConfig
from data import IsaacSimData
from model import LidarCenterNet

# config attributes that model_ckpt/*/args.txt may specify and that affect the model's
# architecture (layer counts, extra input channels, etc.) - anything present in args.txt
# overrides the config default so the constructed model's shapes match the checkpoint.
_CONFIG_OVERRIDE_KEYS = [
    'n_layer', 'use_target_point_image', 'use_ground_plane', 'use_point_pillars',
    'img_vert_anchors', 'img_horz_anchors', 'lidar_vert_anchors', 'lidar_horz_anchors',
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', type=str,
                         default=os.path.join(SCRIPT_DIR, '..', '..', 'data'),
                         help='Root dir containing <scenario>/<route> folders '
                              '(default: container_files/transfuser/data)')
    parser.add_argument('--backbone', type=str, default=None,
                         help='Override config.backbone. Ignored if --checkpoint has an '
                              'args.txt, since its backbone is what the weights match.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--viz_dir', type=str, default='/workspace/viz',
                         help='Hardcoded debug-image output dir used by '
                              'lidar_bev_cam_correspondences')
    parser.add_argument('--checkpoint', type=str,
                         default='/workspace/model_ckpt/models_2022/transfuser',
                         help='Path to a trained .pth file, or a directory containing one '
                              'plus args.txt (e.g. model_ckpt/models_2022/<backbone>). '
                              'Pass "" to use random-initialized weights instead.')
    return parser.parse_args()


def resolve_checkpoint(path):
    """Accept either a specific .pth file or a directory containing one - returns
    (weights_path, args_txt_path_or_None), or (None, None) if nothing usable is found."""
    if not path:
        return None, None
    if os.path.isdir(path):
        pth_files = sorted(Path(path).glob('*.pth'))
        if not pth_files:
            return None, None
        weights_path = str(pth_files[0])
        args_dir = path
    elif os.path.isfile(path):
        weights_path = path
        args_dir = os.path.dirname(path)
    else:
        return None, None
    args_path = os.path.join(args_dir, 'args.txt')
    return weights_path, (args_path if os.path.isfile(args_path) else None)


def fill_missing_model_inputs(batch, config, device):
    missing = [k for k in ('target_point', 'target_point_image', 'speed') if k not in batch]
    if missing:
        print(f"[WARNING] batch is missing {missing} - IsaacSimData.__getitem__ never "
              f"populates these (see data.py:304-305, 320-325). Using zero placeholders; "
              f"forward_ego's target-point / velocity inputs are not really being tested.")

    batch_size = batch['rgb'].shape[0]
    target_point = batch.get('target_point', torch.zeros(batch_size, 2))
    target_point_image = batch.get(
        'target_point_image',
        torch.zeros(batch_size, 1, config.lidar_resolution_height, config.lidar_resolution_width),
    )
    ego_vel = batch.get('speed', torch.zeros(batch_size, 1))

    return (target_point.to(device, dtype=torch.float32),
            target_point_image.to(device, dtype=torch.float32),
            ego_vel.to(device, dtype=torch.float32).reshape(-1, 1))


def main():
    args = parse_args()

    weights_path, args_txt_path = resolve_checkpoint(args.checkpoint)
    train_args = {}
    if args.checkpoint and weights_path is None:
        print(f"[WARNING] No checkpoint found at {args.checkpoint} - using random-initialized weights.")
    elif weights_path:
        if args_txt_path:
            with open(args_txt_path) as f:
                train_args = json.load(f)
            print(f"Loaded training args from {args_txt_path}: {train_args}")
        else:
            print(f"No args.txt next to {weights_path} - assuming current config/CLI args match it.")

    config = GlobalConfig(root_dir=args.data_root, setting='all')

    checkpoint_backbone = train_args.get('backbone')
    if checkpoint_backbone and args.backbone and checkpoint_backbone != args.backbone:
        print(f"[WARNING] --backbone={args.backbone} conflicts with the checkpoint's "
              f"backbone={checkpoint_backbone} - using the checkpoint's, since its "
              f"weights won't load into a different architecture.")
    config.backbone = checkpoint_backbone or args.backbone or config.backbone

    for key in _CONFIG_OVERRIDE_KEYS:
        if key in train_args:
            setattr(config, key, train_args[key])

    if not config.train_data:
        raise SystemExit(f"No <scenario>/<route> folders found under {args.data_root}")

    print(f"backbone={config.backbone}")
    print(f"routes={config.train_data}")

    if config.backbone == 'geometric_fusion':
        os.makedirs(args.viz_dir, exist_ok=True)  # data.py hardcodes /workspace/viz

    dataset = IsaacSimData(root=config.train_data, config=config)
    print(f"Dataset length: {len(dataset)} samples")
    if len(dataset) == 0:
        raise SystemExit("Dataset produced 0 samples - a route needs more than "
                          "seq_len + pred_len + 4 lidar frames to contribute any.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = LidarCenterNet(config, device, config.backbone,
                            image_architecture=train_args.get('image_architecture', 'resnet34'),
                            lidar_architecture=train_args.get('lidar_architecture', 'resnet18'),
                            use_velocity=bool(train_args.get('use_velocity', 1))).to(device)

    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint weights from {weights_path}")
        if missing:
            print(f"  missing keys: {missing}")
        if unexpected:
            print(f"  unexpected keys: {unexpected}")
        print("[NOTE] This is the original CARLA-trained TransFuser release, not anything "
              "trained on this real-robot data - don't expect pred_wp to track gt_waypoints; "
              "this only checks that the checkpoint loads and runs through the pipeline.")

    model.eval()

    print(f"Running the full trajectory: {len(dataset)} steps")

    for step, batch in enumerate(loader):
        if step>5:
            break
        if step == 0:
            print("Batch keys:", list(batch.keys()))
            for key, value in batch.items():
                if torch.is_tensor(value):
                    print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")

        rgb = batch['rgb'].to(device, dtype=torch.float32)
        lidar_bev = batch['lidar'].to(device, dtype=torch.float32)
        ego_vel = batch['velocity'].to(device, dtype=torch.float32)
        target_point = batch['target_point'].to(device, dtype=torch.float32)
        target_point_image = batch['target_point_image'].to(device, dtype=torch.float32)
        # target_point, target_point_image, ego_vel = fill_missing_model_inputs(batch, config, device)
        gt_waypoints = batch['ego_waypoint'].to(device, dtype=torch.float32)

        bev_points = cam_points = None
        if config.backbone == 'geometric_fusion':
            bev_points = batch['bev_points'].to(device, dtype=torch.int64)
            cam_points = batch['cam_points'].to(device, dtype=torch.int64)

        with torch.no_grad():
            pred_wp = model.forward_ego(rgb, lidar_bev, target_point, target_point_image,
                                                 ego_vel, bev_points=bev_points, cam_points=cam_points,
                                                 expert_waypoints=gt_waypoints, save_path=args.viz_dir)

        print(f"step {step}/{len(loader)}: pred_wp shape={tuple(pred_wp.shape)}, "
              f"gt last wp={gt_waypoints[0, -1].cpu().numpy()}, "
              f"pred last wp={pred_wp[0, -1].cpu().numpy()}")

    print(f"Smoke test passed: ran the full trajectory ({len(dataset)} steps) through "
          f"dataloader -> model forward pass. Debug images written to {args.viz_dir}")


if __name__ == '__main__':
    main()
