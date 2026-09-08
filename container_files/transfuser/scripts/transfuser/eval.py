"""
Evaluate the model and document the results.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import GlobalConfig
from data import IsaacSimData, draw_target_point
from model import LidarCenterNet



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', type=str,
                         default=os.path.join(SCRIPT_DIR, '..', '..', 'data'),
                         help='Root dir containing <scenario>/<route> folders '
                              '(default: container_files/transfuser/data)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--viz_dir', type=str, default='/workspace/viz',
                         help='Hardcoded debug-image output dir used by '
                              'lidar_bev_cam_correspondences')
    parser.add_argument('--checkpoint', type=str,
                         default='/workspace/model_ckpt/models_2022/transfuser',
                         help='Path to a trained .pth file, or a directory containing one '
                              'plus args.txt (e.g. model_ckpt/models_2022/<backbone>). '
                              'Pass "" to use random-initialized weights instead.')
    parser.add_argument('--eval_scenario', type=str, default='scenario_1', help="The scenario you would like to evaluate")
    parser.add_argument('--eval_route', type=str, default='1', help="The scenario you would like to evaluate")
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
        print(f"Found checkpoint {weights_path} in directory {path}")
    elif os.path.isfile(path):
        weights_path = path
        print(f"Using checkpoint {weights_path}")
        args_dir = os.path.dirname(path)
    else:
        return None, None
    args_path = os.path.join(args_dir, 'args.txt')
    return weights_path, (args_path if os.path.isfile(args_path) else None)


def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def plot_trajectory_comparison(pred_points, gt_points, viz_dir):
    """Trace the predicted and ground-truth trajectories as lines """
    pred = np.asarray(pred_points)
    gt = np.asarray(gt_points)

    fig, ax = plt.subplots(figsize=(6, 6))
    
    # ax.scatter(gt[:, 0], gt[:, 1], marker='o', s=15, color='tab:green', alpha=0.6, label='ground truth')
    # ax.scatter(pred[:, 0], pred[:, 1], marker='x', s=15, color='tab:red', alpha=0.6, label='predicted')
    ax.plot(gt[:, 0], gt[:, 1], '-o', markersize=3, linewidth=1.5,
            color='#2EC37F', alpha=0.9, label='ground truth')
    ax.plot(pred[:, 0], pred[:, 1], '-x', markersize=4, linewidth=1.5,
            color='#7B6EE8', alpha=0.9, label='predicted')
    # Mark where both paths begin, so the direction of travel is unambiguous.
    ax.plot(0, 0, marker='*', markersize=12, color='#444',
            linestyle='none', label='start')
    ax.set_xlabel('x (m, relative to first step)')
    ax.set_ylabel('y (m, relative to first step)')
    ax.set_title('Predicted vs ground-truth waypoints (anchored on first step)')
    ax.legend()
    ax.set_aspect('equal', adjustable='datalim')

    plot_path = os.path.join(viz_dir, 'trajectory_comparison.png')
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


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

    config = GlobalConfig(root_dir=args.data_root, setting='eval', eval_scenario=args.eval_scenario, eval_route=args.eval_route)

    checkpoint_backbone = train_args.get('backbone')
    config.backbone = checkpoint_backbone
    config.use_target_point_image = bool(train_args.get('use_target_point_image', 1))
    config.n_layer = train_args.get('n_layer', config.n_layer)
    config.use_point_pillars = bool(train_args.get('use_point_pillars', 0))

    if not config.eval_data:
        raise SystemExit(f"No <scenario>/<route> folders found under {args.data_root}")

    print(f"backbone={config.backbone}")
    print(f"routes={config.eval_data}")

    os.makedirs(args.viz_dir, exist_ok=True)  # data.py also hardcodes /workspace/viz for geometric_fusion

    dataset = IsaacSimData(root=config.eval_data, config=config)
    print(f"Dataset length: {len(dataset)} samples")
    if len(dataset) == 0:
        raise SystemExit("Dataset produced 0 samples - a route needs more than "
                          "seq_len + pred_len + 4 lidar frames to contribute any.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(train_args.get('image_architecture', 'regnety_032'))

    model = LidarCenterNet(config, device,
                            image_architecture=train_args.get('image_architecture', 'regnety_032'),
                            lidar_architecture=train_args.get('lidar_architecture', 'regnety_032'),
                            use_velocity=bool(train_args.get('use_velocity', 0))).to(device)

    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        if all(k.startswith('module.') for k in state_dict):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint weights from {weights_path}")
        if missing:
            print(f"  missing keys: {missing}")
        if unexpected:
            print(f"  unexpected keys: {unexpected}")

    model.eval()

    print(f"Running the full trajectory: {len(dataset)} steps")

    #Anchor the frame on the first step, use only odom frame
    theta0 = prev_theta = None
    prev_gt_local = None
    prev_pred_local = None
    anchor_pos = np.zeros(2)
    gt_anchor_pos = np.zeros(2) # undrifted twin of anchor_pos, chained from logged waypoints only - the goal doesn't drift with the model's predictions
    pred_points_anchored = []
    gt_points_anchored = []
    eval_metrics = {'bev_iou': [], 'seg_iou': [], 'depth_rmse': [], 'similarity': []}

    for step, batch in enumerate(loader):
        if step == 0:
            print("Batch keys:", list(batch.keys()))
            for key, value in batch.items():
                if torch.is_tensor(value):
                    print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")

        rgb = batch['rgb'].to(device, dtype=torch.float32)
        lidar_bev = batch['lidar'].to(device, dtype=torch.float32)
        ego_vel = batch['velocity'].to(device, dtype=torch.float32).reshape(-1, 1)

        gt_waypoints = batch['ego_waypoint'].to(device, dtype=torch.float32)

        theta = float(batch['theta'][0])
        if theta0 is None:
            theta0 = theta
        else:
            anchor_pos = anchor_pos + rotation_matrix(prev_theta - theta0) @ prev_pred_local[0]
            gt_anchor_pos = gt_anchor_pos + rotation_matrix(prev_theta - theta0) @ prev_gt_local[0]
        R = rotation_matrix(theta - theta0)

        # Move to anchor frame
        goal_anchored = gt_anchor_pos + R @ batch['target_point'][0].cpu().numpy()

        # Anchor-frame goal 
        goal_local = R.T @ (goal_anchored - anchor_pos)
        batch_size = rgb.shape[0]
        target_point = torch.from_numpy(goal_local).to(device, dtype=torch.float32) \
                            .reshape(1, 2).repeat(batch_size, 1)
        target_point_image = torch.from_numpy(draw_target_point(goal_local)) \
                                  .to(device, dtype=torch.float32) \
                                  .unsqueeze(0).repeat(batch_size, 1, 1, 1)

        bev_gt = batch['bev'].to(device, dtype=torch.long)
        depth_gt = batch['depth'].to(device, dtype=torch.float32)
        semantic_gt = batch['semantic'].squeeze(1).to(device, dtype=torch.long)

        with torch.no_grad():
            pred_wp, step_metrics = model.forward_ego(rgb, lidar_bev, target_point, target_point_image,
                                                 ego_vel=ego_vel, expert_waypoints=gt_waypoints, save_path=args.viz_dir,
                                                 eval=True, bev_gt=bev_gt, depth=depth_gt, semantic=semantic_gt)
        for key in eval_metrics:
            eval_metrics[key].append(step_metrics[f'{key}_mean'])

        gt_local = gt_waypoints[:,0,:].cpu().numpy()
        pred_local = pred_wp[:,0,:].cpu().numpy()

        gt_points_anchored.extend((anchor_pos[:, None] + R @ gt_local.T).T)
        pred_points_anchored.extend((anchor_pos[:, None] + R @ pred_local.T).T)

        prev_theta = theta
        prev_gt_local = gt_local
        prev_pred_local = pred_local

    plot_path = plot_trajectory_comparison(pred_points_anchored, gt_points_anchored, args.viz_dir)
    eval_metrics_summary = {}
    for k, v in eval_metrics.items():
        eval_metrics_summary[f'{k}_mean'] = float(np.mean(v))
        eval_metrics_summary[f'{k}_std'] = float(np.std(v))
    print(f"Evaluation metrics summary: {eval_metrics_summary}")

    print(f"Evaluated the full trajectory ({len(dataset)} steps) through "
          f"dataloader -> model forward pass. Debug images written to {args.viz_dir}")
    print(f"Predicted-vs-ground-truth trajectory plot saved to {plot_path}")


if __name__ == '__main__':
    main()
