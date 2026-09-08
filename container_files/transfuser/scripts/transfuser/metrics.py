import numpy as np
import torch
from torchmetrics.functional import jaccard_index, mean_squared_error
# pip install similaritymeasures
import similaritymeasures


#IoU for segmentation mask and BeV. 
def compute_IoU(pred_mask, gt_mask):
    pred = torch.as_tensor(np.asarray(pred_mask), dtype=torch.long)
    gt = torch.as_tensor(np.asarray(gt_mask), dtype=torch.long)
    scores = []
    for p, g in zip(pred, gt):
        if p.sum() == 0 and g.sum() == 0: # union empty -> torchmetrics has no fixed convention here, original code defined this as a perfect match
            scores.append(1.0)
        else:
            scores.append(jaccard_index(p, g, task="binary").item())
    return float(np.mean(scores)), float(np.std(scores))

#RMSE for depth estimation
def compute_RMSE(pred_depth, gt_depth):
    pred = torch.as_tensor(np.asarray(pred_depth), dtype=torch.float32)
    gt = torch.as_tensor(np.asarray(gt_depth), dtype=torch.float32)
    scores = [mean_squared_error(p, g, squared=False).item() for p, g in zip(pred, gt)]
    return float(np.mean(scores)), float(np.std(scores))

#fretchet distance for trajectory prediction
def compute_fretchet(pred_path, gt_path):
    pred = np.asarray(pred_path, dtype=float)
    gt = np.asarray(gt_path, dtype=float)
    fretchet_score = [similaritymeasures.frechet_dist(p, g) for p, g in zip(pred, gt)]
    return float(np.mean(fretchet_score)), float(np.std(fretchet_score))

#ADE for trajectory prediction
def compute_ADE(pred_path, gt_path):
    pred = np.asarray(pred_path, dtype=float)
    gt = np.asarray(gt_path, dtype=float)
    ade_score = [np.mean(np.linalg.norm(p - g, axis=1)) for p, g in zip(pred, gt)]
    return float(np.mean(ade_score)), float(np.std(ade_score))

#SPL for navigation
def compute_SPL(pred_path, gt_path, success):
    if not success:
        return 0.0
    path_length = len(pred_path)
    optimal_length = len(gt_path)
    spl_score = optimal_length / path_length if path_length != 0 else 0.0
    return spl_score

# compute yaw standard deviation
def compute_yaw_stats(waypoints):
    pts = np.asarray(waypoints, dtype=float)
    deltas = np.diff(pts, axis=0)  # (dx, dy) per segment
    yaw = np.arctan2(deltas[:, 1], deltas[:, 0])  # radians, one per segment

    # Unwrap to avoid false jumps at the -pi/pi boundary
    yaw_unwrapped = np.unwrap(yaw)

    # Std of absolute heading
    yaw_std = np.std(yaw_unwrapped)

    # Turning rate (change in yaw between consecutive segments)
    turning_rate = np.diff(yaw_unwrapped)
    turning_rate_std = np.std(turning_rate)

    return {
        "yaw_rad": yaw_unwrapped,
        "yaw_deg": np.degrees(yaw_unwrapped),
        "yaw_std_rad": yaw_std,
        "yaw_std_deg": np.degrees(yaw_std),
        "turning_rate_std_rad": turning_rate_std,
        "turning_rate_std_deg": np.degrees(turning_rate_std),
    }

