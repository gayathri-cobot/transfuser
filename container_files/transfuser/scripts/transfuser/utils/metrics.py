import numpy as np

#IoU for segmentation mask and BeV
def compute_IoU(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 1.0
    return iou_score

#RMSE 
def compute_RMSE(pred_depth, gt_depth):
    rmse_score = np.sqrt(np.mean((pred_depth - gt_depth) ** 2))
    return rmse_score

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

