import numpy as np

def get_base_to_lidar_transform():
    """
    Get the transformation matrix from base frame to LiDAR frame.
    """
    # Assuming the LiDAR is mounted at the front of the vehicle, facing forward.
    # This is a placeholder; in practice, you would get this from calibration data.
    translation = np.array([0.115, 0.0, 1.71])  # x, y, z in meters
    rotation = np.eye(3)  # No rotation

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation
    transform_matrix[:3, 3] = translation

    return transform_matrix

def get_lidar_to_base_transform():
    """
    Get the transformation matrix from LiDAR frame to base frame.
    """
    # This is the inverse of the base to LiDAR transform.
    base_to_lidar = get_base_to_lidar_transform()
    lidar_to_base = np.linalg.inv(base_to_lidar)
    return lidar_to_base

# temp
def get_lidar_to_bevimage_transform():
    # rot 
    T = np.array([[0, -1, 16],
                  [-1, 0, 32],
                  [0, 0, 1]], dtype=np.float32)
    # scale 
    T[:2, :] *= 8

    return T

def normalize_angle_degree(x):
    x = x % 360.0
    if (x > 180.0):
        x -= 360.0
    return x