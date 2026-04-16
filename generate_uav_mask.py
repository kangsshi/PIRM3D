import numpy as np
from scipy.ndimage import binary_dilation


def generate_custom_uav_mask():
    """ Generate a custom UAV sampling mask with hourglass spiral trajectory
    The mask is designed to simulate realistic UAV sampling in urban 3D airspace,
    with a target volume ratio of 20%.

    Returns:
        np.ndarray: Boolean mask of shape (20, 256, 256), where True indicates sampled voxels
    """
    HEIGHTS = 20
    GRID = 256
    mask = np.zeros((HEIGHTS, GRID, GRID), dtype=np.float32)

    num_points = 200000
    t = np.linspace(0, 1, num_points)

    # 1. Allow sampling across full altitude range from h1(0) to h20(19)
    z_continuous = t * (HEIGHTS - 1)
    z_idx = np.clip(np.round(z_continuous).astype(int), 0, HEIGHTS - 1)

    # 2. Hourglass-shaped radius (larger at low/high altitudes, smaller in the middle)
    mid_z = (HEIGHTS - 1) / 2.0
    r_max = (GRID // 2) - 15
    r_min = (GRID // 5)
    dist_from_mid = np.abs(z_continuous - mid_z) / mid_z
    radius = r_min + (r_max - r_min) * dist_from_mid

    # 3. Spiral trajectory
    num_turns = 15
    theta = t * num_turns * 2 * np.pi
    x = (GRID // 2) + radius * np.cos(theta)
    y = (GRID // 2) + radius * np.sin(theta)
    x_idx = np.clip(np.round(x).astype(int), 0, GRID - 1)
    y_idx = np.clip(np.round(y).astype(int), 0, GRID - 1)

    # 4. Intermittent sampling (high frequency, short pulse sampling)
    gap_frequency = 80
    is_sampling = np.sin(t * gap_frequency * 2 * np.pi) > 0.0

    valid_z, valid_y, valid_x = z_idx[is_sampling], y_idx[is_sampling], x_idx[is_sampling]
    mask[valid_z, valid_y, valid_x] = 1.0

    # 5. Dilate to exactly 20% volume ratio
    target_voxels = int(HEIGHTS * GRID * GRID * 0.20)
    struct = np.ones((3, 3, 3), dtype=bool)

    while mask.sum() < target_voxels:
        mask = binary_dilation(mask, structure=struct).astype(np.float32)

    # Trim excess points
    excess = int(mask.sum() - target_voxels)
    if excess > 0:
        ones_idx = np.argwhere(mask == 1.0)
        drop_indices = np.random.choice(len(ones_idx), excess, replace=False)
        coords_to_drop = tuple(ones_idx[drop_indices].T)
        mask[coords_to_drop] = 0.0

    print(f"Successfully generated UAV sampling mask, volume ratio: {mask.sum() / (HEIGHTS * GRID * GRID) * 100:.2f}%")
    return mask.astype(bool)


if __name__ == "__main__":
    uav_mask = generate_custom_uav_mask()
    np.save('uav_hourglass_mask.npy', uav_mask)
    print("File saved as: uav_hourglass_mask.npy")