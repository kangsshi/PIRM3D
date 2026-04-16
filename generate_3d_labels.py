import os
import cv2
import numpy as np
import glob
from scipy.ndimage import distance_transform_edt, gaussian_filter

DATA_ROOT = '/root/autodl-tmp/3DRadioMap'
OUTPUT_ROOT = os.path.join(DATA_ROOT, 'Precomputed_Labels_3D')
DEPTH_ROOT = os.path.join(DATA_ROOT, 'Precomputed_Depth_3D')
UAV_MASK_PATH = 'uav_hourglass_mask.npy'

GRID = 256
HEIGHTS = 20
MAX_BUILDING_HEIGHT = 19.8
MAX_GLOBAL_DEPTH = 362.6

# ==========================================
# Core Algorithm: Physics-Informed Regression Kriging
# ==========================================
def generate_pseudo_label(gt_3d, known_mask, depth_3d, bldg_mask, sigma=1.0):
    """ Generate pseudo-label via physics-informed regression kriging
    Args:
        gt_3d (np.ndarray): Ground truth 3D radio map of shape (20, 256, 256)
        known_mask (np.ndarray): Boolean mask of known UAV sampling points
        depth_3d (np.ndarray): Physics-guided 3D depth map of shape (20, 256, 256)
        bldg_mask (np.ndarray): Boolean mask of building voxels
        sigma (float): Sigma for Gaussian smoothing
    Returns:
        np.ndarray: Pseudo-label 3D radio map of shape (20, 256, 256)
    """
    # 1. Calculate residual at known points
    residual = np.zeros_like(gt_3d)
    residual[known_mask] = gt_3d[known_mask] - depth_3d[known_mask]

    # 2. Diffuse residual to blind zones via Euclidean Distance Transform
    _, indices = distance_transform_edt(~known_mask, return_indices=True)
    filled_residual = residual[tuple(indices)]

    # 3. Smooth residual with Gaussian filter
    smoothed_residual = gaussian_filter(filled_residual, sigma=sigma)

    # 4. Physics fusion and boundary enforcement
    label = depth_3d + smoothed_residual
    label[known_mask] = gt_3d[known_mask]
    label[bldg_mask] = 0.0

    return np.clip(label, 0.0, 1.0)

# ==========================================
# Single Sample Processing
# ==========================================
def process_single_sample(fpath, split_name, uav_mask):
    """ Process a single radio map sample to generate pseudo-label
    Args:
        fpath (str): Path to the input radio map file
        split_name (str): 'train' or 'test'
        uav_mask (np.ndarray): Predefined UAV sampling mask
    Returns:
        bool: True if processing is successful
    """
    filename = os.path.basename(fpath)
    b_idx = filename.split('_')[0]

    save_dir = os.path.join(OUTPUT_ROOT, split_name)
    save_path = os.path.join(save_dir, filename.replace('.png', '.npz'))

    # 1. Check for existing file (resume processing)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
        return True

    # --- Read Physics-Guided Depth Prior ---
    depth_p = os.path.join(DEPTH_ROOT, split_name, filename.replace('.png', '.npy'))
    if not os.path.exists(depth_p) or os.path.getsize(depth_p) < 1000000:
        return False
    try:
        depth_raw = np.load(depth_p).astype(np.float32)
        depth_3d = np.clip(np.log10(depth_raw + 1.0) / np.log10(MAX_GLOBAL_DEPTH + 1.0), 0.0, 1.0)
    except Exception:
        return False

    # --- Read Ground Truth 3D Radio Map ---
    gt_3d = np.zeros((HEIGHTS, GRID, GRID), dtype=np.float32)
    for z in range(HEIGHTS):
        img_p = os.path.join(DATA_ROOT, split_name, f'h{z + 1}', filename)
        if not os.path.exists(img_p):
            return False
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        gt_3d[z] = img.astype(np.float32) / 255.0

    # --- Extract 3D Building Mask ---
    bldg_3d_mask = np.zeros((HEIGHTS, GRID, GRID), dtype=bool)
    b_path = os.path.join(DATA_ROOT, 'Building_Infomation', 'buildingsWHeight', f'{b_idx}.png')
    b_map = cv2.imread(b_path, cv2.IMREAD_UNCHANGED)
    if b_map is not None:
        if len(b_map.shape) > 2:
            b_map = b_map[:, :, 0]
        b_map_m = b_map.astype(np.float32) / 255.0 * MAX_BUILDING_HEIGHT
        for z in range(HEIGHTS):
            bldg_3d_mask[z] = b_map_m >= (z + 1.0)

    # --- Apply UAV Mask and Exclude Building Voxels ---
    mask_uav = uav_mask.copy()
    mask_uav[bldg_3d_mask] = False

    # --- Generate Pseudo-Label ---
    pseudo_label = generate_pseudo_label(gt_3d, mask_uav, depth_3d, bldg_3d_mask, sigma=1.0)

    # --- Compress to uint8 and Save ---
    pseudo_label_uint8 = (pseudo_label * 255.0).clip(0, 255).astype(np.uint8)
    np.savez_compressed(save_path, y2=pseudo_label_uint8)
    return True

# ==========================================
# Main Processing Pipeline
# ==========================================
def run_generation(split_name):
    """ Run pseudo-label generation for the entire dataset split
    Args:
        split_name (str): 'train' or 'test'
    """
    print(f"\nGenerating pseudo-labels for: {split_name.upper()} ...")
    src_dir = os.path.join(DATA_ROOT, split_name, 'h1')
    png_files = glob.glob(os.path.join(src_dir, '*.png'))
    os.makedirs(os.path.join(OUTPUT_ROOT, split_name), exist_ok=True)

    uav_mask = np.load(UAV_MASK_PATH).astype(bool)

    print(f"Total files to process: {len(png_files)}")

    for i, fpath in enumerate(png_files):
        process_single_sample(fpath, split_name, uav_mask)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(png_files)} files")

if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    run_generation('test')
    run_generation('train')
    print("\nPseudo-label generation process finished successfully!")