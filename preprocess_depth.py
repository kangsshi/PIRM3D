import os
import cv2
import numpy as np
import glob
import re

# =========================================================
# Global Configuration
# =========================================================
DATA_ROOT = '/root/autodl-tmp/3DRadioMap'
OUTPUT_ROOT = os.path.join(DATA_ROOT, 'Precomputed_Depth_3D')

FREQ_MHZ = 5900.0
TX_Z = 1.5
MAX_BUILDING_HEIGHT = 19.8
MAP_SIZE = 256

# Target altitude layers: 1.0 to 20.0 meters (20 layers in total)
TARGET_HEIGHTS = np.arange(1.0, 21.0, 1.0)

# =========================================================
# Core Algorithm
# =========================================================
def calculate_radio_depth(building_map, tx_col, tx_row, tx_z, rx_z, freq_mhz):
    """ Calculate physics-guided radio depth map for a single altitude layer
    Args:
        building_map (np.ndarray): Building height map of shape (256, 256)
        tx_col (int): Transmitter column index
        tx_row (int): Transmitter row index
        tx_z (float): Transmitter altitude in meters
        rx_z (float): Receiver altitude in meters
        freq_mhz (float): Carrier frequency in MHz
    Returns:
        np.ndarray: Radio depth map of shape (256, 256)
    """
    rows, cols = building_map.shape
    depth_map = np.zeros((rows, cols), dtype=np.float32)

    C = 135.0
    alpha = 20.0
    beta = 20.0
    term_freq = beta * np.log10(freq_mhz)

    for r in range(rows):
        for c in range(cols):
            dist_xy = np.sqrt((r - tx_row) ** 2 + (c - tx_col) ** 2)
            dist_z = abs(rx_z - tx_z)
            dist_3d = np.sqrt(dist_xy ** 2 + dist_z ** 2)

            if dist_3d < 1.0:
                path_loss = 0.0
            else:
                path_loss = alpha * np.log10(dist_3d)

            total_samples = 0
            blocked_samples = 0
            steps = int(np.ceil(dist_xy))
            if steps == 0:
                steps = 1

            for k in range(steps + 1):
                t = k / steps
                cur_col = tx_col + (c - tx_col) * t
                cur_row = tx_row + (r - tx_row) * t
                cur_z = tx_z + (rx_z - tx_z) * t

                idx_c = int(round(cur_col))
                idx_r = int(round(cur_row))

                if 0 <= idx_r < rows and 0 <= idx_c < cols:
                    total_samples += 1
                    b_h = building_map[idx_r, idx_c]
                    if cur_z < (b_h - 0.1):
                        blocked_samples += 1

            if total_samples > 0:
                T = (total_samples - blocked_samples) / total_samples
            else:
                T = 1.0

            val = C - term_freq - path_loss
            if val < 0:
                val = 0
            depth_map[r, c] = T * val

    return depth_map

def parse_tx_coordinate(filename, map_height=256):
    """ Parse transmitter coordinates from filename
    Args:
        filename (str): Name of the radio map file
        map_height (int): Size of the map grid
    Returns:
        tuple: (ds_x, ds_y, tx_row, tx_col)
    """
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    match_x = re.search(r'X(\d+)', name)
    match_y = re.search(r'Y(\d+)', name)
    ds_x = int(match_x.group(1)) if match_x else 0
    ds_y = int(match_y.group(1)) if match_y else 0
    tx_row = (map_height - 1) - ds_x
    tx_col = ds_y
    return ds_x, ds_y, tx_row, tx_col

# =========================================================
# Single File Processing
# =========================================================
def process_single_file_3d(fpath, split_name):
    """ Process a single radio map file to generate 3D depth volume
    Args:
        fpath (str): Path to the input radio map file
        split_name (str): 'train' or 'test'
    Returns:
        bool: True if processing is successful
    """
    filename = os.path.basename(fpath)
    save_dir = os.path.join(OUTPUT_ROOT, split_name)
    save_name = filename.replace('.png', '.npy')
    save_path = os.path.join(save_dir, save_name)

    # 1. Check for existing file (resume processing)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return True

    # 2. Read coordinates and building map
    ds_x, ds_y, tx_row, tx_col = parse_tx_coordinate(filename, MAP_SIZE)
    b_idx = filename.split('_')[0]
    b_path = os.path.join(DATA_ROOT, 'Building_Infomation', 'buildingsWHeight', f'{b_idx}.png')

    if os.path.exists(b_path):
        b_map_raw = cv2.imread(b_path, cv2.IMREAD_UNCHANGED)
        if len(b_map_raw.shape) > 2:
            b_map_raw = b_map_raw[:, :, 0]
        b_map = b_map_raw.astype(np.float32) / 255.0 * MAX_BUILDING_HEIGHT
    else:
        b_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)

    # 3. Initialize 3D depth volume (20, 256, 256)
    depth_volume = np.zeros((len(TARGET_HEIGHTS), MAP_SIZE, MAP_SIZE), dtype=np.float32)

    # 4. Iterate over 20 altitude layers and fill the 3D volume
    for i, rx_z in enumerate(TARGET_HEIGHTS):
        depth_volume[i] = calculate_radio_depth(b_map, tx_col, tx_row, TX_Z, rx_z, FREQ_MHZ)

    # 5. Save as a single 3D .npy file (float16 to save space)
    np.save(save_path, depth_volume.astype(np.float16))

    return True

# =========================================================
# Main Processing Pipeline
# =========================================================
def process_dataset(split_name):
    """ Process the entire dataset split
    Args:
        split_name (str): 'train' or 'test'
    """
    print(f"\nProcessing 3D Depth Map volume data: [{split_name.upper()}] ...")

    # Use h1 as the scan base to get all samples
    src_dir = os.path.join(DATA_ROOT, split_name, 'h1')
    if not os.path.exists(src_dir):
        print(f"Directory not found: {src_dir}")
        return

    png_files = glob.glob(os.path.join(src_dir, '*.png'))
    os.makedirs(os.path.join(OUTPUT_ROOT, split_name), exist_ok=True)

    print(f"Total files to process: {len(png_files)}")

    for i, fpath in enumerate(png_files):
        process_single_file_3d(fpath, split_name)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(png_files)} files")

if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    process_dataset('train')
    process_dataset('test')
    print("\nAll 3D radio depth map volumes have been generated successfully!")