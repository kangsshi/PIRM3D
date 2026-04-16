import os
import cv2
import torch
import numpy as np
import random
from collections import defaultdict
from torch.utils.data import Dataset


class UrbanRadio3D_Dataset(Dataset):
    """ Dataset for 3D Radio Map Construction with PIRM3D
    Args:
        data_root (str): Root directory of the UrbanRadio3D dataset
        split (str): 'train' or 'test'
        uav_mask_path (str): Path to the precomputed UAV sampling mask
    """
    def __init__(self, data_root, split='train', uav_mask_path='uav_hourglass_mask.npy'):
        self.data_root = data_root
        self.split = split
        self.max_bldg_height = 19.8
        self.heights = 20
        self.grid = 256
        self.MAX_GLOBAL_DEPTH = 362.6

        # Paths to precomputed data
        self.depth_dir = os.path.join(data_root, 'Precomputed_Depth_3D', split)
        self.labels_dir = os.path.join(self.data_root, 'Precomputed_Labels_3D', split)

        # 1. Scan valid samples (filter by labels_dir existence)
        all_samples = []
        if os.path.exists(self.labels_dir):
            for fname in os.listdir(self.labels_dir):
                if fname.endswith('.npz'):
                    all_samples.append(fname.replace('.npz', '.png'))

        # 2. Group by Building ID (BID)
        bid_dict = defaultdict(list)
        for fname in all_samples:
            try:
                bid = int(fname.split('_')[0])
                bid_dict[bid].append(fname)
            except (IndexError, ValueError):
                continue

        sorted_bids = sorted(list(bid_dict.keys()))
        self.samples = []

        # 3. Apply strict sampling strategy
        # Train: 601 maps, 50 TX per map
        # Test: 50 maps, 50 TX per map
        target_tx_count = 50
        if split == 'train':
            target_bids = sorted_bids[:601]
        elif split == 'test':
            target_bids = sorted_bids[:50]
        else:
            target_bids = sorted_bids
            target_tx_count = float('inf')

        for bid in target_bids:
            tx_list = bid_dict[bid]
            tx_list.sort()
            random.seed(bid)  # Deterministic sampling for reproducibility
            if len(tx_list) > target_tx_count:
                sampled_tx = random.sample(tx_list, target_tx_count)
            else:
                sampled_tx = tx_list
            self.samples.extend(sampled_tx)

        print(f"[Info] Split: {split.upper()} - Selected {len(self.samples)} samples from {len(target_bids)} buildings.")
        self.uav_mask = np.load(uav_mask_path).astype(bool)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Recursion protection for corrupted files
        if not hasattr(self, '_recursion_depth'):
            self._recursion_depth = 0
        if self._recursion_depth > 10:
            raise RuntimeError("Too many corrupted files in sequence. Please check your dataset!")

        filename = self.samples[idx]
        b_idx = filename.split('_')[0]

        # Parse Transmitter (TX) coordinates
        try:
            ds_x = int(filename.split('_X')[1].split('_')[0])
            ds_y = int(filename.split('_Y')[1].split('.')[0])
            tx_row, tx_col = (self.grid - 1) - ds_x, ds_y
        except (IndexError, ValueError):
            self._recursion_depth += 1
            return self.__getitem__((idx + 1) % len(self.samples))

        # --- 1. TX Location Tensor ---
        tx_3d = np.zeros((self.heights, self.grid, self.grid), dtype=np.float32)
        if 0 <= tx_row < self.grid and 0 <= tx_col < self.grid:
            tx_3d[0:2, tx_row, tx_col] = 1.0  # Mark TX on first two altitude layers

        # --- 2. 3D Building Map & Mask ---
        bldg_3d = np.zeros((self.heights, self.grid, self.grid), dtype=np.float32)
        bldg_mask = np.zeros((self.heights, self.grid, self.grid), dtype=bool)
        b_path = os.path.join(self.data_root, 'Building_Infomation', 'buildingsWHeight', f'{b_idx}.png')
        if os.path.exists(b_path):
            b_map_raw = cv2.imread(b_path, cv2.IMREAD_UNCHANGED)
            if b_map_raw is not None:
                if len(b_map_raw.shape) > 2:
                    b_map_raw = b_map_raw[:, :, 0]
                b_map_meters = b_map_raw.astype(np.float32) / 255.0 * self.max_bldg_height
                for z in range(self.heights):
                    is_bldg = b_map_meters >= (z + 1.0)
                    bldg_3d[z][is_bldg] = 1.0
                    bldg_mask[z] = is_bldg

        # --- 3. Physics-Guided Depth Map ---
        depth_path = os.path.join(self.depth_dir, filename.replace('.png', '.npy'))
        depth_3d = np.zeros((self.heights, self.grid, self.grid), dtype=np.float32)

        if os.path.exists(depth_path) and os.path.getsize(depth_path) > 1000000:
            try:
                depth_raw = np.load(depth_path).astype(np.float32)
                depth_3d = np.log10(depth_raw + 1.0) / np.log10(self.MAX_GLOBAL_DEPTH + 1.0)
                depth_3d = np.clip(depth_3d, 0.0, 1.0)
            except Exception:
                self._recursion_depth += 1
                return self.__getitem__((idx + 1) % len(self.samples))
        else:
            self._recursion_depth += 1
            return self.__getitem__((idx + 1) % len(self.samples))

        # --- 4. Pseudo-Label (y2 only) ---
        label_path = os.path.join(self.labels_dir, filename.replace('.png', '.npz'))
        y2_label = np.zeros((self.heights, self.grid, self.grid), dtype=np.float32)

        if os.path.exists(label_path) and os.path.getsize(label_path) > 1024:
            try:
                labels_npz = np.load(label_path)
                y2_label = labels_npz['y2'].astype(np.float32) / 255.0  # Restore to [0.0, 1.0]
            except Exception:
                self._recursion_depth += 1
                return self.__getitem__((idx + 1) % len(self.samples))
        else:
            self._recursion_depth += 1
            return self.__getitem__((idx + 1) % len(self.samples))

        # --- 5. Ground Truth (GT) 3D Radio Map ---
        gt_3d = np.zeros((self.heights, self.grid, self.grid), dtype=np.float32)
        for z in range(self.heights):
            h_folder = f'h{z + 1}'
            img_path = os.path.join(self.data_root, self.split, h_folder, filename)
            if os.path.exists(img_path):
                img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_raw is None:
                    self._recursion_depth += 1
                    return self.__getitem__((idx + 1) % len(self.samples))
                gt_3d[z] = img_raw.astype(np.float32) / 255.0
            else:
                self._recursion_depth += 1
                return self.__getitem__((idx + 1) % len(self.samples))

        # --- 6. Input Sparse Hint & Mask ---
        mask_candidate = self.uav_mask.copy()
        mask_candidate[bldg_mask] = False

        if self.split == 'train':
            random_matrix = np.random.rand(self.heights, self.grid, self.grid)
        else:
            state = np.random.RandomState(idx)  # Deterministic for test
            random_matrix = state.rand(self.heights, self.grid, self.grid)

        input_mask = mask_candidate & (random_matrix < 0.05)  # 5% of UAV mask as input hint
        sparse_hint = np.zeros_like(gt_3d)
        sparse_hint[input_mask] = gt_3d[input_mask]
        mask_float = input_mask.astype(np.float32)

        # --- 7. Loss Mask ---
        loss_mask = np.zeros_like(bldg_mask)
        loss_mask[0, :, :] = True  # Full ground layer (h1)
        loss_mask = loss_mask | self.uav_mask  # Plus 20% UAV sampling
        loss_mask[bldg_mask] = False  # Exclude building voxels

        # Reset recursion protection on successful load
        self._recursion_depth = 0

        # --- 8. Stack 5-Channel Input & Return ---
        x_input = np.stack([tx_3d, bldg_3d, depth_3d, sparse_hint, mask_float], axis=0)

        return {
            'x': torch.from_numpy(x_input).float(),
            'y2': torch.from_numpy(y2_label).float(),
            'gt': torch.from_numpy(gt_3d).float(),
            'input_mask': torch.from_numpy(mask_float).float(),
            'loss_mask': torch.from_numpy(loss_mask).bool()
        }