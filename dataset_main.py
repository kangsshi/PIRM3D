# PIRM3D: physics-inspired 3D radio map construction.
"""One complete route as input; union of 20 routes as supervision."""
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from main_protocol import building_volume, depth_path, digest, load_manifest, png_paths, read_gray
from main_pseudo import BoundedPseudoCache, PSEUDO_VERSION, pseudo_label
from main_trajectories import get_trajectories, route_mask


class UrbanRadio3D_Dataset(Dataset):
    def __init__(self, config, split="train", manifest=None, max_samples=None):
        if split not in ("train", "val", "test"):
            raise ValueError("Invalid dataset split")
        self.cfg, self.split = config, split
        self.manifest = load_manifest(config) if manifest is None else manifest
        self.samples = self.manifest["splits"][split]
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be positive")
            self.samples = self.samples[:max_samples]
        self.shape = tuple(config["shape"])
        self.epoch = 0
        self.geometry = OrderedDict()
        self.cache = BoundedPseudoCache(config["pseudo_cache_dir"], config["pseudo_cache_max_gib"])

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        # Nonpersistent workers receive this epoch at the next loader iteration.
        self.epoch = int(epoch)

    def _geometry(self, scene_id):
        if scene_id in self.geometry:
            self.geometry.move_to_end(scene_id)
            return self.geometry[scene_id]
        occupied, free = building_volume(self.cfg, scene_id)
        coords, offsets, meta = get_trajectories(self.cfg, scene_id, free)
        known = np.zeros(self.shape, dtype=bool)
        known[tuple(coords.T)] = True
        if np.any(known & occupied):
            raise ValueError("Trajectory crosses a building")
        value = occupied, coords, offsets, known, meta
        self.geometry[scene_id] = value
        if len(self.geometry) > 2:
            self.geometry.popitem(last=False)
        return value

    def __getitem__(self, index):
        row = self.samples[index]
        occupied, coords, offsets, known, meta = self._geometry(row["scene_id"])
        paths = png_paths(self.cfg, row)
        gt = np.stack([read_gray(p, self.shape[1:]) for p in paths]).astype(np.float32) / 255.0
        p_depth = depth_path(self.cfg, row)
        if not p_depth.is_file():
            raise FileNotFoundError(f"Missing depth: {p_depth}; run prepare_main.py --depth")
        raw = np.load(p_depth, mmap_mode="r", allow_pickle=False)
        if raw.shape != self.shape or not np.issubdtype(raw.dtype, np.floating):
            raise ValueError(f"Bad raw depth shape/dtype: {p_depth}")
        depth = np.asarray(raw, dtype=np.float32)
        if not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError(f"Invalid raw depth: {p_depth}")
        depth = np.clip(np.log10(depth + 1.0) / np.log10(self.cfg["depth_max_raw"] + 1.0), 0, 1)
        epoch = self.epoch if self.split == "train" else 0
        rng = np.random.default_rng(int(digest([self.cfg["seed"], row["filename"], epoch])[:16], 16))
        selection = self.cfg["train_input_trajectory"] if self.split == "train" else self.cfg["eval_input_trajectory"]
        route_id = int(rng.integers(self.cfg["trajectory_count"])) if selection == "random" else selection
        input_mask = route_mask(coords, offsets, route_id, self.shape)
        measured = np.zeros_like(gt)
        measured[known] = gt[known]
        std = self.cfg["noise_std"]
        if std is not None and std > 0 and self.cfg["noise_scope"] == "all_measurements":
            measured[known] = np.clip(measured[known] + rng.normal(0, std, int(known.sum())), 0, 1)
        hint = np.zeros_like(gt)
        hint[input_mask] = measured[input_mask]
        if std is not None and std > 0 and self.cfg["noise_scope"] == "input":
            hint[input_mask] = np.clip(hint[input_mask] + rng.normal(0, std, int(input_mask.sum())), 0, 1)
        tx = np.zeros(self.shape, dtype=np.float32)
        _, sx, sy = row["filename"][:-4].split("_")
        tx_row, tx_col = self.shape[1] - 1 - int(sx[1:]), int(sy[1:])
        tx[:min(2, self.shape[0]), tx_row, tx_col] = 1
        x = np.stack([tx, occupied.astype(np.float32), depth, hint, input_mask.astype(np.float32)])
        result = {"x": torch.from_numpy(x), "gt": torch.from_numpy(gt),
                  "measured": torch.from_numpy(measured), "loss_mask": torch.from_numpy(known.copy()),
                  "input_mask": torch.from_numpy(input_mask), "trajectory_id": route_id,
                  "filename": row["filename"], "scene_id": row["scene_id"]}
        if self.split == "train":
            def generate():
                label = pseudo_label(measured, known, depth, occupied, self.cfg["gaussian_sigma"],
                                     self.cfg["gaussian_truncate"], self.cfg["spacing_m"])
                return label.astype(np.float16).astype(np.float32)
            key = digest({"version": PSEUDO_VERSION, "manifest": self.manifest["manifest_hash"],
                          "sample": row, "trajectory": meta["spec"],
                          "files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths + [p_depth]],
                          "sigma": self.cfg["gaussian_sigma"], "truncate": self.cfg["gaussian_truncate"],
                          "depth_max": self.cfg["depth_max_raw"]})
            if std is not None and std > 0 and self.cfg["noise_scope"] == "all_measurements":
                y2 = generate()
            else:
                y2 = self.cache.get(key, generate)
            result["y2"] = torch.from_numpy(y2)
        return result
