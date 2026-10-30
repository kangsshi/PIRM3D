# PIRM3D: physics-inspired 3D radio map construction.
"""Prepare an exact scene split and selected depth/trajectories. Never delete raw data."""
import argparse
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from main_protocol import (atomic_json, building_path, building_volume, create_manifest, depth_path,
                           load_config, load_manifest, png_paths, read_gray)
from main_trajectories import get_trajectories
from preprocess_depth import ensure_depth


def storage_report(cfg, manifest):
    selected = {p.resolve() for rows in manifest["splits"].values() for row in rows for p in png_paths(cfg, row)}
    totals = Counter()
    unselected_counts = Counter()
    for source in ("train", "test"):
        for z in range(1, cfg["shape"][0] + 1):
            for path in (Path(cfg["data_root"]) / source / f"h{z}").glob("*.png"):
                size = path.stat().st_size
                totals["raw_total_bytes"] += size
                if path.resolve() in selected:
                    totals["raw_selected_bytes"] += size
                else:
                    totals["raw_unselected_bytes"] += size
                    unselected_counts[source] += 1
    rows = [row for entries in manifest["splits"].values() for row in entries]
    depth_existing = sum(depth_path(cfg, row).stat().st_size for row in rows if depth_path(cfg, row).is_file())
    raw_volume = int(np.prod(cfg["shape"]))
    old_labels = []
    for directory in sorted(Path(cfg["data_root"]).glob("Precomputed_Labels*")):
        if directory.is_dir():
            count, size = 0, 0
            for p in directory.rglob("*.npz"):
                count += 1
                size += p.stat().st_size
            old_labels.append({"directory": str(directory), "files": count, "bytes": size})
    disk = shutil.disk_usage(cfg["data_root"])
    report = {"bytes": dict(totals), "unselected_png_counts": dict(unselected_counts),
              "selected_png_count": len(selected), "selected_depth_existing_bytes": depth_existing,
              "selected_depth_float16_estimate_bytes": len(rows) * raw_volume * 2,
              "all_training_pseudo_float16_uncompressed_bytes": len(manifest["splits"]["train"]) * raw_volume * 2,
              "pseudo_cache_limit_bytes": int(cfg["pseudo_cache_max_gib"] * 1024**3),
              "old_pseudo_label_directories": old_labels, "disk_free_bytes": disk.free,
              "note": "Read-only inventory. Unselected files may still serve baseline/cross-dataset work. "
                      "This program never deletes or moves server data."}
    output = Path(cfg["manifest_path"]).with_name("revision_storage_report.json")
    atomic_json(output, report)
    print("Storage report:", output)
    for key, value in totals.items():
        print(key, f"{value / 1024**3:.3f} GiB")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="main_config.json")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--trajectories", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--check", action="store_true", help="Validate every selected PNG and depth array")
    parser.add_argument("--storage-report", action="store_true")
    parser.add_argument("--smoke-data", action="store_true", help="Prepare trajectories/depth only for 2 train and 2 val samples")
    args = parser.parse_args()
    if not any((args.manifest, args.trajectories, args.depth, args.check, args.storage_report, args.smoke_data)):
        parser.error("Choose --manifest, --trajectories, --depth, --check or --storage-report")
    cfg = load_config(args.config)
    if args.manifest and not Path(cfg["manifest_path"]).exists():
        manifest = create_manifest(cfg)
    else:
        manifest = load_manifest(cfg)
    for split, rows in manifest["splits"].items():
        print(split, "maps", len({r["scene_id"] for r in rows}), "TX samples", len(rows))
    rows = [row for entries in manifest["splits"].values() for row in entries]
    if args.smoke_data:
        rows = manifest["splits"]["train"][:2] + manifest["splits"]["val"][:2]
        args.trajectories, args.depth = True, True
        print("SMOKE ONLY: preparing", len(rows), "train/validation samples; no test data")
    if args.trajectories:
        summaries = []
        for sid in tqdm(sorted({r["scene_id"] for r in rows}), desc="Trajectories", ascii=True):
            _, free = building_volume(cfg, sid)
            _, _, metadata = get_trajectories(cfg, sid, free)
            summaries.append({"scene_id": sid, "union_ratio": metadata["union_full_volume_ratio"],
                              "routes": metadata["routes"]})
        summary_name = "revision_trajectory_summary_smoke.json" if args.smoke_data else "revision_trajectory_summary.json"
        atomic_json(Path(cfg["manifest_path"]).with_name(summary_name), summaries)
        print("Actual union coverage min/max:", min(r["union_ratio"] for r in summaries),
              max(r["union_ratio"] for r in summaries))
    if args.depth:
        counts = Counter(ensure_depth(cfg, row) for row in tqdm(rows, desc="Selected depth", ascii=True))
        print("Depth:", dict(counts))
    if args.check:
        for row in tqdm(rows, desc="Validate selected files", ascii=True):
            read_gray(building_path(cfg["data_root"], row["scene_id"]), cfg["shape"][1:])
            for p in png_paths(cfg, row):
                read_gray(p, cfg["shape"][1:])
            depth = np.load(depth_path(cfg, row), mmap_mode="r", allow_pickle=False)
            if depth.shape != tuple(cfg["shape"]) or not np.issubdtype(depth.dtype, np.floating):
                raise ValueError(f"Bad depth: {depth_path(cfg, row)}")
            if not np.isfinite(depth).all() or (depth < 0).any():
                raise ValueError(f"Invalid depth: {depth_path(cfg, row)}")
        print("All selected raw/depth files passed validation")
    if args.storage_report:
        storage_report(cfg, manifest)


if __name__ == "__main__":
    main()
