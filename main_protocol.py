# PIRM3D: physics-inspired 3D radio map construction.
"""Shared, explicit experiment configuration and immutable scene-level splits."""
import hashlib
import csv
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

NAME_RE = re.compile(r"^(\d+)_X(\d+)_Y(\d+)\.png$")
VERSION = "pirm3d_revision_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def load_config(path="main_config.json"):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg["protocol_version"] != VERSION:
        raise ValueError("Unsupported protocol version")
    for key in ("manifest_path", "data_root", "run_dir", "trajectory_dir", "pseudo_cache_dir"):
        value = Path(cfg[key])
        cfg[key] = str(value if value.is_absolute() else path.parent / value)
    for key in ("train_maps", "val_maps", "test_maps", "tx_per_map", "trajectory_count",
                "epochs", "batch_size", "accumulation_steps", "base_dim"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["trajectory_count"] != 20:
        raise ValueError("This protocol uses exactly 20 trajectories")
    if not 0 < cfg["trajectory_ratio"] * cfg["trajectory_count"] < 1:
        raise ValueError("Invalid trajectory measurement budget")
    if len(cfg["shape"]) != 3 or any(n < 1 for n in cfg["shape"]):
        raise ValueError("shape must contain three positive sizes in z,y,x order")
    if len(cfg["spacing_m"]) != 3 or any(s <= 0 for s in cfg["spacing_m"]):
        raise ValueError("spacing_m must contain three positive spacings in z,y,x order")
    if cfg["noise_std"] is not None and (not np.isfinite(cfg["noise_std"]) or cfg["noise_std"] < 0):
        raise ValueError("noise_std must be null or a nonnegative normalized standard deviation")
    if cfg["noise_scope"] not in ("input", "all_measurements"):
        raise ValueError("noise_scope must be input or all_measurements")
    if cfg["gradient_axes"] not in ("xy", "xyz"):
        raise ValueError("gradient_axes must be xy or xyz")
    if cfg["gaussian_sigma"] < 0 or cfg["gaussian_truncate"] <= 0:
        raise ValueError("Invalid Gaussian settings")
    if cfg["trajectory_speed_mps"] <= 0 or cfg["trajectory_clearance_m"] < 0:
        raise ValueError("Invalid flight settings")
    if cfg["pseudo_cache_max_gib"] < 0 or cfg["num_workers"] < 0:
        raise ValueError("Cache size and num_workers must be nonnegative")
    for key in ("train_input_trajectory", "eval_input_trajectory"):
        value = cfg[key]
        if key == "train_input_trajectory" and value == "random":
            continue
        if not isinstance(value, int) or not 0 <= value < cfg["trajectory_count"]:
            raise ValueError(f"{key} must be a valid trajectory index")
    return cfg


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def building_path(root, scene_id):
    root = Path(root)
    candidates = [root / "Building_Infomation" / "buildingsWHeight" / f"{scene_id}.png",
                  root / "Building_Infomation" / "Building_Infomation" / "buildingsWHeight" / f"{scene_id}.png"]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing building height image; tried: {candidates}")


def read_gray(path, shape):
    with Image.open(path) as image:
        pixels = np.asarray(image)
    if pixels.ndim == 3:
        pixels = pixels[..., 0]
    if pixels.dtype != np.uint8 or pixels.shape != tuple(shape):
        raise ValueError(f"Expected uint8 image {tuple(shape)}: {path}; got {pixels.shape}/{pixels.dtype}")
    return pixels


def building_volume(cfg, scene_id):
    z, y, x = cfg["shape"]
    pixels = read_gray(building_path(cfg["data_root"], scene_id), (y, x))
    heights = pixels.astype(np.float32) / 255.0 * cfg["max_building_height_m"]
    levels = np.arange(1, z + 1, dtype=np.float32) * cfg["spacing_m"][0]
    occupied = heights[None] >= levels[:, None, None]
    flight_free = levels[:, None, None] > heights[None] + cfg["trajectory_clearance_m"]
    return occupied, flight_free


def png_paths(cfg, row):
    return [Path(cfg["data_root"]) / row["source_split"] / f"h{i + 1}" / row["filename"]
            for i in range(cfg["shape"][0])]


def depth_path(cfg, row):
    return (Path(cfg["data_root"]) / "Precomputed_Depth_3D" / row["source_split"] /
            Path(row["filename"]).with_suffix(".npy"))


def manifest_selection(cfg):
    return {key: cfg[key] for key in ("protocol_version", "train_maps", "val_maps", "test_maps",
                                     "tx_per_map", "seed", "shape")}


def validate_manifest(manifest, cfg):
    if manifest["selection"] != manifest_selection(cfg):
        raise ValueError("Manifest selection differs from config; explicitly prepare a new manifest")
    expected_hash = digest({k: v for k, v in manifest.items() if k != "manifest_hash"})
    if manifest.get("manifest_hash") != expected_hash:
        raise ValueError("Manifest content hash mismatch")
    seen = set()
    for split in ("train", "val", "test"):
        rows = manifest["splits"][split]
        ids = {r["scene_id"] for r in rows}
        if len(ids) != cfg[f"{split}_maps"] or len(rows) != len(ids) * cfg["tx_per_map"]:
            raise ValueError(f"Incorrect scene/sample count in {split}")
        if seen & ids:
            raise ValueError("Scene overlap between train/validation/test")
        seen |= ids
        for sid in ids:
            names = [r["filename"] for r in rows if r["scene_id"] == sid]
            if len(set(names)) != cfg["tx_per_map"]:
                raise ValueError(f"Duplicate/incomplete TX samples in scene {sid}")


def load_manifest(cfg):
    path = Path(cfg["manifest_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Prepare the scene manifest first: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest, cfg)
    return manifest


def create_manifest(cfg):
    """Repartition scene IDs across source folders without moving raw data."""
    if Path(cfg["manifest_path"]).exists():
        raise FileExistsError("Manifest already exists; use a different manifest_path for another split")
    groups = {}
    manifest_path = Path(cfg["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path = manifest_path.with_name("revision_completeness_report.json")
    missing_path = manifest_path.with_name("revision_missing_height_files.csv")
    audit = {"required_heights": cfg["shape"][0], "tx_per_map": cfg["tx_per_map"],
             "sources": {}, "missing_height_directories": [], "scenes": []}
    # Keep two filename sets per source, rather than all 20 layers in memory.
    with missing_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["source_split", "scene_id", "filename", "height", "missing_path"])
        for source in ("train", "test"):
            directory = Path(cfg["data_root"]) / source
            h1_names = set()
            for p in sorted((directory / "h1").glob("*.png")):
                match = NAME_RE.fullmatch(p.name)
                if not match:
                    raise ValueError(f"Unrecognized sample name: {p.name}")
                sid, tx_x, tx_y = map(int, match.groups())
                if not (0 <= tx_x < cfg["shape"][1] and 0 <= tx_y < cfg["shape"][2]):
                    raise ValueError(f"Out-of-range transmitter coordinates: {p.name}")
                sample = {"scene_id": sid, "source_split": source, "filename": p.name}
                group = groups.setdefault(sid, {})
                if p.name in group:
                    raise ValueError(f"Duplicate sample in source folders: {p.name}")
                group[p.name] = sample
                h1_names.add(p.name)
            complete = h1_names.copy()
            height_stats = []
            for height in range(1, cfg["shape"][0] + 1):
                folder = directory / f"h{height}"
                if not folder.is_dir():
                    audit["missing_height_directories"].append(str(folder))
                present = h1_names if height == 1 else {p.name for p in folder.glob("*.png") if p.is_file()}
                absent = h1_names - present
                height_stats.append({"height": height, "png_count": len(present),
                                     "missing_h1_counterparts": len(absent)})
                for name in sorted(absent):
                    writer.writerow([source, int(name.split("_")[0]), name, height, str(folder / name)])
                complete.intersection_update(present)
                print(f"[Completeness] {source}/h{height}: missing={len(absent)}, complete_TX_remaining={len(complete)}", flush=True)
            audit["sources"][source] = {"h1_samples": len(h1_names), "complete_samples": len(complete),
                                         "incomplete_samples": len(h1_names - complete), "heights": height_stats}
            for name in h1_names - complete:
                sid = int(name.split("_")[0])
                # Retain raw counts separately; incomplete TXs are excluded before sampling.
                groups[sid][name]["incomplete"] = True
    eligible = []
    for sid in sorted(groups):
        original = groups[sid]
        complete = {name: row for name, row in original.items() if not row.get("incomplete", False)}
        audit["scenes"].append({"scene_id": sid, "h1_tx": len(original), "complete_tx": len(complete),
                                "eligible": len(complete) >= cfg["tx_per_map"]})
        groups[sid] = complete
        if len(complete) >= cfg["tx_per_map"]:
            eligible.append(sid)
    audit["eligible_scene_count"] = len(eligible)
    audit["insufficient_scene_ids"] = [r["scene_id"] for r in audit["scenes"] if not r["eligible"]]
    atomic_json(audit_path, audit)
    print("Completeness report:", audit_path, flush=True)
    print("Missing-file list:", missing_path, flush=True)
    if audit["missing_height_directories"]:
        raise FileNotFoundError(f"Required height directories missing: {audit['missing_height_directories']}. "
                                f"Check folder layout before resampling; see {audit_path}")
    need = sum(cfg[f"{s}_maps"] for s in ("train", "val", "test"))
    if len(eligible) < need:
        raise ValueError(f"Need {need} complete scenes with >= {cfg['tx_per_map']} TX each; "
                         f"found {len(eligible)} eligible scenes. No automatic downsizing. See {audit_path}")
    # Use a seeded scene split across the full pool; physical folder != logical split.
    rng = np.random.default_rng(cfg["seed"])
    order = rng.permutation(eligible).tolist()
    manifest = {"selection": manifest_selection(cfg), "splits": {},
                "unused_scene_ids": sorted(set(groups) - set(order[:need])),
                "eligibility": "TX must have every required height PNG before deterministic selection",
                "excluded_incomplete_scene_ids": audit["insufficient_scene_ids"],
                "note": "New scene-level split across original train/test sources; source paths are preserved."}
    offset = 0
    missing = []
    for split in ("train", "val", "test"):
        ids = sorted(order[offset:offset + cfg[f"{split}_maps"]])
        offset += len(ids)
        chosen = []
        for sid in ids:
            building_path(cfg["data_root"], sid)
            rows = sorted(groups[sid].values(), key=lambda r: r["filename"])
            local = np.random.default_rng(cfg["seed"] + sid)
            indices = sorted(local.choice(len(rows), cfg["tx_per_map"], replace=False).tolist())
            for index in indices:
                row = rows[index]
                missing.extend(str(p) for p in png_paths(cfg, row) if not p.is_file())
                chosen.append(row)
        manifest["splits"][split] = chosen
    if missing:
        raise FileNotFoundError(f"Selected files changed during scan; {len(missing)} missing: {missing[:10]}")
    manifest["manifest_hash"] = digest(manifest)
    validate_manifest(manifest, cfg)
    atomic_json(cfg["manifest_path"], manifest)
    return manifest
