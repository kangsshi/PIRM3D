"""Exclude whole invalid 3D TX samples, preserve raw files and the old manifest."""
import argparse
import csv
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from main_protocol import (NAME_RE, atomic_json, building_path, digest,
                           load_config, manifest_selection, png_paths, read_gray, validate_manifest)


def _file_stamps(cfg, sid, rows):
    paths = [building_path(cfg["data_root"], sid)]
    paths.extend(p for row in rows for p in png_paths(cfg, row))
    return [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in paths]


def _decode_row(cfg, row):
    for p in png_paths(cfg, row):
        try:
            read_gray(p, cfg["shape"][1:])
        except Exception as error:
            return row, str(p), str(error)
    return row, None, None


def prepare_valid(cfg, output="valid_data_v1", resume=False, decode_workers=1):
    output = Path(output).resolve()
    if decode_workers < 1:
        raise ValueError("decode_workers must be positive")
    signature = digest({"selection": manifest_selection(cfg), "data_root": cfg["data_root"]})
    state_path = output / "preparation_state.json"
    if output.exists() and not resume:
        raise FileExistsError("Output already exists; use another --output name")
    if resume:
        if not state_path.is_file():
            raise ValueError("No checkpoint from the new version; use a NEW output directory")
        if json.loads(state_path.read_text(encoding="utf-8"))["signature"] != signature:
            raise ValueError("Resume configuration differs from saved preparation state")
    else:
        output.mkdir(parents=True)
        atomic_json(state_path, {"signature": signature})
    checkpoint_dir = output / "checked_scenes"
    checkpoint_dir.mkdir(exist_ok=True)
    groups = {}
    report = {"sources": {}, "required_maps": sum(cfg[f"{s}_maps"] for s in ("train", "val", "test")),
              "tx_per_map": cfg["tx_per_map"], "shape": cfg["shape"],
              "rule": "Exclude entire TX if any height is missing, empty or undecodable; no data files changed."}
    with (output / "excluded_height_files.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_split", "scene_id", "filename", "height", "reason"])
        for source in ("train", "test"):
            root = Path(cfg["data_root"]) / source
            names = set()
            for p in sorted((root / "h1").glob("*.png")):
                match = NAME_RE.fullmatch(p.name)
                if not match:
                    raise ValueError(f"Unrecognized sample name: {p}")
                sid, x, y = map(int, match.groups())
                if not (0 <= x < cfg["shape"][1] and 0 <= y < cfg["shape"][2]):
                    raise ValueError(f"Invalid TX coordinates: {p}")
                group = groups.setdefault(sid, {})
                if p.name in group:
                    raise ValueError(f"Duplicate source sample: {p.name}")
                group[p.name] = {"source_split": source, "scene_id": sid, "filename": p.name}
                names.add(p.name)
            available = names.copy()
            heights = []
            for height in range(1, cfg["shape"][0] + 1):
                folder = root / f"h{height}"
                present, empty = set(), set()
                for p in folder.glob("*.png"):
                    if p.is_file():
                        (present if p.stat().st_size > 0 else empty).add(p.name)
                unusable = names - present
                for name in sorted(unusable):
                    writer.writerow([source, int(name.split("_")[0]), name, height,
                                     "empty" if name in empty else "missing_or_not_file"])
                available.intersection_update(present)
                item = {"height": height, "empty_pngs": len(empty), "unusable_h1_counterparts": len(unusable),
                        "remaining_candidate_tx": len(available)}
                heights.append(item)
                print(f"[Nonempty] {source}/h{height}: empty={len(empty)}, unusable={len(unusable)}, "
                      f"remaining_TX={len(available)}", flush=True)
            for name in names - available:
                groups[int(name.split("_")[0])].pop(name)
            report["sources"][source] = {"h1_tx": len(names), "nonempty_all_height_tx": len(available), "heights": heights}
    report["scene_candidates"] = [{"scene_id": sid, "nonempty_tx": len(rows)} for sid, rows in sorted(groups.items())]
    eligible = sorted(sid for sid, rows in groups.items() if len(rows) >= cfg["tx_per_map"])
    report["nonempty_eligible_maps"] = len(eligible)
    report_path = output / "valid_data_report.json"
    atomic_json(report_path, report)
    print("Maps with enough nonempty TX:", len(eligible), "/ required:", report["required_maps"], flush=True)
    if len(eligible) < report["required_maps"]:
        raise ValueError(f"After excluding empty/missing TX, only {len(eligible)} maps remain; "
                         f"need {report['required_maps']}. No automatic downsizing. See {report_path}")
    # Decode candidates in deterministic order until enough complete maps are obtained.
    order = np.random.default_rng(cfg["seed"]).permutation(eligible).tolist()
    accepted, failed_maps = [], []
    failure_log = output / "undecodable_candidates.csv"
    append_log = resume and failure_log.exists() and failure_log.stat().st_size > 0
    with failure_log.open("a" if append_log else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not append_log:
            writer.writerow(["scene_id", "source_split", "filename", "bad_path", "error"])
        for sid in order:
            checkpoint_path = checkpoint_dir / f"{sid}.json"
            if resume and checkpoint_path.is_file():
                cached = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                cached_rows = cached["rows"]
                try:
                    unchanged = (cached["signature"] == signature and
                                 len(cached_rows) == cfg["tx_per_map"] and
                                 cached["candidate_names_hash"] == digest(sorted(groups[sid])) and
                                 cached["stamps"] == _file_stamps(cfg, sid, cached_rows))
                except (FileNotFoundError, OSError):
                    unchanged = False
                if unchanged:
                    accepted.append((sid, cached_rows))
                    print(f"[Resume] scene={sid}: accepted_maps={len(accepted)}/{report['required_maps']}", flush=True)
                    if len(accepted) == report["required_maps"]:
                        break
                    continue
            try:
                read_gray(building_path(cfg["data_root"], sid), cfg["shape"][1:])
            except Exception as error:
                failed_maps.append({"scene_id": sid, "reason": "building: " + str(error)})
                continue
            candidates = sorted(groups[sid].values(), key=lambda r: r["filename"])
            indices = np.random.default_rng(cfg["seed"] + sid).permutation(len(candidates))
            healthy = []
            with ThreadPoolExecutor(max_workers=decode_workers) as pool:
                for start in range(0, len(indices), decode_workers):
                    chunk = [candidates[int(i)] for i in indices[start:start + decode_workers]]
                    results = pool.map(lambda row: _decode_row(cfg, row), chunk)
                    for row, bad_path, error in results:
                        if bad_path is not None:
                            writer.writerow([sid, row["source_split"], row["filename"], bad_path, error])
                            f.flush()
                        elif len(healthy) < cfg["tx_per_map"]:
                            healthy.append(row)
                    if len(healthy) == cfg["tx_per_map"]:
                        break
            if len(healthy) == cfg["tx_per_map"]:
                rows = sorted(healthy, key=lambda r: r["filename"])
                accepted.append((sid, rows))
                atomic_json(checkpoint_path, {"signature": signature, "rows": rows,
                                             "candidate_names_hash": digest(sorted(groups[sid])),
                                             "stamps": _file_stamps(cfg, sid, rows)})
            else:
                failed_maps.append({"scene_id": sid, "valid_tx": len(healthy), "reason": "not enough decoded TX"})
            print(f"[Decode] scene={sid}: valid_TX={len(healthy)}, accepted_maps={len(accepted)}/"
                  f"{report['required_maps']}", flush=True)
            if len(accepted) == report["required_maps"]:
                break
    report["decoded_accepted_maps"] = len(accepted)
    report["failed_maps"] = failed_maps
    atomic_json(report_path, report)
    if len(accepted) < report["required_maps"]:
        raise ValueError(f"Only {len(accepted)} maps pass decoding; no reduced manifest created. See {report_path}")
    new_cfg = dict(cfg, manifest_path=str(output / "revision_manifest_valid.json"), run_dir=cfg["run_dir"] + "_valid")
    manifest = {"selection": manifest_selection(new_cfg), "splits": {},
                "unused_scene_ids": sorted(set(groups) - {sid for sid, _ in accepted}),
                "eligibility": "Every selected TX fully decoded at all heights; empty/corrupt TX excluded",
                "note": "Raw files and the previous manifest preserved; selection uses integrity only, not RSS quality."}
    offset = 0
    for split in ("train", "val", "test"):
        selected = sorted(accepted[offset:offset + cfg[f"{split}_maps"]], key=lambda pair: pair[0])
        manifest["splits"][split] = [row for _, rows in selected for row in rows]
        offset += cfg[f"{split}_maps"]
    manifest["manifest_hash"] = digest(manifest)
    validate_manifest(manifest, new_cfg)
    atomic_json(new_cfg["manifest_path"], manifest)
    atomic_json(output / "main_config_valid.json", new_cfg)
    for split, rows in manifest["splits"].items():
        print(split, "maps", len({r['scene_id'] for r in rows}), "TX samples", len(rows))
    print("Use this config for ALL subsequent preparation, comparison and training:", output / "main_config_valid.json")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="main_config.json")
    parser.add_argument("--output", default="valid_data_v1")
    parser.add_argument("--resume", action="store_true", help="Reuse unchanged decoded maps from this version")
    parser.add_argument("--decode-workers", type=int, default=1, help="Concurrent image readers; try 2 or 4")
    args = parser.parse_args()
    prepare_valid(load_config(args.config), args.output, args.resume, args.decode_workers)
