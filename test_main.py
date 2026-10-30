# PIRM3D: physics-inspired 3D radio map construction.
"""Final held-out evaluation using a checkpoint selected only on validation."""
import argparse
import csv
import json
from pathlib import Path

import torch

from dataset_main import UrbanRadio3D_Dataset
from main_protocol import atomic_json, load_config, load_manifest
from main_runtime import evaluate, loader, load_checkpoint, run_signature, seed_everything
from model_3d import ResUNet3D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="main_config.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg["run_dir"] += "_smoke"
        cfg["epochs"] = 1
    manifest = load_manifest(cfg)
    seed_everything(cfg["seed"])
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory = Path(cfg["run_dir"])
    path = directory / "best_model.pth"
    checkpoint = load_checkpoint(path, device)
    if checkpoint["signature"] != run_signature(cfg, manifest):
        raise ValueError("Checkpoint config/data protocol differs from requested test")
    model = ResUNet3D(5, 1, cfg["base_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = UrbanRadio3D_Dataset(cfg, "test", manifest, 2 if args.smoke else None)
    results, rows = evaluate(model, loader(dataset, cfg), device, detailed=True)
    report = {"checkpoint": str(path), "epoch": checkpoint["epoch"], "samples": len(dataset),
              "manifest_hash": manifest["manifest_hash"], "noise_std": cfg["noise_std"],
              "aggregation": "sample-wise macro average", "results": results}
    atomic_json(directory / "test_results.json", report)
    with (directory / "test_per_sample.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
