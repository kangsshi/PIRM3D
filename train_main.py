# PIRM3D: physics-inspired 3D radio map construction.
"""The sole active main-experiment training entry point."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataset_main import UrbanRadio3D_Dataset
from main_loss import PhysicalStructureLoss
from main_protocol import atomic_json, load_config, load_manifest
from main_runtime import (amp_context, atomic_checkpoint, evaluate, loader, load_checkpoint,
                          make_scaler, run_signature, seed_everything)
from model_3d import ResUNet3D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="main_config.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="One epoch, 2 train/val samples in a separate run folder")
    parser.add_argument("--require-cuda", action="store_true", help="Stop if no GPU is available")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = load_manifest(cfg)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: switch to GPU mode and check the active Python environment")
    if args.smoke:
        cfg["run_dir"] += "_smoke"
        cfg["epochs"] = 1
        cfg["num_workers"] = 0
    signature = run_signature(cfg, manifest)
    directory = Path(cfg["run_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "run_config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise ValueError("Run configuration changed; choose a new run_dir")
        if not args.resume:
            raise FileExistsError("Run already exists; pass --resume or choose another run_dir")
    elif args.resume:
        raise FileNotFoundError("Cannot resume: run_config.json missing")
    atomic_json(config_path, {"config": cfg, "manifest_hash": manifest["manifest_hash"], "signature": signature})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(directory / "train.log", encoding="utf-8"), logging.StreamHandler()])
    seed_everything(cfg["seed"])
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maximum = 2 if args.smoke else None
    train = UrbanRadio3D_Dataset(cfg, "train", manifest, maximum)
    validation = UrbanRadio3D_Dataset(cfg, "val", manifest, maximum)
    logging.info("train=%d val=%d; test never loaded; noise_std=%s; device=%s",
                 len(train), len(validation), cfg["noise_std"], device)
    if device.type == "cuda":
        logging.info("GPU=%s; torch=%s", torch.cuda.get_device_name(0), torch.__version__)
    model = ResUNet3D(5, 1, cfg["base_dim"]).to(device)
    criterion = PhysicalStructureLoss(cfg["lambda_structure"], cfg["gradient_axes"], cfg["spacing_m"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    amp = bool(cfg["amp"] and device.type == "cuda")
    scaler = make_scaler(amp)
    start, best = 1, float("inf")
    if args.resume:
        checkpoint = load_checkpoint(directory / "latest_model.pth", device)
        if checkpoint["signature"] != signature:
            raise ValueError("Checkpoint signature mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best_val_rmse"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    for epoch in range(start, cfg["epochs"] + 1):
        train.set_epoch(epoch)
        train_loader = loader(train, cfg, True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {}
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", ascii=True)):
            x = batch["x"].to(device)
            measured = batch["measured"].to(device)
            known = batch["loss_mask"].to(device)
            y2 = batch["y2"].to(device)
            # Account for the final, potentially shorter accumulation group.
            group_start = step // cfg["accumulation_steps"] * cfg["accumulation_steps"]
            group_size = min(cfg["accumulation_steps"], len(train_loader) - group_start)
            with amp_context(amp):
                logits = model(x).squeeze(1)
            # Float32 loss reductions avoid FP16 overflow on 1.3M voxels.
            loss, parts = criterion(logits.float(), measured, y2, known, x[:, 1] > 0.5)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, step {step}")
            scaler.scale(loss / group_size).backward()
            if (step + 1) % cfg["accumulation_steps"] == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if args.smoke:
                logging.info("Smoke step %d: loss=%g; components=%s", step + 1, loss.detach().item(),
                             {name: value.detach().item() for name, value in parts.items()})
            for name, value in {"total": loss, **parts}.items():
                sums[name] = sums.get(name, 0.0) + value.detach().item()
        results, _ = evaluate(model, loader(validation, cfg), device)
        rmse = results["full"]["rmse"]
        if not np.isfinite(rmse):
            raise FloatingPointError("Nonfinite validation RMSE")
        improved = rmse < best
        best = min(best, rmse)
        record = {"epoch": epoch, "train": {k: v / len(train_loader) for k, v in sums.items()},
                  "validation": results["full"], "best_val_rmse": best}
        logging.info("%s", json.dumps(record))
        with (directory / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        checkpoint = {"epoch": epoch, "signature": signature, "config": cfg,
                      "manifest_hash": manifest["manifest_hash"], "best_val_rmse": best,
                      "val_rmse": rmse, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
                      "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        atomic_checkpoint(directory / "latest_model.pth", checkpoint)
        if improved:
            atomic_checkpoint(directory / "best_model.pth", checkpoint)


if __name__ == "__main__":
    main()
