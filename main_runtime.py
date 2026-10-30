"""Shared loaders, checkpoint IO and evaluation; no experiment side effects on import."""
import os
import hashlib
import random
import tempfile
import inspect
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from main_metrics import Metrics, local_ssim_map
from main_protocol import digest


def load_checkpoint(path, device):
    # Only use the newer explicit keyword when the installed torch.load supports it.
    options = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = False
    return torch.load(path, **options)


def make_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def amp_context(enabled):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, cfg, training=False):
    workers = cfg["num_workers"]
    options = dict(batch_size=cfg["batch_size"], shuffle=training, num_workers=workers,
                   pin_memory=torch.cuda.is_available(), drop_last=False,
                   worker_init_fn=seed_worker,
                   generator=torch.Generator().manual_seed(cfg["seed"] + dataset.epoch))
    if workers:
        options.update(prefetch_factor=cfg["prefetch_factor"], persistent_workers=False)
    return DataLoader(dataset, **options)


def atomic_checkpoint(path, data):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(data, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def run_signature(cfg, manifest):
    # epochs may be increased on resume; data, model, loss and optimizer cannot change silently.
    source_names = ("model_3d.py", "dataset_main.py", "main_loss.py", "main_pseudo.py",
                    "main_trajectories.py", "main_protocol.py", "main_metrics.py")
    code = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in source_names}
    return digest({"config": {k: v for k, v in cfg.items() if k not in ("epochs", "num_workers", "prefetch_factor")},
                   "manifest_hash": manifest["manifest_hash"], "sources": code})


@torch.inference_mode()
def evaluate(model, data_loader, device, detailed=False):
    model.eval()
    names = ("full", "known", "blind", "input") if detailed else ("full",)
    trackers = {name: Metrics() for name in names}
    altitude = {str(z + 1): Metrics() for z in range(data_loader.dataset.shape[0])} if detailed else {}
    ssim_values = []
    for batch in tqdm(data_loader, desc="Test" if detailed else "Validation", ascii=True):
        x = batch["x"].to(device)
        gt = batch["gt"].to(device)
        air = x[:, 1] < 0.5
        pred = torch.sigmoid(model(x).squeeze(1))
        pred = pred.masked_fill(~air, 0)
        masks = {"full": air}
        if detailed:
            known = batch["loss_mask"].to(device)
            masks.update(known=known & air, blind=~known & air, input=batch["input_mask"].to(device) & air)
        for name, tracker in trackers.items():
            tracker.update(pred, gt, masks[name], batch["filename"])
        if detailed:
            for z, tracker in altitude.items():
                index = int(z) - 1
                tracker.update(pred[:, index:index + 1], gt[:, index:index + 1],
                               air[:, index:index + 1], batch["filename"])
            p, g, a = pred.cpu().numpy(), gt.cpu().numpy(), air.cpu().numpy()
            for i in range(len(p)):
                per_slice = []
                for z in range(p.shape[1]):
                    window = min(7, p.shape[2], p.shape[3])
                    window -= (window % 2 == 0)
                    if window >= 3 and a[i, z].any():
                        similarity = local_ssim_map(p[i, z], g[i, z], window=window)
                        per_slice.append(float(similarity[a[i, z]].mean()))
                if per_slice:
                    ssim_values.append(float(np.mean(per_slice)))
    # Some evaluation regions can legitimately be empty (e.g. an altitude with no measured points).
    results = {name: tracker.compute() if tracker.rows else None for name, tracker in trackers.items()}
    if detailed:
        results["altitudes"] = {z: t.compute() if t.rows else None for z, t in altitude.items()}
        results["full"]["ssim"] = float(np.mean(ssim_values)) if ssim_values else None
        results["ssim_definition"] = "Mean local SSIM over non-building pixels per slice, then slices and samples"
    if results["full"] is None:
        raise ValueError("No evaluation data")
    return results, trackers["full"].rows
