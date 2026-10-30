"""Small synthetic invariants; never reads or trains on the real radio dataset."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from dataset_main import UrbanRadio3D_Dataset
from main_loss import PhysicalStructureLoss
from main_metrics import local_ssim_map
from main_protocol import create_manifest, load_config, validate_manifest
from main_pseudo import BoundedPseudoCache, pseudo_label
from main_trajectories import generate_trajectories, route_mask
from model_3d import ResUNet3D
from preprocess_depth import ensure_depth


@pytest.fixture
def synthetic(tmp_path):
    cfg = load_config(Path(__file__).parents[1] / "main_config.json")
    cfg.update(data_root=str(tmp_path / "data"), manifest_path=str(tmp_path / "manifest.json"),
               trajectory_dir=str(tmp_path / "routes"), pseudo_cache_dir=str(tmp_path / "cache"),
               run_dir=str(tmp_path / "run"), shape=[4, 32, 32],
               train_maps=1, val_maps=1, test_maps=1, tx_per_map=2,
               trajectory_ratio=0.01, pseudo_cache_max_gib=0, num_workers=0,
               base_dim=2, epochs=1, batch_size=1, accumulation_steps=2, amp=False)
    root = Path(cfg["data_root"])
    building_dir = root / "Building_Infomation" / "buildingsWHeight"
    building_dir.mkdir(parents=True)
    for sid in range(3):
        building = np.zeros((32, 32), dtype=np.uint8)
        building[12:16, 12:16] = 32  # A low obstacle inside the volume.
        Image.fromarray(building).save(building_dir / f"{sid}.png")
        source = "train" if sid < 2 else "test"
        for tx in range(3):
            name = f"{sid}_X{tx+2}_Y{tx+3}.png"
            for z in range(4):
                p = root / source / f"h{z+1}" / name
                p.parent.mkdir(parents=True, exist_ok=True)
                image = np.full((32, 32), 40 + sid * 20 + tx * 8 + z * 10, dtype=np.uint8)
                Image.fromarray(image).save(p)
    manifest = create_manifest(cfg)
    for rows in manifest["splits"].values():
        for row in rows:
            ensure_depth(cfg, row)
    return cfg, manifest


def test_scene_split_exact_and_source_images_untouched(synthetic):
    cfg, manifest = synthetic
    validate_manifest(manifest, cfg)
    sets = [{r["scene_id"] for r in manifest["splits"][s]} for s in ("train", "val", "test")]
    assert all(len(s) == 1 for s in sets)
    assert not (sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])
    assert all(len(rows) == 2 for rows in manifest["splits"].values())
    assert len(list(Path(cfg["data_root"]).glob("*/h*/*.png"))) == 36
    with pytest.raises(FileExistsError):
        create_manifest(cfg)


def test_insufficient_dataset_is_not_silently_reduced(synthetic):
    cfg, _ = synthetic
    cfg = dict(cfg, train_maps=450, val_maps=50, test_maps=150,
               manifest_path=str(Path(cfg["manifest_path"]).with_name("too_large.json")))
    with pytest.raises(ValueError, match="Need 650"):
        create_manifest(cfg)


def test_trajectory_continuity_obstacles_and_budget(synthetic):
    cfg, _ = synthetic
    free = np.ones(cfg["shape"], dtype=bool)
    free[:3, 12:17, 12:17] = False
    points, offsets, meta = generate_trajectories(cfg, 7, free)
    same, same_offsets, _ = generate_trajectories(cfg, 7, free)
    np.testing.assert_array_equal(points, same)
    np.testing.assert_array_equal(offsets, same_offsets)
    known = np.zeros(free.shape, dtype=bool)
    known[tuple(points.T)] = True
    assert np.all(free[known])
    assert int(known.sum()) == 20 * round(free.size * cfg["trajectory_ratio"])
    for i in range(20):
        route = points[offsets[i]:offsets[i+1]].astype(np.int32)
        assert np.all(np.abs(np.diff(route, axis=0)).sum(axis=1) == 1)
        mask = route_mask(points, offsets, i, free.shape)
        assert np.all(~mask | known)
        assert mask.any()
    assert meta["union_voxels"] == int(known.sum())


def test_dataset_masks_clean_truth_noise_and_no_eval_pseudo(synthetic):
    cfg, manifest = synthetic
    clean = UrbanRadio3D_Dataset(cfg, "train", manifest)[0]
    assert clean["x"].shape == (5, 4, 32, 32)
    assert not clean["loss_mask"][0].all()  # h1 is not automatically supervised.
    assert torch.all(~clean["input_mask"] | clean["loss_mask"])
    assert not torch.any(clean["loss_mask"] & (clean["x"][1] > .5))
    assert torch.equal(clean["x"][3][clean["input_mask"]], clean["gt"][clean["input_mask"]])
    noisy_cfg = dict(cfg, noise_std=0.1)
    noisy = UrbanRadio3D_Dataset(noisy_cfg, "train", manifest)[0]
    assert torch.equal(clean["gt"], noisy["gt"])
    assert torch.equal(clean["measured"], noisy["measured"])
    assert not torch.equal(clean["x"][3], noisy["x"][3])
    assert torch.equal(clean["y2"], noisy["y2"])
    all_cfg = dict(noisy_cfg, noise_scope="all_measurements")
    all_noisy = UrbanRadio3D_Dataset(all_cfg, "train", manifest)[0]
    assert not torch.equal(clean["measured"], all_noisy["measured"])
    assert torch.equal(all_noisy["x"][3][all_noisy["input_mask"]],
                       all_noisy["measured"][all_noisy["input_mask"]])
    evaluation = UrbanRadio3D_Dataset(noisy_cfg, "val", manifest)
    first, second = evaluation[0], evaluation[0]
    assert "y2" not in first
    assert torch.equal(first["x"], second["x"])
    assert first["trajectory_id"] == 0


def test_pseudo_blind_truth_not_used_and_known_restored():
    shape = (4, 8, 8)
    known = np.zeros(shape, bool)
    known[:, 0, 0] = True
    measured = np.zeros(shape, np.float32)
    measured[known] = .4
    depth = np.full(shape, .3, np.float32)
    building = np.zeros(shape, bool)
    first = pseudo_label(measured, known, depth, building)
    measured[~known] = 100  # Hidden dense values must not leak into labels.
    second = pseudo_label(measured, known, depth, building)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[known], measured[known])


def test_vertical_gradient_and_signed_gradient_are_constrained():
    gt = torch.full((1, 4, 4, 4), .5)
    pred = torch.full_like(gt, .4)
    pred[:, 2:] = .6
    logits = torch.logit(pred).requires_grad_()
    known = torch.zeros_like(gt, dtype=torch.bool)
    occupied = known.clone()
    loss_xy, parts_xy = PhysicalStructureLoss(gradient_axes="xy")(logits, gt, gt, known, occupied)
    loss_xyz, parts_xyz = PhysicalStructureLoss(gradient_axes="xyz")(logits, gt, gt, known, occupied)
    assert parts_xy["gradient"] == 0
    assert parts_xyz["grad_z"] > 0 and loss_xyz > loss_xy
    loss_xyz.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    # Opposite signed slopes have identical magnitudes but must be penalized.
    target = torch.linspace(.2, .8, 4)[None, None, None].expand(1, 4, 4, 4)
    _, parts = PhysicalStructureLoss()(torch.logit(1-target), gt, target, known, occupied)
    assert parts["grad_x"] > .1


def test_cache_bounded_and_matches_generated_target(tmp_path):
    cache = BoundedPseudoCache(tmp_path / "cache", 1500 / 1024**3)
    rng = np.random.default_rng(1)
    arrays = [rng.random((4, 8, 8), dtype=np.float32) for _ in range(4)]
    for i, values in enumerate(arrays):
        result = cache.get(str(i), lambda values=values: values)
        np.testing.assert_array_equal(result, values.astype(np.float16).astype(np.float32))
        assert sum(p.stat().st_size for p in cache.directory.glob("pseudo_*.npz")) <= cache.max_bytes
    def fail():
        raise AssertionError("Cache hit unexpectedly regenerated")
    np.testing.assert_array_equal(cache.get("3", fail), arrays[3].astype(np.float16).astype(np.float32))


def test_backbone_forward_backward_and_architecture_preserved(synthetic):
    torch.set_num_threads(1)
    cfg, manifest = synthetic
    sample = UrbanRadio3D_Dataset(cfg, "train", manifest)[0]
    model = ResUNet3D(5, 1, 2)
    logits = model(sample["x"].unsqueeze(0)).squeeze(1)
    loss, _ = PhysicalStructureLoss()(logits, sample["measured"].unsqueeze(0), sample["y2"].unsqueeze(0),
                                     sample["loss_mask"].unsqueeze(0), sample["x"][1].unsqueeze(0) > .5)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.outc.weight.grad.abs().sum() > 0
    assert model.down1[0].kernel_size == (1, 2, 2)


def test_training_validation_checkpoint_resume_and_final_test(synthetic, monkeypatch):
    import sys
    import train_main
    import test_main
    cfg, _ = synthetic
    cfg = dict(cfg, epochs=2)
    config_file = Path(cfg["run_dir"]).with_name("test_config.json")
    config_file.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_main.py", "--config", str(config_file)])
    train_main.main()
    directory = Path(cfg["run_dir"])
    history = [json.loads(s) for s in (directory / "history.jsonl").read_text().splitlines()]
    best = torch.load(directory / "best_model.pth", weights_only=False)
    latest = torch.load(directory / "latest_model.pth", weights_only=False)
    assert best["val_rmse"] == min(row["validation"]["rmse"] for row in history)
    assert latest["epoch"] == 2
    # Resume after increasing only the epoch budget; best RMSE must survive reload.
    cfg["epochs"] = 3
    config_file.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_main.py", "--config", str(config_file), "--resume"])
    train_main.main()
    resumed = torch.load(directory / "latest_model.pth", weights_only=False)
    assert resumed["epoch"] == 3 and resumed["best_val_rmse"] <= latest["best_val_rmse"]
    monkeypatch.setattr(sys, "argv", ["test_main.py", "--config", str(config_file)])
    test_main.main()
    report = json.loads((directory / "test_results.json").read_text())
    assert report["samples"] == 2
    assert report["results"]["full"]["ssim"] is not None
    assert report["results"]["full"]["rmse"] >= 0
    assert (directory / "test_per_sample.csv").is_file()


def test_ssim_known_identity_constant_and_opposite_contrast():
    image = np.random.default_rng(0).random((16, 16))
    np.testing.assert_allclose(local_ssim_map(image, image), 1, atol=1e-12)
    zeros, ones = np.zeros((16, 16)), np.ones((16, 16))
    np.testing.assert_allclose(local_ssim_map(zeros, ones), .01**2 / (1 + .01**2))
    assert local_ssim_map(image, 1-image).mean() < 0


def test_storage_inventory_selected_unselected_and_no_raw_deletion(synthetic):
    from prepare_main import storage_report
    cfg, manifest = synthetic
    images = sorted(Path(cfg["data_root"]).glob("*/h*/*.png"))
    before = {str(p): p.read_bytes() for p in images}
    report = storage_report(cfg, manifest)
    assert report["selected_png_count"] == 24
    assert sum(report["unselected_png_counts"].values()) == 12
    assert report["bytes"]["raw_total_bytes"] == report["bytes"]["raw_selected_bytes"] + report["bytes"]["raw_unselected_bytes"]
    assert {str(p): p.read_bytes() for p in images} == before
