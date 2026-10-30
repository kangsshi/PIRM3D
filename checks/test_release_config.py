import json
from pathlib import Path


def test_requested_release_split_and_distinct_output_paths():
    cfg = json.loads((Path(__file__).parents[1] / "main_config.json").read_text())
    assert tuple(cfg[k] for k in ("train_maps", "val_maps", "test_maps", "tx_per_map")) == (500, 50, 100, 200)
    assert sum(cfg[k] for k in ("train_maps", "val_maps", "test_maps")) == 650
    for key in ("manifest_path", "run_dir", "trajectory_dir", "pseudo_cache_dir"):
        assert "500train_tx200" in cfg[key]
    assert cfg["trajectory_count"] == 20
    assert cfg["trajectory_ratio"] == 0.01
    assert cfg["gaussian_sigma"] == 1.0
    assert cfg["lambda_structure"] == 0.1
