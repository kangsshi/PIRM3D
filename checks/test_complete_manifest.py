import json
from pathlib import Path

import pytest

from main_protocol import create_manifest
from test_main_pipeline import synthetic


def test_missing_height_tx_replaced_by_complete_tx(synthetic, tmp_path):
    cfg, old = synthetic
    broken = old["splits"]["train"][0]
    p = Path(cfg["data_root"]) / broken["source_split"] / "h3" / broken["filename"]
    p.unlink()
    cfg = dict(cfg, manifest_path=str(tmp_path / "replacement_manifest.json"))
    new = create_manifest(cfg)
    rows = [r for group in new["splits"].values() for r in group]
    assert len(rows) == 6
    assert broken["filename"] not in {r["filename"] for r in rows}
    audit = json.loads((tmp_path / "revision_completeness_report.json").read_text())
    assert audit["eligible_scene_count"] == 3
    assert str(p) in (tmp_path / "revision_missing_height_files.csv").read_text()


def test_insufficient_complete_scenes_reports_without_reducing_counts(synthetic, tmp_path):
    cfg, old = synthetic
    for row in old["splits"]["train"]:
        (Path(cfg["data_root"]) / row["source_split"] / "h3" / row["filename"]).unlink()
    cfg = dict(cfg, manifest_path=str(tmp_path / "cannot_create.json"))
    with pytest.raises(ValueError, match="found 2 eligible"):
        create_manifest(cfg)
    assert not Path(cfg["manifest_path"]).exists()
    assert (tmp_path / "revision_completeness_report.json").is_file()
