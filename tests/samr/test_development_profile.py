import json

import pytest
from PIL import Image

from orchestration.publication import allowed, content_issues
from orchestration.storage import file_hash, object_hash
from samr.development import load_development_inputs, pixel_hash
from samr.offline import score_profile


def test_exclusion_blocks_renamed_reencoded_cohort_images_and_gt_fields(tmp_path, monkeypatch):
    image = Image.new("RGB", (40, 20), "white")
    path = tmp_path / "renamed.png"
    image.save(path)
    ids = [f"test:{i}" for i in range(665)]
    monkeypatch.setattr("samr.development.COHORT_IDS_HASH", object_hash(ids))
    exclusion = tmp_path / "exclusion.json"
    exclusion.write_text(json.dumps({"cohort_ids_hash": object_hash(ids), "table_ids": ids,
                                     "file_hashes": ["different file encoding"], "pixel_hashes": [pixel_hash(image)]}))
    monkeypatch.setattr("samr.development.EXCLUSION_PAYLOAD_HASH", object_hash(json.loads(exclusion.read_text())))
    manifest = tmp_path / "input.json"
    data = {"split": "development", "source": "synthetic fixture only",
            "images": [{"table_id": "dev:renamed", "image": "renamed.png", "sha256": file_hash(path)}]}
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="overlaps"):
        load_development_inputs(manifest, exclusion)
    data["images"][0]["gt_rows"] = 10
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="excludes GT"):
        load_development_inputs(manifest, exclusion)


def test_offline_score_is_separate_and_never_overwrites_profile():
    rows = [{"table_id": "dev:test", "scale": 1.0, "evaluation_split": "development",
             "prediction_html": "<table/>", "teds": None, "teds_s": None, "em": None, "cer": None}]
    refs = [{"table_id": "dev:test", "html": "<table/>"}]
    def mock_scorer(pairs):
        assert pairs[0]["html"] == refs[0]["html"]
        return [{"teds": 1.0, "teds_s": 1.0, "em": 1, "cer": 0.0}]
    result = score_profile(rows, refs, mock_scorer)
    assert result[0]["scored_offline"] and result[0]["teds"] == 1
    assert rows[0]["teds"] is None
    with pytest.raises(ValueError, match="restricted"):
        score_profile([{**rows[0], "evaluation_split": "test"}], refs, mock_scorer)
    with pytest.raises(ValueError, match="match"):
        score_profile(rows, [], mock_scorer)


def test_publication_scope_excludes_every_real_samr_artifact():
    assert allowed("configs/samr/samr_full.yaml")
    assert allowed("reports/samr/test_report.md")
    for path in ("experiments/samr/profile.jsonl", "reports/samr/audit.json", "configs/samr/local.yaml",
                 "reports/samr/dev.png", "models/original.safetensors", "data/dev.jsonl"):
        assert not allowed(path)
    assert content_issues("reports/samr/test_report.md", b"x" * (1024 * 1024 + 1))
