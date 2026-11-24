from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from samr.complexity import VisualComplexityEstimator
from samr.config import SAMRConfig
from samr.routing import BudgetLedger, route
from samr.scales import ScaleLadder


@pytest.mark.parametrize("size", [(245, 98), (777, 399), (80, 800), (4096, 1024), (65, 96)])
@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 1.75, 2.0])
def test_aspect_ratio_crop_and_native_grid(size, scale):
    ladder = ScaleLadder()
    plan = ladder.plan(size, scale)
    image = Image.new("RGB", size, "red")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 7, 7), fill="blue")
    draw.rectangle((size[0] - 8, size[1] - 8, size[0] - 1, size[1] - 1), fill="green")
    result = plan.prepare(image)
    assert result.size == plan.actual_size
    if scale == 1:
        assert result is image
    for actual, original in zip(plan.actual_size, size):
        assert abs(actual - original * plan.applied_scale) <= 0.5
    assert result.getpixel((0, 0)) == (0, 0, 255)
    assert result.getpixel((result.width - 1, result.height - 1)) == (0, 128, 0)
    width, height = plan.processed_size
    assert width % 32 == height % 32 == 0
    assert plan.estimated_visual_tokens == width * height // 1024
    assert plan.estimated_visual_tokens <= ladder.limits.max_visual_tokens


def test_native_limits_extreme_aspect_and_duplicate_sizes():
    ladder = ScaleLadder()
    with pytest.raises(ValueError, match="aspect"):
        ladder.plan((5000, 10), 1.0)
    assert ladder.plan((10, 10), 1.25).processed_size == ladder.plan((10, 10), 1.0).processed_size
    plan = ladder.plan((8192, 8192), 1.75)
    assert plan.actual_size == (4096, 4096)
    assert plan.estimated_visual_tokens == 16384


def test_budget_boundaries_tokens_count_duplicates_and_time():
    now = [10.0]
    ledger = BudgetLedger(replace(SAMRConfig(), max_visual_tokens=64, max_latency_budget=2), lambda: now[0])
    plan = ScaleLadder().plan((256, 256), 1.0)
    assert ledger.admit(plan) == "admitted"
    ledger.reserve(plan)
    assert ledger.remaining_tokens == 0
    assert ledger.admit(ScaleLadder().plan((256, 256), 1.25)) == "visual_token_budget_exhausted"
    now[0] = 12.0
    assert ledger.admit(plan) == "latency_budget_exhausted"
    assert ledger.started_scales == 1
    ledger = BudgetLedger(replace(SAMRConfig(), max_scales_per_table=1))
    ledger.reserve(plan)
    assert ledger.admit(plan) == "scale_count_budget_exhausted"
    ledger = BudgetLedger(SAMRConfig())
    ledger.reserve(plan)
    assert ledger.admit(plan) == "duplicate_native_resolution"


def test_complexity_has_only_image_and_actual_native_token_features():
    estimator = VisualComplexityEstimator()
    white = Image.new("RGB", (512, 512), "white")
    dense = white.copy()
    draw = ImageDraw.Draw(dense)
    for x in range(2, 510, 6):
        for y in range(2, 510, 6):
            draw.rectangle((x, y, x + 2, y + 3), fill="black")
    plain = estimator.estimate(white, visual_token_count=256)
    complex_image = estimator.estimate(dense, visual_token_count=256)
    assert complex_image.score > plain.score
    assert set(plain.raw) == {"aspect_ratio", "original_area", "edge_density", "component_density",
                              "text_like_density", "visual_token_count"}
    assert plain == estimator.estimate(white, visual_token_count=256)
    with pytest.raises(TypeError):
        estimator.estimate(white, visual_token_count=256, gt_rows=10)
    assert all(0 <= value <= 1 for value in complex_image.normalized.values())


@pytest.mark.parametrize("level,count", [("LOW", 1), ("MEDIUM", 2), ("HIGH", 3)])
def test_deterministic_low_medium_high_route(level, count):
    features = VisualComplexityEstimator().estimate(Image.new("RGB", (256, 256)), visual_token_count=64)
    decision = route(SAMRConfig(), replace(features, level=level))
    assert decision.desired_scales == (1.0, 1.5, 1.75)[:count]
    assert level in decision.reason


def test_every_published_config_loads_and_has_pinned_identities():
    paths = list(Path("configs/samr").glob("*.yaml"))
    assert len(paths) == 12
    for path in paths:
        config = SAMRConfig.load(path)
        assert len(config.as_dict()["model_hash"]) == 64
    assert SAMRConfig.load(Path("configs/samr/samr_full.yaml")).mode == "adaptive"


@pytest.mark.parametrize("update", [{"mode": "training"}, {"scales": (1.0, 3.0)},
                                    {"scales": (True, 1.5, 1.75)}, {"max_latency_budget": True},
                                    {"max_visual_tokens": 0}, {"max_visual_tokens": 64.5},
                                    {"max_scales_per_table": 0}, {"max_latency_budget": float("nan")},
                                    {"routing_policy": "GT"}, {"resolution": "stretched"}])
def test_invalid_config_is_rejected(update):
    with pytest.raises(ValueError):
        replace(SAMRConfig(), **update)
