"""Development-only inputs and profiling, separated from all routing interfaces."""

from __future__ import annotations

import hashlib
import json
from math import ceil, floor
from pathlib import Path

from PIL import Image

from orchestration.storage import atomic_json, file_hash, object_hash, read_json, read_jsonl

COHORT_IDS_HASH = "c2e9f777eb62316db2d26464ad1d97b94c93c4ac7543fee4ec2e691b2467f33e"
EXCLUSION_PAYLOAD_HASH = "53f6baba71c05f07999df1d51e370f4f3bf613f5f42d073d8febf7cf56d513ef"
MAX_DEV_IMAGES = 32


def pixel_hash(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    return hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()).hexdigest()


def build_exclusion_manifest(baseline_pairs: Path, benchmark: Path, output: Path) -> None:
    """Read frozen IDs/crops only to deny their reuse; this never enters an engine."""
    rows = read_jsonl(baseline_pairs)
    ids = [row["id"] for row in rows]
    if len(ids) != 665 or object_hash(ids) != COHORT_IDS_HASH:
        raise ValueError("Expected the exact frozen 665-table exclusion cohort")
    pixels: set[str] = set()
    files: set[str] = set()
    for row in rows:
        path = benchmark / "images" / row["page"]
        files.add(file_hash(path))
        with Image.open(path) as source:
            image = source.convert("RGB")
        pixels.add(pixel_hash(image))
        x, y = row["polygon"][::2], row["polygon"][1::2]
        box = (max(0, floor(min(x))), max(0, floor(min(y))),
               min(image.width, ceil(max(x))), min(image.height, ceil(max(y))))
        pixels.add(pixel_hash(image.crop(box)))
    atomic_json(output, {"cohort_ids_hash": COHORT_IDS_HASH, "table_ids": ids,
                         "file_hashes": sorted(files), "pixel_hashes": sorted(pixels)}, immutable=True)


def load_development_inputs(manifest: Path, exclusion_manifest: Path) -> list[tuple[str, Path, str]]:
    data, excluded = read_json(manifest), read_json(exclusion_manifest)
    if (set(data) != {"split", "source", "images"} or data["split"] != "development"
            or not isinstance(data["source"], str) or not data["source"].strip()
            or "omnidoc" in data["source"].lower()):
        raise ValueError("Only an explicitly identified non-OmniDocBench development source is allowed")
    if (object_hash(excluded) != EXCLUSION_PAYLOAD_HASH
            or excluded["cohort_ids_hash"] != COHORT_IDS_HASH or len(excluded["table_ids"]) != 665
            or object_hash(excluded["table_ids"]) != COHORT_IDS_HASH
            or not excluded["file_hashes"] or not excluded["pixel_hashes"]):
        raise ValueError("Missing frozen benchmark exclusion evidence")
    if not isinstance(data["images"], list) or not 1 <= len(data["images"]) <= MAX_DEV_IMAGES:
        raise ValueError("Development runner accepts 1 to 32 images; never a full evaluation")
    result = []
    ids, pixels_seen = set(), set()
    for row in data["images"]:
        if set(row) != {"table_id", "image", "sha256"}:
            raise ValueError("Development input schema excludes GT and all auxiliary annotations")
        table_id = row["table_id"]
        path = (manifest.parent / row["image"]).resolve()
        if (not isinstance(table_id, str) or not table_id.startswith("dev:") or table_id in ids
                or table_id in excluded["table_ids"] or any("omnidoc" in p.lower() for p in path.parts)):
            raise ValueError("Invalid, repeated or forbidden development image identity")
        digest = file_hash(path)
        with Image.open(path) as image:
            pixel_digest = pixel_hash(image)
        if (digest != row["sha256"] or digest in excluded["file_hashes"]
                or pixel_digest in excluded["pixel_hashes"] or pixel_digest in pixels_seen):
            raise ValueError("Image changed, duplicated or overlaps the frozen test cohort")
        ids.add(table_id)
        pixels_seen.add(pixel_digest)
        result.append((table_id, path, digest))
    return result


def profile_rows(audit: dict) -> list[dict]:
    rows = []
    for scale in audit["scales"]:
        rows.append({"table_id": audit["table_id"], "scale": scale["requested_scale"],
                     "requested_scale": scale["requested_scale"],
                     "actual_resolution": [scale["actual_width"], scale["actual_height"]],
                     "processed_resolution": [scale["processed_width"], scale["processed_height"]],
                     "image_grid_thw": scale["image_grid_thw"], "visual_token_count": scale["visual_token_count"],
                     "latency": scale["inference_latency"], "teds": None, "teds_s": None, "em": None, "cer": None,
                     "parse_status": {"otsl": scale["otsl_validity"], "html": scale["html_parse_status"],
                                      "canonical": scale["canonical_grid"]["status"]},
                     "prediction_html": scale["prediction"]["html"], "eos": scale["prediction"]["eos"],
                     "runtime_contract_valid": scale["runtime_contract_valid"],
                     "truncated": scale["prediction"]["truncated"], "error": scale["prediction"]["error"],
                     "evaluation_split": "development", "scored_offline": False})
    return rows


def write_profile(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
