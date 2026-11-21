#!/usr/bin/env python3
"""Verify complete, aligned 6 m to 1 m PlayCanvas dataset captures."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

EXPECTED = {"train": 8000, "val": 1000, "test": 1000}
FILES = {"rgb", "gt_mask", "pose", "reference_depth", "sensor_rgb"}


def verify_frame(record):
    folder, view, target, scale = record
    frame_id = view["result"]["frame_id"]
    files = {kind: folder / relative for kind, relative in view["result"]["files"].items()}
    assert set(files) == FILES, frame_id
    digest = hashlib.sha256(files["rgb"].read_bytes()).hexdigest()
    with Image.open(files["rgb"]) as rgb, Image.open(files["sensor_rgb"]) as sensor:
        assert rgb.size == (640, 640) and sensor.size == (963, 544), frame_id
        assert rgb.format == sensor.format == "PNG", frame_id
    pose = json.loads(files["pose"].read_text())
    assert pose["frame_id"] == frame_id and pose["gravity"]["source"] == "3dgs_pose", frame_id
    assert pose["height_reference"]["actual_height_m"] == 3, frame_id
    camera = np.array(pose["camera_to_world"])
    assert camera.shape == (4, 4) and np.isfinite(camera).all(), frame_id
    delta = camera[:3, 3] - target
    distance = np.linalg.norm(delta) * scale
    assert abs(distance - pose["approach"]["distance_m"]) < 0.005, frame_id
    assert abs(distance - (6 - 5 * view["index"] / 99)) < 0.005, frame_id
    np.testing.assert_allclose(camera[:3, 2], delta / np.linalg.norm(delta), atol=1e-4)
    np.testing.assert_allclose(camera[:3, :3].T @ pose["gravity"]["vector"], pose["gravity"]["vector_camera"], atol=1e-5)
    with Image.open(files["gt_mask"]) as source:
        mask = np.asarray(source.convert("L"))
    depth = np.load(files["reference_depth"], allow_pickle=False)
    assert mask.shape == depth.shape == (640, 640) and depth.dtype == np.float32, frame_id
    assert np.all((mask == 0) | (mask == 255)), frame_id
    foreground = mask == 255
    count = int(np.count_nonzero(foreground))
    assert count >= 256, f"{frame_id}: tree is obscured"
    valid = np.count_nonzero(np.isfinite(depth[foreground]) & (depth[foreground] > 0))
    assert valid >= 0.95 * count, f"{frame_id}: incomplete depth in the annotated tree"
    return digest, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    partitions = Counter()
    route_counts = Counter()
    sector_splits = defaultdict(set)
    records = []
    for split in EXPECTED:
        for path in sorted((args.dataset / split).glob("approach_*/*_manifest.json")):
            manifest = json.loads(path.read_text())
            assert manifest["complete"] and manifest["renderer"] == "playcanvas"
            assert manifest["assigned_split"] == split and len(manifest["views"]) == 100
            sector_splits[manifest["approach"]["sector"]].add(split)
            route_counts[split] += 1
            partitions[split] += len(manifest["views"])
            records.extend((path.parent, view, np.array(manifest["target_scene"]),
                            manifest["metric_scale"]["meters_per_scene_unit"]) for view in manifest["views"])
    assert dict(partitions) == EXPECTED, partitions
    assert all(len(splits) == 1 for splits in sector_splits.values()), sector_splits
    assert len({view["prefix"] for _, view, _, _ in records}) == 10_000
    with ThreadPoolExecutor(max_workers=6) as tasks:
        results = list(tasks.map(verify_frame, records))
    assert len({digest for digest, _ in results}) == 10_000
    summary = {
        "renderer": "playcanvas", "distance_to_bbox_center_m": [6, 1],
        "frames_per_split": dict(partitions), "paths_per_split": dict(route_counts),
        "split_by_sector": {sector: next(iter(splits)) for sector, splits in sorted(sector_splits.items())},
        "distinct_rgb": len(results), "tree_pixel_range": [min(n for _, n in results), max(n for _, n in results)],
        "validation_scope": "same_tree"
    }
    (args.dataset / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
