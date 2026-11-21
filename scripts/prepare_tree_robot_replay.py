#!/usr/bin/env python3
"""Prepare a camera/IMU replay and its relative DA2 depth for val or test."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

LABELER = Path(__file__).resolve().parents[1]
REPO = LABELER.parent
sys.path.insert(0, str(LABELER))
from server import DepthWorker


def local_pose(exported):
    recorded_rotation = np.asarray(exported["camera_to_world"])[:3, :3]
    gravity = np.asarray(exported["gravity"]["vector_camera"])
    if not np.allclose(gravity, recorded_rotation.T @ exported["gravity"]["vector"]):
        raise ValueError(f"Invalid camera gravity for {exported['frame_id']}")
    up = -gravity / np.linalg.norm(gravity)
    basis = np.eye(3)[np.argmin(np.abs(up))]
    right = np.cross(basis, up)
    right /= np.linalg.norm(right)
    camera_to_local = np.eye(4)
    camera_to_local[:3, :3] = [right, np.cross(up, right), up]
    return {
        "frame_id": exported["frame_id"], "width": exported["width"], "height": exported["height"],
        "intrinsics": exported["intrinsics"], "world_up_axis": "z",
        "camera_to_world": camera_to_local.tolist(),
        "gravity": {"vector": [0, 0, -1], "frame": "world", "source": "simulated_imu"},
    }


def prepare(dataset, split, output):
    index = json.loads((dataset / "index.json").read_text())
    rows = [row for row in index["samples"] if row["split"] == split and row["source"] == "3dgs_render"]
    if len(rows) != 1000 or len({row["frame_id"] for row in rows}) != len(rows):
        raise ValueError(f"Expected the official 1,000 {split} frames")
    pose_directory = output / "imu_pose"
    pose_directory.mkdir(parents=True, exist_ok=True)
    link = output / split
    if not link.is_symlink():
        link.symlink_to(dataset / split, target_is_directory=True)
    elif link.resolve() != (dataset / split).resolve():
        raise ValueError(f"{link} points to another dataset")
    for row in rows:
        original = dataset / row["files"]["pose"]
        local = local_pose(json.loads(original.read_text()))
        pose = pose_directory / (row["frame_id"] + ".json")
        pose.write_text(json.dumps(local) + "\n")
        row["files"] = {name: row["files"][name] for name in ("rgb", "sensor_rgb", "gt_mask")}
        row["files"]["pose"] = str(pose.relative_to(output))
    index.update(gravity_source="simulated_imu", stage_metrics=True, samples=rows)
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return rows


def cache_depth(dataset, rows, output, reuse):
    cache = output / "DA2Cache"
    cache.mkdir(exist_ok=True)
    fingerprint = b'["Depth-Anything-V2-Small",518,"http://127.0.0.1:8027"]'
    byte_count = 640 * 640 * 4
    worker = None
    fresh = 0
    for i, row in enumerate(rows, 1):
        raw = (dataset / row["files"]["sensor_rgb"]).read_bytes()
        file = hashlib.sha256(fingerprint + raw).hexdigest() + ".f32"
        target = cache / file
        if target.exists() and target.stat().st_size == byte_count:
            continue
        retained = next((folder / file for folder in reuse
                         if (folder / file).is_file() and (folder / file).stat().st_size == byte_count), None)
        if retained is not None and not target.exists():
            try:
                os.link(retained, target)
                continue
            except OSError:
                temporary = target.with_suffix(".tmp")
                temporary.write_bytes(retained.read_bytes())
                temporary.replace(target)
                continue
        if worker is None:
            checkpoint, = (Path.home() / ".cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small"
                           / "snapshots").glob("*/depth_anything_v2_vits.pth")
            worker = DepthWorker(REPO / "Depth-Anything-V2", str(checkpoint), "cuda:0")
            worker.load()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (544, 963, 3):
            raise ValueError(f"Invalid sensor RGB for {row['frame_id']}")
        depth = cv2.resize(worker.infer(image), (640, 640), interpolation=cv2.INTER_LINEAR).astype("<f4")
        if not np.isfinite(depth).all():
            raise ValueError(f"Invalid relative depth for {row['frame_id']}")
        temporary = target.with_suffix(".tmp")
        depth.tofile(temporary)
        temporary.replace(target)
        fresh += 1
        if i % 100 == 0:
            print(f"DA2 {i}/{len(rows)}", flush=True)
    print(f"DA2 ready for {len(rows)} frames ({fresh} newly inferred)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--reuse-cache", type=Path, action="append", default=[])
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    rows = prepare(args.dataset.resolve(), args.split, args.output.resolve())
    if args.depth:
        cache_depth(args.dataset, rows, args.output, args.reuse_cache)
    print(f"{args.split} replay ready for evaluation, {len(rows)} unique frames", flush=True)


if __name__ == "__main__":
    main()
