#!/usr/bin/env python3
"""Index LOS tree captures and expose RGB-only PyTorch training samples."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from math import sqrt
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
from PIL import Image


def build_index(root: Path) -> dict:
    root = root.resolve()
    samples = []
    runs = set()
    frame_ids = set()
    for path in sorted(root.rglob("*_manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != "tree-mask-dataset/v1" or not manifest.get("complete"):
            continue
        scene = manifest["scene_id"]
        run = manifest["run_id"]
        if (scene, run) in runs:
            raise ValueError(f"{path}: duplicate scene and run")
        runs.add((scene, run))
        source = manifest["source"]
        if source not in {"3dgs_render", "real_camera"}:
            raise ValueError(f"{path}: unsupported source {source}")
        assigned = manifest.get("assigned_split")
        if assigned not in {None, "train", "val", "test"}:
            raise ValueError(f"{path}: invalid assigned split {assigned}")
        required = {"rgb", "gt_mask", "pose", "sensor_rgb"}
        required.add("calibration_depth" if source == "real_camera" else "reference_depth")
        if len(manifest["views"]) != manifest["expected_view_count"]:
            raise ValueError(f"{path}: incomplete frame list")
        for view in manifest["views"]:
            result = view["result"]
            frame_id = result["frame_id"]
            if frame_id in frame_ids or Path(frame_id).name != frame_id or frame_id in {"", ".", ".."}:
                raise ValueError(f"{path}: invalid or duplicate frame ID {frame_id}")
            frame_ids.add(frame_id)
            if view["prefix"] != frame_id or (result["width"], result["height"]) != (640, 640):
                raise ValueError(f"{path}: malformed view {frame_id}")
            if not required <= result["files"].keys():
                raise ValueError(f"{path}: missing files for {frame_id}")
            files = {}
            for kind in sorted(required):
                resolved = (path.parent / result["files"][kind]).resolve()
                if not resolved.is_relative_to(root) or not resolved.is_file():
                    raise ValueError(f"{path}: missing or external {kind} for {frame_id}")
                files[kind] = resolved.relative_to(root).as_posix()
            pose = json.loads((root / files["pose"]).read_text())
            if pose.get("frame_id") != frame_id:
                raise ValueError(f"{path}: pose belongs to another frame")
            sample = {"frame_id": frame_id, "scene_id": scene, "run_id": run,
                      "segment_index": view["segment_index"], "source": source, "files": files}
            if assigned:
                sample["split"] = assigned
            samples.append(sample)
    if not samples:
        raise ValueError(f"{root}: no complete tree-mask capture manifests")

    real_scenes = {item["scene_id"] for item in samples if item["source"] == "real_camera"}
    explicit_scenes = {item["scene_id"] for item in samples if "split" in item}
    if any(item["scene_id"] in real_scenes and item.get("split", "test") != "test" for item in samples):
        raise ValueError("A tree with real-camera test samples may only appear in test")
    if any(item["scene_id"] in explicit_scenes and "split" not in item for item in samples):
        raise ValueError("All runs of an explicitly split tree must specify assigned_split")
    scenes = sorted({item["scene_id"] for item in samples} - real_scenes - explicit_scenes)
    holdout = max(1, round(len(scenes) * 0.1))
    test = set(scenes[-holdout:]) if len(scenes) >= 3 else set()
    development = [scene for scene in scenes if scene not in test]
    validation = set(development[-holdout:]) if len(development) >= 2 else set()
    development_runs = sorted(run for scene, run in runs if scene in development)
    validation_run = development_runs[-1] if len(scenes) == 1 and len(development_runs) >= 2 else None
    for item in samples:
        if "split" in item:
            continue
        scene = item["scene_id"]
        if scene in test or scene in real_scenes:
            item["split"] = "test"
        elif scene in validation or item["run_id"] == validation_run:
            item["split"] = "val"
        else:
            item["split"] = "train"
    same_tree = validation_run or any(
        len({item["split"] for item in samples if item["scene_id"] == scene}) > 1 for scene in explicit_scenes)
    return {"schema": "tree-mask-index/v1", "validation_scope": "same_tree" if same_tree else "by_tree",
            "samples": samples}


class TreeMaskDataset:
    """Yields (float32 RGB CHW in [0,1], int64 mask HW with IDs 0/1)."""

    def __init__(self, index: Path, split: str = "train"):
        index = Path(index)
        self.root = index.parent
        data = json.loads(index.read_text())
        if data["schema"] != "tree-mask-index/v1":
            raise ValueError(f"{index}: unsupported dataset schema")
        self.samples = [item for item in data["samples"] if item["split"] == split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch

        item = self.samples[index]
        with Image.open(self.root / item["files"]["rgb"]) as image:
            rgb = np.array(image.convert("RGB"))
        with Image.open(self.root / item["files"]["gt_mask"]) as image:
            mask = np.array(image.convert("RGB"))
        if rgb.shape != (640, 640, 3) or mask.shape != rgb.shape:
            raise ValueError(f"{item['frame_id']}: expected aligned 640×640 RGB/mask")
        if not np.all(mask == mask[:, :, :1]) or not np.all((mask == 0) | (mask == 255)):
            raise ValueError(f"{item['frame_id']}: ground truth is not binary")
        return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255), torch.from_numpy((mask[:, :, 0] == 255).astype(np.int64))


def prepare_yolov9(root: Path, samples: list[dict]) -> Path:
    import yaml

    destination = root / "yolov9t"
    if destination.exists():
        raise FileExistsError(f"{destination} already exists")
    counts = Counter(item["split"] for item in samples)
    if any(not counts[split] for split in ("train", "val", "test")):
        raise ValueError("YOLOv9 dataset preparation requires train, val, and test samples")

    with TemporaryDirectory(prefix=".yolov9t-", dir=root) as temporary:
        stage = Path(temporary)
        for split in ("train", "val", "test"):
            for role in ("images", "masks", "labels"):
                (stage / role / split).mkdir(parents=True)

        def convert(item):
            split = item["split"]
            stem = item["frame_id"]
            rgb = root / item["files"]["rgb"]
            image_link = stage / "images" / split / f"{stem}.png"
            image_link.symlink_to(os.path.relpath(rgb, image_link.parent))
            with Image.open(root / item["files"]["gt_mask"]) as image:
                mask = np.asarray(image.convert("L"))
            if mask.shape != (640, 640) or not np.all((mask == 0) | (mask == 255)):
                raise ValueError(f"{stem}: expected a binary 640×640 tree mask")
            # In YOLOv9's semantic loader 255 denotes an ignored pixel.
            ids = (mask == 255).astype(np.uint8)
            Image.fromarray(ids).save(stage / "masks" / split / f"{stem}.png", compress_level=1)
            (stage / "labels" / split / f"{stem}.txt").touch()
            return int(ids.sum()) if split == "train" else 0

        with ThreadPoolExecutor(max_workers=8) as tasks:
            foreground = sum(tasks.map(convert, samples))
        background = counts["train"] * 640 * 640 - foreground
        if not foreground or not background:
            raise ValueError("Both classes must be present in the training masks")
        config = {
            "path": str(destination), "train": "images/train", "val": "images/val", "test": "images/test",
            "nc": 2, "names": {0: "background", 1: "tree"}, "stuff_names": [],
            "semantic_class_weights": [1.0, min(20.0, sqrt(background / foreground))]
        }
        (stage / "dataset.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        stage.rename(destination)
    return destination / "dataset.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--verify-batch", action="store_true")
    parser.add_argument("--prepare-yolov9", action="store_true")
    args = parser.parse_args()
    root = args.dataset.resolve()
    index = root / "index.json"
    data = build_index(root)
    with NamedTemporaryFile(mode="w", dir=root, delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(json.dumps(data, indent=2) + "\n")
            output.close()
            temporary.replace(index)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"index": str(index), "splits": Counter(item["split"] for item in data["samples"]),
                      "validation_scope": data["validation_scope"]}))
    if args.prepare_yolov9:
        print(json.dumps({"yolov9_yaml": str(prepare_yolov9(root, data["samples"]))}))
    if args.verify_batch:
        from torch.utils.data import DataLoader

        dataset = TreeMaskDataset(index)
        if not len(dataset):
            raise ValueError("No training samples in this index")
        rgb, mask = next(iter(DataLoader(dataset, batch_size=4)))
        print(json.dumps({"rgb_shape": list(rgb.shape), "mask_shape": list(mask.shape), "mask_ids": mask.unique().tolist()}))


if __name__ == "__main__":
    main()
