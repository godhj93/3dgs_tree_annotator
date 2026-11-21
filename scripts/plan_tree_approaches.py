#!/usr/bin/env python3
"""Inspect PlayCanvas pilot views and assign disjoint radial approach routes."""

import argparse
from collections import defaultdict, deque
from itertools import combinations
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw
import numpy as np


def inspect(folder):
    manifest = json.loads(next(folder.glob("*_manifest.json")).read_text())
    assert manifest["complete"] and manifest["renderer"] == "playcanvas"
    assert len(manifest["views"]) == 5
    checks = []
    for index, view in enumerate(manifest["views"]):
        depth_file = folder / view["result"]["files"]["reference_depth"]
        mask_file = folder / view["result"]["files"]["gt_mask"]
        mask = np.asarray(Image.open(mask_file).convert("L")) == 255
        depth = np.load(depth_file)
        finite = np.isfinite(depth) & (depth > 0)
        annotated_pixels = mask.sum()
        coverage = (finite & mask).sum() / max(annotated_pixels, 1)
        assert abs(view["distance_to_target_m"] - (6 - 1.25 * index)) <= 0.005
        checks.append({
            "rgb": view["result"]["files"]["rgb"],
            "distance_m": view["distance_to_target_m"],
            "tree_pixels": int(annotated_pixels),
            "tree_depth_fraction": float(coverage),
            "pass": bool(annotated_pixels >= 256 and coverage >= 0.95)
        })
    return {"id": manifest["run_id"], "dir": str(folder), **manifest["approach"], "frames": checks,
            "pass": all(frame["pass"] for frame in checks)}


def previews(routes, output):
    for offset in range(0, len(routes), 6):
        group = routes[offset:offset + 6]
        sheet = Image.new("RGB", (1000, len(group) * 221), (16, 21, 18))
        draw = ImageDraw.Draw(sheet)
        for row, route in enumerate(group):
            for col, frame in enumerate(route["frames"]):
                x, y = col * 200, row * 221
                image = Image.open(Path(route["dir"]) / frame["rgb"]).convert("RGB")
                sheet.paste(image.resize((198, 198), Image.Resampling.LANCZOS), (x, y + 21))
                pin = route["id"] + " " if col == 0 else ""
                label = f'{pin}{frame["distance_m"]:.2f}m | {frame["tree_pixels"]}'
                draw.text((x + 3, y + 3), label, fill="white" if frame["pass"] else "#ff8165")
        sheet.save(output / f"pilot_views_{offset // 6 + 1}.jpg", quality=92)


def assign_routes(passed, test_pilot=None):
    per_sector = defaultdict(list)
    for route in passed:
        per_sector[route["sector"]].append(route["elevation_deg"])
    sectors = sorted(per_sector)
    pairs = [pair for pair in combinations(sectors, 2) if min(pair[1]-pair[0], 12-(pair[1]-pair[0])) >= 3]
    choices = [(a, b) for a in pairs for b in pairs if not set(a).intersection(b)]
    if len(sectors) < 6 or not choices:
        raise ValueError(f"Insufficient 6 m to 1 m coverage for three splits: {sectors}")
    # Favor opposite views in the held-out splits and spread the four held-out sectors.
    def spacing(sectors):
        s = sorted(sectors)
        return min((s[(i + 1) % len(s)] - v) % 12 for i, v in enumerate(s))
    if test_pilot is not None:
        choices = [pair for pair in choices if test_pilot["sector"] in pair[1]]
    val, test = max(choices, key=lambda pair: (spacing(pair[0] + pair[1]), spacing(pair[0])+spacing(pair[1])))
    train = sorted(set(sectors) - set(val) - set(test))
    selection = {"train": (train, 80), "val": (val, 10), "test": (test, 10)}
    rand = random.Random(261119)
    routes = []
    for split, (bins, quota) in selection.items():
        directions = {}
        for sector in bins:
            pool = {(15 + sector * 30 + d, e + h) for e in per_sector[sector]
                    for d in range(-8, 9, 2) for h in (-2, -1, 0, 1, 2)}
            queue = sorted(pool)
            rand.shuffle(queue)
            if split == "test" and test_pilot is not None and sector == test_pilot["sector"]:
                direction = (test_pilot["azimuth_deg"], test_pilot["elevation_deg"])
                queue.remove(direction)
                queue.insert(0, direction)
            directions[sector] = deque(queue)
        for i in range(quota):
            sector = bins[i % len(bins)]
            if not directions[sector]:
                raise ValueError(f"Insufficient unused directions in sector {sector}")
            azimuth, elevation = directions[sector].popleft()
            routes.append({"runId": f"approach_{split}_{i:03d}", "split": split, "sector": sector,
                           "azimuthDeg": azimuth, "elevationDeg": elevation})
    return routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--routes", type=Path, help="Save assignments after visual review")
    parser.add_argument("--exclude", nargs="*", default=[], help="Pilot IDs with washed-out, missing or occluded RGB")
    parser.add_argument("--test-pilot", help="Reserve a passing pilot direction with a DA2 landing reference for test")
    args = parser.parse_args()
    candidates = [inspect(d) for d in sorted(args.pilot.glob("pilot_*")) if d.is_dir()]
    if len(candidates) != 24:
        raise ValueError(f"Expected 24 pilot paths, found {len(candidates)}")
    if args.routes:
        passed = [route for route in candidates if route["pass"] and route["id"] not in args.exclude]
        test_pilot = next((route for route in passed if route["id"] == args.test_pilot), None)
        if args.test_pilot and test_pilot is None:
            raise ValueError(f"Requested test pilot did not pass: {args.test_pilot}")
        selected = assign_routes(passed, test_pilot)
        args.routes.write_text(json.dumps(selected, indent=2) + "\n")
    else:
        previews(candidates, args.pilot)
    print(json.dumps([{"id": route["id"], "pass": route["pass"], "tree_pixels": [
        frame["tree_pixels"] for frame in route["frames"]]} for route in candidates]))


if __name__ == "__main__":
    main()
