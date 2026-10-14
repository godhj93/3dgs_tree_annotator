#!/usr/bin/env python3
"""Pack one synchronized Save Current Set into the native benchmark format."""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path, help="Folder containing RGB, Mask, Depth")
    parser.add_argument("prefix", help="Shared saved frame prefix, without extension")
    parser.add_argument("output", type=Path, help="Output binary fixture path")
    args = parser.parse_args()

    rgb = np.asarray(Image.open(args.export_dir / "RGB" / f"{args.prefix}.png").convert("RGBA"))
    mask = np.asarray(Image.open(args.export_dir / "Mask" / f"{args.prefix}.png").convert("L"))
    depth = np.load(args.export_dir / "Depth" / f"{args.prefix}.npy", allow_pickle=False)
    pose = json.loads((args.export_dir / "Pose" / f"{args.prefix}.json").read_text())
    if rgb.shape != (640, 640, 4) or mask.shape != (640, 640) or depth.shape != (640, 640):
        raise ValueError("RGB, mask, and depth must belong to a 640x640 set")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        file.write(rgb.tobytes())
        file.write(mask.tobytes())
        file.write(np.asarray(depth, dtype="<f4").tobytes())
        k = pose["intrinsics"]
        camera = [k[0][0], k[1][1], k[0][2], k[1][2]]
        camera += [pose["camera_to_world"][row][col] for row in range(3) for col in range(3)]
        file.write(struct.pack("<13fi", *camera, 2 if pose.get("world_up_axis") == "z" else 1))
    print(args.output)


if __name__ == "__main__":
    main()
