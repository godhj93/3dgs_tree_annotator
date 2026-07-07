# 3DGS Tree Annotator

Browser-based 3D Gaussian Splat annotation tool for tree and branch labeling.

This repository packages a customized SuperSplat viewer that can:

- load a 3DGS `.ply` file in the browser
- place and edit multiple 3D bounding boxes
- segment Gaussian centers inside each 3D box
- export the current renderer view as RGB plus per-box mask images
- run spiral camera capture around a chosen point with configurable radii and angle step

## Run

From this repository root:

```bash
python3 -m http.server 8027 --bind 0.0.0.0
```

Then open:

```text
http://localhost:8027/app/
```

For a local PLY, put it somewhere under the repository or under the served parent directory and open:

```text
http://localhost:8027/app/?load=../path/to/model.ply&filename=model.ply&control=fly
```

The app must be served over HTTP. Opening `app/index.html` directly from disk is not reliable because the browser needs to fetch `.ply`, `.wasm`, and app assets.

## Annotation Workflow

1. Load a 3DGS `.ply`.
2. Use `Place 10cm Cube` or numeric fields to place a 3D bbox.
3. Use `Add BBox` / `Remove BBox` for multiple labels.
4. Adjust bbox center, size, and Euler rotation.
5. Set `View mask` radius in pixels.
6. Click `Export View RGB+Mask` for the current renderer camera.

Exported current-view files:

```text
current_rgb.png
current_overlay.png
current_01_<bbox_id>_mask.png
current_02_<bbox_id>_mask.png
current_annotation.json
```

## Spiral Export

The `Auto capture` controls generate renderer-view data by moving the camera around a target point.

- `cap-cx/cy/cz`: spiral target center in current scene units
- `d theta`: angular step in degrees
- radii list: comma/space separated scene-unit radii, e.g. `1,2,3,4,5,6,7,8,9,10`
- `Choose Output Folder`: writes files directly to a selected local folder in Chrome/Edge
- `Run Spiral Export`: captures RGB, overlay, masks, annotations, and `spiral_manifest.json`

Example output:

```text
spiral_0000_r1p000_t0p000_rgb.png
spiral_0000_r1p000_t0p000_overlay.png
spiral_0000_r1p000_t0p000_01_branch_001_mask.png
spiral_0000_r1p000_t0p000_annotation.json
spiral_manifest.json
```

## Data Policy

Do not commit datasets, trained models, exports, or PLY files to this repository. The `.gitignore` blocks common large artifacts such as:

- `*.ply`
- `data/`
- `outputs/`
- `exports/`
- `renders/`

## Credits

This tool is based on SuperSplat by PlayCanvas and includes a customized annotation panel for 3DGS tree labeling.

