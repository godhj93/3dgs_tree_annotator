# 3DGS Tree Annotator

Browser-based 3D Gaussian Splat annotation tool for tree and branch labeling.

This repository packages a customized SuperSplat viewer that can:

- load a 3DGS `.ply` file in the browser
- place and edit multiple 3D bounding boxes
- segment Gaussian centers inside each 3D box
- export the current renderer view as RGB plus per-box mask images
- run spiral camera capture around a chosen point with configurable radii and angle step

The built standalone app lives in `app/`. The editable customized SuperSplat source lives in `supersplat/`.

## Run

Running the already-built app does not require `npm install` or `node_modules`.
Only Python is needed to serve the static files:

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

## Live RGB / Segmentation / DA2 vs 3DGS Depth

Start the annotation app with its persistent Depth Anything V2 Small service:

```bash
bash scripts/serve_with_depth.sh
```

This uses the `da3-ros` conda environment, GPU 0, and the existing DA2 source
at the parent project's `Depth-Anything-V2/`. Override `CONDA_ENV`,
`CUDA_VISIBLE_DEVICES`, or `PYTHON_BIN` as needed. Python needs PyTorch,
OpenCV, NumPy, torchvision, huggingface_hub, FastAPI and Uvicorn. The Small
checkpoint is loaded once from the Hugging Face cache (downloaded on first use).
Use `--da2-source /path/to/Depth-Anything-V2`, `--checkpoint /path/to/model.pth`,
`--device cpu`, `--host`, or `--port` to override server settings.

When served from the parent project, open
`http://localhost:8027/3dgs-splat-labeler/app/` and load a PLY. The bottom-right
BBox Manager shows shared RGB/segmentation, both depth maps, and paired branch-analysis stages.
Add `boxes3d=/annotations/dji_0967_saved_boxes_3d.json` to the DJI model URL
to restore the eight BBoxes saved in `~/Downloads/boxes_3d.json`. A `metricScale`
URL parameter keeps the scene's current metric calibration when importing older
boxes that contain raw scene coordinates.
The DJI Nerfstudio PLY uses its vertical Z axis for Fly left-drag yaw and WASD
movement. Other scenes retain the original Y-up controls. `up=z` and `up=y`
explicitly select either convention.

- The LI-IMX678-MIPI-HL-118H camera preset uses the datasheet's horizontal
  FoV 118° and sensor 3856×2176. The vertical FoV 86.407° is a **pinhole
  approximation** inferred from that aspect ratio, not a measured calibration.
  Lens distortion is not modeled. The nominal focal length is metadata only;
  intrinsics are calculated from FoV. The main viewport preserves sensor aspect.
- The full camera image is rendered at 963×544, then resized to 640×640.
  DA2 runs at the sensor aspect before its raw relative result is resized.
  Resizing the full rectangular FoV into a square intentionally yields different
  `fx` and `fy`; the output is not a central crop or a 118° square frustum.
- Stop moving for 400ms to update. Moving or editing annotations during inference
  invalidates that frame. `Refresh` retries a stationary view after a depth error.
  Uncheck `Live preview` to pause it and restore the regular camera view.
- The visible mask follows the include BBox union minus remove BBoxes.
  All rendered Gaussians participate in opacity compositing, so unannotated
  foreground occludes annotated branches. It uses visible annotated contribution
  ≥50% of total opacity, with total opacity ≥0.05. It does not alter editor selection.
- `Binary mask` switches the preview between a yellow RGB overlay and the binary
  mask. No boxes produces an empty mask. Raw RGB excludes edit handles/highlights.
- DA2 and 3DGS use the same captured RGB, annotation mask, intrinsics, and pose.
  Every comparison row keeps **DA2 on the left and 3DGS on the right**: depth
  support, depth-aware skeleton, separated branches, PCA axes, merged regions,
  gravity-filtered regions, and final landing branches. Source-local IDs are
  not cross-source correspondences.
- **Depth gap / robust range** (default 0.15) acts before skeletonization.
  Each 2×2 masked cell contributes its median valid depth. An eight-neighbor
  jump above the threshold times that source's 2nd–98th percentile depth range
  removes the far-side cell as a one-cell barrier, then the remaining mask is
  thinned at 320×320. Orange shows depth barriers; gray shows missing depth.
  Unknown/nonpositive 3DGS depth is excluded rather than filled from DA2.
  Constant depth preserves the mask. Depth support can therefore change the
  skeleton itself, not just later ranking.
- DA2 keeps its raw inverse-like relative values and dimensionless pseudo-3D
  reconstruction. 3DGS uses its positive camera-forward depth directly with
  camera intrinsics and the current meters-per-scene-unit calibration. Its
  PCA and final geometry use approximate meters, without DA2's inverse mapping
  or image-tangent fallback. Both depth previews use robust percentile colors;
  multiplying the 3DGS scale changes coordinates, not the normalized colors.
- Expand **Shared PCA, gravity and landing filters** to adjust both paths
  together. PCA uses the same maximum axis angle (default 20°) and normalized
  depth-gap condition. All member axes must agree before adjacent regions
  merge. Gravity accepts a merged axis or a reliable component within 20° of
  horizontal by default; **Require each component horizontal** applies the
  stricter gate. Minimum skeleton length is 12 image pixels for DA2 and 10 cm for 3DGS; **Exclude trunks**
  is off by default. These controls use cached depths and do not rerun DA2.
- 3DGS skeleton length is a visible 3D arc length in cm: backproject the
  depth at each skeleton node with the captured intrinsics, sum each undirected
  connection once, and avoid diagonal shortcuts at corners. Missing depth is
  not bridged. Skeleton and final-branch captions show cm, and hovering over a
  skeleton/region shows that link's length. Merged-region lengths sum their
  component links. DA2 does not receive an invented cm conversion.
- The final stage preserves source junction-link IDs and applies the existing
  landing criteria independently: maximum apparent width 24px, minimum 3D
  linearity 0.5, image linearity 0.6, cylinder inlier fraction 0.4, maximum
  relative depth spread 0.3, minimum relative height 0.1, non-green fraction
  0.25, and Top K 3. Green foliage rejection is on. Camera approach allowances
  are 30° for center bearing, end-on deviation, and screen tilt; they affect
  ranking unless **Require angles** is checked. Angle/ratio/projected-width thresholds
  are shared. Skeleton length has separate DA2-pixel and 3DGS-cm controls. Hover over
  either final image to see its source ID and rejection reasons.
- Final mask difference uses white for overlap, cyan for DA2-only, and magenta
  for 3DGS-only pixels. IoU measures mask overlap, not accuracy against ground
  truth. Two empty masks are explicitly reported. **Main overlay** selects
  DA2, 3DGS (default), or final-mask difference; **View** selects skeleton,
  landing candidates, or final segmentation. Both columns use one annotation
  crop; disable **Zoom to annotation** to see the full frame.
- Analyses run in separate workers, tagged by frame, source, and settings
  revision. A source can finish while the other is still running or has failed.
  Camera/annotation changes discard stale responses. The source-specific
  results are available under `annotation.liveSnapshot.analyses.da2` and
  `.analyses['3dgs']`; legacy detection/filter fields remain DA2 aliases.
- The simulated camera pose is read from the current frame. To feed an IMU
  gravity direction pointing **toward the ground** in camera coordinates, call
  `scene.events.invoke('annotation.setImuGravity', {frame:'camera', vector:[x,y,z], captured_at_utc:'...'})`
  with the synchronized sample. `frame:'world'` also accepts a world-frame
  gravity vector. Camera-frame vectors are rotated into world coordinates using
  the captured camera pose. Updating the vector recomputes the gravity and final
  panels for the current frame without rerunning DA2. Pass `null` to use the
  scene's up axis as a simulation fallback. The pose JSON records the gravity
  vector and its source.
- The landing-candidate mode shows branch instances and ranked landing
  candidates over the current RGB frame. A C++17 core uses
  RGB, each depth-prepared annotation mask, its depth values, camera intrinsics, and
  camera rotation. It thins the 320×320 mask, splits centerlines at junctions,
  assigns visible pixels to branches, and ranks slender, long, relatively level
  candidates. DA2 pixels are projected into **dimensionless pseudo-3D** and 3DGS pixels into
  calibrated geometry. Scores are inspection candidates,
  not physical landing or load-bearing decisions. The worker processes a
  stationary synchronized frame; it does not infer while the camera is moving.
- Use **maximize** in the BBox Manager header to open a split view: the live
  camera stays on the left and the enlarged previews and controls stay on the
  right. Drag on the camera to turn the view, or use WASDQE to fly while the
  manager is open. Click the camera after editing a numeric field to resume
  keyboard movement. **Restore** returns to the previous size, including a
  manually dragged size. The upper-left corner handle remains available for resizing.
  Set **Top K** (default 3) at any BBox Manager size to show only the K
  highest-scoring ground-parallel, non-trunk branches in the landing tile.
  The numbers 1, 2, 3... are score ranks; the separate ID is the branch's
  instance ID in that frame. **Show on main
  view** overlays those same candidates on the camera viewport with translucent
  colors from the selected depth source. It is on by default and clears the overlay as
  soon as the camera or annotations change, until a matching frame is ready.
  Detection continues for the main view even while the BBox Manager is hidden.
- **Yellow segmentation** toggles the yellow selection tint in the main view
  and the yellow mask overlay in the segmentation preview. It does not change
  the annotation or the saved mask. **Binary mask** still displays the mask.
- Expand **Branch search** to adjust maximum tilt (5–80°, default 25°), minimum
  visible length (6–80 px, default 12 px), minimum length/width ratio (1–5×,
  default 1.5×), and minimum direction certainty (0–1, default 0.5).
  **Reset search** restores the defaults. These controls filter the current
  detections instantly and do not rerun depth or segmentation.
- **Exclude trunk** is off by default. When enabled, broad upright paths are
  removed from the candidate list and branch ID preview/export while the
  original segmentation mask remains intact. Eligible **Ground-parallel** candidates
  are scored and the selected top K are outlined in cyan and listed by rank.
  Rank is based on a heuristic score, not physical landing safety.
- **Save Figures** in the BBox Manager header saves every current panel figure
  as PNG, including stages below the scroll position. It exports RGB,
  the displayed segmentation, both depth previews, both sources' depth support,
  skeleton, separated regions, PCA axes, merged regions, gravity-filtered regions,
  final branches, and the final difference view (19 individual 640×640 PNGs).
  `overview.png` adds a two-column comparison with titles, captions, frame ID,
  current crop, calibration and filter values. The displayed zoom, colors and
  overlays are captured at button-click time before the folder chooser opens.
  Saving also works for a paused analysis; each image belongs to the captured
  analysis frame. The files go under `Figures/<frame>_figures_<timestamp>/` in
  the chosen output folder. If browser directory access is unavailable, one ZIP
  download contains the same 20 PNG files. A cancelled folder chooser saves no
  files. The button becomes available after both sources finish or report an
  error and at least one source has a result.
- `Save DA2 Set` becomes available after the current frame's DA2 depth and
  branch detection finish, including when only the 3DGS analysis has failed.
  Choose a folder to save `RGB/*.png`, `Mask/*.png`, `Depth/*.npy`,
  `Depth/*_preview.png`, `Pose/*.json`, `Branches/*_instances.png`,
  `Branches/*.json`, `Landing/*_ids.npy`, `Landing/*_preview.png`, and
  `Landing/*.json`, all using one timestamped prefix. The Landing NPY is an
  exact 640×640 `uint32` instance-ID mask (0 = background); its JSON records
  filters, cylinder-fit metrics, rejection reasons, selected junction-split
  link IDs, gravity source,
  and the shared frame ID. The instance PNG stores
  branch IDs as equal RGB byte values (0 = background); the detection JSON has
  rank scores, endpoints, center pseudo-3D coordinates, junctions, estimated
  angle and class, active filters, excluded trunk IDs, horizontal branch IDs,
  selected landing candidate IDs, and the matching frame ID. The instance PNG
  still stores all non-excluded branch IDs so that changing K only affects the
  landing visualization. Remaining IDs are stable when trunks are hidden.
  Images and float32 depth are exactly 640×640. Browsers without folder access
  download flat files with directory prefixes; JSON records those actual names.
- Pose JSON includes the shared frame ID, annotations, nominal camera model,
  intrinsics, XYZ/quaternion, camera-to-world, world-to-camera, and projection.
  Matrices are row-major; the camera uses OpenGL axes (+X right, +Y up, −Z forward).
  Translation is in scene units; calibrated meter position is recorded separately.
  `Copy Pose JSON` copies the shared captured camera pose.

The plain static `serve.sh` still opens the editor, but DA2 requires this Python
service. Legacy current-view/spiral/random/LOS export retains its existing format;
use `Save DA2 Set` for the new synchronized visible-mask/depth dataset.

The CPU postprocessor is shared by the browser WebAssembly worker and the native
benchmark. Rebuild both after editing it:

```bash
./branch_detector/build.sh
./branch_detector/build/benchmark
cd supersplat && npm run build
cd .. && bash scripts/sync_app_from_supersplat.sh
```

`clang++` with a `wasm32` target and `g++` are required. For a real captured
frame, `benchmark` accepts one 640×640 binary fixture containing contiguous
RGBA bytes, mask bytes, little-endian float32 depth, then camera intrinsics,
rotation, and world-up axis. Its p95 covers only
CPU branch postprocessing. Validate the 10 Hz requirement on VOXL2 and Hailo-15
CPUs with representative masks; the included x86 result is not a board claim.
Create a fixture from a saved set using
`python branch_detector/make_fixture.py <export-folder> <shared-prefix> /tmp/frame.bin`,
then run `./branch_detector/build/benchmark /tmp/frame.bin`. On either target
board, `NATIVE_ONLY=1 ./branch_detector/build.sh` builds the same CPU core
without requiring a WebAssembly toolchain.

### Validation

```bash
python -m unittest discover -s tests -p 'test_depth_server.py'
node --test tests/test_figure_export.mjs tests/test_live_view.mjs tests/test_depth_comparison.mjs tests/test_final_landing_analysis.mjs tests/test_pca_link_analysis.mjs tests/test_junction_analysis.mjs
# With the depth server running on 8027, installed Chrome and the DJI PLY:
node tests/integration_depth_comparison.mjs
# Camera, annotations, shared filter controls and legacy DA2 exports:
node tests/integration_live_view.mjs
cd supersplat
./node_modules/.bin/tsc --noEmit
npm run build
```

## Develop / Rebuild

The source under `supersplat/` is the customized SuperSplat app. To edit and rebuild the standalone app:

Use Node.js `>=20.19.0`.

Option A, with conda:

```bash
conda env create -f environment.yml
conda activate 3dgs-tree-annotator
```

Option B, with `nvm`:

```bash
nvm install
nvm use
```

```bash
cd supersplat
npm ci
npm run build
cd ..
bash scripts/sync_app_from_supersplat.sh
```

Or from the repository root:

```bash
npm run build:app
```

This rebuilds `supersplat/dist/` and copies it into `app/`.

`supersplat/package-lock.json` is committed, so `npm ci` recreates the same dependency versions on another machine. Do not commit `supersplat/node_modules/` or `supersplat/dist/`; both are generated and ignored. Commit the changed source files under `supersplat/src/` and the refreshed built app under `app/`.

## Annotation Workflow

1. Load a 3DGS `.ply`.
2. Calibrate scale if needed: use the `Measurement` tool, measure a known length, type the real length in meters, then click `Apply Scale`. Alternatively, select a fitted BBox, choose its X/Y/Z axis under `BBox dimension → 3DGS depth scale`, enter that side's measured real length in meters, and leave the field (or click `Apply Box Scale`). This changes the scene's meters-per-unit scale without resizing the BBox. The calibrated value and the reference BBox/axis/measurement are written to the URL so they survive reload, replacing an older `metricScale` URL reference. The reference is also included in annotation exports and pose/depth metadata; 3DGS depth and cm arc lengths explicitly use this scale.
3. Click `Draw BBox`, then click visible splat points to append 10cm bbox segments.
4. Use `Add +BBox` for regions that should be segmented and `Add Remove BBox` for regions that should be subtracted. Segmentation is `(union of include bboxes) - (union of remove bboxes)`.
5. Use `Undo Step` to revert the last bbox operation, or `Delete BBox` to delete the active bbox itself.
6. Adjust bbox center, size, Euler rotation, and segmentation role. Include boxes are red/blue; remove boxes are orange.
7. Set `View mask` radius in pixels.
8. Click `Export View RGB+Mask` for the current renderer camera.

After scale calibration, bbox center/size fields and spiral capture radii are shown and edited in meters. Exports include both calibrated meter values and raw scene-unit values.
The live panel displays DA2 relative depth alongside 3DGS opacity-weighted depth from the same camera pose. The 3DGS preview and `_3dgs_m.npy` export use the current meters-per-scene-unit calibration; uncovered pixels are NaN. The screen compares both depth-aware analyses; existing branch/landing exports continue to use DA2. New 3DGS branch-label dataset exports are deferred. The calibrated 3DGS depth is approximate geometry, especially around thin or translucent branches.
When multiple bboxes exist, the visible segmented splats and exported mask are the union of include boxes minus the union of remove boxes. The per-bbox records remain in JSON so annotators can see how each 3D box contributed to the semantic mask.

Use `Export 3D` to save the complete bbox state as `boxes_3d.json`. To resume later, load the same 3DGS scene, click `Import 3D`, and select that JSON file. Import restores include/remove roles, centers, sizes, rotations, active selection, and the exported metric scale unless a `metricScale` URL parameter supplies a newer calibration. The current bbox state is placed on the undo stack before replacement.

When an output folder is chosen, RGB and mask images are separated into dataset subdirectories while overlays, annotations, and manifests remain at the selected root:

```text
RGB/current_rgb.png
Mask/current_mask.png
current_overlay.png
current_annotation.json
```

If the browser does not support direct folder access, files fall back to ordinary flat browser downloads.

## Spiral Export

The `Auto capture` controls generate renderer-view data by moving the camera around a target point.

- `cap-cx/cy/cz`: capture target center in calibrated meters
- `d theta`: angular step in degrees
- radii list: comma/space separated calibrated meter radii, e.g. `1,2,3,4,5,6,7,8,9,10`
- `Choose Output Folder`: writes files directly to a selected local folder in Chrome/Edge
- `Run Spiral Export`: captures RGB, overlay, merged union mask, annotations, and `spiral_manifest.json`
- `Run Random Dome Export`: captures random views inside an upper hemisphere while excluding an inner vertical cylinder

After metric calibration, `cap-cx/cy/cz`, spiral radii, random outer hemisphere radius, and random inner cylinder radius are all edited in meters.

Random dome parameters:

- count: number of random views
- outer radius: maximum camera distance as an upper hemisphere around the target
- inner radius: excluded vertical cylinder radius around the target
- seed: optional deterministic random seed

Example output:

```text
RGB/spiral_0000_r1p000_t0p000_rgb.png
Mask/spiral_0000_r1p000_t0p000_mask.png
spiral_0000_r1p000_t0p000_overlay.png
spiral_0000_r1p000_t0p000_annotation.json
spiral_manifest.json
RGB/random_0000_d2p418_rgb.png
Mask/random_0000_d2p418_mask.png
random_0000_d2p418_overlay.png
random_0000_d2p418_annotation.json
random_dome_manifest.json
```

## LOS Waypoint Export

Use this mode when only manually approved camera regions should contribute views:

1. Set the fixed LOS target with `Use BBox Center` or the `cap-cx/cy/cz` fields.
2. Move the viewer camera to a good position and click `Add Current Camera`.
3. Repeat for at least two good camera positions. The purple path shows the ordered piecewise-linear route.
4. Set `steps / segment`. With `N` waypoints and `S` steps, the export contains `(N - 1) * S + 1` views.
5. Choose an output folder and click `Run LOS Path`.

Only camera XYZ position is interpolated. At every generated position, camera orientation is recomputed to look at the same fixed target with zero roll. Add more waypoints when a straight segment between two approved positions would cross a bad viewing region.

Each run receives a timestamped `los_...` prefix so repeated runs do not overwrite prior captures. RGB, overlay, mask, per-view annotation JSON, and a run manifest are exported.

## Data Policy

Do not commit datasets, trained models, exports, or PLY files to this repository. The `.gitignore` blocks common large artifacts such as:

- `*.ply`
- `data/`
- `outputs/`
- `exports/`
- `renders/`

## Credits

This tool is based on SuperSplat by PlayCanvas and includes a customized annotation panel for 3DGS tree labeling.
