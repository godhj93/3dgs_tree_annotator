// Per-link pseudo-3D cylinder screening. DA2 depth remains relative: widths
// and fitted spans below are image-space cues, not physical load estimates.
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
import { cameraDepth, validDepth } from '../static/lib/branch-detector/depth-geometry.mjs';

const WIDTH = 320, FULL = 640;
const isGreenFoliage = (r, g, b) => g >= 35 && g - r >= 10 && g - b >= 8 &&
    g > r * 1.1 && g > b * 1.08;

const principalAxis = (n, sum, outer) => {
    if (n < 6) return { axis: [1, 0, 0], linearity: 0, reliable: false };
    const center = sum.map(value => value / n);
    const covariance = outer.map((value, i) => value / n - center[Math.floor(i / 3)] * center[i % 3]);
    const trace = covariance[0] + covariance[4] + covariance[8];
    let axis = [1, 0.37, 0.11];
    for (let iteration = 0; iteration < 12; iteration++) {
        const next = [0, 1, 2].map(row => covariance[row * 3] * axis[0] +
            covariance[row * 3 + 1] * axis[1] + covariance[row * 3 + 2] * axis[2]);
        const norm = Math.hypot(...next);
        if (norm < 1e-12) break;
        axis = next.map(value => value / norm);
    }
    const eigenvalue = axis.reduce((value, component, row) => value + component *
        (covariance[row * 3] * axis[0] + covariance[row * 3 + 1] * axis[1] +
            covariance[row * 3 + 2] * axis[2]), 0);
    const linearity = trace > 1e-12 ? clamp((3 * eigenvalue / trace - 1) / 2, 0, 1) : 0;
    return { axis, linearity, reliable: trace > 1e-12 };
};

const projectedLine = s => {
    const n = s.woodPx;
    if (n < 6) return { axis: [1, 0], linearity: 0, center: [0, 0] };
    const cx = s.xSum / n, cy = s.ySum / n;
    const xx = s.xxSum / n - cx * cx, yy = s.yySum / n - cy * cy;
    const xy = s.xySum / n - cx * cy;
    const angle = Math.atan2(2 * xy, xx - yy) / 2;
    const major = (xx + yy + Math.hypot(xx - yy, 2 * xy)) / 2;
    const minor = Math.max(0, xx + yy - major);
    return { axis: [Math.cos(angle), Math.sin(angle)],
        linearity: major > 1e-8 ? clamp(1 - minor / major, 0, 1) : 0,
        center: [cx, cy] };
};

export const resolveGravityWorld = (sample, pose) => {
    const fallback = pose.world_up_axis === 'z' ? [0, 0, -1] : [0, -1, 0];
    if (!sample) return { vector: fallback, source: 'scene_up' };
    if (!['camera', 'world'].includes(sample.frame) || !Array.isArray(sample.vector) ||
        sample.vector.length !== 3 || sample.vector.some(value => !Number.isFinite(value))) {
        throw new Error('IMU gravity expects {frame:"camera"|"world", vector:[x,y,z]}');
    }
    const world = sample.frame === 'world' ? sample.vector : [0, 1, 2].map(row =>
        pose.camera_to_world[row][0] * sample.vector[0] +
        pose.camera_to_world[row][1] * sample.vector[1] +
        pose.camera_to_world[row][2] * sample.vector[2]);
    const norm = Math.hypot(...world);
    if (norm < 1e-8) throw new Error('IMU gravity vector must be nonzero');
    return { vector: world.map(value => value / norm), source: 'imu',
        frame: sample.frame, captured_at_utc: sample.captured_at_utc ?? null };
};

// labels is the accepted merged-object gate. sourceLabels keeps the original
// DA2 skeleton links: PCA merging cannot reunite the arms of a Y junction.
export const segmentFinalLanding = ({ labels, sourceLabels = labels, objects,
    sourceObjects = objects, mask, depth, rgb, pose, settings, depthKind = 'relative_inverse' }) => {
    if (labels.length !== WIDTH * WIDTH || sourceLabels.length !== labels.length ||
        mask.length !== FULL * FULL || depth.length !== mask.length || rgb.length !== mask.length * 4) {
        throw new Error('Final landing segmentation expects aligned 320/640 inputs');
    }
    const byId = new Map(sourceObjects.map(object => [object.id, object]));
    const stats = new Map();
    const depthSamples = [];
    let sampled = 0;
    for (let p = 0; p < mask.length; p++) {
        if (!mask[p] || !validDepth(depth[p], depthKind)) continue;
        if (sampled++ % 16 === 0) depthSamples.push(depth[p]);
    }
    depthSamples.sort((a, b) => a - b);
    const low = depthSamples[Math.floor(depthSamples.length * 0.02)] ?? 0;
    const high = depthSamples[Math.floor(depthSamples.length * 0.98)] ?? low;
    const depthRange = Math.max(high - low, 1e-6);
    const rotation = pose.camera_to_world, gravity = pose.gravity.vector;
    const vertical = [0, 1, 2].map(column => -gravity.reduce((sum, value, row) =>
        sum + value * rotation[row][column], 0));
    const fx = pose.intrinsics[0][0], fy = pose.intrinsics[1][1];
    const cx = pose.intrinsics[0][2], cy = pose.intrinsics[1][2];
    const allHeights = [];
    let visible = 0;
    for (let y = 0; y < FULL; y++) for (let x = 0; x < FULL; x++) {
        const p = y * FULL + x, small = (y >> 1) * WIDTH + (x >> 1);
        if (!mask[p] || !validDepth(depth[p], depthKind)) continue;
        const z = cameraDepth(depth[p], depthKind, low, high);
        const pointX = (x - cx) / fx * z, pointY = -(y - cy) / fy * z;
        const height = vertical[0] * pointX + vertical[1] * pointY - vertical[2] * z;
        if (visible++ % 4 === 0) allHeights.push(height);
        const id = sourceLabels[small];
        if (!id || !labels[small] || !byId.has(id)) continue;
        let s = stats.get(id);
        if (!s) {
            s = { mergedId: labels[small], areaPx: 0, woodPx: 0, depthSum: 0, depthSq: 0,
                rgbSum: [0, 0, 0], heights: [], pixels: [], sum3: [0, 0, 0],
                outer3: Array(9).fill(0), xSum: 0, ySum: 0, xxSum: 0, yySum: 0, xySum: 0 };
            stats.set(id, s);
        }
        s.areaPx++;
        const r = rgb[p * 4], g = rgb[p * 4 + 1], b = rgb[p * 4 + 2];
        if (settings.rejectFoliage !== false && isGreenFoliage(r, g, b)) continue;
        s.woodPx++;
        s.pixels.push(p);
        s.xSum += x; s.ySum += y; s.xxSum += x * x; s.yySum += y * y; s.xySum += x * y;
        s.depthSum += depth[p]; s.depthSq += depth[p] * depth[p];
        s.heights.push(height);
        s.rgbSum[0] += r; s.rgbSum[1] += g; s.rgbSum[2] += b;
        const point = [pointX, pointY, -z];
        for (let a = 0; a < 3; a++) {
            s.sum3[a] += point[a];
            for (let column = 0; column < 3; column++) s.outer3[a * 3 + column] += point[a] * point[column];
        }
    }
    allHeights.sort((a, b) => a - b);
    const heightLow = allHeights[Math.floor(allHeights.length * 0.05)] ?? 0;
    const heightHigh = allHeights[Math.floor(allHeights.length * 0.95)] ?? heightLow;
    const heightRange = Math.max(heightHigh - heightLow, 1e-6);
    const results = [], eligiblePixels = new Map();
    for (const [id, s] of stats) {
        const object = byId.get(id);
        const fit = principalAxis(s.woodPx, s.sum3, s.outer3);
        const line = projectedLine(s);
        s.heights.sort((a, b) => a - b);
        const relativeHeight = ((s.heights[Math.floor(s.heights.length / 2)] ?? heightLow) - heightLow) / heightRange;
        const meanDepth = s.woodPx ? s.depthSum / s.woodPx : 0;
        const depthStdRelative = Math.sqrt(Math.max(0, s.depthSq / Math.max(1, s.woodPx) - meanDepth * meanDepth)) / depthRange;
        const centerlineLengthPx = object.centerlineLengthPx ?? 0;
        const centerlineLengthCm = object.centerlineLengthCm ?? null;
        const apparentWidthPx = s.woodPx / Math.max(1, centerlineLengthPx);
        const radius = clamp(apparentWidthPx * 0.75, 1.5, settings.maxWidthPx / 2);
        let minT = Infinity, maxT = -Infinity;
        for (const p of s.pixels) {
            const t = ((p % FULL) - line.center[0]) * line.axis[0] +
                (Math.floor(p / FULL) - line.center[1]) * line.axis[1];
            minT = Math.min(minT, t); maxT = Math.max(maxT, t);
        }
        const bins = new Uint8Array(8), corridor = [];
        let squaredRadius = 0;
        for (const p of s.pixels) {
            const x = p % FULL, y = Math.floor(p / FULL);
            const dx = x - line.center[0], dy = y - line.center[1];
            const t = dx * line.axis[0] + dy * line.axis[1];
            const radial = Math.abs(-dx * line.axis[1] + dy * line.axis[0]);
            if (radial > radius) continue;
            corridor.push(p);
            squaredRadius += radial * radial;
            bins[clamp(Math.floor((t - minT) / Math.max(1, maxT - minT) * 8), 0, 7)] = 1;
        }
        const fittedLengthPx = Number.isFinite(minT) ? maxT - minT : 0;
        const cylinderWidthPx = corridor.length ? 2 * Math.sqrt(3 * squaredRadius / corridor.length) : 0;
        const corridorFraction = corridor.length / Math.max(1, s.woodPx);
        const coverage = bins.reduce((sum, value) => sum + value, 0) / bins.length;
        let longestGap = 0, gap = 0;
        for (const occupied of bins) {
            gap = occupied ? 0 : gap + 1;
            longestGap = Math.max(longestGap, gap);
        }
        const woodFraction = s.woodPx / Math.max(1, s.areaPx);
        const angleToGravityDeg = fit.reliable ? Math.acos(clamp(Math.abs(fit.axis.reduce((sum, value, i) =>
            sum + value * vertical[i], 0)), 0, 1)) * 180 / Math.PI : 0;
        const angleFromHorizontalDeg = 90 - angleToGravityDeg;
        // The pseudo-3D points are already expressed relative to the simulated
        // camera pose. Bearing uses only their direction, so DA2's unknown
        // metric scale does not affect this front-of-camera test.
        const centerCamera = s.sum3.map(value => value / Math.max(1, s.woodPx));
        const centerBearingDeg = Math.atan2(Math.hypot(centerCamera[0], centerCamera[1]),
            -centerCamera[2]) * 180 / Math.PI;
        const sideOnDeg = fit.reliable ? Math.acos(clamp(Math.abs(fit.axis[2]), 0, 1)) * 180 / Math.PI : 0;
        const endOnDeviationDeg = 90 - sideOnDeg;
        const imageTiltDeg = Math.atan2(Math.abs(line.axis[1]), Math.abs(line.axis[0])) * 180 / Math.PI;
        const isTrunk = object.isTrunk ?? (object.trunkCenterlineLengthPx >= 6 &&
            object.trunkCenterlineLengthPx / Math.max(1, centerlineLengthPx) >= 0.35);
        const reasons = [];
        if (!s.areaPx || !fit.reliable) reasons.push('unreliable');
        if (woodFraction < (settings.minWoodyFraction ?? 0.25)) reasons.push('foliage');
        const tooShort = depthKind === 'camera_z_m' && settings.minLengthCm !== undefined ?
            !Number.isFinite(centerlineLengthCm) || centerlineLengthCm < settings.minLengthCm :
            centerlineLengthPx < settings.minLengthPx || fittedLengthPx < settings.minLengthPx * 0.5;
        if (tooShort) reasons.push('short');
        if (fit.linearity < settings.minLinearity ||
            line.linearity < (settings.minProjectedLinearity ?? 0.65)) reasons.push('low_linearity');
        if (apparentWidthPx > settings.maxWidthPx || cylinderWidthPx > settings.maxWidthPx) reasons.push('wide');
        if (corridorFraction < (settings.minCylinderInlierFraction ?? 0.65) ||
            coverage < 0.5 || longestGap > 2) reasons.push('not_cylindrical');
        if (angleFromHorizontalDeg > (settings.maxAngleFromHorizontalDeg ?? 90)) reasons.push('tilted');
        if (settings.strictCameraAngles) {
            if (centerBearingDeg > (settings.maxCenterBearingDeg ?? 90)) reasons.push('off_center');
            if (endOnDeviationDeg > (settings.maxEndOnDeviationDeg ?? 90)) reasons.push('end_on');
            if (imageTiltDeg > (settings.maxImageTiltDeg ?? 90)) reasons.push('screen_tilt');
        }
        if (depthStdRelative > settings.maxDepthStdRelative) reasons.push('depth_spread');
        if (relativeHeight < settings.minRelativeHeight) reasons.push('low_height');
        if (settings.excludeTrunks && isTrunk) reasons.push('trunk');
        // The camera controls are preferences by default: a poor view scores
        // lower but does not make the whole segmentation disappear.
        const preference = (angle, allowance) => 1 / (1 + (angle / Math.max(1, allowance)) ** 2);
        const viewScore = 0.15 + 0.85 *
            preference(centerBearingDeg, settings.maxCenterBearingDeg ?? 90) *
            preference(endOnDeviationDeg, settings.maxEndOnDeviationDeg ?? 90) *
            preference(imageTiltDeg, settings.maxImageTiltDeg ?? 90);
        const score = fittedLengthPx * fit.linearity * corridorFraction * coverage * woodFraction * viewScore /
            (1 + cylinderWidthPx / 12 + depthStdRelative * 2 + angleFromHorizontalDeg / 30);
        results.push({ id, merged_id: s.mergedId, area_px: s.areaPx, woody_area_px: s.woodPx,
            wood_fraction: woodFraction, centerline_length_px: centerlineLengthPx,
            centerline_length_cm: depthKind === 'camera_z_m' ? centerlineLengthCm : null,
            fitted_length_px: fittedLengthPx, apparent_width_px: apparentWidthPx,
            cylinder_width_px: cylinderWidthPx, cylinder_inlier_fraction: corridorFraction,
            longitudinal_coverage: coverage, longest_empty_axial_bins: longestGap,
            projected_linearity: line.linearity,
            linearity: fit.linearity, angle_from_horizontal_deg: angleFromHorizontalDeg,
            center_px: line.center, center_camera_pseudo3d: centerCamera,
            center_camera: centerCamera, geometry_units: depthKind === 'camera_z_m' ? 'approximate_meters' : 'relative_pseudo3d',
            center_bearing_deg: centerBearingDeg, side_on_deg: sideOnDeg,
            end_on_deviation_deg: endOnDeviationDeg,
            image_tilt_deg: imageTiltDeg, view_score: viewScore,
            relative_depth_std: depthStdRelative, relative_height: relativeHeight,
            mean_relative_depth: meanDepth,
            mean_depth: meanDepth,
            mean_rgb: s.rgbSum.map(sum => Math.round(sum / Math.max(1, s.woodPx))),
            is_trunk: isTrunk, score: Number.isFinite(score) ? score : 0, reasons });
        if (!reasons.length) eligiblePixels.set(id, corridor);
    }
    const eligible = results.filter(result => !result.reasons.length)
        .sort((a, b) => b.score - a.score || a.id - b.id);
    const selected = eligible.slice(0, clamp(Math.round(settings.topK), 1, 253));
    const ids = new Uint32Array(FULL * FULL);
    for (const candidate of selected) for (const p of eligiblePixels.get(candidate.id)) ids[p] = candidate.id;
    return { ids, results, eligible, selected, depthRange, heightLow, heightHigh };
};
