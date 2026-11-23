import { cameraDepth } from './depth-geometry.mjs';

// Space screening in camera coordinates: +X right, +Y up, -Z forward.
export const defaultBirdSpace = Object.freeze({ enabled: true,
    widthM: 0.1,
    lengthM: 0.1,
    heightM: 0.1,
    marginM: 0.05,
    checkApproachPath: true,
    approachM: 1,
    checkApproachAngle: true,
    minApproachAngleDeg: 60 });
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const add = (a, b) => a.map((v, i) => v + b[i]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const mul = (a, s) => a.map(v => v * s);
const norm = a => Math.hypot(...a);
const unit = a => (norm(a) > 1e-9 ? mul(a, 1 / norm(a)) : null);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const valid = z => Number.isFinite(z) && z > 0;
export const cornerProbeRadiusM = 0.02;
export const cornerMinHitPixels = 3;
export const defaultMinimumBranchDiameterM = 0.05;

export const validateBirdSpace = (settings) => {
    const s = { ...defaultBirdSpace, ...settings };
    for (const k of ['widthM', 'lengthM', 'heightM']) if (!Number.isFinite(s[k]) || s[k] <= 0 || s[k] > 5) throw new Error(`Invalid bird ${k}`);
    for (const k of ['marginM', 'approachM']) if (!Number.isFinite(s[k]) || s[k] < 0 || s[k] > 10) throw new Error(`Invalid bird ${k}`);
    if (!Number.isFinite(s.minApproachAngleDeg) || s.minApproachAngleDeg < 0 || s.minApproachAngleDeg > 90) throw new Error('Invalid approach angle');
    return s;
};

const median = values => values.sort((a, b) => a - b)[Math.floor(values.length / 2)];

const alignDA2Depth = (depth, metricDepth, mask, inverse) => {
    if (metricDepth?.length !== depth.length) return null;
    const samples = annotated => {
        const pairs = [];
        for (let i = 0; i < depth.length; i++) {
            if ((!annotated || mask?.[i]) && Number.isFinite(depth[i]) && valid(metricDepth[i])) {
                pairs.push([inverse(depth[i]), 1 / metricDepth[i]]);
            }
        }
        return pairs;
    };
    let pairs = samples(true), region = 'annotation';
    if (pairs.length < 8) { pairs = samples(false); region = 'full_frame'; }
    if (pairs.length < 8) return null;
    const samplesUsed = pairs.length;
    const stride = Math.max(1, Math.ceil(pairs.length / 4096));
    pairs = pairs.filter((_, index) => index % stride === 0).sort((a, b) => a[0] - b[0]);

    let slope = median(pairs.map(([x, y]) => y / x)), intercept = 0;
    let method = 'metric_depth_ratio';
    const bins = Array.from({ length: 8 }, (_, i) => {
        const bin = pairs.slice(Math.floor(i * pairs.length / 8), Math.floor((i + 1) * pairs.length / 8));
        return [median(bin.map(pair => pair[0])), median(bin.map(pair => pair[1]))];
    });
    const slopes = [];
    for (let a = 0; a < bins.length; a++) for (let b = a + 1; b < bins.length; b++) {
        const dx = bins[b][0] - bins[a][0];
        if (dx >= 0.05) slopes.push((bins[b][1] - bins[a][1]) / dx);
    }
    const affineSlope = median(slopes);
    if (valid(affineSlope)) {
        const affineIntercept = median(pairs.map(([x, y]) => y - affineSlope * x));
        const residual = (a, b) => median(pairs.map(([x, y]) => Math.abs((a * x + b) / y - 1)));
        if (affineSlope * pairs[Math.floor(pairs.length * 0.02)][0] + affineIntercept > 0 &&
            residual(affineSlope, affineIntercept) < residual(slope, intercept)) {
            slope = affineSlope; intercept = affineIntercept; method = 'metric_inverse_affine';
        }
    }
    return { method, region, samples: samplesUsed, inverse_scale: slope, inverse_offset: intercept };
};

export const buildSpaceScene = (depth, intrinsics, width = 640, height = 640, da2ReferenceM = null, referenceMask = null, metricDepth = null) => {
    const [fx, fy, cx, cy] = intrinsics;
    if (!depth || depth.length !== width * height || ![fx, fy, cx, cy].every(Number.isFinite) || fx <= 0 || fy <= 0) throw new Error('Invalid metric space scene');
    let geometryUnits = 'approximate_meters', calibration = null;
    if (da2ReferenceM !== null || metricDepth) {
        da2ReferenceM ??= 1;
        if (!valid(da2ReferenceM)) throw new Error('Invalid DA2 reference distance');
        const samples = Array.from(depth).filter(Number.isFinite).sort((a, b) => a - b);
        const low = samples[Math.floor(samples.length * 0.02)] ?? 0;
        const high = samples[Math.min(samples.length - 1, Math.floor(samples.length * 0.98))] ?? low;
        const inverse = value => 1 / cameraDepth(value, 'relative_inverse', low, high);
        calibration = alignDA2Depth(depth, metricDepth, referenceMask, inverse);
        geometryUnits = calibration ? 'da2_calibrated_meters' : 'da2_assumed_meters';
        calibration ??= { method: 'assumed_median', median_reference_m: da2ReferenceM,
            inverse_scale: 1 / (inverse(samples[Math.floor(samples.length / 2)] ?? low) * da2ReferenceM), inverse_offset: 0 };
        depth = Float32Array.from(depth, value => {
            if (!Number.isFinite(value)) return NaN;
            const zInverse = calibration.inverse_scale * inverse(value) + calibration.inverse_offset;
            return valid(zInverse) ? 1 / zInverse : NaN;
        });
    }
    const points = new Float32Array(depth.length * 3);
    const pixels = [];
    for (let p = 0; p < depth.length; p++) {
        const z = depth[p];
        if (!valid(z)) continue;
        points[p * 3] = (p % width - cx) / fx * z;
        points[p * 3 + 1] = -(Math.floor(p / width) - cy) / fy * z;
        points[p * 3 + 2] = -z;
        pixels.push(p);
    }
    return { depth, intrinsics, width, height, points, pixels: Uint32Array.from(pixels), geometryUnits, calibration };
};

// Slab intersection gives continuous swept-box collision, without path sampling
// that could step over a thin branch. Bounds are relative to the feet.
export const sweptBoxInterval = (point, travel, low, high) => {
    let enter = 0, leave = 1;
    for (let k = 0; k < 3; k++) {
        if (Math.abs(travel[k]) < 1e-10) {
            if (point[k] < low[k] || point[k] > high[k]) return null;
        } else {
            const a = (point[k] - high[k]) / travel[k];
            const b = (point[k] - low[k]) / travel[k];
            enter = Math.max(enter, Math.min(a, b)); leave = Math.min(leave, Math.max(a, b));
            if (enter > leave) return null;
        }
    }
    return [enter, leave];
};

export const acuteAxisAngle = (axis, direction) => Math.acos(clamp(Math.abs(dot(axis, direction)), 0, 1)) * 180 / Math.PI;

const fitAxis = (points, center) => {
    if (points.length < 6) return null;
    const c = Array(9).fill(0);
    for (const p of points) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) c[a * 3 + b] += (p[a] - center[a]) * (p[b] - center[b]);
    const major = [c[0], c[4], c[8]].indexOf(Math.max(c[0], c[4], c[8]));
    let axis = [0, 1, 2].map(k => (k === major ? 1 : 0));
    for (let i = 0; i < 20; i++) {
        const previous = axis;
        const next = unit([0, 1, 2].map(a => c[a * 3] * previous[0] + c[a * 3 + 1] * previous[1] + c[a * 3 + 2] * previous[2]));
        if (!next) return null;
        axis = next;
    }
    const trace = c[0] + c[4] + c[8];
    const eigen = dot(axis, [0, 1, 2].map(a => c[a * 3] * axis[0] + c[a * 3 + 1] * axis[1] + c[a * 3 + 2] * axis[2]));
    return { axis, linearity: trace > 1e-9 ? clamp((3 * eigen / trace - 1) / 2, 0, 1) : 0 };
};

// Probe the visible silhouette in a metric plane transverse to the branch
// and viewing ray. The mask supplies the boundary; this source's depth
// supplies scale and prevents joining surfaces across a depth discontinuity.
const diameterProfile = (scene, mask, support, anchor, axis) => {
    const [fx, fy, cx, cy] = scene.intrinsics;
    const project = p => [cx + fx * p[0] / -p[2], cy - fy * p[1] / -p[2]];
    const point = Array.from(scene.points.subarray(anchor * 3, anchor * 3 + 3));
    const transverse = cross(point, axis);
    if (norm(transverse) < 0.2 * norm(point)) return null;
    const stepM = 0.0025;
    const normalStep = mul(unit(transverse), stepM);
    const maxSteps = Math.ceil(0.30 / stepM);
    const surfaceGate = Math.min(0.1, 0.035 - point[2] * 0.018);
    const supportGate = Math.min(0.22, 0.08 - point[2] * 0.045);
    const side = direction => {
        for (let n = 1; n <= maxSteps; n++) {
            const at = add(point, mul(normalStep, n * direction));
            if (at[2] >= -0.1) return { n, censored: true };
            const [x, y] = project(at).map(Math.round);
            if (x < 0 || x >= scene.width || y < 0 || y >= scene.height) return { n, censored: true };
            const p = y * scene.width + x;
            if (!mask[p]) return { n, censored: false };
            // Within the same segmented link, allow the surface to curve away
            // from the anchor. Outside it require a much tighter depth match.
            const maxGap = support.has(p) ? supportGate : surfaceGate;
            if (!valid(scene.depth[p]) || Math.abs(scene.depth[p] + at[2]) > maxGap) return { n, censored: true };
        }
        return { n: maxSteps + 1, censored: true };
    };
    const negative = side(-1), positive = side(1);
    return { widthM: (negative.n + positive.n - 1) * stepM, censored: negative.censored || positive.censored };
};

export const evaluateBranchDiameters = ({ scene, candidates, supportPixels, mask, minimumM }) => {
    if (!Number.isFinite(minimumM) || minimumM < 0.005 || minimumM > 0.5) throw new Error('Invalid branch diameter threshold');
    return candidates.map(candidate => {
        const unknown = { id: candidate.id, status: 'unknown', diameter_m: null, profiles: 0 };
        if (!scene || mask?.length !== scene.depth.length || !candidate.center_px?.every(Number.isFinite)) return unknown;
        const pixels = (supportPixels.get(candidate.id) ?? []).filter(p => mask[p] && valid(scene.depth[p]));
        if (pixels.length < 6) return unknown;
        const [u, v] = candidate.center_px;
        const distancePx = p => (p % scene.width - u) ** 2 + (Math.floor(p / scene.width) - v) ** 2;
        const centerPixel = pixels.reduce((best, p) => distancePx(p) < distancePx(best) ? p : best);
        const [fx, fy] = scene.intrinsics, z = scene.depth[centerPixel];
        const x0 = centerPixel % scene.width, y0 = Math.floor(centerPixel / scene.width);
        // Fit to observed pixels in the calibrated image plane. Choosing the
        // sections on the mask also works when a short, curved fragment would
        // fail to line up with extrapolated points along a straight 3D axis.
        const samples = pixels.map(p => ({ p, pos: [(p % scene.width - x0) * z / fx,
            (y0 - Math.floor(p / scene.width)) * z / fy, 0] }))
            .filter(({ p, pos }) => norm(pos) <= 0.18 &&
                Math.abs(scene.depth[p] - z) <= Math.min(0.22, 0.08 + z * 0.04));
        if (samples.length < 6) return unknown;
        const mean = [0, 1, 2].map(i => samples.reduce((sum, { pos }) => sum + pos[i], 0) / samples.length);
        const fit = fitAxis(samples.map(s => s.pos), mean);
        if (!fit || fit.linearity < 0.35) return unknown;
        const bins = new Map();
        for (const { p, pos } of samples) {
            const centered = sub(pos, mean);
            const along = dot(centered, fit.axis);
            const bin = Math.round(along / 0.02);
            if (Math.abs(bin) > 3) continue;
            const error = norm(sub(centered, mul(fit.axis, along))) +
                Math.abs(along - bin * 0.02) * 0.5;
            if (!bins.has(bin) || error < bins.get(bin).error) bins.set(bin, { p, error });
        }
        const support = new Set(pixels);
        const profiles = [...bins].flatMap(([bin, { p }]) => {
            const observed = diameterProfile(scene, mask, support, p, fit.axis);
            return observed === null ? [] : [{ bin, ...observed }];
        });
        if (profiles.length < 3 || !profiles.some(p => p.bin < 0) || !profiles.some(p => p.bin > 0)) {
            return { ...unknown, profiles: profiles.length };
        }
        const middle = values => {
            values.sort((a, b) => a - b);
            return (values[Math.floor(values.length / 2)] + values[Math.floor((values.length - 1) / 2)]) / 2;
        };
        const lower = middle(profiles.map(p => p.widthM));
        const upper = middle(profiles.map(p => p.censored ? Infinity : p.widthM));
        const status = lower + 1e-8 >= minimumM ? 'pass' : upper + 1e-8 < minimumM ? 'below_minimum' : 'unknown';
        return { id: candidate.id, status, diameter_m: status === 'unknown' ? null : status === 'below_minimum' ? upper : lower,
            profiles: profiles.length };
    });
};

export const birdBoxBounds = s => ({
    low: [-s.lengthM / 2 - s.marginM, -s.widthM / 2 - s.marginM, 0],
    high: [s.lengthM / 2 + s.marginM, s.widthM / 2 + s.marginM, s.heightM + s.marginM]
});

export const birdBoxGeometry = (contact, up, settings) => {
    const s = validateBirdSpace(settings);
    up = unit(up);
    if (!up) return null;
    const center = add(contact, mul(up, s.heightM / 2));
    const approach = unit(center);
    if (!approach) return null;
    const horizontal = sub(approach, mul(up, dot(approach, up)));
    const fallback = sub([0, 0, -1], mul(up, dot([0, 0, -1], up)));
    const forward = unit(horizontal) ?? unit(fallback);
    if (!forward) return null;
    const right = unit(cross(forward, up));
    const basis = [forward, right, up];
    const travelLength = s.checkApproachPath ? Math.min(s.approachM, norm(center)) : 0;
    const travel = mul(approach, travelLength);
    const start = sub(contact, travel);
    const { low, high } = birdBoxBounds(s);
    const corners = origin => [0, 1, 2, 3, 4, 5, 6, 7].map(i => basis.reduce((p, b, k) => add(p, mul(b, (i & (1 << k)) ? high[k] : low[k])), [...origin]));
    return { contact,
        center,
        basis,
        approach,
        start,
        travel,
        travelLength,
        low,
        high,
        corners_camera_m: corners(contact),
        start_corners_camera_m: corners(start) };
};

const screenFree = (point, scene, targetPixels, geometry) => {
    const [fx, fy, cx, cy] = scene.intrinsics, z = -point[2];
    if (!valid(z)) return false;
    const x = Math.round(cx + fx * point[0] / z), y = Math.round(cy - fy * point[1] / z);
    if (x < 0 || x >= scene.width || y < 0 || y >= scene.height) return false;
    const p = y * scene.width + x, surface = scene.depth[p];
    if (!valid(surface)) return false;
    // Known free space is in front of the first observed surface. Behind it is
    // unknown, never inferred empty from the absence of reconstructed points.
    if (z < surface - 0.01) return true;
    const q = sub(point, geometry.contact), h = dot(q, geometry.basis[2]);
    return targetPixels.has(p) && h <= 0.005 && h >= -0.05 && Math.abs(z - surface) < 0.05;
};

const inspectCorner = (point, index, scene, targetPixels) => {
    const result = { index, status: 'unknown', hit_pixels: 0, free_pixels: 0, camera_m: point };
    const [fx, fy, cx, cy] = scene.intrinsics;
    const z = -point[2], radius = cornerProbeRadiusM;
    if (z <= radius) return result;
    const u = cx + fx * point[0] / z, v = cy - fy * point[1] / z;
    if (u < 0 || u >= scene.width || v < 0 || v >= scene.height) return result;
    const padX = fx * radius * (1 + Math.abs(point[0]) / z) / (z - radius);
    const padY = fy * radius * (1 + Math.abs(point[1]) / z) / (z - radius);
    const radiusSquared = radius * radius;
    let rays = 0, observedSurfaceRays = 0;
    for (let y = Math.max(0, Math.floor(v - padY)); y <= Math.min(scene.height - 1, Math.ceil(v + padY)); y++) {
        for (let x = Math.max(0, Math.floor(u - padX)); x <= Math.min(scene.width - 1, Math.ceil(u + padX)); x++) {
            const p = y * scene.width + x, depth = scene.depth[p], at = p * 3;
            const inProbe = ((x - u) / fx * z) ** 2 + ((y - v) / fy * z) ** 2 <= radiusSquared;
            if (inProbe) rays++;
            if (!valid(depth)) continue;
            const squaredDistance = (scene.points[at] - point[0]) ** 2 + (scene.points[at + 1] - point[1]) ** 2 + (depth - z) ** 2;
            const inSphere = squaredDistance <= radiusSquared;
            const isSupport = targetPixels.has(p);
            if (inSphere && !isSupport) result.hit_pixels++;
            if (inProbe) {
                if (depth > z + radius) result.free_pixels++;
                else if (isSupport || inSphere) observedSurfaceRays++;
            }
        }
    }
    if (result.hit_pixels >= cornerMinHitPixels) result.status = 'blocked';
    // The landing branch is excluded. A sampled surface inside the probe is
    // also observed, with the hit threshold deciding whether it blocks.
    else if (rays && result.free_pixels + observedSurfaceRays >= Math.ceil(rays * 0.6)) result.status = 'clear';
    return result;
};

const supportedHit = (hits, scene) => {
    for (const p of hits) {
        const x = p % scene.width, y = Math.floor(p / scene.width), at = p * 3;
        let count = 1;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            if ((!dx && !dy) || x + dx < 0 || x + dx >= scene.width || y + dy < 0 || y + dy >= scene.height) continue;
            const other = p + dy * scene.width + dx;
            if (!hits.has(other)) continue;
            const start = other * 3;
            const separation = Math.hypot(scene.points[start] - scene.points[at],
                scene.points[start + 1] - scene.points[at + 1], scene.points[start + 2] - scene.points[at + 2]);
            if (separation <= 2 * cornerProbeRadiusM && ++count >= cornerMinHitPixels) {
                return Array.from(scene.points.subarray(at, at + 3));
            }
        }
    }
    return null;
};

export const assessLandingCorners = (scene, topCorners, targetPixels) =>
    topCorners.map((corner, index) => inspectCorner(corner, index + 4, scene, targetPixels));

export const assessSpaceBox = ({ scene, contact, up, axis, axisReliable = true, targetPixels = new Set(), settings }) => {
    const s = validateBirdSpace(settings), geometry = birdBoxGeometry(contact, up, s);
    const result = { status: 'unknown',
        reasons: [],
        approach_angle_deg: null,
        min_clearance_m: null,
        observed_fraction: 0,
        contact_camera_m: contact,
        top_corners: [],
        geometry,
        geometry_units: scene?.geometryUnits ?? 'approximate_meters' };
    if (!scene || !geometry) {
        result.reasons.push('metric_geometry_missing'); return result;
    }
    result.top_corners = assessLandingCorners(scene, geometry.corners_camera_m.slice(4), targetPixels);
    result.observed_fraction = result.top_corners.filter(corner => corner.status !== 'unknown').length / 4;
    if (result.top_corners.some(corner => corner.status === 'blocked')) result.reasons.push('perch_space_blocked');
    if (result.top_corners.some(corner => corner.status === 'unknown')) result.reasons.push('space_unobserved');
    if (s.checkApproachAngle) {
        if (axis && axisReliable) {
            result.approach_angle_deg = acuteAxisAngle(axis, geometry.approach);
            if (result.approach_angle_deg + 1e-7 < s.minApproachAngleDeg) {
                result.reasons.push('approach_end_on');
            }
        } else {
            result.reasons.push('axis_uncertain');
        }
    }
    if (geometry.travelLength > 0) checkSweptPath(scene, geometry, targetPixels, s, result);
    result.status = result.reasons.some(reason => reason.endsWith('_blocked') || reason === 'approach_end_on') ? 'blocked' :
        result.reasons.length ? 'unknown' : 'clear';
    return result;
};

const checkSweptPath = (scene, geometry, targetPixels, s, result) => {
    const local = p => geometry.basis.map(b => dot(p, b));
    const travel = local(geometry.travel);
    const actualLow = [-s.lengthM / 2, -s.widthM / 2, 0];
    const actualHigh = [s.lengthM / 2, s.widthM / 2, s.heightM];
    const inside = q => q.every((v, k) => v >= geometry.low[k] && v <= geometry.high[k]);
    const bodyLow = actualLow.map(v => v + 0.0001);
    const hits = new Set();
    for (const p of scene.pixels) {
        const at = p * 3;
        const dx = scene.points[at] - geometry.contact[0], dy = scene.points[at + 1] - geometry.contact[1], dz = scene.points[at + 2] - geometry.contact[2];
        const q = geometry.basis.map(b => dx * b[0] + dy * b[1] + dz * b[2]);
        const support = targetPixels.has(p) && q[2] >= -0.05 && q[2] <= 0.005;
        if (inside(q)) continue;
        const fromStart = q.map((v, k) => v + travel[k]);
        if (support) {
            const bodyHit = sweptBoxInterval(fromStart, travel, bodyLow, actualHigh);
            // Target support may enter the bottom safety margin, but may not
            // penetrate the actual body before the terminal 5cm of approach.
            if (!bodyHit || (1 - bodyHit[0]) * geometry.travelLength <= 0.05 + 1e-7) continue;
        }
        if (sweptBoxInterval(fromStart, travel, geometry.low, geometry.high)) hits.add(p);
    }
    const blocker = supportedHit(hits, scene);
    if (blocker) {
        result.blocker_camera_m = blocker;
        result.reasons.push('approach_space_blocked');
        return;
    }
    let observed = 0, total = 0;
    // Inspect the complete swept volume on a <=5cm grid, including its faces.
    // The collision test above is continuous and independent of this grid.
    const counts = geometry.high.map((v, k) => Math.max(1, Math.ceil((v - geometry.low[k]) / 0.05)));
    const steps = Math.ceil(geometry.travelLength / 0.05);
    if ((steps + 1) * counts.reduce((n, count) => n * (count + 1), 1) > 250000) {
        result.reasons.push('inspection_budget');
        return;
    }
    for (let t = 0; t <= steps; t++) {
        const origin = steps ? add(geometry.start, mul(geometry.travel, t / steps)) : geometry.contact;
        for (let x = 0; x <= counts[0]; x++) {
            for (let y = 0; y <= counts[1]; y++) {
                for (let z = 0; z <= counts[2]; z++) {
                    const coords = [x, y, z].map((n, k) => geometry.low[k] + (geometry.high[k] - geometry.low[k]) * n / counts[k]);
                    const point = geometry.basis.reduce((p, b, k) => add(p, mul(b, coords[k])), origin);
                    total++;
                    if (screenFree(point, scene, targetPixels, geometry)) observed++;
                }
            }
        }
    }
    result.approach_observed_fraction = total ? observed / total : 0;
    if (observed !== total && !result.reasons.includes('space_unobserved')) result.reasons.push('space_unobserved');
};

export const evaluateLandingSpace = ({ scene, candidates, supportPixels, up, settings }) => candidates.map((candidate) => {
    const pixels = supportPixels.get(candidate.id) ?? [];
    const points = [], validPixels = [];
    if (scene) {
        for (const p of pixels) {
            if (!valid(scene.depth[p])) continue;
            points.push(Array.from(scene.points.subarray(p * 3, p * 3 + 3))); validPixels.push(p);
        }
    }
    if (!points.length) {
        return { id: candidate.id,
            status: 'unknown',
            reasons: ['metric_geometry_missing'],
            geometry: null,
            approach_angle_deg: null,
            contact_camera_m: null,
            min_clearance_m: null,
            observed_fraction: 0 };
    }
    const contact = [0, 1, 2].map(k => points.reduce((sum, point) => sum + point[k], 0) / points.length);
    const fit = settings.checkApproachAngle ? fitAxis(points, contact) : null;
    const result = assessSpaceBox({ scene,
        contact,
        up,
        axis: fit?.axis,
        axisReliable: !!fit && fit.linearity >= 0.5 && validPixels.length >= pixels.length * 0.8,
        targetPixels: new Set(validPixels),
        settings });
    return { id: candidate.id, ...result, axis_camera: fit?.axis, axis_linearity: fit?.linearity };
});

const selectMetricCandidates = (base, results, topK) => {
    const eligible = results.filter(c => !c.reasons.length).sort((a, b) => Number(a.space?.status !== 'clear') - Number(b.space?.status !== 'clear') || b.score - a.score || a.id - b.id);
    const selected = topK == null ? eligible : eligible.slice(0, clamp(Math.round(topK), 1, 253));
    const ids = new Uint32Array(base.ids.length);
    for (const c of selected) for (const p of base.candidatePixels.get(c.id) ?? []) ids[p] = c.id;
    return { ...base, results, eligible, selected, ids };
};

export const selectDiameterCandidates = (base, assessments, topK) => {
    const byId = new Map(assessments.map(a => [a.id, a]));
    const results = base.results.map(c => {
        const diameter = byId.get(c.id);
        if (!diameter) return c;
        const reason = diameter.status === 'below_minimum' ? 'diameter_below_minimum' :
            diameter.status === 'unknown' ? 'diameter_unobserved' : null;
        return { ...c, diameter, reasons: reason ? [...c.reasons, reason] : c.reasons };
    });
    return selectMetricCandidates(base, results, topK);
};

export const selectSpaceCandidates = (base, assessments, topK) => {
    const byId = new Map(assessments.map(a => [a.id, a]));
    const results = base.results.map((c) => {
        const space = byId.get(c.id);
        return { ...c, space, reasons: [...c.reasons, ...(space?.status === 'blocked' ? space.reasons : [])] };
    });
    return selectMetricCandidates(base, results, topK);
};

export const projectedHeight = (a, b, up) => Math.abs(dot(sub(b, a), unit(up) ?? [0, 0, 0]));
