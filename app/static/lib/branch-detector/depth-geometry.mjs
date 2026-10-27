const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));

export const validDepth = (value, kind) => Number.isFinite(value) && (kind !== 'camera_z_m' || value > 0);

export const robustDepthLimits = (depth, mask, kind = 'relative_inverse') => {
    const values = [];
    for (let i = 0; i < depth.length; i++) if (mask[i] && validDepth(depth[i], kind)) values.push(depth[i]);
    values.sort((a, b) => a - b);
    return [values[Math.floor(values.length * 0.02)] ?? 0,
        values[Math.min(values.length - 1, Math.floor(values.length * 0.98))] ?? 0];
};

export const cameraDepth = (value, kind, low, high) => kind === 'camera_z_m' ? value :
    1 / (0.2 + 0.8 * clamp((value - low) / Math.max(high - low, 1e-6), 0, 1));

// Arc length of the visible skeleton graph, using calibrated camera Z at each
// node. Count each undirected edge once and avoid diagonal shortcuts at elbows.
export const measureSkeletonLengths = (skeleton, labels, mask, depth, intrinsics,
    depthKind = 'camera_z_m', width = 320) => {
    const full = width * 2;
    if (skeleton.length !== width * width || labels.length !== skeleton.length ||
        mask.length !== full * full || depth.length !== mask.length) throw new Error('Skeleton length grid mismatch');
    const byLabel = new Map();
    if (depthKind !== 'camera_z_m') return { totalCm: null, byLabel, validFraction: 0, edges: 0 };
    const [fx, fy, cx, cy] = intrinsics;
    if (![fx, fy, cx, cy].every(Number.isFinite) || fx <= 0 || fy <= 0) throw new Error('Invalid length intrinsics');
    const points = new Float64Array(skeleton.length * 3), valid = new Uint8Array(skeleton.length);
    let nodes = 0, validNodes = 0;
    for (let i = 0; i < skeleton.length; i++) {
        if (!skeleton[i]) continue;
        nodes++;
        const x = i % width, y = Math.floor(i / width), p = y * 2 * full + x * 2, values = [];
        for (const q of [p, p + 1, p + full, p + full + 1]) {
            if (mask[q] && validDepth(depth[q], depthKind)) values.push(depth[q]);
        }
        if (!values.length) continue;
        values.sort((a, b) => a - b);
        const z = values.length % 2 ? values[values.length >> 1] :
            (values[values.length / 2 - 1] + values[values.length / 2]) / 2;
        points[i * 3] = (2 * x + 1 - cx) / fx * z;
        points[i * 3 + 1] = -(2 * y + 1 - cy) / fy * z;
        points[i * 3 + 2] = -z;
        valid[i] = 1; validNodes++;
        if (labels[i]) byLabel.set(labels[i], 0);
    }
    let totalCm = 0, edges = 0;
    for (let i = 0; i < skeleton.length; i++) {
        if (!valid[i]) continue;
        const x = i % width, y = Math.floor(i / width);
        for (const [dx, dy] of [[1, 0], [0, 1], [1, 1], [-1, 1]]) {
            const nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= width || ny >= width) continue;
            const j = ny * width + nx;
            if (!valid[j] || (dx && dy && (valid[y * width + nx] || valid[ny * width + x]))) continue;
            const cm = Math.hypot(points[i * 3] - points[j * 3], points[i * 3 + 1] - points[j * 3 + 1],
                points[i * 3 + 2] - points[j * 3 + 2]) * 100;
            totalCm += cm; edges++;
            if (labels[i] && labels[i] === labels[j]) byLabel.set(labels[i], byLabel.get(labels[i]) + cm);
        }
    }
    return { totalCm: validNodes ? totalCm : null, byLabel, validFraction: nodes ? validNodes / nodes : 0, edges };
};

// Build a separate support mask before thinning. A one-cell far-side barrier
// prevents an occlusion boundary from being treated as a connected branch.
export const prepareDepthMask = (mask, depth, kind = 'relative_inverse', threshold = 0.15, full = 640) => {
    if (full % 2 || mask.length !== full * full || depth.length !== mask.length ||
        !Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new Error('Invalid depth skeleton inputs');
    const width = full / 2, count = width * width;
    const samples = new Float32Array(count).fill(NaN);
    const occupied = new Uint8Array(count), boundary = new Uint8Array(count);
    const missing = new Uint8Array(count), processedMask = new Uint8Array(mask.length);
    let maskPixels = 0, validPixels = 0;
    for (let y = 0; y < width; y++) for (let x = 0; x < width; x++) {
        const i = y * width + x, p = y * 2 * full + x * 2, values = [];
        for (const q of [p, p + 1, p + full, p + full + 1]) {
            if (!mask[q]) continue;
            occupied[i] = 1; maskPixels++;
            if (validDepth(depth[q], kind)) { values.push(depth[q]); validPixels++; }
        }
        values.sort((a, b) => a - b);
        if (values.length) samples[i] = values.length % 2 ? values[values.length >> 1] :
            (values[values.length / 2 - 1] + values[values.length / 2]) / 2;
        else if (occupied[i]) missing[i] = 1;
    }
    const [low, high] = robustDepthLimits(depth, mask, kind), range = high - low;
    if (range > 1e-6) for (let y = 0; y < width; y++) for (let x = 0; x < width; x++) {
        const i = y * width + x, a = samples[i];
        if (!Number.isFinite(a)) continue;
        for (const [dx, dy] of [[1, 0], [0, 1], [1, 1], [-1, 1]]) {
            const nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= width || ny >= width) continue;
            const j = ny * width + nx, b = samples[j];
            if (!Number.isFinite(b) || Math.abs(a - b) <= threshold * range) continue;
            const far = kind === 'camera_z_m' ? (a > b ? i : j) : (a < b ? i : j);
            boundary[far] = 1;
        }
    }
    let boundaryCells = 0, missingCells = 0;
    for (let y = 0; y < width; y++) for (let x = 0; x < width; x++) {
        const i = y * width + x, p = y * 2 * full + x * 2;
        boundaryCells += boundary[i]; missingCells += missing[i];
        if (boundary[i] || missing[i]) continue;
        for (const q of [p, p + 1, p + full, p + full + 1]) {
            if (mask[q] && validDepth(depth[q], kind)) processedMask[q] = mask[q];
        }
    }
    return { processedMask, boundary, missing, boundaryCells, missingCells,
        validFraction: maskPixels ? validPixels / maskPixels : 0, depthLimits: [low, high] };
};
