// DA2 is relative depth. These coordinates are dimensionless pseudo-3D, not meters.
const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export const analyzePcaSegments = (labels, mask, depth, intrinsics, rotation, width = 320) => {
    const full = width * 2;
    if (labels.length !== width * width || mask.length !== full * full || depth.length !== mask.length) {
        throw new Error('PCA expects matching 320-link and 640-mask/depth grids');
    }
    const sampled = new Float32Array(labels.length).fill(NaN);
    const values = [];
    for (let y = 0; y < width; y++) for (let x = 0; x < width; x++) {
        const i = y * width + x;
        if (!labels[i]) continue;
        const p = (y * 2) * full + x * 2;
        let sum = 0, count = 0;
        for (const q of [p, p + 1, p + full, p + full + 1]) {
            if (mask[q] && Number.isFinite(depth[q])) { sum += depth[q]; count++; }
        }
        if (!count) continue;
        sampled[i] = sum / count;
        if (values.length < 32 || i % 16 === 0) values.push(sampled[i]);
    }
    values.sort((a, b) => a - b);
    const low = values[Math.floor(values.length * 0.02)] ?? 0;
    const high = values[Math.floor(values.length * 0.98)] ?? low;
    const range = Math.max(high - low, 1e-6);
    const [fx, fy, cx, cy] = intrinsics;
    const point = (i, d) => {
        const x = i % width, y = Math.floor(i / width);
        const near = clamp((d - low) / range, 0, 1);
        const z = 1 / (0.2 + 0.8 * near);
        const camera = [((2 * x + 1 - cx) / fx) * z, -((2 * y + 1 - cy) / fy) * z, -z];
        return [0, 1, 2].map(k => rotation[k * 3] * camera[0] +
            rotation[k * 3 + 1] * camera[1] + rotation[k * 3 + 2] * camera[2]);
    };
    const stats = new Map();
    for (let i = 0; i < labels.length; i++) {
        const id = labels[i], d = sampled[i];
        if (!id || !Number.isFinite(d)) continue;
        let s = stats.get(id);
        if (!s) { s = { n: 0, sum: [0, 0, 0], outer: new Float64Array(9), image: [0, 0] }; stats.set(id, s); }
        const p = point(i, d);
        s.n++;
        s.image[0] += 2 * (i % width) + 1; s.image[1] += 2 * Math.floor(i / width) + 1;
        for (let a = 0; a < 3; a++) {
            s.sum[a] += p[a];
            for (let b = 0; b < 3; b++) s.outer[a * 3 + b] += p[a] * p[b];
        }
    }
    const objects = [...stats].map(([id, s]) => {
        const center = s.sum.map(v => v / s.n);
        const covariance = s.outer.map((v, i) => v / s.n - center[Math.floor(i / 3)] * center[i % 3]);
        const trace = covariance[0] + covariance[4] + covariance[8];
        let axis = [1, 0.37, 0.11];
        for (let t = 0; t < 16; t++) {
            const next = [0, 1, 2].map(a => covariance[a * 3] * axis[0] +
                covariance[a * 3 + 1] * axis[1] + covariance[a * 3 + 2] * axis[2]);
            const norm = Math.hypot(...next);
            if (norm < 1e-12) break;
            axis = next.map(v => v / norm);
        }
        const eigenvalue = dot(axis, [0, 1, 2].map(a => covariance[a * 3] * axis[0] +
            covariance[a * 3 + 1] * axis[1] + covariance[a * 3 + 2] * axis[2]));
        const linearity = trace > 1e-12 ? clamp((3 * eigenvalue / trace - 1) / 2, 0, 1) : 0;
        const reliable = s.n >= 6 && linearity >= 0.2;
        const project = sign => {
            const world = center.map((v, k) => v + sign * axis[k] * Math.sqrt(Math.max(eigenvalue, 0)) * 2);
            const camera = [0, 1, 2].map(k => rotation[k] * world[0] + rotation[3 + k] * world[1] + rotation[6 + k] * world[2]);
            return camera[2] < -1e-6 ? [cx + fx * camera[0] / -camera[2], cy - fy * camera[1] / -camera[2]] : null;
        };
        return { id, pixels: s.n, center, axis, linearity, reliable,
            moments: { sum: s.sum, outer: Array.from(s.outer) },
            centerPx: s.image.map(v => v / s.n), endpointsPx: reliable ? [project(-1), project(1)] : null };
    }).sort((a, b) => a.id - b.id);
    const byId = new Map(objects.map(o => [o.id, o]));
    const contacts = new Map();
    for (let y = 0; y < width; y++) for (let x = 0; x < width; x++) {
        const i = y * width + x, a = labels[i];
        if (!a || !Number.isFinite(sampled[i])) continue;
        for (const j of [x + 1 < width ? i + 1 : -1, y + 1 < width ? i + width : -1]) {
            const b = j < 0 ? 0 : labels[j];
            if (!b || b === a || !Number.isFinite(sampled[j])) continue;
            const key = `${Math.min(a, b)}:${Math.max(a, b)}`;
            const edge = contacts.get(key) ?? { a: Math.min(a, b), b: Math.max(a, b), contacts: 0, gapSum: 0 };
            edge.contacts++; edge.gapSum += Math.abs(sampled[i] - sampled[j]) / range;
            contacts.set(key, edge);
        }
    }
    const adjacency = [...contacts.values()].map(edge => {
        const left = byId.get(edge.a), right = byId.get(edge.b);
        const angleDeg = Math.acos(clamp(Math.abs(dot(left.axis, right.axis)), 0, 1)) * 180 / Math.PI;
        const depthGap = edge.gapSum / edge.contacts;
        return { a: edge.a, b: edge.b, contacts: edge.contacts, angleDeg, depthGap,
            eligible: left.reliable && right.reliable && depthGap <= 0.15 };
    }).sort((a, b) => a.angleDeg - b.angleDeg);
    return { objects, adjacency };
};

export const mergePcaSegments = (labels, objects, adjacency, maxAngleDeg) => {
    const parent = new Map();
    for (const id of labels) if (id) parent.set(id, id);
    const members = new Map(objects.map(object => [object.id, [object]]));
    const root = id => { while (parent.get(id) !== id) id = parent.get(id); return id; };
    let merges = 0;
    for (const edge of adjacency) {
        if (!edge.eligible || edge.angleDeg > maxAngleDeg) continue;
        const a = root(edge.a), b = root(edge.b);
        if (a === b) continue;
        // Complete-link angular coherence prevents a chain of small turns from folding into one object.
        if (members.get(a).some(left => members.get(b).some(right =>
            Math.acos(clamp(Math.abs(dot(left.axis, right.axis)), 0, 1)) * 180 / Math.PI > maxAngleDeg))) continue;
        const keep = Math.min(a, b), drop = Math.max(a, b);
        parent.set(drop, keep);
        members.get(keep).push(...members.get(drop)); members.delete(drop);
        merges++;
    }
    const merged = new Uint32Array(labels.length);
    for (let i = 0; i < labels.length; i++) if (labels[i]) merged[i] = root(labels[i]);
    return { labels: merged, count: parent.size - merges, merges };
};
