// Junction inspection on the unfiltered 320-square segmentation skeleton.
// Depth Anything values are relative, so a depth gap is only an overlap cue.
const WIDTH = 320;
const FULL_WIDTH = 640;
const OFFSETS = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]];

const neighbors = (index, width, height, visit) => {
    const x = index % width, y = Math.floor(index / width);
    for (const [dx, dy] of OFFSETS) {
        const nx = x + dx, ny = y + dy;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) visit(ny * width + nx);
    }
};

const median = values => {
    if (!values.length) return NaN;
    values.sort((a, b) => a - b);
    const middle = values.length >> 1;
    return values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
};

const junctionCore = (skeleton, width, height) => {
    if (skeleton.length !== width * height) throw new Error('Skeleton size does not match its dimensions');
    const branchpoint = new Uint8Array(skeleton.length);
    const core = new Uint8Array(skeleton.length);
    for (let i = 0; i < skeleton.length; i++) {
        if (!skeleton[i]) continue;
        const x = i % width, y = Math.floor(i / width);
        const ring = OFFSETS.map(([dx, dy]) => {
            const nx = x + dx, ny = y + dy;
            return nx >= 0 && nx < width && ny >= 0 && ny < height ? skeleton[ny * width + nx] : 0;
        });
        let transitions = 0;
        for (let k = 0; k < 8; k++) transitions += !ring[k] && !!ring[(k + 1) & 7] ? 1 : 0;
        if (transitions >= 3) branchpoint[i] = 1;
    }
    for (let i = 0; i < branchpoint.length; i++) {
        if (!branchpoint[i]) continue;
        core[i] = 1;
        neighbors(i, width, height, q => { if (skeleton[q]) core[q] = 1; });
    }
    return core;
};

export const findRawJunctions = (skeleton, width = WIDTH, height = WIDTH) => {
    const core = junctionCore(skeleton, width, height);
    const visited = new Uint8Array(skeleton.length);
    const junctions = [];
    for (let i = 0; i < core.length; i++) {
        if (!core[i] || visited[i]) continue;
        const cluster = [i]; visited[i] = 1;
        let sx = 0, sy = 0;
        for (let head = 0; head < cluster.length; head++) {
            const p = cluster[head]; sx += p % width; sy += Math.floor(p / width);
            neighbors(p, width, height, q => {
                if (core[q] && !visited[q]) { visited[q] = 1; cluster.push(q); }
            });
        }
        const boundary = new Set();
        for (const p of cluster) neighbors(p, width, height, q => {
            if (skeleton[q] && !core[q]) boundary.add(q);
        });
        const seedSeen = new Set();
        const arms = [];
        const armSeeds = [];
        for (const seed of boundary) {
            if (seedSeen.has(seed)) continue;
            const seeds = [seed]; seedSeen.add(seed);
            for (let head = 0; head < seeds.length; head++) neighbors(seeds[head], width, height, q => {
                if (boundary.has(q) && !seedSeen.has(q)) { seedSeen.add(q); seeds.push(q); }
            });
            // Follow each arm away from its junction core. Only a short segment
            // is used, so another junction or a loop cannot redefine this one.
            const queue = seeds.map(p => [p, 1]);
            const armSeen = new Set(seeds);
            const samples = [];
            for (let head = 0; head < queue.length; head++) {
                const [p, distance] = queue[head];
                if (distance >= 3) samples.push(p);
                if (distance >= 12) continue;
                neighbors(p, width, height, q => {
                    if (skeleton[q] && !core[q] && !armSeen.has(q)) {
                        armSeen.add(q); queue.push([q, distance + 1]);
                    }
                });
            }
            // A short arm still needs an identity even if it has no samples
            // three pixels away for a reliable depth comparison.
            arms.push(samples.length ? samples : seeds);
            armSeeds.push(seeds);
        }
        junctions.push({ x: (sx / cluster.length) * 2 + 1, y: (sy / cluster.length) * 2 + 1,
            armCount: arms.length, arms, armSeeds });
    }
    return junctions;
};

export const rawJunctionMarkers = junctions => junctions.map(({ x, y, armCount }) => ({ x, y, armCount }));

// One ID per uninterrupted skeleton link. Fill the visible annotation mask
// from those centerline seeds so the result can be viewed as colored regions.
export const segmentSkeletonLinks = (skeleton, mask, width = WIDTH, height = WIDTH) => {
    if (mask.length !== FULL_WIDTH * FULL_WIDTH || width * 2 !== FULL_WIDTH || height * 2 !== FULL_WIDTH) {
        throw new Error('Link segmentation expects a 320×320 skeleton and 640×640 mask');
    }
    const core = junctionCore(skeleton, width, height);
    const labels = new Uint32Array(skeleton.length);
    let count = 0;
    for (let i = 0; i < skeleton.length; i++) {
        if (!skeleton[i] || core[i] || labels[i]) continue;
        const queue = [i]; labels[i] = ++count;
        for (let head = 0; head < queue.length; head++) neighbors(queue[head], width, height, q => {
            if (skeleton[q] && !core[q] && !labels[q]) { labels[q] = count; queue.push(q); }
        });
    }
    const visible = new Uint8Array(skeleton.length);
    const queue = [];
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const i = y * width + x, p = (y * 2) * FULL_WIDTH + x * 2;
        visible[i] = !!(mask[p] || mask[p + 1] || mask[p + FULL_WIDTH] || mask[p + FULL_WIDTH + 1]);
        if (labels[i] && visible[i]) queue.push(i);
    }
    // Multi-source flood fill stays within the visible mask and assigns joint
    // cores to the nearest link without fusing neighboring link IDs.
    for (let head = 0; head < queue.length; head++) {
        const p = queue[head], x = p % width, y = Math.floor(p / width);
        const assign = q => { if (visible[q] && !labels[q]) { labels[q] = labels[p]; queue.push(q); } };
        if (x > 0) assign(p - 1);
        if (x + 1 < width) assign(p + 1);
        if (y > 0) assign(p - width);
        if (y + 1 < height) assign(p + width);
    }
    return { labels, count };
};

const depthAt = (index, depth, mask, width) => {
    const x = (index % width) * 2, y = Math.floor(index / width) * 2;
    const values = [];
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        const p = (y + dy) * FULL_WIDTH + x + dx;
        if (mask[p] && Number.isFinite(depth[p])) values.push(depth[p]);
    }
    return median(values);
};

const robustDepthRange = (depth, mask) => {
    let count = 0;
    for (let i = 0; i < depth.length; i++) if (mask[i] && Number.isFinite(depth[i])) count++;
    if (count < 8) return 0;
    const stride = Math.max(1, Math.floor(count / 4096));
    const values = [];
    let seen = 0;
    for (let i = 0; i < depth.length; i++) {
        if (!mask[i] || !Number.isFinite(depth[i])) continue;
        if (seen++ % stride === 0) values.push(depth[i]);
    }
    values.sort((a, b) => a - b);
    const low = values[Math.floor(values.length * 0.02)];
    const high = values[Math.min(values.length - 1, Math.floor(values.length * 0.98))];
    return Math.max(0, high - low);
};

export const compareJunctionDepth = (junctions, depth, mask, threshold = 0.15, width = WIDTH) => {
    if (depth.length !== FULL_WIDTH * FULL_WIDTH || mask.length !== depth.length) {
        throw new Error('Junction depth comparison expects matching 640×640 depth and mask');
    }
    const range = robustDepthRange(depth, mask);
    return junctions.map(({ x, y, armCount, arms }) => {
        const estimates = [];
        const spreads = [];
        const armDepths = [];
        for (const arm of arms) {
            const samples = arm.map(p => depthAt(p, depth, mask, width)).filter(Number.isFinite);
            if (samples.length < 2) { armDepths.push(NaN); continue; }
            const center = median(samples);
            armDepths.push(center);
            estimates.push(center);
            spreads.push(median(samples.map(value => Math.abs(value - center))));
        }
        estimates.sort((a, b) => a - b);
        let gap = 0;
        for (let i = 1; i < estimates.length; i++) gap = Math.max(gap, estimates[i] - estimates[i - 1]);
        const relativeGap = range > 1e-6 ? gap / range : 0;
        const relativeNoise = range > 1e-6 ? Math.max(0, ...spreads) / range : 0;
        const status = estimates.length < 3 ? 'uncertain' : range <= 1e-6 ? 'similar-depth' :
            relativeGap >= threshold && relativeGap > relativeNoise * 2 ? 'depth-separated' : 'similar-depth';
        return { x, y, armCount, sampledArms: estimates.length, status,
            relativeGap, relativeNoise, armDepths };
    });
};

// A junction is a boundary between objects, even when two arms have similar
// relative depth. Split uninterrupted 2D links further at sharp DA2 changes,
// then grow each centerline piece only inside its original 2D link region.
export const segmentLinksByJunctionDepth = (junctions, comparisons, skeleton, labels, count, depth, mask,
    threshold = 0.15, width = WIDTH) => {
    if (junctions.length !== comparisons.length) throw new Error('Junction comparisons do not match the raw topology');
    if (skeleton.length !== labels.length || labels.length !== width * width ||
        mask.length !== FULL_WIDTH * FULL_WIDTH || depth.length !== mask.length) {
        throw new Error('Depth link segmentation expects matching skeleton, labels, mask and depth');
    }
    const range = robustDepthRange(depth, mask);
    const core = junctionCore(skeleton, width, width);
    const smallDepth = new Float32Array(labels.length).fill(NaN);
    for (let i = 0; i < skeleton.length; i++) {
        if (skeleton[i] && !core[i] && labels[i]) smallDepth[i] = depthAt(i, depth, mask, width);
    }
    const centerline = new Uint32Array(labels.length);
    let nextId = count;
    const componentSeen = new Uint8Array(count + 1);
    for (let i = 0; i < skeleton.length; i++) {
        if (!skeleton[i] || core[i] || !labels[i] || centerline[i]) continue;
        const base = labels[i];
        const id = componentSeen[base]++ ? ++nextId : base;
        const queue = [i]; centerline[i] = id;
        for (let head = 0; head < queue.length; head++) {
            const p = queue[head];
            neighbors(p, width, width, q => {
                if (!skeleton[q] || core[q] || labels[q] !== base || centerline[q]) return;
                const a = smallDepth[p], b = smallDepth[q];
                if (range > 1e-6 && Number.isFinite(a) && Number.isFinite(b) &&
                    Math.abs(a - b) / range > threshold) return;
                centerline[q] = id; queue.push(q);
            });
        }
    }
    const depthCuts = nextId - count;
    // A 2D link can loop back into the same junction. Split that link by
    // geodesic distance from its distinct arm entrances, so the junction is
    // a true object boundary even when the far ends reconnect.
    for (const junction of junctions) {
        const byComponent = new Map();
        for (const seeds of junction.armSeeds) {
            const id = seeds.map(p => centerline[p]).find(Boolean);
            if (!id) continue;
            if (!byComponent.has(id)) byComponent.set(id, []);
            byComponent.get(id).push(seeds.filter(p => centerline[p] === id));
        }
        for (const [id, arms] of byComponent) {
            if (arms.length < 2) continue;
            const owner = new Uint32Array(labels.length);
            const queue = [];
            arms.forEach((seeds, index) => {
                const armId = index === 0 ? id : ++nextId;
                for (const p of seeds) if (!owner[p]) { owner[p] = armId; queue.push(p); }
            });
            for (let head = 0; head < queue.length; head++) {
                neighbors(queue[head], width, width, q => {
                    if (centerline[q] === id && !owner[q]) { owner[q] = owner[queue[head]]; queue.push(q); }
                });
            }
            for (const p of queue) centerline[p] = owner[p];
        }
    }
    const segmented = new Uint32Array(labels.length);
    const queue = [];
    for (let i = 0; i < centerline.length; i++) if (centerline[i]) {
        segmented[i] = centerline[i]; queue.push(i);
    }
    // Multi-source geodesic fill preserves the original junction boundaries.
    for (let head = 0; head < queue.length; head++) {
        const p = queue[head], x = p % width, y = Math.floor(p / width);
        const assign = q => {
            if (labels[q] === labels[p] && !segmented[q]) {
                segmented[q] = segmented[p]; queue.push(q);
            }
        };
        if (x > 0) assign(p - 1);
        if (x + 1 < width) assign(p + 1);
        if (y > 0) assign(p - width);
        if (y + 1 < width) assign(p + width);
    }
    for (let i = 0; i < labels.length; i++) if (labels[i] && !segmented[i]) segmented[i] = labels[i];
    return { labels: segmented, count: new Set(segmented).size - (segmented.includes(0) ? 1 : 0),
        depthCuts, junctionCuts: nextId - count - depthCuts };
};
