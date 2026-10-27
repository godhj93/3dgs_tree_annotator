import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { measureSkeletonLengths, prepareDepthMask } from '../supersplat/static/lib/branch-detector/depth-geometry.mjs';
import { analyzePcaSegments } from '../supersplat/static/lib/branch-detector/pca-link-analysis.mjs';
import { segmentFinalLanding } from '../supersplat/src/final-landing-analysis.mjs';

const wasm = (await WebAssembly.instantiate(await readFile(new URL('../supersplat/static/lib/branch-detector/branch_detector.wasm', import.meta.url)))).instance.exports;
const n = 640 * 640, rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const thin = mask => {
    new Uint8Array(wasm.memory.buffer, wasm.bd_mask(), n).set(mask);
    wasm.bd_run_raw_skeleton();
    return new Uint8Array(new Uint8Array(wasm.memory.buffer, wasm.bd_raw_skeleton(), 320 * 320));
};
const components = skeleton => {
    const seen = new Uint8Array(skeleton.length); let count = 0;
    for (let i = 0; i < skeleton.length; i++) {
        if (!skeleton[i] || seen[i]) continue;
        count++; const queue = [i]; seen[i] = 1;
        for (let head = 0; head < queue.length; head++) {
            const p = queue[head], x = p % 320, y = Math.floor(p / 320);
            for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
                const nx = x + dx, ny = y + dy, q = ny * 320 + nx;
                if (nx < 0 || nx >= 320 || ny < 0 || ny >= 320 || !skeleton[q] || seen[q]) continue;
                seen[q] = 1; queue.push(q);
            }
        }
    }
    return count;
};
const horizontal = () => {
    const mask = new Uint8Array(n);
    for (let y = 312; y < 328; y++) for (let x = 80; x < 560; x++) mask[y * 640 + x] = 255;
    return mask;
};

test('constant depth preserves connected skeleton, thin diagonals and image-edge support', () => {
    const mask = horizontal();
    for (let x = 0; x < 80; x++) mask[x * 640 + x] = 255;
    const prepared = prepareDepthMask(mask, new Float32Array(n).fill(3), 'camera_z_m');
    assert.deepEqual(prepared.processedMask, mask);
    assert.equal(prepared.boundaryCells, 0);
    assert.equal(prepared.validFraction, 1);
    assert.deepEqual(thin(prepared.processedMask), thin(mask));
    assert.equal(prepareDepthMask(new Uint8Array(n), new Float32Array(n), 'camera_z_m').processedMask.some(Boolean), false);
});

test('depth boundary splits support before thinning with correct DA2 and 3DGS near direction', () => {
    const mask = horizontal(), gs = new Float32Array(n), da = new Float32Array(n);
    for (let i = 0; i < n; i++) { gs[i] = i % 640 < 320 ? 2 : 5; da[i] = i % 640 < 320 ? 10 : 1; }
    const a = prepareDepthMask(mask, gs, 'camera_z_m'), b = prepareDepthMask(mask, da, 'relative_inverse');
    assert.equal(components(thin(mask)), 1);
    assert.equal(components(thin(a.processedMask)), 2);
    assert.deepEqual(a.processedMask, b.processedMask);
    assert.ok(a.boundaryCells > 0);
    assert.equal(a.processedMask[320 * 640 + 318], 255, 'near side remains');
    assert.equal(a.processedMask[320 * 640 + 320], 0, 'far side forms the barrier');
    const scaled = prepareDepthMask(mask, gs.map(value => value * 2), 'camera_z_m');
    assert.deepEqual(scaled.processedMask, a.processedMask);
});

test('missing and nonpositive 3DGS depths are excluded without borrowing DA2 support', () => {
    const mask = horizontal(), depth = new Float32Array(n).fill(3);
    for (let y = 312; y < 328; y++) for (let x = 300; x < 340; x++) depth[y * 640 + x] = NaN;
    const prepared = prepareDepthMask(mask, depth, 'camera_z_m');
    assert.ok(prepared.validFraction < 1 && prepared.missingCells > 0);
    assert.equal(prepared.processedMask[320 * 640 + 320], 0);
    assert.equal(components(thin(prepared.processedMask)), 2);
    assert.equal(prepareDepthMask(mask, new Float32Array(n).fill(-1), 'camera_z_m').processedMask.some(Boolean), false);
});

test('metric PCA reconstructs a known slanted branch and preserves orientation under scale', () => {
    const labels = new Uint32Array(320 * 320), mask = new Uint8Array(n), depth = new Float32Array(n);
    const slope = 0.4, intrinsics = [320, 320, 320, 320];
    for (let x = 70; x < 250; x++) {
        labels[160 * 320 + x] = 1;
        const z = 3 / (1 - slope * ((2 * x + 1 - 320) / 320));
        for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
            const p = (320 + dy) * 640 + 2 * x + dx; mask[p] = 255; depth[p] = z;
        }
    }
    const a = analyzePcaSegments(labels, mask, depth, intrinsics, rotation, 320, 'camera_z_m').objects[0];
    const b = analyzePcaSegments(labels, mask, depth.map(value => value * 2), intrinsics, rotation, 320, 'camera_z_m').objects[0];
    const expected = [1, 0, -slope].map(value => value / Math.hypot(1, slope));
    assert.ok(Math.abs(a.axis.reduce((sum, value, i) => sum + value * expected[i], 0)) > 0.9999);
    a.center.forEach((value, i) => assert.ok(Math.abs(b.center[i] - value * 2) < 1e-6));
    assert.ok(Math.abs(a.axis.reduce((sum, value, i) => sum + value * b.axis[i], 0)) > 0.99999);
});

test('WASM metric branch centers and final geometry follow camera Z, not the DA2 inverse mapping', () => {
    const mask = horizontal(), depth = new Float32Array(n).fill(2), rgb = new Uint8Array(n * 4).fill(110);
    const run = multiplier => {
        new Uint8Array(wasm.memory.buffer, wasm.bd_mask(), n).set(mask);
        new Uint8Array(wasm.memory.buffer, wasm.bd_rgb(), n * 4).set(rgb);
        new Float32Array(wasm.memory.buffer, wasm.bd_depth(), n).set(depth.map(value => value * multiplier));
        new Float32Array(wasm.memory.buffer, wasm.bd_rotation(), 9).set(rotation);
        wasm.bd_set_depth_kind(1); wasm.bd_run(320, 320, 320, 320, 1);
        return Array.from(new Float32Array(wasm.memory.buffer, wasm.bd_branches(), wasm.bd_fields()));
    };
    const a = run(1), b = run(2);
    assert.ok(Math.abs(a[16] + 2) < 1e-3);
    assert.ok(Math.abs(b[16] + 4) < 1e-3);
    assert.ok(Math.abs(a[20] - b[20]) < 1e-3);
    const labels = new Uint32Array(320 * 320); for (let y = 156; y < 164; y++) for (let x = 40; x < 280; x++) labels[y * 320 + x] = 1;
    const result = segmentFinalLanding({ labels, objects: [{ id: 1, centerlineLengthPx: 480 }], mask, depth, rgb,
        depthKind: 'camera_z_m', pose: { intrinsics: [[320, 0, 320], [0, 320, 320], [0, 0, 1]],
            camera_to_world: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], gravity: { vector: [0, -1, 0] } },
        settings: { maxWidthPx: 24, minLengthPx: 12, minLinearity: 0.5, maxDepthStdRelative: 0.3,
            minRelativeHeight: 0, rejectFoliage: false, topK: 3 } });
    assert.ok(Math.abs(result.results[0].center_camera[2] + 2) < 1e-6);
    assert.equal(result.results[0].geometry_units, 'approximate_meters');
});

test('3D skeleton arc length measures horizontal, diagonal and elbow paths in cm without duplicate edges', () => {
    const width = 20, full = width * 2, depth = new Float32Array(full * full).fill(2);
    const measure = coordinates => {
        const skeleton = new Uint8Array(width * width), labels = new Uint32Array(width * width), mask = new Uint8Array(full * full);
        for (const [x, y] of coordinates) {
            skeleton[y * width + x] = labels[y * width + x] = 1;
            for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) mask[(y * 2 + dy) * full + x * 2 + dx] = 255;
        }
        return measureSkeletonLengths(skeleton, labels, mask, depth, [200, 200, 20, 20], 'camera_z_m', width);
    };
    assert.ok(Math.abs(measure(Array.from({length: 10}, (_, x) => [x + 3, 10])).totalCm - 18) < 1e-9);
    assert.ok(Math.abs(measure(Array.from({length: 10}, (_, x) => [x + 3, x + 3])).totalCm - 18 * Math.SQRT2) < 1e-9);
    const elbow = measure([[5,5],[6,5],[6,6]]);
    assert.equal(elbow.edges, 2); assert.ok(Math.abs(elbow.byLabel.get(1) - 4) < 1e-9);
});

test('metric arc length scales with depth, follows 3D slopes and never bridges missing samples', () => {
    const width = 20, full = 40, skeleton = new Uint8Array(width * width), labels = new Uint32Array(width * width);
    const mask = new Uint8Array(full * full), depth = new Float32Array(full * full);
    const k = [30,30,20,20], slope = 0.4;
    for (let x = 3; x <= 16; x++) {
        skeleton[10 * width + x] = labels[10 * width + x] = 1;
        const z = 3 / (1 - slope * (2 * x + 1 - 20) / k[0]);
        for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
            const p = (20 + dy) * full + x * 2 + dx; mask[p] = 255; depth[p] = z;
        }
    }
    const a = measureSkeletonLengths(skeleton, labels, mask, depth, k, 'camera_z_m', width);
    const b = measureSkeletonLengths(skeleton, labels, mask, depth.map(value => value * 2), k, 'camera_z_m', width);
    const z0 = 3/(1-slope*(7-20)/30), z1 = 3/(1-slope*(33-20)/30);
    const expected = Math.hypot((33-20)/30*z1-(7-20)/30*z0, -(21-20)/30*(z1-z0), z1-z0)*100;
    assert.ok(Math.abs(a.totalCm - expected) < 1e-4);
    assert.ok(Math.abs(b.totalCm - a.totalCm * 2) < 1e-5);
    for (let dy=0;dy<2;dy++) for(let dx=0;dx<2;dx++) depth[(20+dy)*full+20+dx]=NaN;
    const missing = measureSkeletonLengths(skeleton, labels, mask, depth, k, 'camera_z_m', width);
    assert.equal(missing.edges, a.edges - 2); assert.ok(missing.totalCm < a.totalCm);
    assert.equal(measureSkeletonLengths(skeleton, labels, mask, depth, k, 'relative_inverse', width).totalCm, null);
});

test('3DGS minimum cm length rejects a physically short branch even when its pixel skeleton is long', () => {
    const mask = horizontal(), labels = new Uint32Array(320*320), depth = new Float32Array(n).fill(2);
    for (let y=156;y<164;y++) for(let x=40;x<280;x++) labels[y*320+x]=1;
    const input = { labels, mask, depth, rgb: new Uint8Array(n*4).fill(110), depthKind:'camera_z_m',
        pose:{intrinsics:[[320,0,320],[0,320,320],[0,0,1]],camera_to_world:[[1,0,0],[0,1,0],[0,0,1]],gravity:{vector:[0,-1,0]}},
        settings:{minLengthPx:12,minLengthCm:10,minLinearity:.5,maxWidthPx:24,maxDepthStdRelative:.3,minRelativeHeight:0,rejectFoliage:false,topK:3}};
    const short = segmentFinalLanding({...input,objects:[{id:1,centerlineLengthPx:480,centerlineLengthCm:8}]});
    const long = segmentFinalLanding({...input,objects:[{id:1,centerlineLengthPx:480,centerlineLengthCm:12}]});
    assert.ok(short.results[0].reasons.includes('short'));
    assert.ok(!long.results[0].reasons.includes('short'));
    assert.equal(long.results[0].centerline_length_cm,12);
});
