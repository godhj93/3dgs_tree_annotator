import assert from 'node:assert/strict';
import { test } from 'node:test';

import { analyzePcaSegments, mergePcaSegments } from '../supersplat/static/lib/branch-detector/pca-link-analysis.mjs';

const width = 320, full = 640;
const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const camera = [500, 500, 320, 320];
const makeScene = () => {
    const labels = new Uint32Array(width * width);
    const mask = new Uint8Array(full * full);
    const depth = new Float32Array(full * full).fill(0.5);
    const fill = (id, x0, x1, y0, y1, d = 0.5) => {
        for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
            labels[y * width + x] = id;
            for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
                const p = (y * 2 + dy) * full + x * 2 + dx;
                mask[p] = 255; depth[p] = d;
            }
        }
    };
    return { labels, mask, depth, fill,
        analyze: () => analyzePcaSegments(labels, mask, depth, camera, identity) };
};

test('adjacent parallel DA2 segments merge as the angle threshold increases', () => {
    const scene = makeScene();
    scene.fill(1, 120, 139, 150, 153);
    scene.fill(2, 140, 159, 150, 153);
    const { objects, adjacency } = scene.analyze();
    assert.equal(objects.length, 2);
    assert.ok(objects.every(object => object.reliable && object.linearity > 0.8));
    assert.equal(adjacency.length, 1);
    assert.ok(adjacency[0].angleDeg < 1);
    assert.equal(adjacency[0].eligible, true);
    const merged = mergePcaSegments(scene.labels, objects, adjacency, 5);
    assert.equal(merged.count, 1);
    assert.equal(merged.labels[151 * width + 145], 1);
});

test('an orthogonal neighbor remains separate below its measured angle', () => {
    const scene = makeScene();
    scene.fill(1, 120, 140, 150, 153);
    scene.fill(2, 140, 143, 154, 174);
    const { objects, adjacency } = scene.analyze();
    assert.equal(adjacency.length, 1);
    assert.ok(adjacency[0].angleDeg > 70);
    assert.equal(mergePcaSegments(scene.labels, objects, adjacency, 20).count, 2);
    assert.equal(mergePcaSegments(scene.labels, objects, adjacency, 90).count, 1);
});

test('a DA2 depth discontinuity blocks even parallel PCA axes', () => {
    const scene = makeScene();
    scene.fill(1, 120, 139, 150, 153, 0.2);
    scene.fill(2, 140, 159, 150, 153, 0.8);
    const { objects, adjacency } = scene.analyze();
    assert.ok(adjacency[0].angleDeg < 1);
    assert.ok(adjacency[0].depthGap > 0.15);
    assert.equal(adjacency[0].eligible, false);
    assert.equal(mergePcaSegments(scene.labels, objects, adjacency, 90).count, 2);
});

test('a group without finite DA2 stays visible and unmerged', () => {
    const scene = makeScene();
    scene.fill(1, 120, 139, 150, 153);
    scene.fill(2, 140, 159, 150, 153);
    for (let y = 150 * 2; y < 154 * 2; y++) for (let x = 140 * 2; x < 160 * 2; x++) {
        scene.depth[y * full + x] = NaN;
    }
    const { objects, adjacency } = scene.analyze();
    assert.equal(objects.length, 1);
    assert.equal(adjacency.length, 0);
    assert.equal(mergePcaSegments(scene.labels, objects, adjacency, 90).count, 2);
});
