import assert from 'node:assert/strict';
import { test } from 'node:test';

import { compareJunctionDepth, findRawJunctions, rawJunctionMarkers, segmentLinksByJunctionDepth, segmentSkeletonLinks } from '../supersplat/static/lib/branch-detector/junction-analysis.mjs';

const width = 320;
const skeleton = new Uint8Array(width * width);
for (let x = 145; x <= 175; x++) skeleton[160 * width + x] = 1;
for (let y = 145; y <= 175; y++) skeleton[y * width + 160] = 1;
const topology = findRawJunctions(skeleton);
const mask = new Uint8Array(640 * 640);
const depth = new Float32Array(640 * 640);
for (let y = 0; y < width; y++) for (let x = 0; x < width; x++) {
    if (!skeleton[y * width + x]) continue;
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        mask[(y * 2 + dy) * 640 + x * 2 + dx] = 255;
    }
}

test('2D junction detection returns the crossing without consulting depth', () => {
    assert.equal(topology.length, 1);
    assert.equal(topology[0].armCount, 4);
    assert.deepEqual(rawJunctionMarkers(topology).map(marker => marker.armCount), [4]);
});

test('links are independently labeled between junctions and fill only visible mask pixels', () => {
    const { labels, count } = segmentSkeletonLinks(skeleton, mask);
    assert.equal(count, 4);
    const armIds = [labels[150 * width + 160], labels[160 * width + 170],
        labels[170 * width + 160], labels[160 * width + 150]];
    assert.equal(new Set(armIds).size, 4);
    assert.ok(armIds.every(Boolean));
    assert.equal(labels[0], 0);
    assert.ok(armIds.includes(labels[160 * width + 160]), 'junction core should inherit a neighboring link color');
});

test('relative depth distinguishes overlapping arms from similar-depth arms', () => {
    depth.fill(0.5);
    const same = compareJunctionDepth(topology, depth, mask);
    assert.equal(same[0].status, 'similar-depth');
    for (let y = 0; y < width; y++) for (let x = 0; x < width; x++) {
        if (!skeleton[y * width + x]) continue;
        const value = y === 160 ? 0.2 : 0.8;
        for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
            depth[(y * 2 + dy) * 640 + x * 2 + dx] = value;
        }
    }
    const separated = compareJunctionDepth(topology, depth, mask);
    assert.equal(separated[0].status, 'depth-separated');
    assert.ok(separated[0].relativeGap > 0.15);
    assert.equal(separated[0].sampledArms, 4);
});

test('every junction arm remains a distinct segment at both similar and separated depth', () => {
    const base = segmentSkeletonLinks(skeleton, mask);
    const flat = new Float32Array(640 * 640).fill(0.5);
    const same = segmentLinksByJunctionDepth(topology, compareJunctionDepth(topology, flat, mask),
        skeleton, base.labels, base.count, flat, mask);
    assert.equal(same.count, 4);
    const layered = new Float32Array(640 * 640);
    for (let y = 145; y <= 175; y++) for (let x = 145; x <= 175; x++) {
        if (!skeleton[y * width + x]) continue;
        const value = y === 160 ? 0.2 : 0.8;
        for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
            layered[(y * 2 + dy) * 640 + x * 2 + dx] = value;
        }
    }
    const separated = segmentLinksByJunctionDepth(topology, compareJunctionDepth(topology, layered, mask),
        skeleton, base.labels, base.count, layered, mask);
    assert.equal(separated.count, 4);
    const armIds = [separated.labels[160 * width + 150], separated.labels[160 * width + 170],
        separated.labels[150 * width + 160], separated.labels[170 * width + 160]];
    assert.equal(new Set(armIds).size, 4);
    const missing = new Float32Array(640 * 640).fill(NaN);
    const unknown = segmentLinksByJunctionDepth(topology,
        compareJunctionDepth(topology, missing, mask), skeleton, base.labels, base.count, missing, mask);
    assert.equal(unknown.count, base.count);
});

test('DA2 splits a continuous 2D link at an abrupt depth change', () => {
    const line = new Uint8Array(width * width);
    const lineMask = new Uint8Array(640 * 640);
    const lineDepth = new Float32Array(640 * 640).fill(NaN);
    for (let x = 130; x <= 190; x++) {
        line[160 * width + x] = 1;
        for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
            const p = (160 * 2 + dy) * 640 + x * 2 + dx;
            lineMask[p] = 255; lineDepth[p] = x < 160 ? 0.2 : 0.8;
        }
    }
    const base = segmentSkeletonLinks(line, lineMask);
    assert.equal(base.count, 1);
    const topology = findRawJunctions(line);
    const result = segmentLinksByJunctionDepth(topology,
        compareJunctionDepth(topology, lineDepth, lineMask), line, base.labels, base.count, lineDepth, lineMask);
    assert.equal(result.count, 2);
    assert.equal(result.depthCuts, 1);
    assert.notEqual(result.labels[160 * width + 145], result.labels[160 * width + 175]);
});

test('a short junction arm still has a separate object ID', () => {
    const short = new Uint8Array(width * width);
    const shortMask = new Uint8Array(640 * 640);
    for (let x = 150; x <= 170; x++) short[160 * width + x] = 1;
    for (let y = 150; y <= 162; y++) short[y * width + 160] = 1;
    for (let i = 0; i < short.length; i++) if (short[i]) {
        const x = i % width, y = Math.floor(i / width);
        for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
            shortMask[(y * 2 + dy) * 640 + x * 2 + dx] = 255;
        }
    }
    const junctions = findRawJunctions(short);
    const links = segmentSkeletonLinks(short, shortMask);
    const armIds = junctions[0].arms.map(arm => arm.map(p => links.labels[p]).find(Boolean));
    assert.equal(junctions[0].armCount, 4);
    assert.ok(armIds.every(Boolean));
    assert.equal(new Set(armIds).size, 4);
});

test('two arms of one returning link split at their junction entrances', () => {
    const loop = new Uint8Array(width * width);
    const loopMask = new Uint8Array(640 * 640);
    const flat = new Float32Array(640 * 640).fill(0.5);
    for (let x = 140; x <= 180; x++) {
        loop[160 * width + x] = 1;
        for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
            loopMask[(160 * 2 + dy) * 640 + x * 2 + dx] = 255;
        }
    }
    const base = segmentSkeletonLinks(loop, loopMask);
    const left = 160 * width + 145, right = 160 * width + 175;
    assert.equal(base.labels[left], base.labels[right]);
    const junction = { arms: [[left], [right]], armSeeds: [[left], [right]] };
    const result = segmentLinksByJunctionDepth([junction], [{}], loop, base.labels, base.count, flat, loopMask);
    assert.equal(result.junctionCuts, 1);
    assert.notEqual(result.labels[left], result.labels[right]);
});

test('insufficient arm depths remain uncertain', () => {
    depth.fill(Number.NaN);
    const result = compareJunctionDepth(topology, depth, mask);
    assert.equal(result[0].status, 'uncertain');
    assert.equal(result[0].sampledArms, 0);
});
