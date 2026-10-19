import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';

const wasm = await readFile(new URL('../supersplat/static/lib/branch-detector/branch_detector.wasm', import.meta.url));
const { instance } = await WebAssembly.instantiate(wasm);
const core = instance.exports;
const mask = new Uint8Array(core.memory.buffer, core.bd_mask(), 640 * 640);
const depth = new Float32Array(core.memory.buffer, core.bd_depth(), 640 * 640);
const skeleton = new Uint8Array(core.memory.buffer, core.bd_raw_skeleton(), 320 * 320);
const mark = (x, y) => { mask[(y * 2) * 640 + x * 2] = 255; };

test('raw skeleton retains every segmented component without depth or size filtering', () => {
    mask.fill(0);
    depth.fill(Number.NaN);
    mark(0, 0); // border pixel
    mark(20, 20); // isolated speck that branch filtering normally discards
    mask[(40 * 2 + 1) * 640 + 60 * 2 + 1] = 255; // any of the 2x2 source pixels counts
    for (let y = 90; y <= 98; y++) for (let x = 80; x <= 130; x++) mark(x, y);
    const count = core.bd_run_raw_skeleton();
    assert.equal(skeleton[0], 1);
    assert.equal(skeleton[20 * 320 + 20], 1);
    assert.equal(skeleton[40 * 320 + 60], 1);
    assert.ok(count > 3 && count < 9 * 51 + 3, 'thinning should reduce the thick segmented region');
    assert.equal(count, skeleton.reduce((sum, pixel) => sum + pixel, 0));
    assert.equal(skeleton[250 * 320 + 250], 0);
});

test('empty segmentation yields an empty raw skeleton', () => {
    mask.fill(0);
    assert.equal(core.bd_run_raw_skeleton(), 0);
    assert.equal(skeleton.reduce((sum, pixel) => sum + pixel, 0), 0);
});
