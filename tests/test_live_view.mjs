import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import test from 'node:test';
import ts from '../supersplat/node_modules/typescript/lib/typescript.js';

const source = readFileSync(new URL('../supersplat/src/live-view-utils.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText;
const { SENSOR_PROFILE, sensorIntrinsics, visibleMask, rowMajorMatrix, numpyFloat32, numpyUint32, parseBoxScaleReference } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);

test('BBox scale reference is accepted only when it agrees with the applied 3DGS scale', () => {
    const reference = { source:'bbox',box_id:'branch_001',axis:'X',scene_length:.25,actual_length_m:.5 };
    assert.deepEqual(parseBoxScaleReference(JSON.parse(JSON.stringify(reference)),2),reference);
    assert.equal(parseBoxScaleReference(reference,3),null);
    assert.equal(parseBoxScaleReference({...reference,actual_length_m:0},2),null);
    assert.equal(parseBoxScaleReference({...reference,axis:'Q'},2),null);
    assert.equal(parseBoxScaleReference(reference,NaN),null);
});

test('640-square intrinsics preserve both full sensor fields of view', () => {
    const k = sensorIntrinsics();
    const horizontal = 2 * Math.atan(320 / k[0][0]) * 180 / Math.PI;
    const vertical = 2 * Math.atan(320 / k[1][1]) * 180 / Math.PI;
    assert.ok(Math.abs(horizontal - 118) < 1e-10);
    assert.ok(Math.abs(vertical - SENSOR_PROFILE.vertical_fov_degrees) < 1e-10);
    assert.equal(k[0][2], 320);
    assert.equal(k[1][2], 320);
    assert.ok(Math.abs(k[1][1] / k[0][0] - 3856 / 2176) < 1e-12);
});

test('opaque non-annotated foreground suppresses an annotated branch', () => {
    // A fully annotated background, then a foreground with opacity .99.
    const contribution = Uint8Array.from([3, 0, 0, 255, 230, 0, 0, 255, 2, 0, 0, 5, 0, 0, 0, 0]);
    const mask = visibleMask(contribution);
    assert.deepEqual(Array.from(mask), [0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255]);
});

test('camera translation occupies the last column in row-major JSON', () => {
    const matrix = rowMajorMatrix([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 4, 5, 1]);
    assert.deepEqual(matrix, [[1, 0, 0, 3], [0, 1, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]]);
});

test('NPY export retains shape and raw float32 values', () => {
    const original = Float32Array.of(.1234567, 100.25, 0, 3.75, 2.5, 99);
    const encoded = numpyFloat32(original, 3, 2);
    assert.deepEqual(Array.from(encoded.slice(0, 8)), [147, 78, 85, 77, 80, 89, 1, 0]);
    const headerSize = new DataView(encoded.buffer).getUint16(8, true);
    const header = new TextDecoder().decode(encoded.slice(10, 10 + headerSize));
    assert.match(header, /'shape': \(2, 3\)/);
    assert.equal((10 + headerSize) % 64, 0);
    assert.deepEqual(Array.from(new Float32Array(encoded.buffer, 10 + headerSize)), Array.from(original));
    writeFileSync('/tmp/annotation_live_npy_test.npy', encoded);
    assert.throws(() => numpyFloat32(original, 4, 2), /shape mismatch/);
});

test('final segmentation NPY retains full uint32 object IDs', () => {
    const ids = Uint32Array.of(0, 255, 65536, 4294967295);
    const encoded = numpyUint32(ids, 2, 2);
    const headerSize = new DataView(encoded.buffer).getUint16(8, true);
    const header = new TextDecoder().decode(encoded.slice(10, 10 + headerSize));
    assert.match(header, /'descr': '<u4'/);
    assert.deepEqual(Array.from(new Uint32Array(encoded.buffer, 10 + headerSize)), Array.from(ids));
});
