import assert from 'node:assert/strict';
import test from 'node:test';
import { resolveGravityWorld, segmentFinalLanding } from '../supersplat/src/final-landing-analysis.mjs';

test('camera-frame IMU gravity is rotated into the scene world frame', () => {
    const pose = { world_up_axis: 'z', camera_to_world: [
        [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    ] };
    assert.deepEqual(resolveGravityWorld({ frame: 'camera', vector: [0, -2, 0] }, pose).vector, [1, 0, 0]);
    assert.deepEqual(resolveGravityWorld(null, pose), { vector: [0, 0, -1], source: 'scene_up' });
    assert.throws(() => resolveGravityWorld({ frame: 'world', vector: [0, 0, 0] }, pose), /nonzero/);
});

test('final mask keeps only slender passing objects inside visible annotation', () => {
    const labels = new Uint32Array(320 * 320);
    for (let y = 20; y < 22; y++) for (let x = 20; x < 30; x++) labels[y * 320 + x] = 1;
    for (let y = 40; y < 60; y++) for (let x = 40; x < 50; x++) labels[y * 320 + x] = 2;
    const mask = new Uint8Array(640 * 640);
    const depth = new Float32Array(640 * 640);
    const rgb = new Uint8Array(640 * 640 * 4);
    for (let y = 0; y < 640; y++) for (let x = 0; x < 640; x++) {
        const p = y * 640 + x, id = labels[(y >> 1) * 320 + (x >> 1)];
        if (!id) continue;
        mask[p] = 255; depth[p] = id === 1 ? 0.5 : 0.7;
        rgb[p * 4] = id === 1 ? 100 : 10;
        rgb[p * 4 + 1] = 40; rgb[p * 4 + 2] = 20; rgb[p * 4 + 3] = 255;
    }
    // An occluding pixel inside the 320-cell must never enter the final ID map.
    mask[40 * 640 + 40] = 0;
    const objects = [
        { id: 1, centerlineLengthPx: 20, reliable: true, linearity: 0.9, angleToGravityDeg: 85, isTrunk: false },
        { id: 2, centerlineLengthPx: 20, reliable: true, linearity: 0.9, angleToGravityDeg: 85, isTrunk: false }
    ];
    const pose = { camera_to_world: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        intrinsics: [[100, 0, 320], [0, 100, 320], [0, 0, 1]],
        gravity: { vector: [0, -1, 0], source: 'imu' } };
    const settings = { minLengthPx: 12, minLinearity: 0.5, maxWidthPx: 10,
        maxDepthStdRelative: 0.3, minRelativeHeight: 0.15, excludeTrunks: true, topK: 3 };
    const result = segmentFinalLanding({ labels, objects, mask, depth, rgb, pose, settings });
    assert.deepEqual(result.selected.map(object => object.id), [1]);
    assert.ok(result.results.find(object => object.id === 2).reasons.includes('foliage'));
    assert.equal(result.results.find(object => object.id === 1).mean_rgb[0], 100);
    assert.equal(result.ids[40 * 640 + 40], 0);
    assert.deepEqual([...new Set(result.ids)].sort(), [0, 1]);
    const heightOnly = segmentFinalLanding({ labels, objects, mask, depth, rgb, pose,
        settings: { ...settings, maxWidthPx: 100, rejectFoliage: false } });
    assert.deepEqual(heightOnly.selected.map(object => object.id), [1]);
    assert.ok(heightOnly.results.find(object => object.id === 2).reasons.includes('low_height'));
    const strict = segmentFinalLanding({ labels, objects, mask, depth, rgb, pose,
        settings: { ...settings, minLinearity: 0.95 } });
    assert.equal(strict.selected.length, 0);
});

test('a merged Y stays two cylindrical link IDs and green pine foliage is removed', () => {
    const labels = new Uint32Array(320 * 320), sourceLabels = new Uint32Array(labels.length);
    const mask = new Uint8Array(640 * 640), depth = new Float32Array(mask.length);
    const rgb = new Uint8Array(mask.length * 4);
    const paint = (x, y, id, green = false) => {
        const small = y * 320 + x;
        labels[small] = 10; sourceLabels[small] = id;
        for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
            const p = (2 * y + dy) * 640 + 2 * x + dx;
            mask[p] = 255; depth[p] = 0.5;
            rgb[p * 4] = green ? 40 : 130;
            rgb[p * 4 + 1] = green ? 100 : 85;
            rgb[p * 4 + 2] = green ? 35 : 55;
            rgb[p * 4 + 3] = 255;
        }
    };
    for (let x = 80; x < 120; x++) paint(x, 100, 1);
    for (let x = 120; x < 160; x++) paint(x, 100 + Math.floor((x - 120) / 5), 2, x >= 147 && x < 154);
    for (let y = 115; y < 127; y++) for (let x = 135; x < 150; x++) paint(x, y, 3, true);
    const objects = [1, 2, 3].map(id => ({ id, centerlineLengthPx: id === 3 ? 30 : 80,
        reliable: true, linearity: 0.95, isTrunk: false }));
    const pose = { camera_to_world: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        intrinsics: [[500, 0, 320], [0, 500, 320], [0, 0, 1]],
        gravity: { vector: [0, -1, 0], source: 'imu' } };
    const settings = { minLengthPx: 12, minLinearity: 0.5, maxWidthPx: 24,
        maxDepthStdRelative: 0.3, minRelativeHeight: -1,
        minWoodyFraction: 0.25, rejectFoliage: true, excludeTrunks: true, topK: 3 };
    const result = segmentFinalLanding({ labels, sourceLabels, objects,
        sourceObjects: objects, mask, depth, rgb, pose, settings });
    assert.deepEqual(result.selected.map(candidate => candidate.id).sort((a, b) => a - b), [1, 2]);
    assert.deepEqual([...new Set(result.ids)].sort((a, b) => a - b), [0, 1, 2]);
    assert.ok(result.results.find(candidate => candidate.id === 3).reasons.includes('foliage'));
    assert.equal(result.ids[200 * 640 + 180], 1);
    assert.equal(result.ids[208 * 640 + 280], 2);
    assert.equal(result.ids[212 * 640 + 300], 0);
    for (let x = 133; x < 155; x++) {
        const y = 100 + Math.floor((x - 120) / 5);
        for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
            const p = ((2 * y + dy) * 640 + 2 * x + dx) * 4;
            rgb[p] = 40; rgb[p + 1] = 100; rgb[p + 2] = 35;
        }
    }
    const broken = segmentFinalLanding({ labels, sourceLabels, objects,
        sourceObjects: objects, mask, depth, rgb, pose, settings });
    assert.ok(broken.results.find(candidate => candidate.id === 2).reasons.includes('not_cylindrical'));
    assert.ok(!broken.selected.some(candidate => candidate.id === 2));
});

test('camera-relative filters keep a frontal transverse branch and reject side, tilted and end-on views', () => {
    const labels = new Uint32Array(320 * 320), mask = new Uint8Array(640 * 640);
    const depth = new Float32Array(mask.length), rgb = new Uint8Array(mask.length * 4);
    const paint = (x, y, id, value = 0.5) => {
        labels[y * 320 + x] = id;
        for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
            const p = (2 * y + dy) * 640 + 2 * x + dx;
            mask[p] = 255; depth[p] = value;
            rgb[p * 4] = 130; rgb[p * 4 + 1] = 85;
            rgb[p * 4 + 2] = 55; rgb[p * 4 + 3] = 255;
        }
    };
    for (let x = 145; x < 175; x++) paint(x, 160, 1);
    for (let y = 140; y < 175; y++) paint(195, y, 2);
    for (let x = 255; x < 285; x++) paint(x, 160, 3);
    for (let x = 145; x < 175; x++) paint(x, 200, 4, 0.1 + 0.8 * (x - 145) / 29);
    const objects = [1, 2, 3, 4].map(id => ({ id, centerlineLengthPx: 60,
        reliable: true, linearity: 0.95, isTrunk: false }));
    const pose = { camera_to_world: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        intrinsics: [[300, 0, 320], [0, 300, 320], [0, 0, 1]],
        gravity: { vector: [0, -1, 0], source: 'imu' } };
    const settings = { minLengthPx: 12, minLinearity: 0.2, maxWidthPx: 24,
        maxDepthStdRelative: 1, minRelativeHeight: -1, excludeTrunks: false, topK: 4,
        maxCenterBearingDeg: 30, maxEndOnDeviationDeg: 30, maxImageTiltDeg: 30,
        strictCameraAngles: true };
    const result = segmentFinalLanding({ labels, objects, mask, depth, rgb, pose, settings });
    assert.deepEqual(result.selected.map(candidate => candidate.id), [1]);
    assert.ok(result.results.find(candidate => candidate.id === 2).reasons.includes('screen_tilt'));
    assert.ok(result.results.find(candidate => candidate.id === 3).reasons.includes('off_center'));
    assert.ok(result.results.find(candidate => candidate.id === 4).reasons.includes('end_on'));
    const frontal = result.selected[0];
    assert.ok(frontal.center_bearing_deg < 2);
    assert.ok(frontal.side_on_deg > 85);
    assert.ok(frontal.image_tilt_deg < 2);
    const soft = segmentFinalLanding({ labels, objects, mask, depth, rgb, pose,
        settings: { ...settings, strictCameraAngles: false } });
    assert.ok(soft.selected.some(candidate => candidate.id === 2));
    assert.ok(soft.selected.some(candidate => candidate.id === 3));
    assert.ok(soft.selected.some(candidate => candidate.id === 4));
    assert.ok(soft.results.every(candidate =>
        !candidate.reasons.some(reason => ['off_center', 'end_on', 'screen_tilt'].includes(reason))));
    const allNinety = segmentFinalLanding({ labels, objects, mask, depth, rgb, pose,
        settings: { ...settings, maxCenterBearingDeg: 90, maxEndOnDeviationDeg: 90,
            maxImageTiltDeg: 90 } });
    assert.deepEqual(allNinety.selected.map(candidate => candidate.id).sort((a, b) => a - b), [1, 2, 3, 4]);
});
