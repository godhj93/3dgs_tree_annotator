import assert from 'node:assert/strict';
import test from 'node:test';
import { defaultBirdSpace, buildSpaceScene, assessSpaceBox, birdBoxGeometry,
    sweptBoxInterval, selectSpaceCandidates, projectedHeight, evaluateLandingSpace,
    defaultMinimumBranchDiameterM, evaluateBranchDiameters, selectDiameterCandidates } from '../supersplat/static/lib/branch-detector/landing-space-analysis.mjs';

const contact = [0, -1, -3], up = [0, 1, 0], axis = [1, 0, 0];
const poseOnly = { ...defaultBirdSpace, checkApproachPath: false, checkApproachAngle: false };
const size = 256, intrinsics = [320, 320, 128, 128];
const flatDepth = (value = 10) => new Float32Array(size * size).fill(value);
const depthPixel = ([x,y,z]) => [Math.round(intrinsics[2] - intrinsics[0] * x / z), Math.round(intrinsics[3] + intrinsics[1] * y / z)];
const scene = (points = [], depth = flatDepth()) => {
    for (const point of points) {
        const [x,y] = depthPixel(point);
        depth[y * size + x] = -point[2];
    }
    return buildSpaceScene(depth, intrinsics, size, size);
};
const cluster = (point, count) => Array.from({length: count}, (_, i) => [point[0] + (i - 1) * .009, point[1], point[2]]);
const assess = (points = [], extra = {}) => assessSpaceBox({ scene: scene(points), contact, up, axis,
    settings: poseOnly, ...extra });

test('the default profile enables both approach checks', () => {
    assert.equal(defaultBirdSpace.checkApproachPath, true);
    assert.equal(defaultBirdSpace.checkApproachAngle, true);
});

test('metric diameter measures the transverse silhouette and rejects a thin branch before top K', () => {
    assert.equal(defaultMinimumBranchDiameterM, 0.05);
    const k = [800, 800, 128, 128], depth = flatDepth(2);
    const mask = new Uint8Array(depth.length);
    const supportPixels = new Map([[1, []], [2, []]]);
    const candidates = [{ id: 1, center_px: [128, 80], score: 10, reasons: [] },
        { id: 2, center_px: [128, 160], score: 1, reasons: [] }];
    for (const [id, v, diameter] of [[1, 80, 0.04], [2, 160, 0.06]]) {
        for (let x = 85; x < 172; x++) for (let y = v - 16; y <= v + 16; y++) {
            if (Math.abs(y - v) * 2 / k[1] >= diameter / 2) continue;
            const p = y * size + x;
            mask[p] = 255;
            supportPixels.get(id).push(p);
        }
    }
    const input = { scene: buildSpaceScene(depth, k, size, size), candidates, supportPixels, mask,
        minimumM: defaultMinimumBranchDiameterM };
    const measured = evaluateBranchDiameters(input);
    assert.deepEqual(measured.map(a => a.status), ['below_minimum', 'pass']);
    assert.ok(Math.abs(measured[0].diameter_m - 0.04) < 0.004);
    assert.ok(Math.abs(measured[1].diameter_m - 0.06) < 0.004);
    const selected = selectDiameterCandidates({ ids: new Uint32Array(depth.length), results: candidates,
        candidatePixels: supportPixels }, measured, 1);
    assert.deepEqual(selected.selected.map(c => c.id), [2]);
    assert.deepEqual(selected.results[0].reasons, ['diameter_below_minimum']);
    assert.ok(supportPixels.get(1).every(p => selected.ids[p] === 0));
    const unknown = evaluateBranchDiameters({ ...input, scene: null });
    assert.ok(unknown.every(a => a.status === 'unknown' && a.diameter_m === null));
    const calibratedDa2 = buildSpaceScene(flatDepth(1), k, size, size, 1, mask, depth);
    assert.equal(calibratedDa2.geometryUnits, 'da2_calibrated_meters');
    assert.deepEqual(evaluateBranchDiameters({ ...input, scene: calibratedDa2 }), measured);
});

test('the pose box is oriented with the added side and top clearance', () => {
    const g = birdBoxGeometry(contact, up, poseOnly);
    assert.deepEqual(g.low, [-.1, -.1, 0]);
    assert.deepEqual(g.high.map(v => Math.round(v * 100)), [10, 10, 15]);
    assert.equal(g.travelLength, 0);
    const spans = g.high.map((v, k) => v - g.low[k]);
    assert.deepEqual(spans.map(v => Math.round(v * 100)), [20, 20, 15]);
    const turned = birdBoxGeometry([-3, -1, 0], up, poseOnly);
    assert.ok(Math.abs(turned.basis[0][0] + 1) < 1e-9);
    // Width and length must rotate with the approach, not stay in camera axes.
    assert.ok(Math.abs(g.basis[1][0]) > .99 && Math.abs(turned.basis[1][2]) > .99);
});

test('upper corners pass in free space and only a supported corner is marked blocked', () => {
    assert.deepEqual(assess().top_corners.map(corner => corner.status), ['clear','clear','clear','clear']);
    const corner = birdBoxGeometry(contact, up, defaultBirdSpace).corners_camera_m[4];
    assert.equal(assess([corner]).status, 'clear');
    const blocked = assess(cluster(corner, 3));
    assert.equal(blocked.status, 'blocked');
    assert.deepEqual(blocked.reasons, ['perch_space_blocked']);
    assert.deepEqual(blocked.top_corners.map(value => value.status), ['blocked','clear','clear','clear']);
    assert.equal(blocked.top_corners[0].hit_pixels, 3);
    assert.equal(assess(cluster(corner, 3), {settings: {...poseOnly, marginM: 0}}).status, 'clear');
});

test('an obstacle only on the approach is rejected even with a clear final pose', () => {
    const settings = {...poseOnly, checkApproachPath: true};
    const g = birdBoxGeometry(contact, up, settings);
    const midpoint = g.start.map((v, k) => v + g.travel[k] * .5 + up[k] * .05);
    assert.equal(assess([midpoint]).status, 'clear');
    assert.equal(assess([midpoint], {settings}).status, 'unknown');
    const result = assess(cluster(midpoint, 3), {settings});
    assert.ok(result.reasons.includes('approach_space_blocked'));
    assert.ok(!result.reasons.includes('perch_space_blocked'));
});

test('swept collision is continuous and cannot step over a thin obstacle', () => {
    assert.ok(sweptBoxInterval([.01, .01, .487], [0, 0, 1], [-.02, -.02, -.001], [.02, .02, .001]));
    assert.equal(sweptBoxInterval([.1, .01, .487], [0, 0, 1], [-.02, -.02, -.001], [.02, .02, .001]), null);
});

test('the pose check samples upper corners above uneven support', () => {
    assert.equal(assess(cluster([0, -.95, -3], 3)).status, 'clear');
    const corner = birdBoxGeometry(contact, up, defaultBirdSpace).corners_camera_m[7];
    const points = cluster(corner, 3), pixels = points.map(point => {
        const [x,y] = depthPixel(point);
        return y * size + x;
    });
    assert.equal(assess(points).top_corners[3].status, 'blocked');
    const ownSupport = assess(points, {targetPixels: new Set(pixels)}).top_corners[3];
    assert.equal(ownSupport.status, 'clear');
    assert.equal(ownSupport.hit_pixels, 0);
});

test('the 1 cm box checks nearby pixels outside the landing branch', () => {
    const points = Array.from({length: 17}, (_, i) => [(i - 8) * 3 / intrinsics[0], contact[1], contact[2]]);
    const pixels = points.map(point => {
        const [x, y] = depthPixel(point);
        return y * size + x;
    });
    const settings = {...poseOnly, widthM: .01, lengthM: .01, heightM: .01, marginM: 0};
    const input = {scene: scene(points), candidates: [{id: 2}], up, settings};
    const [clear] = evaluateLandingSpace({...input, supportPixels: new Map([[2, pixels]])});
    assert.equal(clear.status, 'clear');
    assert.ok(clear.top_corners.every(corner => corner.hit_pixels === 0));

    const support = pixels.filter((_, index) => index < 7 || index > 9);
    const [blocked] = evaluateLandingSpace({...input, supportPixels: new Map([[2, support]])});
    assert.equal(blocked.status, 'blocked');
    assert.deepEqual(blocked.reasons, ['perch_space_blocked']);
});

test('the 1 cm corner probe excludes the landing branch at its observed depth', () => {
    const [cx, cy] = depthPixel(contact);
    const depth = flatDepth();
    const support = new Set();
    for (let y = cy - 3; y <= cy + 3; y++) {
        for (let x = cx - 8; x <= cx + 8; x++) {
            const pixel = y * size + x;
            depth[pixel] = -contact[2];
            support.add(pixel);
        }
    }
    const settings = {...poseOnly, widthM: .01, lengthM: .01, heightM: .01, marginM: 0};
    const result = assess([], {scene: scene([], depth), settings, targetPixels: support});
    assert.equal(result.status, 'clear');
    assert.ok(result.top_corners.every(corner => corner.hit_pixels === 0));

    const foreground = depth.slice();
    for (const pixel of support) foreground[pixel] = -contact[2] - .15;
    const foregroundScene = scene([], foreground);
    assert.equal(assess([], {scene: foregroundScene, settings, targetPixels: support}).status, 'clear');
    const occluded = assess([], {scene: foregroundScene, settings});
    assert.equal(occluded.status, 'unknown');
    assert.deepEqual(occluded.reasons, ['space_unobserved']);

    const corner = result.top_corners[0].camera_m;
    const [x, y] = depthPixel(corner);
    support.delete(y * size + x);
    const sparse = assess([corner], {scene: scene([corner], foreground), settings, targetPixels: support});
    assert.equal(sparse.top_corners[0].hit_pixels, 1);
    assert.equal(sparse.status, 'clear');
});

test('59 degree approach fails, 60 and 90 pass, using robot-to-target direction', () => {
    const g = birdBoxGeometry(contact, up, defaultBirdSpace);
    for (const angle of [59, 60, 90]) {
        const rad = angle * Math.PI / 180;
        const tangent = g.approach.map((v, k) => v * Math.cos(rad) + g.basis[1][k] * Math.sin(rad));
        assert.equal(assess([], {axis: tangent}).status, 'clear');
        const result = assess([], { axis: tangent, settings: {...poseOnly, checkApproachAngle: true} });
        assert.ok(Math.abs(result.approach_angle_deg - angle) < 1e-6);
        assert.equal(result.status, angle < 60 ? 'blocked' : 'clear');
    }
});

test('occlusion, invalid depth, and uncertain axes remain unknown rather than empty', () => {
    assert.equal(assess([], { axisReliable: false }).status, 'clear');
    assert.equal(assess([], { axisReliable: false, settings: {...poseOnly, checkApproachAngle: true} }).status, 'unknown');
    assert.equal(assess([], { scene: scene([], flatDepth(NaN)) }).status, 'unknown');
    const occluded = assess([], { scene: scene([], flatDepth(1)) });
    assert.equal(occluded.status, 'unknown'); assert.ok(occluded.reasons.includes('space_unobserved'));
    assert.equal(assess([], { scene: null }).status, 'unknown');
});

test('short approach is clamped to current position', () => {
    const g = birdBoxGeometry([0, -.05, -.4], up, {...defaultBirdSpace, checkApproachPath: true});
    assert.ok(Math.abs(g.travelLength - .4) < 1e-9);
});

test('space filtering reranks every eligible object before Top K and preserves ID pixels', () => {
    const candidates = [1, 2, 3, 4].map(id => ({ id, score: 10 - id, reasons: [] }));
    const base = { ids: new Uint32Array(5), results: candidates, eligible: candidates, selected: candidates.slice(0, 2),
        candidatePixels: new Map(candidates.map(c => [c.id, [c.id]])) };
    const result = selectSpaceCandidates(base, [{ id: 1, status: 'blocked', reasons: ['perch_space_blocked'] },
        { id: 2, status: 'unknown', reasons: ['space_unobserved'] },
        { id: 3, status: 'clear', reasons: [] }, { id: 4, status: 'clear', reasons: [] }], 3);
    assert.deepEqual(result.selected.map(c => c.id), [3, 4, 2]);
    assert.deepEqual(Array.from(result.ids), [0, 0, 2, 3, 4]);
    assert.equal(result.results.find(c => c.id === 2).reasons.length, 0);
    assert.deepEqual(base.results[0].reasons, []);
    assert.deepEqual(selectSpaceCandidates(base, [], null).selected.map(c=>c.id), [1, 2, 3, 4]);
});

test('height calibration uses vertical height rather than the diagonal', () => {
    assert.equal(projectedHeight([0, 0, 0], [3, 4, 0], up), 4);
    assert.equal(projectedHeight([0, 0, 0], [3, 4, 0], [0, 0, -1]), 0);
});

test('no metric pixels produces a warning assessment for each source-local object', () => {
    const result = evaluateLandingSpace({ scene: null, candidates: [{ id: 17 }],
        supportPixels: new Map([[17, [1, 2]]]), up, settings: defaultBirdSpace });
    assert.equal(result[0].id, 17); assert.equal(result[0].status, 'unknown');
});

test('a single visible segmentation pixel anchors an upright box without an axis fit', () => {
    const [x,y] = depthPixel(contact);
    const result = evaluateLandingSpace({ scene: scene([contact]), candidates: [{id: 2}],
        supportPixels: new Map([[2, [y * size + x]]]), up, settings: poseOnly });
    assert.equal(result[0].status, 'clear');
    assert.equal(result[0].top_corners.length, 4);
    assert.equal(result[0].geometry.low[2], 0);
});

test('DA2 space uses its own relative depth and assumed scale', () => {
    const depth = Float32Array.of(-1, 0, 1, NaN);
    const relative = buildSpaceScene(depth, [100,100,1,1], 2, 2, 1.5);
    assert.equal(relative.geometryUnits, 'da2_assumed_meters');
    assert.equal(relative.depth[1], 1.5);
    assert.ok(relative.depth[0] > relative.depth[1] && relative.depth[1] > relative.depth[2]);
    assert.ok(Number.isNaN(relative.depth[3]));
    assert.deepEqual([...buildSpaceScene(depth, [100,100,1,1], 2, 2).pixels], [2]);
});

test('metric DA2 alignment recovers the scale and offset from overlapping calibrated depth', () => {
    const depth = Float32Array.from({length: 16}, (_, i) => i);
    const metric = depth.map(value => 1 / (0.1 + 0.15 * (0.2 + 0.8 * value / 15)));
    const mask = new Uint8Array(depth.length).fill(1);
    mask[0] = 0;
    metric[0] = 100;
    const fitted = buildSpaceScene(depth, [100,100,1,1], 4, 4, 1, mask, metric);
    assert.equal(fitted.geometryUnits, 'da2_calibrated_meters');
    assert.equal(fitted.calibration.method, 'metric_inverse_affine');
    assert.equal(fitted.calibration.region, 'annotation');
    assert.equal(fitted.calibration.samples, 15);
    for (let i = 1; i < depth.length; i++) assert.ok(Math.abs(fitted.depth[i] - metric[i]) < 1e-5);
    assert.ok(Math.abs(fitted.depth[0] - 1 / 0.13) < 1e-5);
});
