#!/usr/bin/env node
import { readFile, mkdir, stat, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { parseArgs } from 'node:util';

import { withTreeDatasetBrowser } from './tree_dataset_browser.mjs';

const { positionals, values } = parseArgs({ allowPositionals: true, options: {
    shard: { type: 'string', default: '0' },
    shards: { type: 'string', default: '1' },
    frame: { type: 'string' },
    origin: { type: 'string', default: 'http://127.0.0.1:8027' }
} });
const [sourceRoot, output] = positionals;
const shard = Number(values.shard);
const shards = Number(values.shards);
if (positionals.length !== 2 || !Number.isInteger(shard) || !Number.isInteger(shards) ||
    shard < 0 || shard >= shards) {
    throw new Error('Usage: render_tree_training_rgb.mjs SOURCE OUTPUT [--shard I --shards N] [--frame FRAME_ID]');
}

const width = 1280;
const height = 720;
const index = JSON.parse(await readFile(path.join(sourceRoot, 'index.json'), 'utf8'));
const splitOrder = ['val', 'train', 'test'];
const rows = splitOrder.flatMap(split => index.samples.filter(row => row.split === split));
if (rows.length !== 10000 || new Set(rows.map(row => row.frame_id)).size !== rows.length ||
    rows.some(row => !/^approach_(train|val|test)_\d{3}_\d{4}$/.test(row.frame_id))) {
    throw new Error('Expected the fixed set of 10,000 approach poses');
}
const frames = values.frame ? rows.filter(row => row.frame_id === values.frame) :
    rows.filter((_, i) => i % shards === shard);
if (!frames.length) throw new Error('No matching frames');

const url = new URL('/3dgs-splat-labeler/app/', values.origin);
url.search = new URLSearchParams({
    control: 'fly', load: '/exports/dji_0967_supersplat/dji_0967_splat.ply',
    filename: 'dji_0967_splat.ply', transforms: '/data/dji_0967_3dgs/transforms.json',
    dataparser: '/outputs/unnamed/splatfacto/2026-07-06_221130/dataparser_transforms.json',
    boxes3d: '/boxes_3d_261118.json'
}).toString();

async function capture(position, target, expectedProjection, width, height) {
    const { scene } = window;
    const camera = scene.camera;
    const vector = camera.mainCamera.getPosition();
    camera.setPose(vector.clone().set(...position), vector.clone().set(...target), 0);
    camera.onUpdate(0);
    const frame = await scene.events.invoke('render.annotationSnapshot', width, height, true);
    if (frame.position.some((value, i) => Math.abs(value - position[i]) > 1e-6) ||
        [0, 5].some(i => Math.abs(frame.projection[i] - expectedProjection[i]) > 1e-6)) {
        throw new Error('Capture pose or field of view differs from the source');
    }

    const pixels = new ImageData(width, height);
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    pixels.data.set(frame.rgb);
    for (let i = 3; i < pixels.data.length; i += 4) pixels.data[i] = 255;
    context.putImageData(pixels, 0, 0);
    const rgb = canvas.toDataURL('image/png').split(',')[1];
    for (let i = 0; i < frame.membership.length; i += 4) {
        const alpha = frame.membership[i + 3];
        const mask = alpha / 255 >= 0.05 && frame.membership[i] >= alpha * 0.5 ? 255 : 0;
        pixels.data[i] = pixels.data[i + 1] = pixels.data[i + 2] = mask;
    }
    context.putImageData(pixels, 0, 0);
    return { rgb, mask: canvas.toDataURL('image/png').split(',')[1], rotation: frame.quaternion };
}

async function save(filename, encoded) {
    const pending = `${filename}.${process.pid}.tmp`;
    await writeFile(pending, Buffer.from(encoded, 'base64'));
    await rename(pending, filename);
}

async function exists(filename) {
    try {
        return (await stat(filename)).size > 0;
    } catch (error) {
        if (error.code === 'ENOENT') return false;
        throw error;
    }
}

for (const split of splitOrder) {
    for (const folder of ['RGB', 'Mask']) {
        await mkdir(path.join(output, split, folder), { recursive: true });
    }
}

const flags = JSON.parse(process.env.TREE_BROWSER_FLAGS ??
    '["--use-angle=vulkan","--enable-features=Vulkan","--disable-vulkan-surface"]');
flags.push('--disable-background-timer-throttling', '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows');
await withTreeDatasetBrowser(url.href, async evaluate => {
    const limit = Date.now() + 180000;
    for (;;) {
        const renderer = await evaluate(`(() => {
            const { scene } = window;
            const panel = scene.events.invoke('annotation.panel');
            panel.liveView.enabled = false;
            if (!panel.activeSplat() || panel.heightScaleReference?.actual_height_m !== 3 ||
                panel.liveView.capturing) return null;
            const source = panel.liveView.getSource();
            if (!source) throw new Error('Missing tree annotations');
            panel.liveView.updateMembership(source);
            // Only the aligned color and membership targets are needed here.
            scene.camera.picker.prepareDepth = () => {};
            scene.camera.picker.readDepthMap = async () => null;
            const gl = document.createElement('canvas').getContext('webgl2');
            const ext = gl?.getExtension('WEBGL_debug_renderer_info');
            return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'WebGL';
        })()`);
        if (renderer) {
            console.log(renderer);
            break;
        }
        if (Date.now() > limit) throw new Error('Calibrated source splat did not load');
        await delay(250);
    }

    const start = performance.now();
    for (const [i, row] of frames.entries()) {
        const rgbPath = path.join(output, row.split, 'RGB', row.frame_id + '.png');
        const maskPath = path.join(output, row.split, 'Mask', row.frame_id + '.png');
        if (await exists(rgbPath) && await exists(maskPath)) continue;
        const pose = JSON.parse(await readFile(path.join(sourceRoot, row.files.pose), 'utf8'));
        const target = pose.approach.target_world_m.map(value => value / pose.meters_per_scene_unit);
        const expectedProjection = pose.projection_matrix.flat();
        const result = await evaluate(`(${capture})(${JSON.stringify(pose.position)}, ${JSON.stringify(target)},
            ${JSON.stringify(expectedProjection)}, ${width}, ${height})`);
        const rotationMatch = Math.abs(result.rotation.reduce((dot, value, j) =>
            dot + value * pose.quaternion_xyzw[j], 0));
        if (Math.abs(rotationMatch - 1) > 1e-6) throw new Error(`${row.frame_id}: orientation mismatch`);
        await Promise.all([save(rgbPath, result.rgb), save(maskPath, result.mask)]);
        if (i === 0 || (i + 1) % 50 === 0 || i + 1 === frames.length) {
            console.log(`${i + 1}/${frames.length} ${row.frame_id} ${((performance.now() - start) / 1000).toFixed(1)}s`);
        }
    }
}, { flags });
