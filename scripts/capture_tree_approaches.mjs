#!/usr/bin/env node
import { createWriteStream } from 'node:fs';
import { mkdir, mkdtemp, readFile, rename, rm, stat } from 'node:fs/promises';
import { randomBytes } from 'node:crypto';
import { createServer, request } from 'node:http';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';
import { setTimeout as delay } from 'node:timers/promises';
import { parseArgs } from 'node:util';

import { withTreeDatasetBrowser } from './tree_dataset_browser.mjs';

const { values } = parseArgs({ options: {
    output: { type: 'string' }, routes: { type: 'string' }, pilot: { type: 'boolean' },
    shard: { type: 'string', default: '0' }, shards: { type: 'string', default: '1' },
    steps: { type: 'string' }, origin: { type: 'string', default: 'http://127.0.0.1:8027' }
} });
const steps = Number(values.steps ?? (values.pilot ? 4 : 99));
const shard = Number(values.shard);
const shards = Number(values.shards);
if (!values.output || !!values.pilot === !!values.routes || !Number.isInteger(steps) || steps < 1 || steps > 1000 ||
    !Number.isInteger(shard) || !Number.isInteger(shards) || shard < 0 || shard >= shards) {
    throw new Error('Usage: node scripts/capture_tree_approaches.mjs (--pilot | --routes ROUTES.json) --output DIR [--shards N --shard I]');
}
const root = path.resolve(values.output);
const upstream = new URL(values.origin);
const folders = new Set(['RGB', 'Mask', 'Pose', 'Depth', 'SensorRGB']);
const splitValues = new Set(['train', 'val', 'test']);
const pilotRoutes = Array.from({ length: 12 }, (_, sector) => [8, 18].map(elevationDeg => ({
    azimuthDeg: 15 + sector * 30, elevationDeg, sector, runId: `pilot_${sector}_${elevationDeg}`
}))).flat();
const routes = values.pilot ? pilotRoutes : JSON.parse(await readFile(values.routes, 'utf8'));
if (new Set(routes.map(r => r.runId)).size !== routes.length || routes.some(route =>
    !/^[a-z0-9_]+$/.test(route.runId) || !Number.isInteger(route.sector) || route.sector < 0 || route.sector >= 12 ||
    ![route.azimuthDeg, route.elevationDeg].every(Number.isFinite) ||
    route.azimuthDeg < route.sector * 30 || route.azimuthDeg >= (route.sector + 1) * 30 ||
    route.elevationDeg <= -90 || route.elevationDeg >= 90 ||
    (!values.pilot && !splitValues.has(route.split)))) throw new Error('Invalid approach routes');

const capability = '/__capture_' + randomBytes(16).toString('hex') + '/';
let activeDirectory;
let activeRoute;
let writes = 0;

async function writeCapture(req, res) {
    const parts = req.url.slice(capability.length).split('/');
    const validName = /^[a-z0-9_]+\.(json|png|npy)$/;
    const folder = parts.length === 2 ? parts[0] : '';
    const name = parts.at(-1);
    if (!activeDirectory || parts.length > 2 || !validName.test(name) ||
        (folder ? !folders.has(folder) : name !== activeRoute.runId + '_manifest.json') ||
        !name.startsWith(activeRoute.runId + '_')) {
        res.writeHead(400).end();
        return;
    }
    const destination = path.join(activeDirectory, folder, name);
    const temporary = destination + '.tmp';
    try {
        await pipeline(req, createWriteStream(temporary, { flags: 'wx' }));
        await rename(temporary, destination);
        if (folder === 'Pose') writes++;
        res.writeHead(201).end();
    } catch (error) {
        await rm(temporary, { force: true });
        console.error(error.message);
        res.writeHead(500).end();
    }
}

const server = createServer((req, res) => {
    if (req.method === 'PUT' && req.url.startsWith(capability)) {
        void writeCapture(req, res);
        return;
    }
    if (req.method !== 'GET' && req.method !== 'HEAD') {
        res.writeHead(405).end();
        return;
    }
    // Host the viewer and capture sink on the same loopback origin. Keep static reads streaming.
    const target = new URL(req.url, upstream);
    if (target.origin !== upstream.origin) {
        res.writeHead(400).end();
        return;
    }
    const proxy = request(target, { method: req.method, headers: {
        ...(req.headers.range ? { range: req.headers.range } : {})
    } }, response => {
        res.writeHead(response.statusCode, response.headers);
        response.pipe(res);
    });
    proxy.on('error', error => {
        if (res.headersSent) res.destroy(error);
        else res.writeHead(502).end();
    });
    res.on('close', () => proxy.destroy());
    proxy.end();
});

async function startPath(route, sink, steps) {
    const panel = window.scene.events.invoke('annotation.panel');
    const target = panel.spiralCenter();
    const scale = panel.metersPerSceneUnit;
    const az = route.azimuthDeg * Math.PI / 180;
    const elevation = route.elevationDeg * Math.PI / 180;
    const ray = target.clone().set(Math.cos(elevation) * Math.cos(az), Math.cos(elevation) * Math.sin(az), Math.sin(elevation));
    panel.losWaypoints = [6, 1].map(distance => ray.clone().mulScalar(distance / scale).add(target));
    panel.losStepsInput.value = String(steps);

    const folder = (prefix = '') => ({
        async getDirectoryHandle(name) {
            return folder(prefix + name + '/');
        },
        async getFileHandle(name) {
            return { async createWritable() { return {
                async write(blob) {
                    const response = await fetch(sink + prefix + name, { method: 'PUT', body: blob });
                    if (!response.ok) throw new Error(`Capture write ${prefix + name}: HTTP ${response.status}`);
                },
                async close() {}
            }; } };
        }
    });
    panel.outputDirHandle = folder();
    window.approachCapture = { done: false, error: null };
    panel.runLosPathExport(true, route).then(() => {
        window.approachCapture.done = true;
    }, error => {
        window.approachCapture = { done: true, error: error.message };
    });
}

async function verifyPath(directory, route) {
    const manifest = JSON.parse(await readFile(path.join(directory, route.runId + '_manifest.json'), 'utf8'));
    if (!manifest.complete || manifest.renderer !== 'playcanvas' || manifest.views.length !== steps + 1 ||
        manifest.approach?.azimuth_deg !== route.azimuthDeg || manifest.approach?.elevation_deg !== route.elevationDeg ||
        manifest.assigned_split !== route.split) throw new Error(`${route.runId}: wrong manifest`);
    for (const [index, view] of manifest.views.entries()) {
        if (Math.abs(view.distance_to_target_m - (6 - 5 * index / steps)) > 0.005 ||
            !Number.isFinite(view.distance_to_target_m)) throw new Error(`${route.runId}: pose distance mismatch at ${index}`);
        for (const file of Object.values(view.result.files)) {
            if (!(await stat(path.join(directory, file))).size) throw new Error(`${route.runId}: empty ${file}`);
        }
    }
}

try {
    await mkdir(root, { recursive: true });
    await new Promise((resolve, reject) => {
        server.once('error', reject);
        server.listen(0, '127.0.0.1', resolve);
    });
    const local = new URL(`http://127.0.0.1:${server.address().port}/3dgs-splat-labeler/app/`);
    local.search = new URLSearchParams({
        control: 'fly', load: '/exports/dji_0967_supersplat/dji_0967_splat.ply',
        filename: 'dji_0967_splat.ply', transforms: '/data/dji_0967_3dgs/transforms.json',
        dataparser: '/outputs/unnamed/splatfacto/2026-07-06_221130/dataparser_transforms.json',
        boxes3d: '/boxes_3d_261118.json'
    }).toString();
    const flags = JSON.parse(process.env.TREE_BROWSER_FLAGS ?? '["--use-angle=swiftshader","--enable-unsafe-swiftshader"]');
    flags.push('--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows');
    await withTreeDatasetBrowser(local.href, async evaluate => {
        let ready = false;
        const limit = Date.now() + 180000;
        while (!ready && Date.now() < limit) {
            ready = await evaluate(`(() => {
                const panel = window.scene.events.invoke('annotation.panel');
                panel.liveView.enabled = false;
                return Boolean(panel.activeSplat() && panel.heightScaleReference?.actual_height_m === 3 && !panel.liveView.capturing);
            })()`);
            if (!ready) await delay(250);
        }
        if (!ready) throw new Error('Could not load calibrated DJI tree');
        const info = await evaluate(`(() => {
            const panel = window.scene.events.invoke('annotation.panel');
            panel.setSpiralCenterFromBox();
            const gl = document.createElement('canvas').getContext('webgl2');
            const ext = gl?.getExtension('WEBGL_debug_renderer_info');
            return {renderer: ext && gl.getParameter(ext.UNMASKED_RENDERER_WEBGL),
                center: panel.spiralCenter().toString(), metersPerSceneUnit: panel.metersPerSceneUnit};
        })()`);
        console.log(JSON.stringify(info));
        for (const [index, route] of routes.entries()) {
            if (index % shards !== shard) continue;
            const splitDir = route.split ? path.join(root, route.split) : root;
            const destination = path.join(splitDir, route.runId);
            try {
                await stat(destination);
                await verifyPath(destination, route);
                continue;
            } catch (error) {
                if (error.code !== 'ENOENT' || error.path !== destination) throw error;
            }
            await mkdir(splitDir, { recursive: true });
            activeDirectory = await mkdtemp(path.join(splitDir, '.capture-'));
            await Promise.all([...folders].map(name => mkdir(path.join(activeDirectory, name))));
            activeRoute = route;
            writes = 0;
            const started = performance.now();
            try {
                await evaluate(`(${startPath})(${JSON.stringify(route)}, ${JSON.stringify(capability)}, ${steps})`);
                let lastWrite = 0;
                let updated = Date.now();
                for (;;) {
                    const progress = await evaluate('window.approachCapture');
                    if (progress.error) throw new Error(progress.error);
                    if (progress.done) break;
                    if (writes !== lastWrite) {
                        lastWrite = writes;
                        updated = Date.now();
                    } else if (Date.now() - updated > 120000) {
                        throw new Error(`${route.runId}: no frames written for two minutes`);
                    }
                    await delay(500);
                }
                await verifyPath(activeDirectory, route);
                await rename(activeDirectory, destination);
                console.log(`${route.runId}: ${steps + 1} frames in ${((performance.now() - started) / 1000).toFixed(1)}s`);
            } catch (error) {
                console.error(`Incomplete path kept at ${activeDirectory}`);
                throw error;
            } finally {
                activeDirectory = undefined;
                activeRoute = undefined;
            }
        }
    }, { flags });
} finally {
    server.closeAllConnections();
    server.close();
}
