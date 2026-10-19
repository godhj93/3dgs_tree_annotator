import { compareJunctionDepth, findRawJunctions, rawJunctionMarkers, segmentLinksByJunctionDepth, segmentSkeletonLinks } from './junction-analysis.mjs';
import { analyzePcaSegments } from './pca-link-analysis.mjs';

const corePromise = (async () => {
    const response = await fetch(new URL('./branch_detector.wasm', import.meta.url));
    if (!response.ok) throw new Error(`Branch detector WASM HTTP ${response.status}`);
    const { instance } = await WebAssembly.instantiate(await response.arrayBuffer());
    return instance.exports;
})();
const pixels = 640 * 640;
let latestRaw = null;

self.onmessage = async ({ data }) => {
    const started = performance.now();
    try {
        const core = await corePromise;
        if (data.type === 'raw-skeleton') {
            if (data.mask.length !== pixels) throw new Error('Raw skeleton expects a 640×640 segmentation mask');
            new Uint8Array(core.memory.buffer, core.bd_mask(), pixels).set(data.mask);
            const count = core.bd_run_raw_skeleton();
            const skeleton = new Uint8Array(new Uint8Array(core.memory.buffer, core.bd_raw_skeleton(), 320 * 320));
            const rawJunctions = findRawJunctions(skeleton);
            const { labels: linkLabels, count: linkCount } = segmentSkeletonLinks(skeleton, data.mask);
            latestRaw = { frameId: data.frameId, skeleton: new Uint8Array(skeleton),
                junctions: rawJunctions, linkLabels, linkCount };
            self.postMessage({ type: 'raw-skeleton', frameId: data.frameId, skeleton, count,
                junctions: rawJunctionMarkers(rawJunctions), linkLabels, linkCount,
                width: 320, height: 320, source: 'visible_segmentation_unfiltered' },
            [skeleton.buffer]);
            return;
        }
        if (data.rgb.length !== pixels * 4 || data.mask.length !== pixels || data.depth.length !== pixels) {
            throw new Error('Branch detector expects synchronized 640×640 inputs');
        }
        const memory = core.memory.buffer;
        new Uint8Array(memory, core.bd_rgb(), pixels * 4).set(data.rgb);
        new Uint8Array(memory, core.bd_mask(), pixels).set(data.mask);
        new Float32Array(memory, core.bd_depth(), pixels).set(data.depth);
        new Float32Array(memory, core.bd_rotation(), 9).set(data.rotation);
        if (latestRaw?.frameId !== data.frameId) {
            core.bd_run_raw_skeleton();
            const skeleton = new Uint8Array(memory, core.bd_raw_skeleton(), 320 * 320);
            const { labels: linkLabels, count: linkCount } = segmentSkeletonLinks(skeleton, data.mask);
            latestRaw = { frameId: data.frameId, skeleton: new Uint8Array(skeleton),
                junctions: findRawJunctions(skeleton), linkLabels, linkCount };
        }
        const junctionComparison = compareJunctionDepth(latestRaw.junctions, data.depth, data.mask);
        const depthLinks = segmentLinksByJunctionDepth(latestRaw.junctions, junctionComparison,
            latestRaw.skeleton, latestRaw.linkLabels, latestRaw.linkCount, data.depth, data.mask);
        const depthJunctionArmIds = latestRaw.junctions.map(junction => junction.armSeeds.map(seeds =>
            seeds.map(index => depthLinks.labels[index]).find(Boolean) ?? 0));
        const pca = analyzePcaSegments(depthLinks.labels, data.mask, data.depth, data.intrinsics, data.rotation);
        core.bd_run(data.intrinsics[0], data.intrinsics[1], data.intrinsics[2], data.intrinsics[3], data.worldUp);
        const count = core.bd_branch_count();
        const fields = core.bd_fields();
        const raw = new Float32Array(memory, core.bd_branches(), count * fields);
        const branches = Array.from({ length: count }, (_, i) => {
            const b = raw.subarray(i * fields, i * fields + fields);
            return {
                id: b[0], score: b[1], centerline_length_px: b[2], area_px: b[3],
                width_px: b[4], straightness: b[5], inclination_to_world_up: b[6],
                relative_depth_consistency: b[7], center_px: [b[8], b[9]],
                endpoints_px: [[b[10], b[11]], [b[12], b[13]]],
                center_pseudo3d_world_axes: [b[14], b[15], b[16]],
                mean_rgb: [b[17], b[18], b[19]],
                vertical_alignment: b[20], orientation_confidence: b[21],
                is_trunk: b[22] > 0.5, projected_up_alignment: b[23],
                angle_to_ground_deg: Math.asin(Math.max(0, Math.min(1, b[20]))) * 180 / Math.PI
            };
        }).sort((a, b) => b.score - a.score);
        const junctions = Array.from(new Int16Array(memory, core.bd_junctions(), core.bd_junction_count() * 2));
        const instanceMask = new Uint8Array(new Uint8Array(memory, core.bd_labels(), pixels));
        const trunkIds = new Set(branches.filter(branch => branch.is_trunk).map(branch => branch.id));
        const pcaById = new Map(pca.objects.map(object => {
            object.centerlineLengthPx = 0;
            object.trunkCenterlineLengthPx = 0;
            object.detectorCenterlineLengthPx = 0;
            return [object.id, object];
        }));
        const pixelOffsets = [0, 1, 640, 641];
        for (let i = 0; i < latestRaw.skeleton.length; i++) {
            if (!latestRaw.skeleton[i]) continue;
            const object = pcaById.get(depthLinks.labels[i]);
            if (!object) continue;
            object.centerlineLengthPx += 2;
            const x = i % 320, y = Math.floor(i / 320);
            const p = y * 1280 + x * 2;
            let assigned = 0, trunk = 0;
            for (const offset of pixelOffsets) {
                const q = p + offset;
                const id = instanceMask[q];
                if (!data.mask[q] || !id) continue;
                assigned++;
                if (trunkIds.has(id)) trunk++;
            }
            if (assigned) object.detectorCenterlineLengthPx += 2;
            if (trunk && trunk * 2 >= assigned) object.trunkCenterlineLengthPx += 2;
        }
        self.postMessage({ frameId: data.frameId, branches, junctions, instanceMask,
            rawJunctions: rawJunctionMarkers(latestRaw.junctions), junctionComparison,
            depthLinkLabels: depthLinks.labels, depthLinkCount: depthLinks.count,
            depthLinkCuts: depthLinks.depthCuts, depthJunctionCuts: depthLinks.junctionCuts,
            depthJunctionArmIds,
            pcaObjects: pca.objects, pcaAdjacency: pca.adjacency,
            cpu_ms: performance.now() - started, geometry_units: 'relative_pseudo3d',
            ranking: 'heuristic_candidate_only' }, [instanceMask.buffer, depthLinks.labels.buffer]);
    } catch (error) {
        self.postMessage({ frameId: data.frameId, error: error.message });
    }
};
