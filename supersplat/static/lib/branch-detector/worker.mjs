const corePromise = (async () => {
    const response = await fetch(new URL('./branch_detector.wasm', import.meta.url));
    if (!response.ok) throw new Error(`Branch detector WASM HTTP ${response.status}`);
    const { instance } = await WebAssembly.instantiate(await response.arrayBuffer());
    return instance.exports;
})();
const pixels = 640 * 640;

self.onmessage = async ({ data }) => {
    const started = performance.now();
    try {
        const core = await corePromise;
        if (data.rgb.length !== pixels * 4 || data.mask.length !== pixels || data.depth.length !== pixels) {
            throw new Error('Branch detector expects synchronized 640×640 inputs');
        }
        const memory = core.memory.buffer;
        new Uint8Array(memory, core.bd_rgb(), pixels * 4).set(data.rgb);
        new Uint8Array(memory, core.bd_mask(), pixels).set(data.mask);
        new Float32Array(memory, core.bd_depth(), pixels).set(data.depth);
        new Float32Array(memory, core.bd_rotation(), 9).set(data.rotation);
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
        self.postMessage({ frameId: data.frameId, branches, junctions, instanceMask,
            cpu_ms: performance.now() - started, geometry_units: 'relative_pseudo3d',
            ranking: 'heuristic_candidate_only' }, [instanceMask.buffer]);
    } catch (error) {
        self.postMessage({ frameId: data.frameId, error: error.message });
    }
};
