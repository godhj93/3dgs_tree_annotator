/* global self */
import { buildSpaceScene, evaluateBranchDiameters, evaluateLandingSpace } from './landing-space-analysis.mjs?v=20261123.5';
let sceneKey = null, scene = null, mask = null;
self.onmessage = ({ data }) => {
    const { requestId, key } = data;
    try {
        const started = performance.now();
        if (data.sceneKey !== sceneKey) {
            mask = data.referenceMask ?? null;
            scene = data.depth ? buildSpaceScene(data.depth, data.intrinsics, 640, 640, data.da2ReferenceM ?? null,
                mask, data.metricReferenceDepth) : null;
            sceneKey = data.sceneKey;
        }
        const usableScene = data.requireMetric && scene?.geometryUnits !== 'da2_calibrated_meters' ? null : scene;
        const diameters = data.minimumDiameterM == null ? [] :
            evaluateBranchDiameters({ ...data, scene: usableScene, mask, minimumM: data.minimumDiameterM });
        const excluded = new Set(diameters.filter(d => d.status !== 'pass').map(d => d.id));
        const assessments = data.settings.enabled ? evaluateLandingSpace({
            ...data, scene: usableScene, candidates: data.candidates.filter(c => !excluded.has(c.id))
        }) : [];
        self.postMessage({ requestId, key, diameters, assessments, calibration: usableScene?.calibration ?? null, cpu_ms: performance.now() - started });
    } catch (error) {
        self.postMessage({ requestId, key, error: error.message });
    }
};
