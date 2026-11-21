/* global self */
import { buildSpaceScene, evaluateLandingSpace } from './landing-space-analysis.mjs?v=20261119.2';
let sceneKey = null, scene = null;
self.onmessage = ({ data }) => {
    const { requestId, key } = data;
    try {
        const started = performance.now();
        if (data.sceneKey !== sceneKey) {
            scene = data.depth ? buildSpaceScene(data.depth, data.intrinsics, 640, 640, data.da2ReferenceM ?? null,
                data.referenceMask, data.metricReferenceDepth) : null;
            sceneKey = data.sceneKey;
        }
        const usableScene = data.requireMetric && scene?.geometryUnits !== 'da2_calibrated_meters' ? null : scene;
        const assessments = evaluateLandingSpace({ ...data, scene: usableScene });
        self.postMessage({ requestId, key, assessments, calibration: usableScene?.calibration ?? null, cpu_ms: performance.now() - started });
    } catch (error) {
        self.postMessage({ requestId, key, error: error.message });
    }
};
