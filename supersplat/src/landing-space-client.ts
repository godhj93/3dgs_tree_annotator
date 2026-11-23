import type { BirdSpaceSettings, BranchDiameterAssessment, Da2SpaceCalibration, SpaceAssessment } from '../static/lib/branch-detector/landing-space-analysis.mjs';
import type { FinalLandingResult } from './final-landing-analysis.mjs';

// One reusable Worker/cache per source. Replies carry the complete filter key;
// callers also check the snapshot identity before applying a result.
class LandingSpaceClient {
    channels = new Map<string, { worker: Worker; sceneKey: string }>();
    nextId = 0;

    assess(source: string, key: string, sceneKey: string, depth: Float32Array | null,
        pose: any, base: FinalLandingResult, settings: BirdSpaceSettings, minimumDiameterM: number | null, da2ReferenceM?: number,
        annotationMask?: HTMLCanvasElement, metricDepth?: Float32Array, requireMetric = false): Promise<{
            assessments: SpaceAssessment[]; diameters: BranchDiameterAssessment[]; cpuMs: number; calibration: Da2SpaceCalibration | null;
        }> {
        let channel = this.channels.get(source);
        if (!channel) {
            channel = { worker: new Worker(new URL('static/lib/branch-detector/landing-space-worker.mjs?v=20261123.5', document.baseURI),
                { type: 'module' }),
            sceneKey: '' };
            this.channels.set(source, channel);
        }
        const requestId = ++this.nextId;
        const worker = channel.worker;
        const k = pose.intrinsics;
        const up = [0, 1, 2].map(column => -pose.gravity.vector.reduce((sum: number, value: number, row: number) => sum + value * pose.camera_to_world[row][column], 0));
        const sceneDepth = channel.sceneKey === sceneKey ? undefined : depth?.slice();
        const metricReferenceDepth = sceneDepth ? metricDepth?.slice() : undefined;
        let referenceMask: Uint8Array<ArrayBuffer> | undefined;
        if (sceneDepth && annotationMask) {
            const { width, height } = annotationMask;
            const rgba = annotationMask.getContext('2d').getImageData(0, 0, width, height).data;
            referenceMask = new Uint8Array(width * height);
            for (let p = 0; p < referenceMask.length; p++) referenceMask[p] = rgba[p * 4];
        }
        const transfer = sceneDepth ? [sceneDepth.buffer] : [];
        if (referenceMask) transfer.push(referenceMask.buffer);
        if (metricReferenceDepth) transfer.push(metricReferenceDepth.buffer);
        channel.sceneKey = sceneKey;
        return new Promise((resolve, reject) => {
            const controller = new AbortController();
            const timeout = setTimeout(() => {
                controller.abort(); this.channels.delete(source); worker.terminate(); reject(new Error('Space inspection timed out'));
            }, 30000);
            worker.addEventListener('message', ({ data }: MessageEvent) => {
                if (data.requestId !== requestId || data.key !== key) return;
                clearTimeout(timeout); controller.abort();
                if (data.error) {
                    channel.sceneKey = ''; reject(new Error(data.error));
                } else resolve({ assessments: data.assessments, diameters: data.diameters, cpuMs: data.cpu_ms, calibration: data.calibration });
            }, { signal: controller.signal });
            worker.addEventListener('error', (event: ErrorEvent) => {
                clearTimeout(timeout); controller.abort(); this.channels.delete(source); worker.terminate(); reject(new Error(event.message));
            }, { signal: controller.signal });
            worker.postMessage({ requestId,
                key,
                sceneKey,
                depth: sceneDepth,
                da2ReferenceM,
                referenceMask,
                metricReferenceDepth,
                requireMetric,
                intrinsics: [k[0][0], k[1][1], k[0][2], k[1][2]],
                up,
                minimumDiameterM,
                candidates: base.eligible,
                supportPixels: base.supportPixels,
                settings }, transfer);
        });
    }
}

export { LandingSpaceClient };
