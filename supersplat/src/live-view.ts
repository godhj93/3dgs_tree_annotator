import { ADDRESS_CLAMP_TO_EDGE, FILTER_NEAREST, PIXELFORMAT_R8, Texture, Vec3 } from 'playcanvas';

import { ElementType } from './element';
import { comparisonStages, DepthComparisonView } from './depth-comparison-view';
import type { DepthSource, ComparisonSourceView, ComparisonStage } from './depth-comparison-view';
import { Events } from './events';
import { encodeFigures, figureArchive, freezeFigures } from './figure-export';
import type { Figure } from './figure-export';
import { SENSOR_PROFILE, numpyFloat32, numpyUint32, rowMajorMatrix, sensorIntrinsics, visibleMask } from './live-view-utils';
import type { BoxScaleReference } from './live-view-utils';
import { Scene } from './scene';
import { Splat } from './splat';
import { State } from './splat-state';
import { resolveGravityWorld, segmentFinalLanding } from './final-landing-analysis.mjs';
import type { GravitySample, FinalLandingResult } from './final-landing-analysis.mjs';

type AnnotationSource = {
    splat: Splat | null;
    boxes: any[];
    metersPerSceneUnit: number;
    scaleReference?: BoxScaleReference | null;
    contains: (point: Vec3) => boolean;
};

type LiveSnapshot = {
    frameId: string;
    fingerprint: string;
    rgb: HTMLCanvasElement;
    mask: HTMLCanvasElement;
    pose: any;
    depth: Float32Array | null;
    gsDepth: Float32Array | null;
    depthSource?: DepthSource;
    depthKind?: 'relative_inverse' | 'camera_z_m';
    analysisMask?: HTMLCanvasElement;
    analyses?: Partial<Record<DepthSource, LiveSnapshot>>;
    analysisStatus?: 'pending' | 'ready' | 'error';
    analysisError?: string;
    analysisRevision?: number;
    analysisGravityKey?: string;
    boundary?: Uint8Array;
    missing?: Uint8Array;
    validFraction?: number;
    boundaryCells?: number;
    missingCells?: number;
    skeletonLengthCm?: number | null;
    metricLengthCoverage?: number;
    depthInfo?: any;
    rawSkeleton?: Uint8Array;
    rawSkeletonCount?: number;
    rawJunctions?: JunctionMarker[];
    rawOverlay?: HTMLCanvasElement;
    linkLabels?: Uint32Array;
    linkCount?: number;
    linkOverlay?: HTMLCanvasElement;
    depthLinkOverlay?: HTMLCanvasElement;
    pcaMergedOverlay?: HTMLCanvasElement;
    pcaMergedLabels?: Uint32Array;
    pcaMergedCount?: number;
    pcaMergedAngle?: number;
    pcaMergedObjects?: GravityObject[];
    gravityMergedLabels?: Uint32Array;
    gravityMergedOverlay?: HTMLCanvasElement;
    gravityMergedObjects?: GravityObject[];
    gravityMergedKey?: string;
    gravityEligibleCount?: number;
    gravityComponentEligibleIds?: Set<number>;
    gravityFilteredLabels?: Uint32Array;
    gravityFilteredOverlay?: HTMLCanvasElement;
    gravityFilterKey?: string;
    finalLanding?: FinalLandingResult;
    finalLandingKey?: string;
    finalLandingOverlay?: HTMLCanvasElement;
    finalLandingCpuMs?: number;
    comparisonCrop?: { x: number; y: number; size: number };
    detection?: BranchDetection;
};

type JunctionMarker = { x: number; y: number; armCount: number };
type DepthJunctionMarker = JunctionMarker & {
    sampledArms: number;
    status: 'similar-depth' | 'depth-separated' | 'uncertain';
    relativeGap: number;
    relativeNoise: number;
    armDepths: number[];
};

type PcaObject = { id: number; pixels: number; center: number[]; axis: number[];
    linearity: number; reliable: boolean; centerPx: number[]; endpointsPx: number[][] | null;
    centerlineLengthPx: number; trunkCenterlineLengthPx: number; detectorCenterlineLengthPx: number;
    centerlineLengthCm?: number | null;
    moments: { sum: number[]; outer: number[] } };
type PcaAdjacency = { a: number; b: number; contacts: number; angleDeg: number; depthGap: number; eligible: boolean };
type GravityObject = { id: number; pixels: number; axis: number[]; linearity: number;
    reliable: boolean; angleToGravityDeg: number | null; centerlineLengthPx: number;
    centerlineLengthCm?: number | null;
    trunkCenterlineLengthPx: number; detectorCenterlineLengthPx: number; isTrunk: boolean;
    memberIds: number[] };

type BranchDetection = {
    frameId: string;
    branches: any[];
    junctions: number[];
    rawJunctions: JunctionMarker[];
    junctionComparison: DepthJunctionMarker[];
    depthLinkLabels: Uint32Array;
    depthLinkCount: number;
    depthLinkCuts: number;
    depthJunctionCuts: number;
    depthJunctionArmIds: number[][];
    pcaObjects: PcaObject[];
    pcaAdjacency: PcaAdjacency[];
    instanceMask: Uint8Array;
    cpu_ms: number;
    geometry_units: string;
    ranking: string;
    rawSkeleton?: Uint8Array;
    rawSkeletonCount?: number;
    linkLabels?: Uint32Array;
    linkCount?: number;
    processedMask?: Uint8Array;
    boundary?: Uint8Array;
    missing?: Uint8Array;
    boundaryCells?: number;
    missingCells?: number;
    validFraction?: number;
    skeletonLengthCm?: number | null;
    metricLengthCoverage?: number;
};

type BranchView = {
    branches: any[];
    horizontal: any[];
    landing: any[];
    excludedTrunkIds: number[];
    instanceMask: Uint8Array;
    landingMask: Uint8Array;
};

const pixelCanvas = (pixels: Uint8Array | Uint8ClampedArray, width: number, height: number) => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(pixels), width, height), 0, 0);
    return canvas;
};

const squareCanvas = (source: HTMLCanvasElement, smooth: boolean) => {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 640;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = smooth;
    ctx.drawImage(source, 0, 0, 640, 640);
    return canvas;
};

const squareDepth = (source: Float32Array, width: number, height: number, metersPerSceneUnit: number) => {
    const result = new Float32Array(640 * 640);
    for (let y = 0; y < 640; y++) {
        const sourceY = Math.min(height - 1, Math.floor((y + 0.5) * height / 640));
        for (let x = 0; x < 640; x++) {
            const sourceX = Math.min(width - 1, Math.floor((x + 0.5) * width / 640));
            result[y * 640 + x] = source[sourceY * width + sourceX] * metersPerSceneUnit;
        }
    }
    return result;
};

const pngBlob = (canvas: HTMLCanvasElement) => new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(blob => (blob ? resolve(blob) : reject(new Error('PNG encoding failed'))), 'image/png');
});

const depthCanvas = (depth: Float32Array, nearIsLow = false) => {
    const sorted = Array.from(depth).filter(Number.isFinite).sort((a, b) => a - b);
    const lo = sorted[Math.floor(sorted.length * 0.02)] ?? 0;
    const hi = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.98))] ?? lo;
    const pixels = new Uint8ClampedArray(depth.length * 4);
    const stops = [[28, 35, 110], [0, 180, 210], [100, 220, 80], [255, 215, 35], [235, 55, 35]];
    for (let i = 0; i < depth.length; i++) {
        if (!Number.isFinite(depth[i])) {
            pixels.set([5, 10, 16, 255], i * 4);
            continue;
        }
        const fraction = Math.max(0, Math.min(1, (depth[i] - lo) / Math.max(hi - lo, 1e-8)));
        const value = (nearIsLow ? 1 - fraction : fraction) * 4;
        const index = Math.min(3, Math.floor(value));
        const alpha = value - index;
        for (let c = 0; c < 3; c++) pixels[i * 4 + c] = stops[index][c] * (1 - alpha) + stops[index + 1][c] * alpha;
        pixels[i * 4 + 3] = 255;
    }
    return pixelCanvas(pixels, 640, 640);
};

const branchPalette = Array.from({ length: 255 }, (_, id) => {
    if (id === 0) return [5, 10, 16];
    const hue = (id * 137.508) % 360;
    const h = hue / 60, c = 210, x = c * (1 - Math.abs(h % 2 - 1));
    const sectors = [[c, x, 0], [x, c, 0], [0, c, x], [0, x, c], [x, 0, c], [c, 0, x]];
    return sectors[Math.floor(h)].map(v => Math.round(v + 28));
});

const instanceCanvas = (ids: Uint8Array, color: boolean) => {
    const pixels = new Uint8ClampedArray(ids.length * 4);
    for (let i = 0; i < ids.length; i++) {
        const id = ids[i];
        const rgb = color ? branchPalette[id] : [id, id, id];
        pixels[i * 4] = rgb[0]; pixels[i * 4 + 1] = rgb[1]; pixels[i * 4 + 2] = rgb[2];
        pixels[i * 4 + 3] = color && !id ? 0 : 255;
    }
    return pixelCanvas(pixels, 640, 640);
};

const finalLandingCanvas = (ids: Uint32Array) => {
    const pixels = new Uint8ClampedArray(ids.length * 4);
    for (let i = 0; i < ids.length; i++) {
        if (!ids[i]) continue;
        const color = branchPalette[((ids[i] - 1) % 254) + 1];
        pixels.set(color, i * 4);
        pixels[i * 4 + 3] = 220;
    }
    return pixelCanvas(pixels, 640, 640);
};

const principalAxisFromMoments = (pixels: number, sum: number[], outer: number[]) => {
    if (!pixels) return { axis: [1, 0, 0], linearity: 0, reliable: false };
    const center = sum.map(value => value / pixels);
    const covariance = outer.map((value, i) => value / pixels -
        center[Math.floor(i / 3)] * center[i % 3]);
    const trace = covariance[0] + covariance[4] + covariance[8];
    let axis = [1, 0.37, 0.11];
    for (let iteration = 0; iteration < 16; iteration++) {
        const next = [0, 1, 2].map(row => covariance[row * 3] * axis[0] +
            covariance[row * 3 + 1] * axis[1] + covariance[row * 3 + 2] * axis[2]);
        const norm = Math.hypot(...next);
        if (norm < 1e-12) break;
        axis = next.map(value => value / norm);
    }
    const eigenvalue = axis.reduce((sumValue, value, row) => sumValue + value *
        (covariance[row * 3] * axis[0] + covariance[row * 3 + 1] * axis[1] +
            covariance[row * 3 + 2] * axis[2]), 0);
    const linearity = trace > 1e-12 ? Math.max(0, Math.min(1, (3 * eigenvalue / trace - 1) / 2)) : 0;
    return { axis, linearity, reliable: pixels >= 6 && linearity >= 0.2 };
};

class LiveView {
    enabled = true;
    capturing = false;
    busy = false;
    saving = false;
    revision = 0;
    fingerprint = '';
    attempted = '';
    changedAt = performance.now();
    last: LiveSnapshot | null = null;
    memberTexture: Texture | null = null;
    memberSplat: Splat | null = null;
    memberKey = '';
    status: HTMLDivElement;
    rgbCanvas: HTMLCanvasElement;
    maskCanvas: HTMLCanvasElement;
    depthPreview: HTMLCanvasElement;
    gsDepthPreview: HTMLCanvasElement;
    gsDepthTitle: HTMLDivElement;
    branchCanvas: HTMLCanvasElement;
    branchList: HTMLDivElement;
    junctionNoDepthCanvas: HTMLCanvasElement;
    junctionDepthCanvas: HTMLCanvasElement;
    junctionNoDepthLabel: HTMLDivElement;
    junctionDepthLabel: HTMLDivElement;
    junctionSummary: HTMLDivElement;
    junctionCheckbox: HTMLInputElement;
    pcaInputCanvas: HTMLCanvasElement;
    pcaMergedCanvas: HTMLCanvasElement;
    pcaInputLabel: HTMLDivElement;
    pcaMergedLabel: HTMLDivElement;
    pcaSummary: HTMLDivElement;
    pcaAngleSlider: HTMLInputElement;
    pcaAngleNumber: HTMLInputElement;
    gravityInputCanvas: HTMLCanvasElement;
    gravityFilteredCanvas: HTMLCanvasElement;
    gravityInputLabel: HTMLDivElement;
    gravityFilteredLabel: HTMLDivElement;
    gravitySummary: HTMLDivElement;
    gravityAngleSlider: HTMLInputElement;
    gravityAngleNumber: HTMLInputElement;
    gravityLengthSlider: HTMLInputElement;
    gravityLengthNumber: HTMLInputElement;
    gsLengthSlider: HTMLInputElement;
    gsLengthNumber: HTMLInputElement;
    gravityComponentCheckbox: HTMLInputElement;
    gravityTrunkCheckbox: HTMLInputElement;
    finalInputCanvas: HTMLCanvasElement;
    finalOutputCanvas: HTMLCanvasElement;
    finalInputLabel: HTMLDivElement;
    finalOutputLabel: HTMLDivElement;
    finalSummary: HTMLDivElement;
    finalDiagnostics: HTMLDivElement;
    finalWidthSlider: HTMLInputElement;
    finalLinearitySlider: HTMLInputElement;
    finalProjectedLinearitySlider: HTMLInputElement;
    finalCylinderInlierSlider: HTMLInputElement;
    finalDepthSlider: HTMLInputElement;
    finalHeightSlider: HTMLInputElement;
    finalWoodSlider: HTMLInputElement;
    finalFoliageCheckbox: HTMLInputElement;
    finalBearingSlider: HTMLInputElement;
    finalEndOnSlider: HTMLInputElement;
    finalImageTiltSlider: HTMLInputElement;
    finalStrictCameraCheckbox: HTMLInputElement;
    finalTopKInput: HTMLInputElement;
    detectorWorker: Worker | null = null;
    comparisonView: DepthComparisonView;
    comparisonRenderQueued = false;
    depthGapThreshold = 0.15;
    analysisRevision = 0;
    analysisRequestId = 0;
    analysisWorkers: Partial<Record<DepthSource, Worker>> = {};
    analysisCancel: Partial<Record<DepthSource, () => void>> = {};
    comparisonDifferenceOverlay: HTMLCanvasElement | null = null;
    imuGravity: GravitySample | null = null;
    excludeTrunk = false;
    horizontalAngle = 25;
    minBranchLength = 12;
    minLengthWidthRatio = 1.5;
    minDirectionConfidence = 0.5;
    landingTopK = 3;
    showMainLanding = true;
    showJunctions = false;
    pcaMaxAngle = 20;
    gravityTolerance = 20;
    minGravityLength = 12;
    minSkeletonLengthCm = 10;
    requireGravityComponents = false;
    excludeGravityTrunks = false;
    finalMaxWidthPx = 24;
    finalMinLinearity = 0.5;
    finalMinProjectedLinearity = 0.6;
    finalMinCylinderInlier = 0.4;
    finalMaxDepthStdRelative = 0.3;
    finalMinRelativeHeight = 0.1;
    finalMinWoodyFraction = 0.25;
    finalRejectFoliage = true;
    finalMaxCenterBearing = 30;
    finalMaxEndOn = 30;
    finalMaxImageTilt = 30;
    finalStrictCameraAngles = false;
    finalTopK = 3;
    viewMode: 'raw' | 'candidates' | 'final' = 'raw';
    showYellowSegmentation = true;
    mainOverlay: HTMLCanvasElement;
    saveButton: HTMLButtonElement;
    figuresButton: HTMLButtonElement;
    savingFigures = false;
    binaryInput: HTMLInputElement;
    savedCamera: { fov: number; ortho: boolean; aspect: number | null };
    wrapper: HTMLDivElement;
    originalContainerStyle: string;
    retryTimer: ReturnType<typeof setTimeout> | null = null;

    constructor(
        readonly events: Events,
        readonly scene: Scene,
        readonly manager: HTMLDivElement,
        readonly getSource: () => AnnotationSource,
        readonly getOutputFolder: () => any,
        readonly chooseOutputFolder: () => Promise<any>,
        readonly writeFile: (name: string, blob: Blob, folder: any) => Promise<void>
    ) {
        this.installStyle();
        this.figuresButton = this.manager.querySelector('[data-action="save-figures"]');
        this.mount();
        this.savedCamera = { fov: scene.camera.fov, ortho: scene.camera.ortho, aspect: scene.camera.sensorAspect };
        const container = document.getElementById('canvas-container');
        this.originalContainerStyle = container.style.cssText;
        this.wrapper = document.createElement('div');
        this.wrapper.id = 'sensor-frame';
        container.parentElement.insertBefore(this.wrapper, container);
        this.wrapper.append(container);
        this.wrapper.addEventListener('pointerdown', () => {
            if (this.manager.classList.contains('maximized') && this.manager.contains(document.activeElement)) {
                (document.activeElement as HTMLElement).blur();
            }
        }, true);
        this.mainOverlay = document.createElement('canvas');
        this.mainOverlay.id = 'landing-overlay';
        // Preserve the camera aspect so rank labels are not stretched with the 640-square detector mask.
        this.mainOverlay.width = SENSOR_PROFILE.native_width / 4;
        this.mainOverlay.height = SENSOR_PROFILE.native_height / 4;
        this.mainOverlay.setAttribute('aria-label', 'Landing candidates on the main camera view');
        this.mainOverlay.style.visibility = 'hidden';
        this.wrapper.append(this.mainOverlay);
        new ResizeObserver(() => this.layoutCamera()).observe(this.wrapper);
        this.applyProfile();
        events.function('annotation.membershipTexture', (splat: Splat) => (splat === this.memberSplat ? this.memberTexture : null));
        events.function('annotation.showSegmentationTint', () => this.showYellowSegmentation);
        events.function('annotation.liveSnapshot', () => this.last);
        events.function('annotation.setImuGravity', (sample: GravitySample | null) => {
            if (sample) resolveGravityWorld(sample, { world_up_axis: 'z',
                camera_to_world: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] });
            this.imuGravity = sample ? { frame: sample.frame, vector: [...sample.vector] as [number, number, number],
                captured_at_utc: sample.captured_at_utc } : null;
            if (this.last) {
                this.last.pose.gravity = resolveGravityWorld(this.imuGravity, this.last.pose);
                this.last.pcaMergedAngle = undefined;
                this.last.gravityMergedKey = undefined;
                this.last.finalLandingKey = undefined;
                this.drawPcaPanel(this.last);
            }
        });
        events.function('annotation.getImuGravity', () => this.imuGravity);
        events.on('update', () => this.tick());
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) this.refresh();
        });
    }

    installStyle() {
        const style = document.createElement('style');
        style.textContent = `
            #sensor-frame { position: relative; display:flex; flex:1 1 auto; min-height:0; background:#05080b; overflow:hidden; }
            #landing-overlay { position:absolute; z-index:3; pointer-events:none; }
            .bbox-manager { width:min(520px, calc(100vw - 390px)); max-height:calc(100vh - 24px); }
            .bbox-manager .bbox-preview-body { overflow-y:auto; }
            .bbox-manager .bbox-manager-list { min-height:90px; max-height:180px; flex:0 0 auto; }
            .live-view { flex:0 0 auto; margin-bottom:12px; padding-bottom:10px; border-bottom:1px solid #344758; }
            .live-profile { font-size:11px; color:#adc3d5; margin-bottom:7px; }
            .live-controls { display:flex; align-items:center; flex-wrap:wrap; gap:7px; margin-bottom:7px; }
            .live-controls label { display:flex; align-items:center; gap:3px; }
            .live-controls input { width:auto; }
            .live-controls select { border:1px solid #52677d; border-radius:3px; background:#101d2b; color:#e6f4ff; padding:3px 5px; }
            .live-controls input[type="range"] { width:104px; accent-color:#55dce9; }
            .live-controls [hidden], .landing-search[hidden] { display:none !important; }
            .landing-topk-control input { width:64px; box-sizing:border-box; padding:3px 5px; border:1px solid #52677d; border-radius:3px; background:#101d2b; color:#e6f4ff; }
            .landing-search { margin:4px 0 10px; padding:6px 8px; border:1px solid #344758; border-radius:5px; background:#0d1822; }
            .landing-search summary { cursor:pointer; color:#81d4fa; font-size:12px; font-weight:600; }
            .landing-search-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px 12px; margin-top:8px; }
            .landing-setting { display:flex; flex-direction:column; min-width:0; gap:3px; color:#cddce8; font-size:11px; }
            .landing-setting-heading { display:flex; justify-content:space-between; gap:4px; }
            .landing-setting input[type="range"] { width:100%; accent-color:#55dce9; }
            .landing-setting small { color:#8da9bd; line-height:1.2; }
            .landing-search-footer { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-top:7px; color:#8da9bd; font-size:10px; }
            .live-grid { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:7px; }
            .bbox-manager.maximized .live-grid { max-width:1120px; margin:0 auto; }
            .bbox-manager.maximized .live-tile canvas { max-width:min(100%, 44vh); margin:0 auto; }
            .live-tile { min-width:0; border:1px solid #2b3d4f; border-radius:5px; background:#050a10; overflow:hidden; }
            .live-tile-title { color:#81d4fa; font-size:11px; padding:4px 6px; }
            .live-tile canvas { display:block; width:100%; height:auto; aspect-ratio:1; }
            .live-branch-list { font-size:10px; line-height:1.35; padding:5px 6px; color:#cddce8; max-height:125px; overflow:auto; }
            .bbox-manager.maximized .live-branch-list { max-height:220px; }
            .live-branch-list div { margin-bottom:3px; }
            .live-junction-tile { grid-column:1 / -1; }
            .live-junction-tile .live-tile-title { display:flex; align-items:center; justify-content:space-between; gap:8px; }
            .live-junction-toggle { display:flex; align-items:center; gap:4px; color:#e2eff8; white-space:nowrap; cursor:pointer; }
            .live-junction-toggle input { margin:0; accent-color:#55dce9; }
            .live-junction-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; padding:6px; }
            .live-junction-view { min-width:0; }
            .live-junction-view canvas { display:block; width:100%; height:auto; aspect-ratio:1; }
            .live-junction-label { min-height:27px; padding:4px 1px; color:#cddce8; font-size:10px; }
            .live-junction-summary { padding:1px 7px 7px; color:#a9bfd0; font-size:10px; line-height:1.35; }
            .live-pca-controls { display:flex; align-items:center; gap:7px; flex-wrap:wrap; padding:4px 7px; color:#d9eaf4; font-size:11px; }
            .live-pca-controls input[type="range"] { flex:1 1 125px; min-width:100px; accent-color:#55dce9; }
            .live-pca-controls input[type="number"] { width:58px; padding:3px; box-sizing:border-box; border:1px solid #52677d; border-radius:3px; background:#101d2b; color:#e6f4ff; }
            .live-status { font-size:11px; color:#adc3d5; margin-top:7px; min-height:28px; overflow-wrap:anywhere; }
            .live-view[data-stale="true"] .live-grid { opacity:.55; }
            .live-view[data-state="error"] .live-status { color:#ffb59e; }
            .live-save { display:flex; gap:7px; flex-wrap:wrap; margin-top:7px; }
            .live-save button:disabled { opacity:.4; cursor:default; }
            @media(max-width:760px) { .bbox-manager { width: min(430px,calc(100vw - 24px)); } }
        `;
        document.head.append(style);
    }

    button(label: string, action: () => void) {
        const button = document.createElement('button');
        button.textContent = label;
        button.addEventListener('click', action);
        return button;
    }

    mount() {
        const root = document.createElement('section');
        root.className = 'live-view';
        root.dataset.stale = 'true';
        const profile = document.createElement('div');
        profile.className = 'live-profile';
        profile.textContent = 'LI-IMX678 · FoV 118° H / 86.407° V (pinhole approximation) · 640×640';
        const controls = document.createElement('div');
        controls.className = 'live-controls';
        const enabledLabel = document.createElement('label');
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = true;
        input.addEventListener('change', () => {
            this.enabled = input.checked;
            this.applyProfile();
            this.refresh();
            if (!this.enabled) this.message('paused', 'Preview paused. Camera controls restored.');
        });
        enabledLabel.append(input, 'Live preview');
        const maskLabel = document.createElement('label');
        this.binaryInput = document.createElement('input');
        this.binaryInput.type = 'checkbox';
        this.binaryInput.addEventListener('change', () => this.drawMask());
        maskLabel.append(this.binaryInput, 'Binary mask');
        controls.append(enabledLabel, maskLabel, this.button('Refresh', () => this.refresh()));
        const yellowLabel = document.createElement('label');
        const yellowInput = document.createElement('input');
        yellowInput.type = 'checkbox'; yellowInput.checked = true;
        yellowInput.setAttribute('aria-label', 'Show yellow segmentation');
        yellowInput.addEventListener('change', () => {
            this.showYellowSegmentation = yellowInput.checked;
            this.drawMask();
            this.scene.forceRender = true;
        });
        yellowLabel.append(yellowInput, 'Yellow segmentation');
        controls.append(yellowLabel);
        const viewLabel = document.createElement('label');
        const viewSelect = document.createElement('select');
        viewSelect.setAttribute('aria-label', 'Branch visualization');
        for (const [value, title] of [['raw', 'Depth-aware skeleton'], ['candidates', 'Landing candidates'],
            ['final', 'Final segmentation']]) {
            const option = document.createElement('option'); option.value = value; option.textContent = title;
            viewSelect.append(option);
        }
        viewLabel.append('View', viewSelect);
        controls.append(viewLabel);
        const topKLabel = document.createElement('label');
        topKLabel.className = 'landing-topk-control';
        const topKInput = document.createElement('input');
        topKInput.type = 'number'; topKInput.min = '1'; topKInput.max = '253'; topKInput.step = '1';
        topKInput.value = String(this.landingTopK);
        topKInput.setAttribute('aria-label', 'Landing candidate top K');
        topKInput.addEventListener('input', () => {
            const value = Number(topKInput.value);
            if (!Number.isInteger(value) || value < 1) return;
            this.landingTopK = Math.min(253, value);
            if (this.last?.detection) this.drawBranches(this.last);
        });
        topKInput.addEventListener('change', () => { topKInput.value = String(this.landingTopK); });
        topKLabel.append('Top K', topKInput);
        controls.append(topKLabel);
        const mainLabel = document.createElement('label');
        const mainInput = document.createElement('input');
        mainInput.type = 'checkbox'; mainInput.checked = true;
        mainInput.setAttribute('aria-label', 'Show landing candidates on main view');
        mainInput.addEventListener('change', () => {
            this.showMainLanding = mainInput.checked;
            if (this.last?.detection) this.drawBranches(this.last);
            else this.hideMainLanding();
        });
        mainLabel.append(mainInput, 'Show on main view');
        controls.append(mainLabel);

        const search = document.createElement('details');
        search.className = 'landing-search'; search.open = true;
        const searchSummary = document.createElement('summary');
        searchSummary.textContent = 'Branch search · adjust landing candidates';
        const searchGrid = document.createElement('div');
        searchGrid.className = 'landing-search-grid';
        const setting = (name: string, aria: string, min: number, max: number, step: number,
            initial: number, unit: string, hint: string, apply: (value: number) => void) => {
            const label = document.createElement('label'); label.className = 'landing-setting';
            const heading = document.createElement('span'); heading.className = 'landing-setting-heading';
            const valueLabel = document.createElement('strong'); valueLabel.textContent = `${initial}${unit}`;
            heading.append(name, valueLabel);
            const slider = document.createElement('input');
            slider.type = 'range'; slider.min = String(min); slider.max = String(max);
            slider.step = String(step); slider.value = String(initial);
            slider.setAttribute('aria-label', aria);
            slider.addEventListener('input', () => {
                const value = Number(slider.value);
                valueLabel.textContent = `${value}${unit}`;
                apply(value);
                if (this.last?.detection) this.drawBranches(this.last);
            });
            const help = document.createElement('small'); help.textContent = hint;
            label.append(heading, slider, help);
            searchGrid.append(label);
            return { slider, valueLabel };
        };
        const angle = setting('Max tilt from ground', 'Maximum ground-parallel angle', 5, 80, 5,
            25, '°', 'Larger angle includes more sloped branches.', value => { this.horizontalAngle = value; });
        const length = setting('Min visible length', 'Minimum branch length', 6, 80, 2,
            12, 'px', 'Larger length removes short fragments.', value => { this.minBranchLength = value; });
        const shape = setting('Min length / width', 'Minimum branch length to width ratio', 1, 5, 0.25,
            1.5, '×', 'Larger ratio removes wide blobs.', value => { this.minLengthWidthRatio = value; });
        const confidence = setting('Min direction certainty', 'Minimum direction confidence', 0, 1, 0.1,
            0.5, '', 'Larger value removes ambiguous directions.', value => { this.minDirectionConfidence = value; });
        const trunkLabel = document.createElement('label');
        const trunkInput = document.createElement('input');
        trunkInput.type = 'checkbox';
        trunkInput.setAttribute('aria-label', 'Exclude trunk');
        trunkInput.addEventListener('change', () => {
            this.excludeTrunk = trunkInput.checked;
            if (this.last?.detection) this.drawBranches(this.last);
        });
        trunkLabel.append(trunkInput, 'Exclude trunk');
        const searchFooter = document.createElement('div'); searchFooter.className = 'landing-search-footer';
        const reset = this.button('Reset search', () => {
            this.horizontalAngle = 25; this.minBranchLength = 12;
            this.minLengthWidthRatio = 1.5; this.minDirectionConfidence = 0.5;
            this.excludeTrunk = false; trunkInput.checked = false;
            for (const [control, value, suffix] of [[angle, 25, '°'], [length, 12, 'px'],
                [shape, 1.5, '×'], [confidence, 0.5, '']] as const) {
                control.slider.value = String(value);
                control.valueLabel.textContent = `${value}${suffix}`;
            }
            if (this.last?.detection) this.drawBranches(this.last);
        });
        searchFooter.append(trunkLabel, reset);
        search.append(searchSummary, searchGrid, searchFooter);
        topKLabel.hidden = true;
        mainLabel.hidden = true;
        search.hidden = true;
        viewSelect.addEventListener('change', () => {
            this.viewMode = viewSelect.value === 'candidates' ? 'candidates' :
                (viewSelect.value === 'final' ? 'final' : 'raw');
            const raw = this.viewMode === 'raw';
            topKLabel.hidden = this.viewMode !== 'candidates';
            mainLabel.hidden = raw;
            search.hidden = this.viewMode !== 'candidates';
            if (this.last) this.drawBranches(this.last);
        });
        const grid = document.createElement('div');
        grid.className = 'live-grid';
        const tile = (title: string) => {
            const item = document.createElement('div');
            item.className = 'live-tile';
            const heading = document.createElement('div');
            heading.className = 'live-tile-title';
            heading.textContent = title;
            const view = document.createElement('canvas');
            view.width = view.height = 640;
            item.append(heading, view);
            grid.append(item);
            return view;
        };
        this.rgbCanvas = tile('RGB') as HTMLCanvasElement;
        this.maskCanvas = tile('Segmentation · visible branches') as HTMLCanvasElement;
        this.depthPreview = tile('DA2 relative · blue far / red near') as HTMLCanvasElement;
        this.gsDepthPreview = tile('3DGS depth · calibrated meters · blue far / red near') as HTMLCanvasElement;
        this.gsDepthTitle = this.gsDepthPreview.parentElement.querySelector('.live-tile-title') as HTMLDivElement;
        this.branchCanvas = tile('Raw skeleton · RGB overlay') as HTMLCanvasElement;
        this.branchList = document.createElement('div');
        this.branchList.className = 'live-branch-list';
        this.branchList.textContent = 'Waiting for visible segmentation…';
        this.branchCanvas.parentElement.append(this.branchList);
        const junctionTile = document.createElement('div');
        junctionTile.className = 'live-tile live-junction-tile';
        const junctionTitle = document.createElement('div');
        junctionTitle.className = 'live-tile-title';
        const junctionHeading = document.createElement('span');
        junctionHeading.textContent = 'Links / junctions · 2D vs DA2';
        const junctionToggle = document.createElement('label');
        junctionToggle.className = 'live-junction-toggle';
        this.junctionCheckbox = document.createElement('input');
        this.junctionCheckbox.type = 'checkbox';
        this.junctionCheckbox.setAttribute('aria-label', 'Show junctions');
        this.junctionCheckbox.addEventListener('change', () => {
            this.showJunctions = this.junctionCheckbox.checked;
            if (this.last) this.drawJunctionComparison(this.last);
        });
        junctionToggle.append(this.junctionCheckbox, 'Show junctions');
        junctionTitle.append(junctionHeading, junctionToggle);
        const junctionGrid = document.createElement('div');
        junctionGrid.className = 'live-junction-grid';
        const junctionView = (label: string) => {
            const view = document.createElement('div'); view.className = 'live-junction-view';
            const canvas = document.createElement('canvas'); canvas.width = canvas.height = 640;
            const caption = document.createElement('div'); caption.className = 'live-junction-label';
            caption.textContent = label;
            view.append(canvas, caption); junctionGrid.append(view);
            return { canvas, caption };
        };
        const plain = junctionView('2D only · waiting for skeleton');
        const guided = junctionView('With DA2 · waiting for relative depth');
        this.junctionNoDepthCanvas = plain.canvas;
        this.junctionDepthCanvas = guided.canvas;
        this.junctionNoDepthLabel = plain.caption;
        this.junctionDepthLabel = guided.caption;
        this.junctionSummary = document.createElement('div');
        this.junctionSummary.className = 'live-junction-summary';
        this.junctionSummary.textContent = 'Each visible link has its own color. Check Show junctions to mark branching points.';
        junctionTile.append(junctionTitle, junctionGrid, this.junctionSummary);
        grid.append(junctionTile);
        const pcaTile = document.createElement('div');
        pcaTile.className = 'live-tile live-junction-tile live-pca-tile';
        const pcaTitle = document.createElement('div');
        pcaTitle.className = 'live-tile-title';
        pcaTitle.textContent = 'PCA axis merging · DA2 segments';
        const pcaControls = document.createElement('div');
        pcaControls.className = 'live-pca-controls';
        const pcaAngleLabel = document.createElement('label');
        pcaAngleLabel.textContent = 'Maximum relative axis angle';
        this.pcaAngleSlider = document.createElement('input');
        this.pcaAngleSlider.type = 'range'; this.pcaAngleSlider.min = '0'; this.pcaAngleSlider.max = '90';
        this.pcaAngleSlider.step = '1'; this.pcaAngleSlider.value = String(this.pcaMaxAngle);
        this.pcaAngleSlider.setAttribute('aria-label', 'Maximum PCA axis angle');
        this.pcaAngleNumber = document.createElement('input');
        this.pcaAngleNumber.type = 'number'; this.pcaAngleNumber.min = '0'; this.pcaAngleNumber.max = '90';
        this.pcaAngleNumber.step = '1'; this.pcaAngleNumber.value = String(this.pcaMaxAngle);
        this.pcaAngleNumber.setAttribute('aria-label', 'Maximum PCA axis angle degrees');
        const updateAngle = (value: number) => {
            if (!Number.isFinite(value)) return;
            this.pcaMaxAngle = Math.max(0, Math.min(90, Math.round(value)));
            this.pcaAngleSlider.value = this.pcaAngleNumber.value = String(this.pcaMaxAngle);
            if (this.last) this.drawPcaPanel(this.last);
        };
        this.pcaAngleSlider.addEventListener('input', () => updateAngle(Number(this.pcaAngleSlider.value)));
        this.pcaAngleNumber.addEventListener('change', () => updateAngle(Number(this.pcaAngleNumber.value)));
        pcaControls.append(pcaAngleLabel, this.pcaAngleSlider, this.pcaAngleNumber, document.createTextNode('°'));
        const pcaGrid = document.createElement('div'); pcaGrid.className = 'live-junction-grid';
        const pcaView = (label: string) => {
            const view = document.createElement('div'); view.className = 'live-junction-view';
            const canvas = document.createElement('canvas'); canvas.width = canvas.height = 640;
            const caption = document.createElement('div'); caption.className = 'live-junction-label';
            caption.textContent = label; view.append(canvas, caption); pcaGrid.append(view);
            return { canvas, caption };
        };
        const pcaInput = pcaView('DA2 segments · waiting for depth');
        const pcaMerged = pcaView('PCA merged · waiting for depth');
        this.pcaInputCanvas = pcaInput.canvas; this.pcaInputLabel = pcaInput.caption;
        this.pcaMergedCanvas = pcaMerged.canvas; this.pcaMergedLabel = pcaMerged.caption;
        this.pcaSummary = document.createElement('div'); this.pcaSummary.className = 'live-junction-summary';
        this.pcaSummary.textContent = 'PCA will fit each DA2 segment after relative depth arrives.';
        pcaTile.append(pcaTitle, pcaControls, pcaGrid, this.pcaSummary);
        grid.append(pcaTile);
        const gravityTile = document.createElement('div');
        gravityTile.className = 'live-tile live-junction-tile live-gravity-tile';
        const gravityTitle = document.createElement('div');
        gravityTitle.className = 'live-tile-title';
        gravityTitle.textContent = 'Gravity angle filter · merged PCA axis';
        const gravityControls = document.createElement('div');
        gravityControls.className = 'live-pca-controls';
        const gravityAngleLabel = document.createElement('label');
        gravityAngleLabel.textContent = 'Maximum deviation from 90°';
        this.gravityAngleSlider = document.createElement('input');
        this.gravityAngleSlider.type = 'range'; this.gravityAngleSlider.min = '0'; this.gravityAngleSlider.max = '90';
        this.gravityAngleSlider.step = '1'; this.gravityAngleSlider.value = String(this.gravityTolerance);
        this.gravityAngleSlider.setAttribute('aria-label', 'Maximum gravity angle deviation');
        this.gravityAngleNumber = document.createElement('input');
        this.gravityAngleNumber.type = 'number'; this.gravityAngleNumber.min = '0'; this.gravityAngleNumber.max = '90';
        this.gravityAngleNumber.step = '1'; this.gravityAngleNumber.value = String(this.gravityTolerance);
        this.gravityAngleNumber.setAttribute('aria-label', 'Maximum gravity angle deviation degrees');
        const updateGravityAngle = (value: number) => {
            if (!Number.isFinite(value)) return;
            this.gravityTolerance = Math.max(0, Math.min(90, Math.round(value)));
            this.gravityAngleSlider.value = this.gravityAngleNumber.value = String(this.gravityTolerance);
            if (this.last) this.drawGravityPanel(this.last);
        };
        this.gravityAngleSlider.addEventListener('input', () => updateGravityAngle(Number(this.gravityAngleSlider.value)));
        this.gravityAngleNumber.addEventListener('change', () => updateGravityAngle(Number(this.gravityAngleNumber.value)));
        gravityControls.append(gravityAngleLabel, this.gravityAngleSlider, this.gravityAngleNumber,
            document.createTextNode('°'));
        const gravityLengthControls = document.createElement('div');
        gravityLengthControls.className = 'live-pca-controls';
        const gravityLengthLabel = document.createElement('label');
        gravityLengthLabel.textContent = 'DA2 minimum skeleton length';
        this.gravityLengthSlider = document.createElement('input');
        this.gravityLengthSlider.type = 'range'; this.gravityLengthSlider.min = '0'; this.gravityLengthSlider.max = '200';
        this.gravityLengthSlider.step = '1'; this.gravityLengthSlider.value = String(this.minGravityLength);
        this.gravityLengthSlider.setAttribute('aria-label', 'Minimum skeleton length');
        this.gravityLengthNumber = document.createElement('input');
        this.gravityLengthNumber.type = 'number'; this.gravityLengthNumber.min = '0'; this.gravityLengthNumber.max = '200';
        this.gravityLengthNumber.step = '1'; this.gravityLengthNumber.value = String(this.minGravityLength);
        this.gravityLengthNumber.setAttribute('aria-label', 'Minimum skeleton length pixels');
        const updateGravityLength = (value: number) => {
            if (!Number.isFinite(value)) return;
            this.minGravityLength = Math.max(0, Math.min(200, Math.round(value)));
            this.gravityLengthSlider.value = this.gravityLengthNumber.value = String(this.minGravityLength);
            if (this.last) this.drawGravityPanel(this.last);
        };
        this.gravityLengthSlider.addEventListener('input', () => updateGravityLength(Number(this.gravityLengthSlider.value)));
        this.gravityLengthNumber.addEventListener('change', () => updateGravityLength(Number(this.gravityLengthNumber.value)));
        const gravityTrunkLabel = document.createElement('label');
        gravityTrunkLabel.className = 'live-junction-toggle';
        this.gravityTrunkCheckbox = document.createElement('input');
        this.gravityTrunkCheckbox.type = 'checkbox';
        this.gravityTrunkCheckbox.checked = this.excludeGravityTrunks;
        this.gravityTrunkCheckbox.setAttribute('aria-label', 'Exclude trunk objects');
        this.gravityTrunkCheckbox.addEventListener('change', () => {
            this.excludeGravityTrunks = this.gravityTrunkCheckbox.checked;
            if (this.last) this.drawGravityPanel(this.last);
        });
        gravityTrunkLabel.append(this.gravityTrunkCheckbox, document.createTextNode('Exclude trunks'));
        const gravityComponentLabel = document.createElement('label');
        gravityComponentLabel.className = 'live-junction-toggle';
        this.gravityComponentCheckbox = document.createElement('input');
        this.gravityComponentCheckbox.type = 'checkbox';
        this.gravityComponentCheckbox.checked = this.requireGravityComponents;
        this.gravityComponentCheckbox.setAttribute('aria-label', 'Require every PCA component horizontal');
        this.gravityComponentCheckbox.addEventListener('change', () => {
            this.requireGravityComponents = this.gravityComponentCheckbox.checked;
            if (this.last) this.drawGravityPanel(this.last);
        });
        gravityComponentLabel.append(this.gravityComponentCheckbox,
            document.createTextNode('Require each component horizontal'));
        gravityLengthControls.append(gravityLengthLabel, this.gravityLengthSlider, this.gravityLengthNumber,
            document.createTextNode('px'), gravityComponentLabel, gravityTrunkLabel);
        const gsLengthControls = document.createElement('div');
        gsLengthControls.className = 'live-pca-controls';
        const gsLengthLabel = document.createElement('label'); gsLengthLabel.textContent = '3DGS minimum skeleton length';
        this.gsLengthSlider = document.createElement('input');
        this.gsLengthSlider.type = 'range'; this.gsLengthSlider.min = '0'; this.gsLengthSlider.max = '200';
        this.gsLengthSlider.step = '1'; this.gsLengthSlider.value = String(this.minSkeletonLengthCm);
        this.gsLengthSlider.setAttribute('aria-label', '3DGS minimum skeleton length cm slider');
        this.gsLengthNumber = document.createElement('input');
        this.gsLengthNumber.type = 'number'; this.gsLengthNumber.min = '0'; this.gsLengthNumber.step = '1';
        this.gsLengthNumber.value = String(this.minSkeletonLengthCm);
        this.gsLengthNumber.setAttribute('aria-label', '3DGS minimum skeleton length cm');
        const updateGsLength = (value: number) => {
            if (!Number.isFinite(value) || value < 0) return;
            this.minSkeletonLengthCm = value;
            this.gsLengthSlider.value = this.gsLengthNumber.value = String(value);
            if (this.last) this.drawGravityPanel(this.last);
        };
        this.gsLengthSlider.addEventListener('input', () => updateGsLength(Number(this.gsLengthSlider.value)));
        this.gsLengthNumber.addEventListener('change', () => updateGsLength(Number(this.gsLengthNumber.value)));
        gsLengthControls.append(gsLengthLabel, this.gsLengthSlider, this.gsLengthNumber, document.createTextNode('cm'));
        const gravityGrid = document.createElement('div'); gravityGrid.className = 'live-junction-grid';
        const gravityView = (label: string) => {
            const view = document.createElement('div'); view.className = 'live-junction-view';
            const canvas = document.createElement('canvas'); canvas.width = canvas.height = 640;
            const caption = document.createElement('div'); caption.className = 'live-junction-label';
            caption.textContent = label; view.append(canvas, caption); gravityGrid.append(view);
            return { canvas, caption };
        };
        const gravityInput = gravityView('PCA merged · waiting for depth');
        const gravityFiltered = gravityView('Near-horizontal · waiting for depth');
        this.gravityInputCanvas = gravityInput.canvas; this.gravityInputLabel = gravityInput.caption;
        this.gravityFilteredCanvas = gravityFiltered.canvas; this.gravityFilteredLabel = gravityFiltered.caption;
        this.gravitySummary = document.createElement('div'); this.gravitySummary.className = 'live-junction-summary';
        this.gravitySummary.textContent = 'Merged PCA axes will be checked against gravity.';
        gravityTile.append(gravityTitle, gravityControls, gravityLengthControls, gsLengthControls, gravityGrid, this.gravitySummary);
        grid.append(gravityTile);
        const finalTile = document.createElement('div');
        finalTile.className = 'live-tile live-junction-tile live-final-tile';
        const finalTitle = document.createElement('div'); finalTitle.className = 'live-tile-title';
        finalTitle.textContent = 'Final landing candidate segmentation';
        const finalControls = document.createElement('div'); finalControls.className = 'live-pca-controls';
        const cameraControls = document.createElement('div'); cameraControls.className = 'live-pca-controls';
        const cameraHeading = document.createElement('strong'); cameraHeading.textContent = 'Camera approach';
        cameraControls.append(cameraHeading);
        const strictCameraLabel = document.createElement('label'); strictCameraLabel.className = 'live-junction-toggle';
        this.finalStrictCameraCheckbox = document.createElement('input'); this.finalStrictCameraCheckbox.type = 'checkbox';
        this.finalStrictCameraCheckbox.checked = this.finalStrictCameraAngles;
        this.finalStrictCameraCheckbox.setAttribute('aria-label', 'Require camera angles');
        this.finalStrictCameraCheckbox.addEventListener('change', () => {
            this.finalStrictCameraAngles = this.finalStrictCameraCheckbox.checked;
            if (this.last) this.drawFinalPanel(this.last);
        });
        strictCameraLabel.append(this.finalStrictCameraCheckbox, 'Require angles');
        cameraControls.append(strictCameraLabel);
        const finalSetting = (name: string, aria: string, min: number, max: number, step: number,
            initial: number, suffix: string, update: (value: number) => void,
            container = finalControls) => {
            const label = document.createElement('label'); label.textContent = name;
            const slider = document.createElement('input'); slider.type = 'range';
            slider.min = String(min); slider.max = String(max); slider.step = String(step); slider.value = String(initial);
            slider.setAttribute('aria-label', aria);
            const value = document.createElement('span'); value.textContent = `${initial}${suffix}`;
            slider.addEventListener('input', () => {
                update(Number(slider.value)); value.textContent = `${slider.value}${suffix}`;
                if (this.last) this.drawFinalPanel(this.last);
            });
            container.append(label, slider, value);
            return slider;
        };
        this.finalWidthSlider = finalSetting('Max apparent width', 'Maximum landing branch width', 4, 80, 2,
            this.finalMaxWidthPx, 'px', value => { this.finalMaxWidthPx = value; });
        this.finalLinearitySlider = finalSetting('Min 3D linearity', 'Minimum landing PCA linearity', 0, 1, 0.05,
            this.finalMinLinearity, '', value => { this.finalMinLinearity = value; });
        this.finalProjectedLinearitySlider = finalSetting('Min image linearity', 'Minimum projected branch linearity', 0, 1, 0.05,
            this.finalMinProjectedLinearity, '', value => { this.finalMinProjectedLinearity = value; });
        this.finalCylinderInlierSlider = finalSetting('Min cylinder inliers', 'Minimum projected cylinder inlier fraction', 0, 1, 0.05,
            this.finalMinCylinderInlier, '', value => { this.finalMinCylinderInlier = value; });
        this.finalDepthSlider = finalSetting('Max relative depth spread', 'Maximum landing relative depth spread', 0.05, 1, 0.05,
            this.finalMaxDepthStdRelative, '', value => { this.finalMaxDepthStdRelative = value; });
        this.finalHeightSlider = finalSetting('Min relative height', 'Minimum landing relative height', 0, 0.5, 0.05,
            this.finalMinRelativeHeight, '', value => { this.finalMinRelativeHeight = value; });
        this.finalWoodSlider = finalSetting('Min non-green fraction', 'Minimum non-green branch fraction', 0, 1, 0.05,
            this.finalMinWoodyFraction, '', value => { this.finalMinWoodyFraction = value; });
        const foliageLabel = document.createElement('label'); foliageLabel.className = 'live-junction-toggle';
        this.finalFoliageCheckbox = document.createElement('input'); this.finalFoliageCheckbox.type = 'checkbox';
        this.finalFoliageCheckbox.checked = this.finalRejectFoliage;
        this.finalFoliageCheckbox.setAttribute('aria-label', 'Reject green foliage');
        this.finalFoliageCheckbox.addEventListener('change', () => {
            this.finalRejectFoliage = this.finalFoliageCheckbox.checked;
            if (this.last) this.drawFinalPanel(this.last);
        });
        foliageLabel.append(this.finalFoliageCheckbox, 'Reject green foliage'); finalControls.append(foliageLabel);
        this.finalBearingSlider = finalSetting('Max center angle', 'Maximum camera center bearing angle', 0, 90, 5,
            this.finalMaxCenterBearing, '°', value => { this.finalMaxCenterBearing = value; }, cameraControls);
        this.finalEndOnSlider = finalSetting('Max axis toward camera', 'Maximum branch axis angle toward camera; 90 degrees allows any direction', 0, 90, 5,
            this.finalMaxEndOn, '°', value => { this.finalMaxEndOn = value; }, cameraControls);
        this.finalImageTiltSlider = finalSetting('Max screen tilt', 'Maximum branch tilt in camera image', 0, 90, 5,
            this.finalMaxImageTilt, '°', value => { this.finalMaxImageTilt = value; }, cameraControls);
        const finalTopKLabel = document.createElement('label'); finalTopKLabel.textContent = 'Top K';
        this.finalTopKInput = document.createElement('input'); this.finalTopKInput.type = 'number';
        this.finalTopKInput.min = '1'; this.finalTopKInput.max = '253'; this.finalTopKInput.step = '1';
        this.finalTopKInput.value = String(this.finalTopK);
        this.finalTopKInput.setAttribute('aria-label', 'Final landing top K');
        this.finalTopKInput.addEventListener('input', () => {
            const value = Number(this.finalTopKInput.value);
            if (!Number.isInteger(value) || value < 1) return;
            this.finalTopK = Math.min(253, value);
            if (this.last) this.drawFinalPanel(this.last);
        });
        finalTopKLabel.append(this.finalTopKInput); finalControls.append(finalTopKLabel);
        const finalGrid = document.createElement('div'); finalGrid.className = 'live-junction-grid';
        const finalView = (caption: string) => {
            const view = document.createElement('div'); view.className = 'live-junction-view';
            const canvas = document.createElement('canvas'); canvas.width = canvas.height = 640;
            const label = document.createElement('div'); label.className = 'live-junction-label';
            label.textContent = caption; view.append(canvas, label); finalGrid.append(view);
            return { canvas, label };
        };
        const finalInput = finalView('Gravity-filtered regions · waiting for depth');
        const finalOutput = finalView('Final landing mask · waiting for depth');
        this.finalInputCanvas = finalInput.canvas; this.finalInputLabel = finalInput.label;
        this.finalOutputCanvas = finalOutput.canvas; this.finalOutputLabel = finalOutput.label;
        this.finalInputCanvas.addEventListener('mousemove', (event: MouseEvent) => {
            const snapshot = this.last;
            if (!snapshot?.detection || !snapshot.finalLanding) return;
            const rect = this.finalInputCanvas.getBoundingClientRect();
            const crop = this.comparisonCrop(snapshot);
            const x = Math.floor(crop.x + (event.clientX - rect.left) / rect.width * crop.size);
            const y = Math.floor(crop.y + (event.clientY - rect.top) / rect.height * crop.size);
            if (x < 0 || x >= 640 || y < 0 || y >= 640) return;
            const small = (y >> 1) * 320 + (x >> 1);
            const id = snapshot.detection.depthLinkLabels[small];
            if (!id) {
                this.finalDiagnostics.textContent = this.finalDiagnostics.dataset.default;
                return;
            }
            const candidate = snapshot.finalLanding.results.find(item => item.id === id);
            if (candidate) {
                this.finalDiagnostics.textContent = `ID ${id}: ${candidate.reasons.length ?
                    `rejected by ${candidate.reasons.join(', ')}` :
                    (snapshot.finalLanding.selected.some(item => item.id === id) ? 'selected' : 'eligible, outside Top K')}. ` +
                    `Wood ${(candidate.wood_fraction * 100).toFixed(0)}%, 3D line ${candidate.linearity.toFixed(2)}, ` +
                    `image line ${candidate.projected_linearity.toFixed(2)}, cylinder inliers ` +
                    `${(candidate.cylinder_inlier_fraction * 100).toFixed(0)}%, relative height ` +
                    `${candidate.relative_height.toFixed(2)}, length ${candidate.centerline_length_px}px.`;
                return;
            }
            const mergedId = snapshot.gravityMergedLabels?.[small] ?? 0;
            const merged = snapshot.gravityMergedObjects?.find(item => item.id === mergedId);
            const source = snapshot.detection.pcaObjects.find(item => item.id === id);
            const componentAngle = source?.reliable ? 90 - Math.acos(Math.min(1,
                Math.abs(source.axis.reduce((sum, value, i) => sum + value * snapshot.pose.gravity.vector[i], 0)))) *
                180 / Math.PI : null;
            const componentPasses = merged?.memberIds.some(memberId =>
                snapshot.gravityComponentEligibleIds?.has(memberId));
            const cause = !mergedId ? (this.requireGravityComponents ? 'component gravity gate' : 'no merged object') :
                !merged?.reliable ? 'unreliable merged PCA' :
                90 - merged.angleToGravityDeg > this.gravityTolerance &&
                    (this.requireGravityComponents || !componentPasses) ? 'gravity angle' :
                merged.centerlineLengthPx < this.minGravityLength ? 'skeleton length' :
                this.excludeGravityTrunks && merged.isTrunk ? 'trunk exclusion' : 'missing valid depth pixels';
            this.finalDiagnostics.textContent = `ID ${id}: removed before final screening (${cause}). ` +
                `Component deviation ${componentAngle === null ? 'unreliable' : `${componentAngle.toFixed(1)}°`}; ` +
                `merged deviation ${merged?.reliable ? `${(90 - merged.angleToGravityDeg).toFixed(1)}°` : 'unavailable'}; ` +
                `allowed ≤${this.gravityTolerance}°.`;
        });
        this.finalInputCanvas.addEventListener('mouseleave', () => {
            this.finalDiagnostics.textContent = this.finalDiagnostics.dataset.default;
        });
        this.finalSummary = document.createElement('div'); this.finalSummary.className = 'live-junction-summary';
        this.finalSummary.textContent = 'Width, 3D linearity and depth spread will refine the gravity-filtered objects.';
        this.finalDiagnostics = document.createElement('div');
        this.finalDiagnostics.className = 'live-junction-summary';
        this.finalDiagnostics.textContent = 'Candidate rejection reasons will appear here.';
        this.finalDiagnostics.dataset.default = this.finalDiagnostics.textContent;
        finalTile.append(finalTitle, finalControls, cameraControls, finalGrid, this.finalSummary, this.finalDiagnostics);
        grid.append(finalTile);
        this.status = document.createElement('div');
        this.status.className = 'live-status';
        const save = document.createElement('div');
        save.className = 'live-save';
        this.saveButton = this.button('Save DA2 Set', () => {
            this.save().catch(error => this.message('error', `Save error: ${error.message}`));
        });
        this.saveButton.disabled = true;
        save.append(this.saveButton, this.button('Copy Pose JSON', () => {
            if (this.last) navigator.clipboard.writeText(JSON.stringify(this.last.pose, null, 2)).catch(error => this.message('error', error.message));
        }));
        root.append(profile, controls, search, grid, this.status, save);
        const sharedControls = [...grid.querySelectorAll<HTMLElement>('.live-pca-controls')];
        for (const panel of [junctionTile, pcaTile, gravityTile, finalTile, this.branchCanvas.parentElement]) {
            panel.style.display = 'none';
        }
        this.comparisonView = new DepthComparisonView(grid, sharedControls, value => {
            this.depthGapThreshold = value;
            this.analysisRevision++;
            if (this.last) this.reanalyzeDepths(this.last);
        }, () => this.queueComparisonRender());
        this.manager.querySelector('.bbox-preview-body').prepend(root);
        this.message('waiting', 'Move the camera or edit a box; preview updates after 400ms of stability.');
    }

    layoutCamera() {
        const container = document.getElementById('canvas-container');
        if (!this.enabled) {
            container.style.cssText = this.originalContainerStyle;
            this.hideMainLanding();
            return;
        }
        const aspect = SENSOR_PROFILE.native_width / SENSOR_PROFILE.native_height;
        const width = Math.min(this.wrapper.clientWidth, this.wrapper.clientHeight * aspect);
        const height = width / aspect;
        Object.assign(container.style, {
            position: 'absolute',
            flexGrow: '0',
            width: `${width}px`,
            height: `${height}px`,
            left: `${(this.wrapper.clientWidth - width) / 2}px`,
            top: `${(this.wrapper.clientHeight - height) / 2}px`
        });
        Object.assign(this.mainOverlay.style, {
            width: `${width}px`, height: `${height}px`,
            left: `${(this.wrapper.clientWidth - width) / 2}px`,
            top: `${(this.wrapper.clientHeight - height) / 2}px`
        });
    }

    applyProfile() {
        const camera = this.scene.camera;
        camera.sensorAspect = this.enabled ? SENSOR_PROFILE.native_width / SENSOR_PROFILE.native_height : this.savedCamera.aspect;
        camera.ortho = this.enabled ? false : this.savedCamera.ortho;
        this.events.fire('camera.setFov', this.enabled ? 118 : this.savedCamera.fov);
        camera.fov = this.enabled ? 118 : this.savedCamera.fov;
        this.layoutCamera();
        camera.rebuildRenderTargets();
        this.scene.forceRender = true;
    }

    message(state: string, text: string) {
        this.status.textContent = text;
        this.status.parentElement.dataset.state = state;
    }

    refresh() {
        this.revision++;
        this.attempted = '';
        this.changedAt = performance.now() - 400;
        this.saveButton.disabled = true;
        this.figuresButton.disabled = this.savingFigures || !this.canSaveFigures();
        this.hideMainLanding();
    }

    sourceKey(source: AnnotationSource) {
        const splat = source.splat;
        return JSON.stringify([source.boxes, source.metersPerSceneUnit, source.scaleReference,
            splat?.changedCounter, splat?.splatData.numSplats,
            splat ? Array.from(splat.worldTransform.data) : null]);
    }

    viewFingerprint(source: AnnotationSource) {
        const camera = this.scene.camera;
        return JSON.stringify([Array.from(camera.displayTransform.data, v => Number(v.toFixed(7))),
            this.sourceKey(source), this.revision,
            (this.scene.getElementsByType(ElementType.splat) as Splat[]).map(splat => [splat.name, splat.visible,
                splat.changedCounter, Array.from(splat.worldTransform.data), splat.transparency,
                splat.brightness, splat.saturation, splat.tintClr.r, splat.tintClr.g, splat.tintClr.b])]);
    }

    tick() {
        if (!this.enabled || document.hidden) {
            this.hideMainLanding();
            return;
        }
        const source = this.getSource();
        if (!source.splat) {
            this.message('waiting', 'Load a 3DGS scene to start the preview.');
            this.saveButton.disabled = true;
            this.hideMainLanding();
            return;
        }
        const key = this.viewFingerprint(source);
        if (key !== this.fingerprint) {
            this.fingerprint = key;
            this.changedAt = performance.now();
            this.saveButton.disabled = true;
            this.figuresButton.disabled = this.savingFigures || !this.canSaveFigures();
            this.status.parentElement.dataset.stale = 'true';
            this.message('moving', 'View changed · waiting for 400ms of stability…');
            this.hideMainLanding();
        }
        if (this.capturing) return;
        if (this.busy || this.scene.camera.targetSizeOverride || this.scene.lockedRenderMode ||
            this.getLegacyCaptureRunning() || (this.manager.classList.contains('collapsed') &&
                (this.viewMode === 'raw' || !this.showMainLanding)) ||
            performance.now() - this.changedAt < 400 || key === this.attempted) return;
        this.attempted = key;
        this.capture(source, key).catch((error) => {
            this.message('error', `Preview error: ${error.message}`);
        });
    }

    getLegacyCaptureRunning() {
        return this.events.functions.has('annotation.batchCaptureRunning') && this.events.invoke('annotation.batchCaptureRunning');
    }

    updateMembership(source: AnnotationSource) {
        const key = this.sourceKey(source);
        if (source.splat === this.memberSplat && key === this.memberKey) return;
        const splat = source.splat;
        if (splat !== this.memberSplat || this.memberTexture?.width !== splat.stateTexture.width ||
            this.memberTexture?.height !== splat.stateTexture.height) {
            this.memberTexture?.destroy();
            this.memberTexture = new Texture(this.scene.graphicsDevice, {
                name: 'annotationMembership',
                width: splat.stateTexture.width,
                height: splat.stateTexture.height,
                format: PIXELFORMAT_R8,
                mipmaps: false,
                minFilter: FILTER_NEAREST,
                magFilter: FILTER_NEAREST,
                addressU: ADDRESS_CLAMP_TO_EDGE,
                addressV: ADDRESS_CLAMP_TO_EDGE
            });
        }
        const data = this.memberTexture.lock() as Uint8Array;
        data.fill(0);
        const centers = splat.entity.gsplat.instance.sorter.centers as Float32Array;
        const state = splat.splatData.getProp('state') as Uint8Array;
        const world = splat.worldTransform;
        const point = new Vec3();
        for (let i = 0; i < splat.splatData.numSplats; i++) {
            if ((state[i] & (State.locked | State.deleted)) !== 0) continue;
            point.set(centers[i * 3], centers[i * 3 + 1], centers[i * 3 + 2]);
            world.transformPoint(point, point);
            if (source.contains(point)) data[i] = 255;
        }
        this.memberTexture.unlock();
        this.memberKey = key;
        this.memberSplat = splat;
    }

    async capture(source: AnnotationSource, key: string) {
        this.busy = true;
        this.capturing = true;
        const frameId = `live_${Date.now()}_${this.revision}`;
        const started = performance.now();
        let result: any;
        try {
            this.message('rendering', 'Rendering synchronized RGB, visible mask and camera pose…');
            this.updateMembership(source);
            result = await this.events.invoke('render.annotationSnapshot', 963, 544);
        } finally {
            this.capturing = false;
            if (!result) this.busy = false;
        }
        try {
            if (!this.enabled || key !== this.viewFingerprint(this.getSource())) return;
            // RGB is already composited against black in the offscreen target.
            // Make it opaque to avoid applying Gaussian alpha a second time.
            for (let i = 3; i < result.rgb.length; i += 4) result.rgb[i] = 255;
            const originalRgb = pixelCanvas(result.rgb, 963, 544);
            const mask = squareCanvas(pixelCanvas(visibleMask(result.membership), 963, 544), false);
            const pose: any = {
                frame_id: frameId,
                captured_at_utc: new Date().toISOString(),
                width: 640,
                height: 640,
                camera_profile: SENSOR_PROFILE,
                intrinsics: sensorIntrinsics(),
                coordinate_system: 'PlayCanvas scene world; camera +X right, +Y up, -Z forward',
                world_up_axis: this.scene.camera.upAxis,
                matrix_layout: 'row-major',
                units: 'scene',
                position: result.position,
                quaternion_xyzw: result.quaternion,
                position_meters: result.position.map((v: number) => v * source.metersPerSceneUnit),
                meters_per_scene_unit: source.metersPerSceneUnit,
                camera_to_world: rowMajorMatrix(result.world),
                world_to_camera: rowMajorMatrix(result.view),
                projection_matrix: rowMajorMatrix(result.projection),
                near: result.near,
                far: result.far,
                segmentation: { type: 'visible_gaussian_contribution',
                    opacity_threshold: 0.05,
                    contribution_ratio_threshold: 0.5,
                    annotations: source.boxes }
            };
            pose.gravity = resolveGravityWorld(this.imuGravity, pose);
            const gsDepth = squareDepth(result.depthScene, 963, 544, source.metersPerSceneUnit);
            pose.gs_depth = { model: '3DGS opacity-weighted Gaussian-center depth',
                units: 'approximate meters', meters_per_scene_unit: source.metersPerSceneUnit,
                scale_reference: source.scaleReference ?? null,
                calibration_source: source.scaleReference ? 'measured_bbox' : 'scene_scale',
                skeleton_length_method: 'sum_of_visible_3d_skeleton_edges_cm',
                background: 'NaN', width: 640, height: 640 };
            const snapshot: LiveSnapshot = { frameId, fingerprint: key, rgb: squareCanvas(originalRgb, true), mask,
                pose, depth: null, gsDepth };
            this.last = snapshot;
            this.figuresButton.disabled = true;
            this.hideMainLanding();
            this.rgbCanvas.getContext('2d').drawImage(snapshot.rgb, 0, 0);
            this.drawMask();
            this.depthPreview.getContext('2d').clearRect(0, 0, 640, 640);
            this.gsDepthPreview.getContext('2d').drawImage(depthCanvas(gsDepth, true), 0, 0);
            this.gsDepthTitle.textContent = source.scaleReference ?
                `3DGS depth · BBox ${source.scaleReference.box_id} ${source.scaleReference.axis} = ${(source.scaleReference.actual_length_m * 100).toFixed(1)} cm` :
                `3DGS depth · approximate scale (1 unit = ${source.metersPerSceneUnit.toFixed(4)} m)`;
            this.branchCanvas.getContext('2d').clearRect(0, 0, 640, 640);
            this.branchList.textContent = 'Skeletonizing visible segmentation…';
            this.drawJunctionComparison(snapshot);
            this.drawPcaPanel(snapshot);
            this.status.parentElement.dataset.frameId = frameId;
            this.status.parentElement.dataset.stale = 'false';
            this.message('skeletonizing', 'RGB / mask / pose ready · computing raw skeleton…');
            const raw = await this.detectRawSkeleton(snapshot);
            if (!this.enabled || key !== this.viewFingerprint(this.getSource())) return;
            snapshot.rawSkeleton = raw.skeleton;
            snapshot.rawSkeletonCount = raw.count;
            snapshot.rawJunctions = raw.junctions;
            snapshot.linkLabels = raw.linkLabels;
            snapshot.linkCount = raw.linkCount;
            this.drawBranches(snapshot);
            this.drawJunctionComparison(snapshot);
            this.drawPcaPanel(snapshot);
            this.comparisonCrop(snapshot); // Same baseline crop for both sources.
            snapshot.analyses = {};
            const gsTask = this.analyzeDepthSource(snapshot, '3dgs');
            const daTask = (async () => {
                const png = originalRgb.toDataURL('image/png').split(',')[1];
                const response = await fetch('/api/depth', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ frame_id: frameId, rgb_png_base64: png }),
                    signal: AbortSignal.timeout(30000)
                });
                if (!this.enabled || snapshot !== this.last || key !== this.viewFingerprint(this.getSource())) return;
                if (!response.ok) {
                    let reason = `HTTP ${response.status}`;
                    try {
                        reason = (await response.json()).detail ?? reason;
                    } catch { /* static-server response */ }
                    const hint = response.status === 404 || response.status === 405 ?
                        ' Start scripts/serve_with_depth.sh to enable depth.' : ' Use Refresh to retry.';
                    throw new Error(`${reason}.${hint} RGB, mask and pose remain available.`);
                }
                const info = await response.json();
                if (!this.enabled || snapshot !== this.last || key !== this.viewFingerprint(this.getSource())) return;
                if (info.frame_id !== frameId || info.width !== 640 || info.height !== 640 || info.dtype !== '<f4') {
                    throw new Error('Depth response does not match the captured frame');
                }
                const bytes = Uint8Array.from(atob(info.depth_f32_base64), c => c.charCodeAt(0));
                if (bytes.length !== 640 * 640 * 4) throw new Error('Unexpected depth data length');
                const view = new DataView(bytes.buffer);
                const depth = new Float32Array(640 * 640);
                for (let i = 0; i < depth.length; i++) depth[i] = view.getFloat32(i * 4, true);
                if (!depth.every(Number.isFinite)) throw new Error('Depth contains non-finite values');
                snapshot.depth = depth;
                snapshot.depthInfo = { model: info.model,
                    kind: info.depth_kind,
                    units: 'relative',
                    dtype: '<f4',
                    inference_ms: info.inference_ms,
                    input_size: info.input_size };
                snapshot.pose.depth = snapshot.depthInfo;
                this.depthPreview.getContext('2d').drawImage(depthCanvas(depth), 0, 0);
                await this.analyzeDepthSource(snapshot, 'da2');
            })().catch(error => {
                if (snapshot !== this.last || key !== this.viewFingerprint(this.getSource())) return;
                snapshot.analyses.da2 = { ...this.sourceSnapshot(snapshot, 'da2'),
                    analysisStatus: 'error', analysisError: error.message };
                this.queueComparisonRender();
            });
            this.message('inferring', 'Comparing depth-aware skeletons · DA2 and 3DGS…');
            await Promise.allSettled([gsTask, daTask]);
            if (!this.enabled || snapshot !== this.last || key !== this.viewFingerprint(this.getSource())) return;
            const ready = Object.values(snapshot.analyses).filter(result => result.analysisStatus === 'ready').length;
            this.saveButton.disabled = this.saving || snapshot.analyses.da2?.analysisStatus !== 'ready';
            this.queueComparisonRender();
            this.message(ready === 2 ? 'ready' : 'partial', `${frameId} · ${ready}/2 depth analyses ready · total ${Math.round(performance.now() - started)}ms`);
        } finally {
            this.busy = false;
        }
    }

    drawMask() {
        if (!this.last) return;
        const ctx = this.maskCanvas.getContext('2d');
        const title = this.maskCanvas.parentElement.querySelector('.live-tile-title');
        title.textContent = this.binaryInput.checked ? 'Segmentation · binary mask' :
            (this.showYellowSegmentation ? 'Segmentation · yellow overlay' : 'Segmentation hidden · RGB');
        ctx.clearRect(0, 0, 640, 640);
        if (this.binaryInput.checked) {
            ctx.drawImage(this.last.mask, 0, 0);
        } else {
            ctx.drawImage(this.last.rgb, 0, 0);
            if (!this.showYellowSegmentation) return;
            const pixels = this.last.mask.getContext('2d').getImageData(0, 0, 640, 640);
            for (let i = 0; i < pixels.data.length; i += 4) {
                const hit = pixels.data[i] > 0;
                pixels.data[i] = 255;
                pixels.data[i + 1] = 210;
                pixels.data[i + 2] = 0;
                pixels.data[i + 3] = hit ? 155 : 0;
            }
            ctx.drawImage(pixelCanvas(pixels.data, 640, 640), 0, 0);
        }
    }

    detectRawSkeleton(snapshot: LiveSnapshot): Promise<{ skeleton: Uint8Array; count: number; junctions: JunctionMarker[]; linkLabels: Uint32Array; linkCount: number }> {
        if (!this.detectorWorker) {
            this.detectorWorker = new Worker(new URL('static/lib/branch-detector/worker.mjs', document.baseURI), { type: 'module' });
        }
        const worker = this.detectorWorker;
        const rgba = snapshot.mask.getContext('2d').getImageData(0, 0, 640, 640).data;
        const mask = new Uint8Array(640 * 640);
        for (let i = 0; i < mask.length; i++) mask[i] = rgba[i * 4];
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                worker.terminate(); this.detectorWorker = null;
                reject(new Error('Raw skeleton worker timed out'));
            }, 10000);
            worker.onmessage = ({ data }) => {
                if (data.frameId !== snapshot.frameId) return;
                clearTimeout(timeout); worker.onerror = null;
                if (data.error) reject(new Error(data.error));
                else if (data.type !== 'raw-skeleton' || data.width !== 320 || data.height !== 320 ||
                    data.skeleton?.length !== 320 * 320 || !Array.isArray(data.junctions) ||
                    data.linkLabels?.length !== 320 * 320 || !Number.isInteger(data.linkCount)) {
                    reject(new Error('Unexpected raw skeleton response'));
                } else resolve({ skeleton: data.skeleton, count: data.count, junctions: data.junctions,
                    linkLabels: data.linkLabels, linkCount: data.linkCount });
            };
            worker.onerror = (event) => {
                clearTimeout(timeout);
                worker.terminate(); this.detectorWorker = null;
                reject(new Error(`Raw skeleton worker failed: ${event.message}`));
            };
            worker.postMessage({ type: 'raw-skeleton', frameId: snapshot.frameId, mask }, [mask.buffer]);
        });
    }

    detectBranches(snapshot: LiveSnapshot, source?: DepthSource): Promise<BranchDetection> {
        if (!this.detectorWorker) {
            this.detectorWorker = new Worker(new URL('static/lib/branch-detector/worker.mjs', document.baseURI), { type: 'module' });
        }
        if (source) this.analysisCancel[source]?.();
        const worker = source ? new Worker(new URL('static/lib/branch-detector/worker.mjs', document.baseURI), { type: 'module' }) : this.detectorWorker;
        const requestId = ++this.analysisRequestId;
        if (source) this.analysisWorkers[source] = worker;
        const rgba = new Uint8Array(snapshot.rgb.getContext('2d').getImageData(0, 0, 640, 640).data);
        const maskRgba = snapshot.mask.getContext('2d').getImageData(0, 0, 640, 640).data;
        const mask = new Uint8Array(640 * 640);
        for (let i = 0; i < mask.length; i++) mask[i] = maskRgba[i * 4];
        const depth = new Float32Array(snapshot.depth);
        const pose = snapshot.pose, k = pose.intrinsics;
        const rotation = new Float32Array([
            pose.camera_to_world[0][0], pose.camera_to_world[0][1], pose.camera_to_world[0][2],
            pose.camera_to_world[1][0], pose.camera_to_world[1][1], pose.camera_to_world[1][2],
            pose.camera_to_world[2][0], pose.camera_to_world[2][1], pose.camera_to_world[2][2]
        ]);
        return new Promise((resolve, reject) => {
            const cleanup = () => {
                clearTimeout(timeout); worker.onerror = null;
                if (source) {
                    worker.terminate();
                    if (this.analysisWorkers[source] === worker) {
                        delete this.analysisCancel[source]; delete this.analysisWorkers[source];
                    }
                }
            };
            const timeout = setTimeout(() => {
                cleanup();
                if (!source) { worker.terminate(); this.detectorWorker = null; }
                reject(new Error('Branch worker timed out'));
            }, 30000);
            if (source) this.analysisCancel[source] = () => { cleanup(); reject(new Error('Analysis superseded')); };
            worker.onmessage = ({ data }) => {
                if (data.frameId !== snapshot.frameId || (source && (data.requestId !== requestId ||
                    data.depthSource !== source || data.analysisRevision !== snapshot.analysisRevision))) return;
                cleanup();
                if (data.error) reject(new Error(data.error));
                else resolve(data as BranchDetection);
            };
            worker.onerror = (event) => { cleanup(); reject(new Error(`Branch worker failed: ${event.message}`)); };
            worker.postMessage({ frameId: snapshot.frameId, rgb: rgba, mask, depth,
                depthSource: source, requestId, analysisRevision: snapshot.analysisRevision,
                depthKind: snapshot.depthKind ?? 'relative_inverse', depthAwareSkeleton: !!source,
                gapThreshold: this.depthGapThreshold,
                rotation, intrinsics: [k[0][0], k[1][1], k[0][2], k[1][2]],
                worldUp: pose.world_up_axis === 'z' ? 2 : 1 },
            [rgba.buffer, mask.buffer, depth.buffer, rotation.buffer]);
        });
    }

    sourceSnapshot(root: LiveSnapshot, source: DepthSource): LiveSnapshot {
        return { frameId: root.frameId, fingerprint: root.fingerprint, rgb: root.rgb, mask: root.mask,
            pose: { ...root.pose }, depth: source === 'da2' ? root.depth : root.gsDepth, gsDepth: root.gsDepth,
            depthSource: source, depthKind: source === 'da2' ? 'relative_inverse' : 'camera_z_m',
            comparisonCrop: this.comparisonCrop(root), analysisStatus: 'pending', analysisRevision: this.analysisRevision };
    }

    async analyzeDepthSource(root: LiveSnapshot, source: DepthSource) {
        const analysis = this.sourceSnapshot(root, source);
        root.analyses ??= {};
        root.analyses[source] = analysis;
        this.queueComparisonRender();
        if (!analysis.depth) return;
        try {
            const result = await this.detectBranches(analysis, source);
            if (root !== this.last || root.analyses[source] !== analysis ||
                root.fingerprint !== this.viewFingerprint(this.getSource())) return;
            const rgba = new Uint8Array(640 * 640 * 4);
            for (let i = 0; i < result.processedMask.length; i++) {
                rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = result.processedMask[i];
                rgba[i * 4 + 3] = 255;
            }
            analysis.mask = pixelCanvas(rgba, 640, 640);
            analysis.detection = result;
            analysis.rawSkeleton = result.rawSkeleton;
            analysis.rawSkeletonCount = result.rawSkeletonCount;
            analysis.rawJunctions = result.rawJunctions;
            analysis.linkLabels = result.linkLabels;
            analysis.linkCount = result.linkCount;
            analysis.boundary = result.boundary;
            analysis.missing = result.missing;
            analysis.validFraction = result.validFraction;
            analysis.boundaryCells = result.boundaryCells;
            analysis.missingCells = result.missingCells;
            analysis.skeletonLengthCm = result.skeletonLengthCm;
            analysis.metricLengthCoverage = result.metricLengthCoverage;
            analysis.analysisStatus = 'ready';
        } catch (error) {
            if (root !== this.last || root.analyses[source] !== analysis) return;
            analysis.analysisStatus = 'error'; analysis.analysisError = error.message;
        }
        this.queueComparisonRender();
    }

    reanalyzeDepths(root: LiveSnapshot) {
        if (root !== this.last || root.fingerprint !== this.viewFingerprint(this.getSource())) return;
        this.saveButton.disabled = true;
        this.figuresButton.disabled = true;
        const tasks = (['da2', '3dgs'] as const).filter(source => source === 'da2' ? root.depth : root.gsDepth)
            .map(source => this.analyzeDepthSource(root, source));
        Promise.allSettled(tasks).then(() => {
            if (root !== this.last) return;
            this.saveButton.disabled = this.saving || root.analyses?.da2?.analysisStatus !== 'ready';
            this.queueComparisonRender();
        });
    }

    queueComparisonRender() {
        if (this.comparisonRenderQueued || !this.last?.analyses || !this.comparisonView) return;
        this.comparisonRenderQueued = true;
        queueMicrotask(() => {
            this.comparisonRenderQueued = false;
            if (this.last?.analyses) this.renderDepthComparison(this.last);
        });
    }

    computeSourceFilters(analysis: LiveSnapshot) {
        analysis.pose.gravity = this.last.pose.gravity;
        const gravityKey = JSON.stringify(analysis.pose.gravity);
        if (analysis.analysisGravityKey !== gravityKey) {
            analysis.pcaMergedAngle = undefined;
            analysis.analysisGravityKey = gravityKey;
        }
        this.mergePcaLabels(analysis);
        this.computeGravity(analysis);
        this.computeFinal(analysis);
    }

    renderDepthComparison(root: LiveSnapshot) {
        const views: Partial<Record<DepthSource, ComparisonSourceView>> = {};
        const composite = (overlay: HTMLCanvasElement, alpha = 0.65) => {
            const canvas = document.createElement('canvas'); canvas.width = canvas.height = 640;
            const ctx = canvas.getContext('2d'); ctx.globalAlpha = alpha; ctx.drawImage(root.rgb, 0, 0);
            ctx.globalAlpha = 1; ctx.drawImage(overlay, 0, 0); return canvas;
        };
        for (const source of ['da2', '3dgs'] as const) {
            const analysis = root.analyses[source];
            if (!analysis || analysis.analysisStatus !== 'ready') {
                views[source] = { status: analysis?.analysisStatus === 'error' ? `error: ${analysis.analysisError}` : 'computing…' };
                continue;
            }
            this.computeSourceFilters(analysis);
            const detection = analysis.detection, final = analysis.finalLanding;
            analysis.linkOverlay ??= this.linkOverlay(analysis, analysis.linkLabels);
            analysis.depthLinkOverlay ??= this.linkOverlay(analysis, detection.depthLinkLabels);
            const boundaryPixels = new Uint8ClampedArray(640 * 640 * 4);
            const original = root.mask.getContext('2d').getImageData(0, 0, 640, 640).data;
            for (let p = 0; p < 640 * 640; p++) {
                if (!original[p * 4]) continue;
                const small = (Math.floor(p / 640) >> 1) * 320 + ((p % 640) >> 1);
                if (analysis.missing[small]) boundaryPixels.set([160, 165, 175, 220], p * 4);
                else if (analysis.boundary[small]) boundaryPixels.set([255, 145, 25, 240], p * 4);
            }
            const skeleton = document.createElement('canvas'); skeleton.width = skeleton.height = 640;
            this.paintRawSkeleton(skeleton.getContext('2d'), analysis);
            const pca = composite(this.linkOverlay(analysis, detection.depthLinkLabels));
            const ctx = pca.getContext('2d'); ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2;
            for (const object of detection.pcaObjects) {
                if (!object.reliable || !object.endpointsPx?.every(Boolean)) continue;
                const [a, b] = object.endpointsPx;
                ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
            }
            const images: Record<ComparisonStage, HTMLCanvasElement> = {
                boundary: composite(pixelCanvas(boundaryPixels, 640, 640)), skeleton,
                links: composite(this.linkOverlay(analysis, detection.depthLinkLabels)), pca,
                merged: composite(analysis.pcaMergedOverlay), gravity: composite(analysis.gravityFilteredOverlay),
                final: composite(analysis.finalLandingOverlay, 0.45)
            };
            const reasons = new Map<string, number>();
            for (const candidate of final.results) for (const reason of candidate.reasons) reasons.set(reason, (reasons.get(reason) ?? 0) + 1);
            const reference = root.pose.gs_depth.scale_reference as BoxScaleReference | null;
            const metric = source === '3dgs';
            const lengthLabel = metric ?
                (analysis.skeletonLengthCm == null ? '3D length unavailable' : `~${analysis.skeletonLengthCm.toFixed(1)} cm visible skeleton`) :
                `${analysis.rawSkeletonCount} skeleton pixels`;
            const calibrationLabel = !metric ? '' : reference ?
                ` · BBox ${reference.box_id} ${reference.axis} ${(reference.actual_length_m * 100).toFixed(1)} cm` : ' · estimated scale';
            const selectedLengths = final.selected.map(candidate => metric ?
                `ID ${candidate.id}: ~${candidate.centerline_length_cm?.toFixed(1) ?? '?'} cm` :
                `ID ${candidate.id}: ${candidate.centerline_length_px.toFixed(0)} px`).join(' · ');
            views[source] = { status: `ready · ${(analysis.validFraction * 100).toFixed(1)}% valid depth${calibrationLabel} · CPU ${detection.cpu_ms.toFixed(1)}ms`, images,
                captions: {
                    boundary: `${analysis.boundaryCells} boundary cells · ${analysis.missingCells} missing cells`,
                    skeleton: `${lengthLabel} · ${analysis.rawJunctions.length} junctions · hover for branch length`,
                    links: `${detection.depthLinkCount} regions`,
                    pca: `${detection.pcaObjects.filter(object => object.reliable).length} reliable 3D axes`,
                    merged: `${analysis.pcaMergedCount} merged regions`,
                    gravity: `${new Set(analysis.gravityFilteredLabels).size - (analysis.gravityFilteredLabels.includes(0) ? 1 : 0)} retained regions · min length ${metric ? `${this.minSkeletonLengthCm} cm` : `${this.minGravityLength} px`}`,
                    final: `${final.selected.length} shown / ${final.eligible.length} eligible · ${selectedLengths} · rejected: ${[...reasons].map(([key, count]) => `${key} ${count}`).join(', ') || 'none'}`
                },
                inspect: (x, y) => {
                    if (x < 0 || x >= 640 || y < 0 || y >= 640) return '';
                    const id = detection.depthLinkLabels[(y >> 1) * 320 + (x >> 1)];
                    const candidate = final.results.find(item => item.id === id);
                    const object = detection.pcaObjects.find(item => item.id === id);
                    const length = metric ? `~${object?.centerlineLengthCm?.toFixed(1) ?? '?'} cm` : `${object?.centerlineLengthPx ?? 0} px`;
                    return candidate ? `ID ${id} · skeleton ${length} · ${candidate.reasons.length ? candidate.reasons.join(', ') : 'eligible'} · angle ${candidate.angle_from_horizontal_deg.toFixed(1)}° · line ${candidate.linearity.toFixed(2)} · width ${candidate.apparent_width_px.toFixed(1)}px` :
                        object ? `ID ${id} · skeleton ${length} · filtered before final stage` : 'No valid branch at this pixel';
                } };
        }
        const da = root.analyses.da2, gs = root.analyses['3dgs'];
        const diffPixels = new Uint8ClampedArray(640 * 640 * 4);
        let overlap = 0, daOnly = 0, gsOnly = 0;
        if (da?.analysisStatus === 'ready' && gs?.analysisStatus === 'ready') {
            for (let p = 0; p < 640 * 640; p++) {
                const a = !!da.finalLanding.ids[p], b = !!gs.finalLanding.ids[p];
                if (a && b) { overlap++; diffPixels.set([255, 255, 255, 225], p * 4); }
                else if (a) { daOnly++; diffPixels.set([0, 235, 245, 230], p * 4); }
                else if (b) { gsOnly++; diffPixels.set([255, 40, 205, 230], p * 4); }
            }
        }
        this.comparisonDifferenceOverlay = pixelCanvas(diffPixels, 640, 640);
        const union = overlap + daOnly + gsOnly;
        const summary = da?.analysisStatus === 'ready' && gs?.analysisStatus === 'ready' ?
            `Final mask comparison · white overlap ${overlap}px · cyan DA2 only ${daOnly}px · magenta 3DGS only ${gsOnly}px · ${union ? `IoU ${(100 * overlap / union).toFixed(1)}%` : 'both masks empty'} · IDs are local to each source.` :
            'Final mask difference appears when both analyses finish.';
        this.comparisonView.render(views, this.comparisonCrop(root), composite(this.comparisonDifferenceOverlay), summary);
        if (root.fingerprint === this.viewFingerprint(this.getSource())) {
            const results = Object.values(root.analyses), ready = results.filter(value => value.analysisStatus === 'ready').length;
            const failed = results.some(value => value.analysisStatus === 'error');
            this.message(ready === 2 ? 'ready' : failed ? 'partial' : 'inferring',
                `${root.frameId} · ${ready}/2 depth analyses ready${failed ? ' · inspect the source error above' : ''}`);
            this.saveButton.disabled = this.saving || da?.analysisStatus !== 'ready';
        }
        this.figuresButton.disabled = this.savingFigures || !this.canSaveFigures(root);
        // Legacy fields and exports continue to represent the DA2 analysis.
        if (da?.analysisStatus === 'ready') {
            const fields = ['detection', 'rawSkeleton', 'rawSkeletonCount', 'rawJunctions', 'linkLabels', 'linkCount',
                'rawOverlay', 'linkOverlay', 'depthLinkOverlay',
                'pcaMergedLabels', 'pcaMergedCount', 'pcaMergedObjects', 'pcaMergedAngle', 'pcaMergedOverlay',
                'gravityMergedLabels', 'gravityMergedObjects', 'gravityMergedOverlay', 'gravityMergedKey',
                'gravityEligibleCount', 'gravityComponentEligibleIds', 'gravityFilteredLabels', 'gravityFilteredOverlay',
                'gravityFilterKey', 'finalLanding', 'finalLandingKey', 'finalLandingOverlay', 'finalLandingCpuMs'] as const;
            for (const field of fields) (root as any)[field] = da[field];
            root.analysisMask = da.mask; root.depthKind = 'relative_inverse';
        }
        if (!this.enabled || !this.showMainLanding || root.fingerprint !== this.viewFingerprint(this.getSource())) {
            this.hideMainLanding(); return;
        }
        const source = this.comparisonView.overlaySource, selected = source === 'difference' ? null : root.analyses[source];
        let overlay = source === 'difference' ? this.comparisonDifferenceOverlay : null;
        let hasContent = source === 'difference' && union > 0;
        if (selected?.analysisStatus === 'ready') overlay = this.viewMode === 'raw' ? selected.rawOverlay :
            this.viewMode === 'final' ? selected.finalLandingOverlay : instanceCanvas(this.branchView(selected).landingMask, true);
        if (selected?.analysisStatus === 'ready') hasContent = this.viewMode === 'raw' ? selected.rawSkeletonCount > 0 :
            this.viewMode === 'final' ? selected.finalLanding.selected.length > 0 : this.branchView(selected).landing.length > 0;
        if (!overlay) { this.hideMainLanding(); return; }
        const ctx = this.mainOverlay.getContext('2d'); ctx.clearRect(0, 0, this.mainOverlay.width, this.mainOverlay.height);
        ctx.drawImage(overlay, 0, 0, this.mainOverlay.width, this.mainOverlay.height);
        this.mainOverlay.dataset.frameId = root.frameId;
        this.mainOverlay.dataset.depthSource = source;
        this.mainOverlay.style.visibility = hasContent ? 'visible' : 'hidden';
    }

    branchView(snapshot: LiveSnapshot): BranchView {
        const detection = snapshot.detection;
        const excludedTrunkIds = this.excludeTrunk ? detection.branches.filter(branch => branch.is_trunk).map(branch => branch.id) : [];
        const branches = detection.branches.filter(branch => !this.excludeTrunk || !branch.is_trunk).map(branch => ({
            ...branch,
            is_horizontal: !branch.is_trunk && branch.angle_to_ground_deg <= this.horizontalAngle &&
                branch.orientation_confidence >= this.minDirectionConfidence &&
                (snapshot.depthKind === 'camera_z_m' ? Number.isFinite(branch.centerline_length_cm) &&
                    branch.centerline_length_cm >= this.minSkeletonLengthCm : branch.centerline_length_px >= this.minBranchLength) &&
                branch.centerline_length_px / Math.max(branch.width_px, 1e-6) >= this.minLengthWidthRatio
        }));
        const horizontal = branches.filter(branch => branch.is_horizontal)
            .sort((a, b) => b.score - a.score || a.id - b.id);
        const landing = horizontal.slice(0, this.landingTopK);
        const instanceMask = new Uint8Array(detection.instanceMask);
        if (excludedTrunkIds.length) {
            const excluded = new Set(excludedTrunkIds);
            for (let i = 0; i < instanceMask.length; i++) if (excluded.has(instanceMask[i])) instanceMask[i] = 0;
        }
        const landingIds = new Set(landing.map(branch => branch.id));
        const landingMask = new Uint8Array(instanceMask.length);
        for (let i = 0; i < instanceMask.length; i++) {
            if (landingIds.has(instanceMask[i])) landingMask[i] = instanceMask[i];
        }
        return { branches, horizontal, landing, excludedTrunkIds, instanceMask, landingMask };
    }

    hideMainLanding() {
        this.mainOverlay.style.visibility = 'hidden';
        this.mainOverlay.dataset.frameId = '';
    }

    drawMainLanding(snapshot: LiveSnapshot, view: BranchView) {
        if (!this.enabled || this.viewMode !== 'candidates' || !this.showMainLanding || snapshot !== this.last ||
            snapshot.fingerprint !== this.viewFingerprint(this.getSource())) {
            this.hideMainLanding();
            return;
        }
        const ctx = this.mainOverlay.getContext('2d');
        const width = this.mainOverlay.width, height = this.mainOverlay.height;
        ctx.clearRect(0, 0, width, height);
        const ranks = new Uint8Array(254);
        for (const [index, branch] of view.landing.entries()) ranks[branch.id] = index + 1;
        const colors = [[0, 245, 255], [255, 209, 75], [255, 113, 194], [134, 255, 137]];
        const pixels = new Uint8ClampedArray(640 * 640 * 4);
        for (let y = 0; y < 640; y++) for (let x = 0; x < 640; x++) {
            const p = y * 640 + x, id = view.landingMask[p], rank = ranks[id];
            if (!rank) continue;
            const color = colors[(rank - 1) % colors.length];
            const edge = x === 0 || x === 639 || y === 0 || y === 639 ||
                view.landingMask[p - 1] !== id || view.landingMask[p + 1] !== id ||
                view.landingMask[p - 640] !== id || view.landingMask[p + 640] !== id;
            const offset = p * 4;
            pixels[offset] = color[0]; pixels[offset + 1] = color[1]; pixels[offset + 2] = color[2];
            pixels[offset + 3] = edge ? 255 : 105;
        }
        ctx.drawImage(pixelCanvas(pixels, 640, 640), 0, 0, width, height);
        ctx.font = 'bold 16px sans-serif';
        ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        for (const [index, branch] of view.landing.entries()) {
            const x = branch.center_px[0] * width / 640;
            const y = branch.center_px[1] * height / 640;
            const color = colors[index % colors.length];
            const label = `Rank ${index + 1} · ID ${branch.id}`;
            const textWidth = ctx.measureText(label).width;
            const lx = x > width * 0.375 ? Math.max(6, x - textWidth - 30) :
                Math.min(width - textWidth - 14, x + 18);
            const ly = Math.min(height - 28, Math.max(24, y - 23));
            ctx.strokeStyle = `rgb(${color.join(',')})`; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(lx, ly + 9); ctx.stroke();
            ctx.fillStyle = '#071923'; ctx.fillRect(lx, ly, textWidth + 12, 22);
            ctx.strokeRect(lx, ly, textWidth + 12, 22);
            ctx.fillStyle = '#ffffff'; ctx.fillText(label, lx + 6, ly + 11);
        }
        this.mainOverlay.dataset.frameId = snapshot.frameId;
        this.mainOverlay.style.visibility = view.landing.length ? 'visible' : 'hidden';
    }

    drawMainFinal(snapshot: LiveSnapshot) {
        if (!this.enabled || this.viewMode !== 'final' || !this.showMainLanding ||
            snapshot !== this.last || snapshot.fingerprint !== this.viewFingerprint(this.getSource()) ||
            !snapshot.finalLanding) {
            this.hideMainLanding();
            return;
        }
        const ctx = this.mainOverlay.getContext('2d');
        ctx.clearRect(0, 0, this.mainOverlay.width, this.mainOverlay.height);
        ctx.drawImage(snapshot.finalLandingOverlay, 0, 0, this.mainOverlay.width, this.mainOverlay.height);
        this.drawFinalMarkers(ctx, snapshot);
        this.mainOverlay.dataset.frameId = snapshot.frameId;
        this.mainOverlay.style.visibility = snapshot.finalLanding.selected.length ? 'visible' : 'hidden';
    }

    drawFinalMarkers(ctx: CanvasRenderingContext2D, snapshot: LiveSnapshot, crop?: { x: number; y: number; size: number }) {
        if (!snapshot.finalLanding) return;
        const source = crop ?? { x: 0, y: 0, size: 640 };
        for (const [rank, candidate] of snapshot.finalLanding.selected.entries()) {
            const x = (candidate.center_px[0] - source.x) * ctx.canvas.width / source.size;
            const y = (candidate.center_px[1] - source.y) * ctx.canvas.height / source.size;
            if (x < 0 || y < 0 || x >= ctx.canvas.width || y >= ctx.canvas.height) continue;
            const color = branchPalette[((candidate.id - 1) % 254) + 1];
            ctx.beginPath(); ctx.arc(x, y, 16, 0, Math.PI * 2);
            ctx.strokeStyle = '#071923'; ctx.lineWidth = 6; ctx.stroke();
            ctx.strokeStyle = `rgb(${color.join(',')})`; ctx.lineWidth = 3; ctx.stroke();
            ctx.font = 'bold 16px sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
            const label = `#${rank + 1}`;
            ctx.fillStyle = '#071923'; ctx.fillRect(x + 17, y - 12, 31, 24);
            ctx.fillStyle = '#ffffff'; ctx.fillText(label, x + 21, y);
        }
    }

    comparisonCrop(snapshot: LiveSnapshot) {
        if (snapshot.comparisonCrop) return snapshot.comparisonCrop;
        let minX = 320, minY = 320, maxX = -1, maxY = -1;
        snapshot.rawSkeleton?.forEach((value, index) => {
            if (!value) return;
            const x = index % 320, y = Math.floor(index / 320);
            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
        });
        if (maxX < 0) return { x: 0, y: 0, size: 640 };
        const span = Math.max(maxX - minX + 1, maxY - minY + 1) * 2;
        const size = Math.min(640, Math.max(96, span * 1.5));
        const x = Math.max(0, Math.min(640 - size, minX + maxX + 1 - size / 2));
        const y = Math.max(0, Math.min(640 - size, minY + maxY + 1 - size / 2));
        snapshot.comparisonCrop = { x, y, size };
        return snapshot.comparisonCrop;
    }

    drawComparisonLayer(ctx: CanvasRenderingContext2D, image: HTMLCanvasElement, snapshot: LiveSnapshot) {
        const crop = this.comparisonCrop(snapshot);
        ctx.drawImage(image, crop.x, crop.y, crop.size, crop.size, 0, 0, 640, 640);
    }

    paintRawSkeleton(ctx: CanvasRenderingContext2D, snapshot: LiveSnapshot, detail = false) {
        ctx.clearRect(0, 0, 640, 640);
        if (detail) this.drawComparisonLayer(ctx, snapshot.rgb, snapshot);
        else ctx.drawImage(snapshot.rgb, 0, 0);
        if (!snapshot.rawSkeleton) return;
        if (!snapshot.rawOverlay) {
            const mask = (snapshot.analysisMask ?? snapshot.mask).getContext('2d').getImageData(0, 0, 640, 640).data;
            const pixels = new Uint8ClampedArray(640 * 640 * 4);
            for (let y = 0; y < 640; y++) for (let x = 0; x < 640; x++) {
                const i = y * 640 + x;
                if (!mask[i * 4] || !snapshot.rawSkeleton[(y >> 1) * 320 + (x >> 1)]) continue;
                pixels[i * 4] = 255;
                pixels[i * 4 + 1] = 30;
                pixels[i * 4 + 2] = 215;
                pixels[i * 4 + 3] = 255;
            }
            snapshot.rawOverlay = pixelCanvas(pixels, 640, 640);
        }
        if (detail) this.drawComparisonLayer(ctx, snapshot.rawOverlay, snapshot);
        else ctx.drawImage(snapshot.rawOverlay, 0, 0);
    }

    drawRawSkeleton(snapshot: LiveSnapshot) {
        this.paintRawSkeleton(this.branchCanvas.getContext('2d'), snapshot);
        this.branchCanvas.parentElement.querySelector('.live-tile-title').textContent = 'Raw skeleton · magenta on RGB';
        if (!snapshot.rawSkeleton) {
            this.branchList.textContent = 'Skeletonizing visible segmentation…';
            return;
        }
        this.branchList.textContent = `${snapshot.rawSkeletonCount} skeleton pixels at 320×320 · visible segmentation only · no depth, size, angle, trunk or ranking filters`;
    }

    paintLinkSegmentation(ctx: CanvasRenderingContext2D, snapshot: LiveSnapshot, depthAware: boolean) {
        ctx.clearRect(0, 0, 640, 640);
        this.drawComparisonLayer(ctx, snapshot.rgb, snapshot);
        const labels = depthAware ? snapshot.detection?.depthLinkLabels : snapshot.linkLabels;
        if (!labels) return;
        const cacheKey = depthAware ? 'depthLinkOverlay' : 'linkOverlay';
        if (!snapshot[cacheKey]) {
            snapshot[cacheKey] = this.linkOverlay(snapshot, labels);
        }
        this.drawComparisonLayer(ctx, snapshot[cacheKey], snapshot);
    }

    linkOverlay(snapshot: LiveSnapshot, labels: Uint32Array) {
        const mask = (snapshot.analysisMask ?? snapshot.mask).getContext('2d').getImageData(0, 0, 640, 640).data;
        const pixels = new Uint8ClampedArray(640 * 640 * 4);
        for (let y = 0; y < 640; y++) for (let x = 0; x < 640; x++) {
            const p = y * 640 + x, id = labels[(y >> 1) * 320 + (x >> 1)];
            if (!id || !mask[p * 4]) continue;
            const color = branchPalette[((id - 1) % 254) + 1];
            pixels[p * 4] = color[0]; pixels[p * 4 + 1] = color[1]; pixels[p * 4 + 2] = color[2];
            pixels[p * 4 + 3] = 225;
        }
        return pixelCanvas(pixels, 640, 640);
    }

    buildPcaGroups(snapshot: LiveSnapshot, allowedIds?: Set<number>) {
        const detection = snapshot.detection;
        const parent = new Map<number, number>();
        for (const id of detection.depthLinkLabels) if (id && (!allowedIds || allowedIds.has(id))) parent.set(id, id);
        const members = new Map<number, PcaObject[]>(detection.pcaObjects
            .filter(object => !allowedIds || allowedIds.has(object.id)).map(object => [object.id, [object]]));
        const root = (value: number) => {
            while (parent.get(value) !== value) value = parent.get(value);
            return value;
        };
        let merges = 0;
        for (const edge of detection.pcaAdjacency) {
            if (!edge.eligible || edge.angleDeg > this.pcaMaxAngle || !parent.has(edge.a) || !parent.has(edge.b)) continue;
            const a = root(edge.a), b = root(edge.b);
            if (a === b) continue;
            const coherent = members.get(a).every(left => members.get(b).every(right =>
                Math.acos(Math.min(1, Math.abs(left.axis.reduce((sum, value, k) =>
                    sum + value * right.axis[k], 0)))) * 180 / Math.PI <= this.pcaMaxAngle));
            if (!coherent) continue;
            const keep = Math.min(a, b), drop = Math.max(a, b);
            parent.set(drop, keep);
            members.get(keep).push(...members.get(drop)); members.delete(drop);
            merges++;
        }
        const merged = new Uint32Array(detection.depthLinkLabels.length);
        for (let i = 0; i < merged.length; i++) {
            const id = detection.depthLinkLabels[i]; if (parent.has(id)) merged[i] = root(id);
        }
        const pooled = new Map<number, { pixels: number; sum: number[]; outer: number[];
            centerlineLengthPx: number; centerlineLengthCm: number | null;
            trunkCenterlineLengthPx: number; detectorCenterlineLengthPx: number }>();
        for (const id of parent.keys()) {
            const group = root(id);
            if (!pooled.has(group)) pooled.set(group, { pixels: 0, sum: [0, 0, 0], outer: Array(9).fill(0),
                centerlineLengthPx: 0, centerlineLengthCm: 0, trunkCenterlineLengthPx: 0, detectorCenterlineLengthPx: 0 });
        }
        for (const object of detection.pcaObjects) {
            if (!parent.has(object.id)) continue;
            const group = pooled.get(root(object.id));
            group.pixels += object.pixels;
            group.centerlineLengthPx += object.centerlineLengthPx;
            group.centerlineLengthCm = group.centerlineLengthCm == null || object.centerlineLengthCm == null ?
                null : group.centerlineLengthCm + object.centerlineLengthCm;
            group.trunkCenterlineLengthPx += object.trunkCenterlineLengthPx;
            group.detectorCenterlineLengthPx += object.detectorCenterlineLengthPx;
            for (let i = 0; i < 3; i++) group.sum[i] += object.moments.sum[i];
            for (let i = 0; i < 9; i++) group.outer[i] += object.moments.outer[i];
        }
        const gravity = snapshot.pose.gravity.vector as number[];
        const objects: GravityObject[] = [...pooled].map(([id, moments]) => {
            const fit = principalAxisFromMoments(moments.pixels, moments.sum, moments.outer);
            return { id, pixels: moments.pixels, axis: fit.axis, linearity: fit.linearity,
                reliable: fit.reliable, angleToGravityDeg: fit.reliable ?
                    Math.acos(Math.min(1, Math.abs(fit.axis.reduce((sum, value, i) =>
                        sum + value * gravity[i], 0)))) * 180 / Math.PI : null,
                centerlineLengthPx: moments.centerlineLengthPx,
                centerlineLengthCm: moments.centerlineLengthCm,
                trunkCenterlineLengthPx: moments.trunkCenterlineLengthPx,
                detectorCenterlineLengthPx: moments.detectorCenterlineLengthPx,
                isTrunk: moments.trunkCenterlineLengthPx >= 6 &&
                    moments.trunkCenterlineLengthPx / Math.max(1, moments.centerlineLengthPx) >= 0.35,
                memberIds: members.get(id).map(object => object.id) };
        }).sort((a, b) => a.id - b.id);
        return { labels: merged, count: parent.size - merges, objects, merges };
    }

    mergePcaLabels(snapshot: LiveSnapshot) {
        if (snapshot.pcaMergedAngle === this.pcaMaxAngle && snapshot.pcaMergedLabels && snapshot.pcaMergedObjects) return 0;
        const merged = this.buildPcaGroups(snapshot);
        snapshot.pcaMergedLabels = merged.labels;
        snapshot.pcaMergedCount = merged.count;
        snapshot.pcaMergedAngle = this.pcaMaxAngle;
        snapshot.pcaMergedObjects = merged.objects;
        snapshot.pcaMergedOverlay = this.linkOverlay(snapshot, merged.labels);
        snapshot.gravityFilteredLabels = undefined;
        snapshot.gravityFilteredOverlay = undefined;
        snapshot.gravityFilterKey = undefined;
        return merged.merges;
    }

    drawPcaPanel(snapshot: LiveSnapshot) {
        this.queueComparisonRender();
        const input = this.pcaInputCanvas.getContext('2d');
        const output = this.pcaMergedCanvas.getContext('2d');
        if (!snapshot.detection?.pcaObjects) {
            for (const ctx of [input, output]) {
                ctx.clearRect(0, 0, 640, 640);
                ctx.globalAlpha = 0.35; this.drawComparisonLayer(ctx, snapshot.rgb, snapshot); ctx.globalAlpha = 1;
            }
            this.pcaInputLabel.textContent = 'DA2 segments · waiting for relative depth';
            this.pcaMergedLabel.textContent = 'PCA merged · waiting for relative depth';
            this.pcaSummary.textContent = 'Each DA2 segment will get a pseudo-3D PCA axis.';
            this.drawGravityPanel(snapshot);
            return;
        }
        const detection = snapshot.detection;
        this.paintLinkSegmentation(input, snapshot, true);
        const crop = this.comparisonCrop(snapshot), zoom = 640 / crop.size;
        let reliable = 0;
        for (const object of detection.pcaObjects) {
            if (!object.reliable || !object.endpointsPx?.[0] || !object.endpointsPx?.[1]) continue;
            const [[x0, y0], [x1, y1]] = object.endpointsPx;
            const dx = (x1 - x0) * zoom, dy = (y1 - y0) * zoom;
            const length = Math.hypot(dx, dy);
            if (length < 1e-6) continue;
            reliable++;
            const half = Math.min(55, Math.max(12, length / 2));
            const cx = (object.centerPx[0] - crop.x) * zoom;
            const cy = (object.centerPx[1] - crop.y) * zoom;
            input.beginPath(); input.moveTo(cx - dx / length * half, cy - dy / length * half);
            input.lineTo(cx + dx / length * half, cy + dy / length * half);
            input.strokeStyle = '#06131b'; input.lineWidth = 6; input.stroke();
            input.strokeStyle = '#f8fcff'; input.lineWidth = 2.5; input.stroke();
            input.beginPath(); input.arc(cx, cy, 3, 0, Math.PI * 2);
            input.fillStyle = '#06212b'; input.fill();
        }
        const merges = this.mergePcaLabels(snapshot);
        output.clearRect(0, 0, 640, 640);
        this.drawComparisonLayer(output, snapshot.rgb, snapshot);
        this.drawComparisonLayer(output, snapshot.pcaMergedOverlay, snapshot);
        this.pcaInputLabel.textContent = `DA2 input · ${detection.depthLinkCount} objects · ${reliable} reliable PCA axes (white)`;
        this.pcaMergedLabel.textContent = `PCA merged · ${snapshot.pcaMergedCount} objects · ${detection.depthLinkCount - snapshot.pcaMergedCount} merges`;
        this.pcaSummary.textContent = `Adjacent objects merge when every pair of reliable 3D axes differs by ≤${this.pcaMaxAngle}° and the local DA2 gap is ≤15% of the frame range. Axis direction is sign-free; relative depth yields dimensionless pseudo-3D, not metric geometry.`;
        this.drawGravityPanel(snapshot);
    }

    computeGravity(snapshot: LiveSnapshot) {
        const gravity = snapshot.pose.gravity.vector as number[];
        const mergeKey = `${this.pcaMaxAngle}:${this.gravityTolerance}:${this.requireGravityComponents}:${gravity.join(',')}`;
        if (snapshot.gravityMergedKey !== mergeKey || !snapshot.gravityMergedLabels) {
            const eligible = new Set(snapshot.detection.pcaObjects.filter(object => object.reliable &&
                90 - Math.acos(Math.min(1, Math.abs(object.axis.reduce((sum, value, i) =>
                    sum + value * gravity[i], 0)))) * 180 / Math.PI <= this.gravityTolerance)
                .map(object => object.id));
            if (this.requireGravityComponents) {
                const merged = this.buildPcaGroups(snapshot, eligible);
                snapshot.gravityMergedLabels = merged.labels;
                snapshot.gravityMergedObjects = merged.objects;
                snapshot.gravityMergedOverlay = this.linkOverlay(snapshot, merged.labels);
            } else {
                snapshot.gravityMergedLabels = snapshot.pcaMergedLabels;
                snapshot.gravityMergedObjects = snapshot.pcaMergedObjects;
                snapshot.gravityMergedOverlay = snapshot.pcaMergedOverlay;
            }
            snapshot.gravityEligibleCount = eligible.size;
            snapshot.gravityComponentEligibleIds = eligible;
            snapshot.gravityMergedKey = mergeKey;
            snapshot.gravityFilterKey = undefined;
        }
        const objects = snapshot.gravityMergedObjects;
        const eligibleComponents = snapshot.gravityComponentEligibleIds ?? new Set<number>();
        const lengthPasses = (object: GravityObject) => snapshot.depthKind === 'camera_z_m' ?
            Number.isFinite(object.centerlineLengthCm) && object.centerlineLengthCm >= this.minSkeletonLengthCm :
            object.centerlineLengthPx >= this.minGravityLength;
        const accepted = new Set(objects.filter(object => object.reliable &&
            (90 - object.angleToGravityDeg <= this.gravityTolerance ||
                (!this.requireGravityComponents && object.memberIds.some(id =>
                    eligibleComponents.has(id)))) &&
            lengthPasses(object) &&
            (!this.excludeGravityTrunks || !object.isTrunk)).map(object => object.id));
        const filterKey = `${mergeKey}:${snapshot.depthKind === 'camera_z_m' ? this.minSkeletonLengthCm : this.minGravityLength}:${this.excludeGravityTrunks}`;
        if (snapshot.gravityFilterKey !== filterKey || !snapshot.gravityFilteredLabels) {
            const labels = new Uint32Array(snapshot.gravityMergedLabels.length);
            for (let i = 0; i < labels.length; i++) {
                const id = snapshot.gravityMergedLabels[i];
                if (accepted.has(id)) labels[i] = id;
            }
            snapshot.gravityFilteredLabels = labels;
            snapshot.gravityFilteredOverlay = this.linkOverlay(snapshot, labels);
            snapshot.gravityFilterKey = filterKey;
        }
        return { objects, accepted };
    }

    drawGravityPanel(snapshot: LiveSnapshot) {
        this.queueComparisonRender();
        const input = this.gravityInputCanvas.getContext('2d');
        const output = this.gravityFilteredCanvas.getContext('2d');
        if (!snapshot.pcaMergedLabels || !snapshot.pcaMergedObjects) {
            for (const ctx of [input, output]) {
                ctx.clearRect(0, 0, 640, 640);
                ctx.globalAlpha = 0.35; this.drawComparisonLayer(ctx, snapshot.rgb, snapshot); ctx.globalAlpha = 1;
            }
            this.gravityInputLabel.textContent = 'PCA merged · waiting for relative depth';
            this.gravityFilteredLabel.textContent = 'Near-horizontal · waiting for relative depth';
            this.gravitySummary.textContent = 'Merged PCA axes will be checked against gravity.';
            this.drawFinalPanel(snapshot);
            return;
        }
        const { objects, accepted } = this.computeGravity(snapshot);
        const eligibleComponents = snapshot.gravityComponentEligibleIds ?? new Set<number>();
        input.clearRect(0, 0, 640, 640);
        this.drawComparisonLayer(input, snapshot.rgb, snapshot);
        this.drawComparisonLayer(input, snapshot.gravityMergedOverlay, snapshot);
        output.clearRect(0, 0, 640, 640);
        output.globalAlpha = 0.45; this.drawComparisonLayer(output, snapshot.rgb, snapshot); output.globalAlpha = 1;
        this.drawComparisonLayer(output, snapshot.gravityFilteredOverlay, snapshot);
        this.gravityInputLabel.textContent = `PCA merged · ${objects.length} objects · ${snapshot.gravityEligibleCount} / ${snapshot.detection.pcaObjects.length} components individually horizontal`;
        this.gravityFilteredLabel.textContent = `Near-horizontal · ${accepted.size} / ${objects.length} objects kept`;
        const trunkCount = objects.filter(object => object.isTrunk).length;
        const gravityRejected = { angle: 0, short: 0, trunk: 0, unreliable: 0 };
        for (const object of objects) {
            if (!object.reliable) gravityRejected.unreliable++;
            if (object.reliable && 90 - object.angleToGravityDeg > this.gravityTolerance &&
                (this.requireGravityComponents || !object.memberIds.some(id => eligibleComponents.has(id)))) {
                gravityRejected.angle++;
            }
            if (object.centerlineLengthPx < this.minGravityLength) gravityRejected.short++;
            if (this.excludeGravityTrunks && object.isTrunk) gravityRejected.trunk++;
        }
        this.gravitySummary.textContent = `${this.requireGravityComponents ?
            'Individual components and merged axes must both pass' :
            'Merged axis or a component must pass'} the ${this.gravityTolerance}° horizontal deviation. ` +
            `Raw-skeleton centerlines ≥${this.minGravityLength}px. Rejected merged objects: angle ${gravityRejected.angle}, ` +
            `short ${gravityRejected.short}, trunk ${gravityRejected.trunk}, unreliable ${gravityRejected.unreliable}. ` +
            `Trunk exclusion ${this.excludeGravityTrunks ? 'on' : 'off'} (${trunkCount} trunk-like objects); gravity: ${snapshot.pose.gravity.source === 'imu' ? 'IMU' : 'scene up'}. ` +
            `DA2 geometry is relative pseudo-3D.`;
        this.drawFinalPanel(snapshot);
    }

    computeFinal(snapshot: LiveSnapshot) {
        const key = `${snapshot.gravityFilterKey}:${this.finalMaxWidthPx}:${this.finalMinLinearity}:` +
            `${this.finalMinProjectedLinearity}:${this.finalMinCylinderInlier}:` +
            `${this.finalMaxDepthStdRelative}:${this.finalMinRelativeHeight}:${this.finalMinWoodyFraction}:` +
            `${this.finalRejectFoliage}:${this.finalMaxCenterBearing}:${this.finalMaxEndOn}:` +
            `${this.finalMaxImageTilt}:${this.finalStrictCameraAngles}:${this.finalTopK}`;
        if (snapshot.finalLandingKey !== key || !snapshot.finalLanding) {
            const started = performance.now();
            const maskRgba = (snapshot.analysisMask ?? snapshot.mask).getContext('2d').getImageData(0, 0, 640, 640).data;
            const mask = new Uint8Array(640 * 640);
            for (let i = 0; i < mask.length; i++) mask[i] = maskRgba[i * 4];
            const rgb = new Uint8Array(snapshot.rgb.getContext('2d').getImageData(0, 0, 640, 640).data);
            snapshot.finalLanding = segmentFinalLanding({ labels: snapshot.gravityFilteredLabels,
                sourceLabels: snapshot.detection.depthLinkLabels,
                objects: snapshot.gravityMergedObjects, sourceObjects: snapshot.detection.pcaObjects,
                mask, depth: snapshot.depth, rgb, pose: snapshot.pose, depthKind: snapshot.depthKind,
                settings: { minLengthPx: this.minGravityLength, minLinearity: this.finalMinLinearity,
                    minLengthCm: snapshot.depthKind === 'camera_z_m' ? this.minSkeletonLengthCm : undefined,
                    minProjectedLinearity: this.finalMinProjectedLinearity,
                    minCylinderInlierFraction: this.finalMinCylinderInlier,
                    maxWidthPx: this.finalMaxWidthPx, maxDepthStdRelative: this.finalMaxDepthStdRelative,
                    minRelativeHeight: this.finalMinRelativeHeight, minWoodyFraction: this.finalMinWoodyFraction,
                    rejectFoliage: this.finalRejectFoliage,
                    maxAngleFromHorizontalDeg: this.gravityTolerance,
                    maxCenterBearingDeg: this.finalMaxCenterBearing,
                    maxEndOnDeviationDeg: this.finalMaxEndOn,
                    maxImageTiltDeg: this.finalMaxImageTilt,
                    strictCameraAngles: this.finalStrictCameraAngles,
                    excludeTrunks: this.excludeGravityTrunks, topK: this.finalTopK } });
            snapshot.finalLandingOverlay = finalLandingCanvas(snapshot.finalLanding.ids);
            snapshot.finalLandingCpuMs = performance.now() - started;
            snapshot.finalLandingKey = key;
        }
    }

    drawFinalPanel(snapshot: LiveSnapshot) {
        this.queueComparisonRender();
        const input = this.finalInputCanvas.getContext('2d');
        const output = this.finalOutputCanvas.getContext('2d');
        if (!snapshot.gravityFilteredLabels || !snapshot.depth || !snapshot.gravityMergedObjects) {
            for (const ctx of [input, output]) {
                ctx.clearRect(0, 0, 640, 640);
                ctx.globalAlpha = 0.35; ctx.drawImage(snapshot.rgb, 0, 0); ctx.globalAlpha = 1;
            }
            this.finalInputLabel.textContent = 'Gravity-filtered regions · waiting for relative depth';
            this.finalOutputLabel.textContent = 'Final landing mask · waiting for relative depth';
            this.finalSummary.textContent = 'Width, 3D linearity and depth spread will refine the gravity-filtered objects.';
            this.finalDiagnostics.textContent = 'Candidate rejection reasons will appear after relative depth arrives.';
            this.finalDiagnostics.dataset.default = this.finalDiagnostics.textContent;
            return;
        }
        this.computeFinal(snapshot);
        input.clearRect(0, 0, 640, 640);
        this.drawComparisonLayer(input, snapshot.rgb, snapshot);
        this.drawComparisonLayer(input, snapshot.gravityFilteredOverlay, snapshot);
        output.clearRect(0, 0, 640, 640);
        output.globalAlpha = 0.45; this.drawComparisonLayer(output, snapshot.rgb, snapshot); output.globalAlpha = 1;
        this.drawComparisonLayer(output, snapshot.finalLandingOverlay, snapshot);
        this.drawFinalMarkers(output, snapshot, this.comparisonCrop(snapshot));
        const result = snapshot.finalLanding;
        this.finalInputLabel.textContent = `Gravity filtered · ${snapshot.gravityMergedObjects.length} merged objects · zoomed`;
        this.finalOutputLabel.textContent = `Final mask · ${result.selected.length} / ${result.eligible.length} eligible shown · zoomed`;
        const ranks = result.selected.map((item, index) =>
            `#${index + 1} ID ${item.id} · center ${item.center_bearing_deg.toFixed(0)}° · ` +
            `axis toward ${item.end_on_deviation_deg.toFixed(0)}° · ` +
            `screen ${item.image_tilt_deg.toFixed(0)}° · view fit ${Math.round(item.view_score * 100)}%`).join(' | ');
        const rejected = { height: 0, width: 0, linearity: 0, depth: 0, foliage: 0, cylinder: 0,
            offCenter: 0, endOn: 0, screenTilt: 0 };
        for (const candidate of result.results) {
            if (candidate.reasons.includes('low_height')) rejected.height++;
            if (candidate.reasons.includes('wide')) rejected.width++;
            if (candidate.reasons.includes('low_linearity')) rejected.linearity++;
            if (candidate.reasons.includes('depth_spread')) rejected.depth++;
            if (candidate.reasons.includes('foliage')) rejected.foliage++;
            if (candidate.reasons.includes('not_cylindrical')) rejected.cylinder++;
            if (candidate.reasons.includes('off_center')) rejected.offCenter++;
            if (candidate.reasons.includes('end_on')) rejected.endOn++;
            if (candidate.reasons.includes('screen_tilt')) rejected.screenTilt++;
        }
        this.finalSummary.textContent = `One DA2 junction-split link per ID; green foliage ${this.finalRejectFoliage ? 'removed' : 'kept'}. ` +
            `Wood-only PCA linearity ≥${this.finalMinLinearity}, image linearity ≥${this.finalMinProjectedLinearity}, ` +
            `cylinder inliers ≥${this.finalMinCylinderInlier}, projected cylinder width ≤${this.finalMaxWidthPx}px, ` +
            `relative depth spread ≤${this.finalMaxDepthStdRelative}, relative height ≥${this.finalMinRelativeHeight}. ` +
            `Camera ${this.finalStrictCameraAngles ? 'required' : 'ranking only'}: center ≤${this.finalMaxCenterBearing}°, ` +
            `axis toward camera ≤${this.finalMaxEndOn}°, screen tilt ≤${this.finalMaxImageTilt}°. ` +
            `Rejected: foliage ${rejected.foliage}, non-cylinder ${rejected.cylinder}, low height ${rejected.height}, ` +
            `wide ${rejected.width}, non-linear ${rejected.linearity}, depth ${rejected.depth}, ` +
            `off-center ${rejected.offCenter}, end-on ${rejected.endOn}, screen tilt ${rejected.screenTilt}. ` +
            `${ranks || 'No candidate passes all filters.'} ` +
            `Final CPU ${snapshot.finalLandingCpuMs.toFixed(1)}ms. ` +
            `DA2 depth and width are not metric; verify physical clearance and load capacity separately.`;
        const closeRejected = result.results.filter(candidate => candidate.reasons.length && candidate.woody_area_px >= 6)
            .sort((a, b) => a.reasons.length - b.reasons.length || b.score - a.score).slice(0, 6);
        this.finalDiagnostics.textContent = closeRejected.length ?
            `Hover over the left image to inspect an ID. Near misses: ${closeRejected.map(candidate =>
                `ID ${candidate.id} (${candidate.reasons.join(', ')})`).join(' · ')}` :
            'Hover over the left image to inspect an ID and its rejection reasons.';
        this.finalDiagnostics.dataset.default = this.finalDiagnostics.textContent;
        if (this.viewMode === 'final') this.drawBranches(snapshot);
    }

    drawJunctionComparison(snapshot: LiveSnapshot) {
        this.queueComparisonRender();
        const plain = this.junctionNoDepthCanvas.getContext('2d');
        const guided = this.junctionDepthCanvas.getContext('2d');
        const compared = snapshot.detection?.junctionComparison;
        if (!this.showJunctions) {
            this.paintLinkSegmentation(plain, snapshot, false);
            this.junctionNoDepthLabel.textContent = snapshot.linkLabels ?
                `2D only · ${snapshot.linkCount} colored links` : '2D only · waiting for raw skeleton';
            if (compared) {
                this.paintLinkSegmentation(guided, snapshot, true);
                const depthCount = snapshot.detection.depthLinkCount;
                this.junctionDepthLabel.textContent = `With DA2 · ${depthCount} separate objects (2D: ${snapshot.linkCount}, extra cuts: ${snapshot.detection.depthLinkCuts + snapshot.detection.depthJunctionCuts})`;
                const separated = compared.filter(item => item.status === 'depth-separated').length;
                const similar = compared.filter(item => item.status === 'similar-depth').length;
                const uncertain = compared.length - separated - similar;
                this.junctionSummary.textContent = `Zoomed to the raw skeleton. Every junction arm keeps a separate object ID, including similar-depth arms. DA2 can add cuts where depth jumps within a link; no junction arms are rejoined. Junctions: ${separated} depth-disagree / ${similar} similar / ${uncertain} uncertain.`;
            } else {
                guided.clearRect(0, 0, 640, 640);
                guided.globalAlpha = 0.35; this.drawComparisonLayer(guided, snapshot.rgb, snapshot); guided.globalAlpha = 1;
                this.junctionDepthLabel.textContent = 'With DA2 · waiting for relative depth';
                this.junctionSummary.textContent = 'Colors identify 2D links between junctions. Depth-guided grouping appears when DA2 finishes.';
            }
            return;
        }
        this.paintRawSkeleton(plain, snapshot, true);
        const markers = (ctx: CanvasRenderingContext2D, items: JunctionMarker[], color: (item: JunctionMarker) => string) => {
            ctx.lineWidth = 3;
            const crop = this.comparisonCrop(snapshot), zoom = 640 / crop.size;
            for (const item of items) {
                ctx.beginPath(); ctx.arc((item.x - crop.x) * zoom, (item.y - crop.y) * zoom,
                    Math.max(8, 8 * zoom), 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(5, 12, 18, 0.8)'; ctx.fill();
                ctx.strokeStyle = color(item); ctx.stroke();
            }
        };
        if (snapshot.rawJunctions) {
            markers(plain, snapshot.rawJunctions, () => '#ffe45c');
            this.junctionNoDepthLabel.textContent = `2D only · ${snapshot.rawJunctions.length} topological junctions`;
        } else {
            this.junctionNoDepthLabel.textContent = '2D only · waiting for raw skeleton';
        }
        if (compared) {
            this.paintRawSkeleton(guided, snapshot, true);
            markers(guided, compared, item => {
                const status = (item as DepthJunctionMarker).status;
                return status === 'depth-separated' ? '#ff775c' :
                    status === 'similar-depth' ? '#50efbf' : '#becbd6';
            });
            const separated = compared.filter(item => item.status === 'depth-separated').length;
            const similar = compared.filter(item => item.status === 'similar-depth').length;
            const uncertain = compared.length - separated - similar;
            this.junctionDepthLabel.textContent = `With DA2 · ${separated} depth-disagree / ${similar} similar / ${uncertain} uncertain`;
            this.junctionSummary.textContent = 'Yellow: every 2D junction · teal: similar arm depths · orange: depth disagreement · gray: insufficient depth. Gap ≥15% of this frame’s robust relative-depth range is a separation cue, not proof of physical connectivity.';
        } else {
            guided.clearRect(0, 0, 640, 640);
            guided.globalAlpha = 0.35; this.drawComparisonLayer(guided, snapshot.rgb, snapshot); guided.globalAlpha = 1;
            this.junctionDepthLabel.textContent = 'With DA2 · waiting for relative depth';
            this.junctionSummary.textContent = 'Yellow: every 2D junction. Depth comparison appears when DA2 finishes.';
        }
    }

    drawBranches(snapshot: LiveSnapshot) {
        this.queueComparisonRender();
        if (this.viewMode === 'raw') {
            this.drawRawSkeleton(snapshot);
            this.hideMainLanding();
            return;
        }
        if (this.viewMode === 'final') {
            this.branchCanvas.parentElement.querySelector('.live-tile-title').textContent = 'Final landing segmentation · top K';
            const ctx = this.branchCanvas.getContext('2d');
            ctx.clearRect(0, 0, 640, 640);
            ctx.globalAlpha = 0.35; ctx.drawImage(snapshot.rgb, 0, 0); ctx.globalAlpha = 1;
            if (snapshot.finalLanding) {
                ctx.drawImage(snapshot.finalLandingOverlay, 0, 0);
                this.drawFinalMarkers(ctx, snapshot);
                this.branchList.replaceChildren();
                const heading = document.createElement('div');
                heading.textContent = `${snapshot.finalLanding.selected.length} shown / ${snapshot.finalLanding.eligible.length} eligible · gravity: ${snapshot.pose.gravity.source === 'imu' ? 'IMU' : 'scene fallback'}`;
                this.branchList.append(heading);
                for (const [index, candidate] of snapshot.finalLanding.selected.entries()) {
                    const line = document.createElement('div');
                    line.textContent = `Rank ${index + 1} · ID ${candidate.id} · center ${candidate.center_bearing_deg.toFixed(0)}° · ` +
                        `axis toward ${candidate.end_on_deviation_deg.toFixed(0)}° · ` +
                        `screen tilt ${candidate.image_tilt_deg.toFixed(0)}° · view fit ${Math.round(candidate.view_score * 100)}%`;
                    this.branchList.append(line);
                }
                this.drawMainFinal(snapshot);
            } else {
                this.branchList.textContent = 'Waiting for gravity, depth and final segmentation…';
                this.hideMainLanding();
            }
            return;
        }
        this.branchCanvas.parentElement.querySelector('.live-tile-title').textContent = 'Landing candidates · top K';
        if (!snapshot.detection) {
            this.branchCanvas.getContext('2d').drawImage(snapshot.rgb, 0, 0);
            this.branchList.textContent = 'Waiting for synchronized depth and branch candidates…';
            this.hideMainLanding();
            return;
        }
        const detection = snapshot.detection;
        const view = this.branchView(snapshot);
        const ctx = this.branchCanvas.getContext('2d');
        ctx.fillStyle = '#050a10'; ctx.fillRect(0, 0, 640, 640);
        ctx.globalAlpha = 0.3; ctx.drawImage(snapshot.rgb, 0, 0); ctx.globalAlpha = 1;
        ctx.drawImage(instanceCanvas(view.landingMask, true), 0, 0);
        if (view.landing.length) {
            const ids = new Set(view.landing.map(branch => branch.id));
            const edge = new Uint8ClampedArray(640 * 640 * 4);
            for (let y = 1; y < 639; y++) for (let x = 1; x < 639; x++) {
                const p = y * 640 + x, id = view.landingMask[p];
                if (!ids.has(id)) continue;
                if (view.landingMask[p - 1] === id && view.landingMask[p + 1] === id &&
                    view.landingMask[p - 640] === id && view.landingMask[p + 640] === id) continue;
                edge[p * 4] = 0; edge[p * 4 + 1] = 245; edge[p * 4 + 2] = 255; edge[p * 4 + 3] = 255;
            }
            ctx.drawImage(pixelCanvas(edge, 640, 640), 0, 0);
        }
        for (const [index, branch] of view.landing.entries()) {
            const [x, y] = branch.center_px;
            ctx.beginPath(); ctx.arc(x, y, 13, 0, 2 * Math.PI);
            ctx.fillStyle = '#071923'; ctx.fill();
            ctx.strokeStyle = '#00f5ff'; ctx.lineWidth = 2; ctx.stroke();
            ctx.fillStyle = '#ffffff'; ctx.font = 'bold 15px sans-serif';
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(String(index + 1), x, y);
        }
        this.branchList.replaceChildren();
        const heading = document.createElement('div');
        heading.textContent = `Landing top ${this.landingTopK} · ${view.landing.length} shown / ${view.horizontal.length} eligible`;
        heading.style.color = '#55dce9';
        this.branchList.append(heading);
        if (!view.landing.length) {
            const line = document.createElement('div');
            line.textContent = 'No ground-parallel branch candidates in this frame.';
            this.branchList.append(line);
        }
        for (const [index, branch] of view.landing.entries()) {
            const line = document.createElement('div');
            line.textContent = `Rank ${index + 1} · ID ${branch.id} · ~${branch.angle_to_ground_deg.toFixed(0)}° · score ${branch.score.toFixed(2)}`;
            this.branchList.append(line);
        }
        const note = document.createElement('div');
        note.textContent = `Rank = heuristic score order · ID = branch instance · ${view.excludedTrunkIds.length} trunk ID(s) excluded · CPU ${detection.cpu_ms.toFixed(1)}ms`;
        this.branchList.append(note);
        this.drawMainLanding(snapshot, view);
    }

    canSaveFigures(snapshot = this.last) {
        return !!snapshot?.analyses &&
            (['da2', '3dgs'] as const).every(source => ['ready', 'error'].includes(snapshot.analyses[source]?.analysisStatus)) &&
            Object.values(snapshot.analyses).some(analysis => analysis.analysisStatus === 'ready');
    }

    collectFigures(): Figure[] {
        const title = (canvas: HTMLCanvasElement) => canvas.parentElement.querySelector('.live-tile-title')?.textContent ?? '';
        const figures: Figure[] = [
            { name: 'rgb', title: 'RGB', caption: title(this.rgbCanvas), canvas: this.rgbCanvas },
            { name: 'segmentation', title: 'Segmentation', caption: title(this.maskCanvas), canvas: this.maskCanvas },
            { name: 'da2_depth', title: 'DA2 depth', caption: title(this.depthPreview), canvas: this.depthPreview },
            { name: '3dgs_depth', title: '3DGS depth', caption: title(this.gsDepthPreview), canvas: this.gsDepthPreview }
        ];
        for (const [stage, label] of comparisonStages) {
            for (const source of ['da2', '3dgs'] as const) {
                const tile = this.comparisonView.tiles.get(`${source}:${stage}`);
                figures.push({ name: `${source}_${stage}`, title: `${source === 'da2' ? 'DA2' : '3DGS'} · ${label.split(' · ')[0]}`,
                    caption: tile.caption.textContent, canvas: tile.canvas });
            }
        }
        figures.push({ name: 'final_difference', title: 'Final mask difference',
            caption: this.comparisonView.summary.textContent, canvas: this.comparisonView.difference });
        return figures;
    }

    async saveFigures() {
        if (this.savingFigures) return null;
        const snapshot = this.last;
        if (!this.canSaveFigures(snapshot)) throw new Error('Wait for the current figures to finish updating');
        // Freeze every displayed canvas and caption before the directory dialog
        // yields control. Later camera/filter changes cannot mix saved frames.
        const figures = freezeFigures(this.collectFigures());
        const prefix = `${snapshot.frameId}_figures_${Date.now()}`;
        const reference = snapshot.pose.gs_depth.scale_reference as BoxScaleReference | null;
        const crop = this.comparisonView.crop;
        const description = `${snapshot.frameId} · captured ${snapshot.pose.captured_at_utc}. ` +
            `Displayed crop: x ${crop.x.toFixed(1)}, y ${crop.y.toFixed(1)}, size ${crop.size.toFixed(1)}. ` +
            `Depth gap ${this.depthGapThreshold}; PCA angle ${this.pcaMaxAngle}°; gravity tolerance ${this.gravityTolerance}°; ` +
            `minimum skeleton: DA2 ${this.minGravityLength} px / 3DGS ${this.minSkeletonLengthCm} cm; ` +
            `final Top K ${this.finalTopK}. ` +
            (reference ? `3DGS calibration: BBox ${reference.box_id} ${reference.axis} = ${reference.actual_length_m * 100} cm.` :
                `3DGS scale: ${snapshot.pose.meters_per_scene_unit} m per scene unit (approximate).`);
        this.savingFigures = true;
        this.figuresButton.disabled = true;
        this.figuresButton.textContent = 'Saving…';
        try {
            const folder = this.getOutputFolder() ?? await this.chooseOutputFolder();
            const files = await encodeFigures(figures, 'BBox Manager · DA2 vs 3DGS', description);
            let destination: string;
            if (folder) {
                const figuresDir = await folder.getDirectoryHandle('Figures', { create: true });
                const output = await figuresDir.getDirectoryHandle(prefix, { create: true });
                for (const file of files) await this.writeFile(file.name, file.blob, output);
                destination = `${folder.name}/Figures/${prefix}/`;
            } else {
                destination = `${prefix}.zip`;
                await this.writeFile(destination, await figureArchive(files), null);
            }
            const message = `Saved ${files.length} PNG figures, including overview.png · ${destination}`;
            this.figuresButton.title = message;
            this.status.textContent = message;
            return { count: files.length, destination, frameId: snapshot.frameId };
        } catch (error) {
            if (error.name === 'AbortError') {
                this.status.textContent = 'Figure save cancelled';
                return null;
            }
            throw error;
        } finally {
            this.savingFigures = false;
            this.figuresButton.textContent = 'Save Figures';
            this.figuresButton.disabled = !this.canSaveFigures();
        }
    }

    async save() {
        const snapshot = this.last;
        if (this.saving || !this.enabled || !snapshot?.depth || !snapshot.gsDepth || !snapshot.detection ||
            snapshot.fingerprint !== this.viewFingerprint(this.getSource())) {
            throw new Error('Wait for the current frame and branch detection to finish');
        }
        this.saving = true;
        this.saveButton.disabled = true;
        try {
            const folder = this.getOutputFolder() ?? await this.chooseOutputFolder();
            const prefix = `${snapshot.frameId}_save_${Date.now()}`;
            const groupNames = ['RGB', 'Mask', 'Depth', 'Pose', 'Branches', 'Landing'];
            const groups: any[] = folder ? await Promise.all(groupNames.map(name => folder.getDirectoryHandle(name, { create: true }))) : [null, null, null, null, null, null];
            const names = { rgb: `RGB/${prefix}.png`,
                mask: `Mask/${prefix}.png`,
                depth: `Depth/${prefix}.npy`,
                depth_preview: `Depth/${prefix}_preview.png`,
                gs_depth: `Depth/${prefix}_3dgs_m.npy`,
                gs_depth_preview: `Depth/${prefix}_3dgs_preview.png`,
                pose: `Pose/${prefix}.json`,
                branch_instances: `Branches/${prefix}_instances.png`,
                branch_detection: `Branches/${prefix}.json`,
                landing_ids: `Landing/${prefix}_ids.npy`,
                landing_preview: `Landing/${prefix}_preview.png`,
                landing_analysis: `Landing/${prefix}.json` };
            if (!folder) {
                Object.keys(names).forEach((key) => {
                    names[key as keyof typeof names] = names[key as keyof typeof names].replace('/', '_');
                });
            }
            const pose = { ...snapshot.pose, files: names };
            const view = this.branchView(snapshot);
            this.drawFinalPanel(snapshot);
            const final = snapshot.finalLanding;
            const detection = { frame_id: snapshot.frameId, captured_at_utc: snapshot.pose.captured_at_utc,
                geometry_units: snapshot.detection.geometry_units, ranking: snapshot.detection.ranking,
                cpu_ms: snapshot.detection.cpu_ms, width: 640, height: 640,
                instance_id_semantics: 'each RGB channel equals branch id; zero is background',
                filters: { exclude_trunk: this.excludeTrunk, horizontal_angle_max_deg: this.horizontalAngle,
                    min_branch_length_px: this.minBranchLength,
                    min_length_width_ratio: this.minLengthWidthRatio,
                    min_direction_confidence: this.minDirectionConfidence,
                    landing_top_k: this.landingTopK },
                excluded_trunk_ids: view.excludedTrunkIds,
                horizontal_branch_ids: view.horizontal.map(branch => branch.id),
                landing_candidate_ids: view.landing.map(branch => branch.id),
                branches: view.branches, junctions_px: snapshot.detection.junctions,
                files: names };
            const landing = { frame_id: snapshot.frameId, captured_at_utc: snapshot.pose.captured_at_utc,
                width: 640, height: 640, id_map: 'uint32 NPY; zero is background; nonzero values are junction-split DA2 link IDs',
                depth_units: snapshot.depthInfo.units, gravity: snapshot.pose.gravity,
                final_cpu_ms: snapshot.finalLandingCpuMs,
                filters: { max_gravity_deviation_deg: this.gravityTolerance,
                    require_each_component_horizontal: this.requireGravityComponents,
                    merge_max_axis_angle_deg: this.pcaMaxAngle, min_centerline_length_px: this.minGravityLength,
                    exclude_trunks: this.excludeGravityTrunks, min_linearity: this.finalMinLinearity,
                    min_projected_linearity: this.finalMinProjectedLinearity,
                    min_cylinder_inlier_fraction: this.finalMinCylinderInlier,
                    max_apparent_width_px: this.finalMaxWidthPx,
                    max_relative_depth_std: this.finalMaxDepthStdRelative,
                    min_relative_height: this.finalMinRelativeHeight,
                    reject_green_foliage: this.finalRejectFoliage,
                    min_non_green_fraction: this.finalMinWoodyFraction,
                    require_camera_angles: this.finalStrictCameraAngles,
                    max_camera_center_bearing_deg: this.finalMaxCenterBearing,
                    max_camera_end_on_deviation_deg: this.finalMaxEndOn,
                    min_camera_side_on_deg: 90 - this.finalMaxEndOn,
                    max_camera_image_tilt_deg: this.finalMaxImageTilt,
                    top_k: this.finalTopK },
                selected_ids: final.selected.map(candidate => candidate.id),
                candidates: final.results, files: names };
            const landingPreview = document.createElement('canvas');
            landingPreview.width = landingPreview.height = 640;
            const landingCtx = landingPreview.getContext('2d');
            landingCtx.drawImage(snapshot.rgb, 0, 0);
            landingCtx.drawImage(snapshot.finalLandingOverlay, 0, 0);
            const files = [
                { name: `${prefix}.png`, group: 0, blob: await pngBlob(snapshot.rgb) },
                { name: `${prefix}.png`, group: 1, blob: await pngBlob(snapshot.mask) },
                { name: `${prefix}.npy`, group: 2, blob: new Blob([numpyFloat32(snapshot.depth, 640, 640).buffer], { type: 'application/octet-stream' }) },
                { name: `${prefix}_preview.png`, group: 2, blob: await pngBlob(depthCanvas(snapshot.depth)) },
                { name: `${prefix}_3dgs_m.npy`, group: 2, blob: new Blob([numpyFloat32(snapshot.gsDepth, 640, 640).buffer], { type: 'application/octet-stream' }) },
                { name: `${prefix}_3dgs_preview.png`, group: 2, blob: await pngBlob(depthCanvas(snapshot.gsDepth, true)) },
                { name: `${prefix}.json`, group: 3, blob: new Blob([JSON.stringify(pose, null, 2)], { type: 'application/json' }) },
                { name: `${prefix}_instances.png`, group: 4, blob: await pngBlob(instanceCanvas(view.instanceMask, false)) },
                { name: `${prefix}.json`, group: 4, blob: new Blob([JSON.stringify(detection, null, 2)], { type: 'application/json' }) },
                { name: `${prefix}_ids.npy`, group: 5, blob: new Blob([numpyUint32(final.ids, 640, 640).buffer], { type: 'application/octet-stream' }) },
                { name: `${prefix}_preview.png`, group: 5, blob: await pngBlob(landingPreview) },
                { name: `${prefix}.json`, group: 5, blob: new Blob([JSON.stringify(landing, null, 2)], { type: 'application/json' }) }
            ];
            for (const file of files) {
                const name = folder ? file.name : `${groupNames[file.group]}_${file.name}`;
                await this.writeFile(name, file.blob, groups[file.group]);
            }
            this.message('ready', `Saved synchronized 640×640 RGB, mask, DA2 and 3DGS depth, pose, branches and final landing segmentation · ${prefix}`);
        } finally {
            this.saving = false;
            this.saveButton.disabled = !this.enabled || !this.last?.detection || this.last.fingerprint !== this.viewFingerprint(this.getSource());
        }
    }
}

export { LiveView };
