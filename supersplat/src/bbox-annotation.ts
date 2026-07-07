import { Color, Mat4, Ray, Vec3 } from 'playcanvas';

import { ElementType } from './element';
import { Events } from './events';
import { Scene } from './scene';
import { Splat } from './splat';
import { State } from './splat-state';

type Box = {
    id: string;
    label: string;
    center: [number, number, number];
    size: [number, number, number];
    euler: [number, number, number];
};

type Frame = {
    file_path: string;
    width: number;
    height: number;
    fx: number;
    fy: number;
    cx: number;
    cy: number;
    k1: number;
    k2: number;
    k3: number;
    p1: number;
    p2: number;
    transform_matrix: number[][];
};

type ProjectionMode = 'train-to-data' | 'splat-local' | 'splat-world' | 'local-applied' | 'local-applied-inverse';

type ProjectedBox = {
    bbox: [number, number, number, number];
    visibleCorners: number;
};

type ViewMask = {
    box: Box;
    maskFile: string;
    area: number;
    bbox: number[] | null;
    projectedPoints: number;
    pixels: Uint8ClampedArray;
};

type HandleDrag = {
    index: number;
    pointerId: number;
    oppositeWorld: Vec3;
    planePoint: Vec3;
    planeNormal: Vec3;
    rotation: number[];
};

const cornerSigns = [
    [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
];

const edges = [
    [0, 1], [1, 3], [3, 2], [2, 0],
    [4, 5], [5, 7], [7, 6], [6, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
];

const lineColor = new Color(1, 0.08, 0.02, 1);
const activeLineColor = new Color(0.1, 0.7, 1, 1);
const defaultCubeMeters = 0.1;
const defaultMaskRadiusPx = 2;
const segmentDebounceMs = 100;
const metricStorageKey = '3dgs-tree-annotator:meters-per-scene-unit';

const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));
const deg = (v: number) => v * Math.PI / 180;

const mat3Mul = (a: number[], b: number[]) => {
    const o = new Array(9).fill(0);
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            o[r * 3 + c] = a[r * 3 + 0] * b[0 * 3 + c] +
                a[r * 3 + 1] * b[1 * 3 + c] +
                a[r * 3 + 2] * b[2 * 3 + c];
        }
    }
    return o;
};

const mat3Vec = (m: number[], v: number[]) => {
    return [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2]
    ];
};

const mat3TransposeVec = (m: number[], v: number[]) => {
    return [
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2]
    ];
};

const rotationMatrix = (rx: number, ry: number, rz: number) => {
    const sx = Math.sin(deg(rx));
    const cx = Math.cos(deg(rx));
    const sy = Math.sin(deg(ry));
    const cy = Math.cos(deg(ry));
    const sz = Math.sin(deg(rz));
    const cz = Math.cos(deg(rz));
    const Rx = [1, 0, 0, 0, cx, -sx, 0, sx, cx];
    const Ry = [cy, 0, sy, 0, 1, 0, -sy, 0, cy];
    const Rz = [cz, -sz, 0, sz, cz, 0, 0, 0, 1];
    return mat3Mul(mat3Mul(Rz, Ry), Rx);
};

const invertRigid = (c2w: number[]) => {
    const r00 = c2w[0], r01 = c2w[1], r02 = c2w[2], tx = c2w[3];
    const r10 = c2w[4], r11 = c2w[5], r12 = c2w[6], ty = c2w[7];
    const r20 = c2w[8], r21 = c2w[9], r22 = c2w[10], tz = c2w[11];
    return [
        r00, r10, r20, -(r00 * tx + r10 * ty + r20 * tz),
        r01, r11, r21, -(r01 * tx + r11 * ty + r21 * tz),
        r02, r12, r22, -(r02 * tx + r12 * ty + r22 * tz),
        0, 0, 0, 1
    ];
};

const transformRowMajor = (m: number[], p: number[]) => {
    return [
        m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
        m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
        m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11]
    ];
};

const downloadText = (filename: string, text: string) => {
    const blob = new Blob([text], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
};

const downloadBlob = (filename: string, blob: Blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
};

const rgbaToPngBlob = (pixels: Uint8Array | Uint8ClampedArray, width: number, height: number) => {
    return new Promise<Blob>((resolve, reject) => {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            reject(new Error('2D canvas unavailable'));
            return;
        }
        const data = new Uint8ClampedArray(pixels.length);
        data.set(pixels);
        ctx.putImageData(new ImageData(data, width, height), 0, 0);
        canvas.toBlob((blob) => {
            if (blob) {
                resolve(blob);
            } else {
                reject(new Error('PNG encode failed'));
            }
        }, 'image/png');
    });
};

const formatCount = (value: number) => value.toLocaleString(undefined, { maximumFractionDigits: 0 });

const cloneBox = (box: Box): Box => ({
    id: box.id,
    label: box.label,
    center: [box.center[0], box.center[1], box.center[2]],
    size: [box.size[0], box.size[1], box.size[2]],
    euler: [box.euler[0], box.euler[1], box.euler[2]]
});

const safeFilePart = (value: string) => {
    return (value || 'bbox').trim().replace(/[^\w.-]+/g, '_').replace(/^_+|_+$/g, '') || 'bbox';
};

class BBoxAnnotation {
    events: Events;
    scene: Scene;
    root: HTMLDivElement;
    previewRoot: HTMLDivElement;
    previewImage: HTMLImageElement;
    previewCanvas: HTMLCanvasElement;
    previewStatus: HTMLDivElement;
    frameLabel: HTMLSpanElement;
    frameSlider: HTMLInputElement;
    projectionModeSelect: HTMLSelectElement;
    useSplatTransformInput: HTMLInputElement;
    showSegmentProjectionInput: HTMLInputElement;
    boxSelect: HTMLSelectElement;
    drawButton: HTMLButtonElement;
    cubeButton: HTMLButtonElement;
    autoSegmentInput: HTMLInputElement;
    maskRadiusInput: HTMLInputElement;
    spiralCenterInputs = new Map<string, HTMLInputElement>();
    spiralRadiiInput: HTMLInputElement;
    spiralThetaInput: HTMLInputElement;
    outputStatus: HTMLDivElement;
    segmentStatus: HTMLDivElement;
    canvasContainer: HTMLElement | null;
    handleSvg: SVGSVGElement | null = null;
    handleNodes: SVGCircleElement[] = [];
    hoverNode: SVGCircleElement | null = null;
    handleDrag: HandleDrag | null = null;
    handleRay = new Ray();
    hoverPoint: Vec3 | null = null;
    hoverPending: { nx: number; ny: number } | null = null;
    hoverPicking = false;
    segmentActive = false;
    segmentRunning = false;
    segmentDirty = false;
    segmentTimer: ReturnType<typeof setTimeout> | null = null;
    projectionTimer: ReturnType<typeof setTimeout> | null = null;
    segmentPoint = new Vec3();
    projectionPoint = new Vec3();
    projectionWorldToSplat = new Mat4();
    segmentLastCount = 0;
    inputs = new Map<string, HTMLInputElement>();
    sliders = new Map<string, HTMLInputElement>();
    frames: Frame[] = [];
    frameIndex = 0;
    transformsUrl: URL | null = null;
    appliedTransform: number[] | null = null;
    appliedTransformInverse: number[] | null = null;
    dataparserTransform: number[] | null = null;
    dataparserTransformInverse: number[] | null = null;
    dataparserScale = 1;
    projectionMode: ProjectionMode = 'train-to-data';
    visible = true;
    touched = false;
    drawMode = false;
    cubeMode = false;
    drawing = false;
    drawPointerId: number | null = null;
    drawStart: Vec3 | null = null;
    drawRight = new Vec3(1, 0, 0);
    drawUp = new Vec3(0, 1, 0);
    drawForward = new Vec3(0, 0, -1);
    drawEuler: [number, number, number] = [0, 0, 0];
    drawRay = new Ray();
    drawDepth = 0.1;
    metersPerSceneUnit = 1;
    metricInput: HTMLInputElement;
    cubeMetersInput: HTMLInputElement;
    calibrationMeasuredInput: HTMLInputElement;
    calibrationActualInput: HTMLInputElement;
    metricSummary: HTMLDivElement;
    lastMeasuredSceneLength = 0;
    outputDirHandle: any = null;
    spiralRunning = false;
    activeBoxIndex = 0;
    box: Box = {
        id: 'branch_001',
        label: 'branch',
        center: [0, 0, 0],
        size: [1, 1, 1],
        euler: [0, 0, 0]
    };
    boxes: Box[] = [this.box];

    constructor(events: Events, scene: Scene) {
        this.events = events;
        this.scene = scene;
        this.canvasContainer = document.getElementById('canvas-container');
        const url = new URL(window.location.href);
        this.initMetricScale(url);
        this.installStyle();
        this.root = this.createPanel();
        this.previewRoot = this.createPreviewPanel();
        document.body.appendChild(this.root);
        document.body.appendChild(this.previewRoot);
        this.handleSvg = this.createHandleOverlay();
        if (this.canvasContainer && this.handleSvg) {
            this.canvasContainer.appendChild(this.handleSvg);
        }
        this.installDrawHandlers();

        events.on('scene.boundChanged', () => {
            if (!this.touched) {
                this.resetToScene();
            } else {
                this.syncSliders();
            }
        });
        events.on('prerender', () => this.draw());
        events.on('postrender', () => this.updateHandles());
        events.on('splat.moved', (splat: Splat) => {
            if (splat === this.activeSplat()) {
                this.scheduleProjectionRefresh();
                if (this.segmentActive) {
                    this.scheduleSegmentUpdate();
                }
            }
        });

        events.function('annotation.box3d', () => this.box);
        events.function('annotation.boxes3d', () => this.boxes.map(box => cloneBox(box)));
        events.function('metric.metersPerSceneUnit', () => this.metersPerSceneUnit);
        events.function('metric.sceneUnitsPerMeter', () => 1 / Math.max(this.metersPerSceneUnit, 1e-9));
        events.on('metric.scale.set', (value: number, source = 'external') => {
            this.setMetersPerSceneUnit(value, source);
        });
        events.on('measure.length.changed', (sceneLength: number, metersLength: number) => {
            if (!Number.isFinite(sceneLength) || sceneLength <= 0) return;
            this.lastMeasuredSceneLength = sceneLength;
            if (this.calibrationMeasuredInput) {
                this.calibrationMeasuredInput.value = sceneLength.toFixed(6);
            }
            if (this.calibrationActualInput && !document.activeElement?.isSameNode(this.calibrationActualInput)) {
                this.calibrationActualInput.value = Number.isFinite(metersLength) && metersLength > 0 ?
                    metersLength.toFixed(4) :
                    this.sceneToMeters(sceneLength).toFixed(4);
            }
        });

        const transforms = url.searchParams.get('transforms');
        if (transforms) {
            this.loadTransforms(decodeURIComponent(transforms));
        }
        const dataparserTransforms = url.searchParams.get('dataparser') ??
            url.searchParams.get('dataparserTransforms') ??
            url.searchParams.get('dataparser-transforms');
        if (dataparserTransforms) {
            this.loadDataparserTransforms(decodeURIComponent(dataparserTransforms));
        }
        const metricScale = url.searchParams.get('metricScale') ?? url.searchParams.get('metric-scale');
        if (metricScale) {
            this.loadMetricScale(decodeURIComponent(metricScale));
        }
    }

    initMetricScale(url: URL) {
        const metersPerSceneUnit = Number(url.searchParams.get('metersPerSceneUnit') ?? url.searchParams.get('mpsu'));
        const sceneUnitsPerMeter = Number(url.searchParams.get('sceneUnitsPerMeter') ?? url.searchParams.get('supm'));
        const stored = Number(localStorage.getItem(metricStorageKey) ?? '');
        if (Number.isFinite(metersPerSceneUnit) && metersPerSceneUnit > 0) {
            this.metersPerSceneUnit = metersPerSceneUnit;
        } else if (Number.isFinite(sceneUnitsPerMeter) && sceneUnitsPerMeter > 0) {
            this.metersPerSceneUnit = 1 / sceneUnitsPerMeter;
        } else if (Number.isFinite(stored) && stored > 0) {
            this.metersPerSceneUnit = stored;
        }
    }

    sceneToMeters(value: number) {
        return value * this.metersPerSceneUnit;
    }

    metersToScene(value: number) {
        return value / Math.max(this.metersPerSceneUnit, 1e-9);
    }

    boxMetric(box: Box) {
        return {
            center: box.center.map(value => this.sceneToMeters(value)),
            size: box.size.map(value => this.sceneToMeters(value))
        };
    }

    setMetersPerSceneUnit(value: number, source = 'manual') {
        if (!Number.isFinite(value) || value <= 0) {
            this.status('metric scale error: value must be positive');
            return;
        }

        this.readInputs(false);
        const spiralCenter = this.spiralCenterInputs?.size ? this.spiralCenter() : null;
        let spiralRadii: number[] | null = null;
        try {
            spiralRadii = this.spiralRadiiInput ? this.spiralRadii() : null;
        } catch (err) {
            spiralRadii = null;
        }
        this.metersPerSceneUnit = value;
        localStorage.setItem(metricStorageKey, `${value}`);
        this.writeMetricInputs();
        this.writeInputs();
        this.writeSpiralInputsFromSceneValues(spiralCenter, spiralRadii);
        this.events.fire('metric.scale.changed', value);
        this.scene.forceRender = true;
        this.drawPreviewOverlay();
        this.updateHandles();
        if (this.segmentActive) {
            this.scheduleSegmentUpdate();
        }
        this.status(`metric scale applied from ${source}: 1 scene unit = ${value.toFixed(6)} m`);
    }

    applyScaleCalibration() {
        const measuredScene = Number(this.calibrationMeasuredInput?.value || this.lastMeasuredSceneLength);
        const actualMeters = Number(this.calibrationActualInput?.value);
        if (!Number.isFinite(measuredScene) || measuredScene <= 0) {
            this.status('calibration error: measure a length first or enter measured scene length');
            return;
        }
        if (!Number.isFinite(actualMeters) || actualMeters <= 0) {
            this.status('calibration error: actual length must be positive meters');
            return;
        }
        this.lastMeasuredSceneLength = measuredScene;
        this.setMetersPerSceneUnit(actualMeters / measuredScene, 'calibration');
        this.calibrationActualInput.value = actualMeters.toFixed(4);
    }

    installStyle() {
        const style = document.createElement('style');
        style.textContent = `
            .bbox-annotation,
            .bbox-preview {
                position: fixed;
                z-index: 30;
                color: #e8edf3;
                background: rgba(11, 16, 22, 0.93);
                border: 1px solid rgba(126, 146, 166, 0.32);
                border-radius: 8px;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.34);
                font: 12px/1.35 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                overflow: hidden;
            }
            .bbox-annotation {
                left: 12px;
                bottom: 12px;
                width: 360px;
            }
            .bbox-preview {
                right: 12px;
                bottom: 12px;
                width: min(440px, calc(100vw - 390px));
                min-width: 300px;
                min-height: 250px;
            }
            .bbox-annotation.collapsed .bbox-body,
            .bbox-preview.collapsed .bbox-preview-body { display: none; }
            .bbox-preview.resized .bbox-preview-body {
                height: calc(100% - 39px);
                display: flex;
                flex-direction: column;
            }
            .bbox-head,
            .bbox-preview-head {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 8px;
                padding: 8px 10px;
                border-bottom: 1px solid rgba(126, 146, 166, 0.24);
                background: rgba(20, 30, 40, 0.92);
                font-weight: 700;
            }
            .bbox-head button,
            .bbox-preview-head button,
            .bbox-annotation button,
            .bbox-preview button {
                color: #edf2f6;
                background: #172331;
                border: 1px solid #334658;
                border-radius: 6px;
                padding: 5px 7px;
                cursor: pointer;
                font: inherit;
            }
            .bbox-annotation button:hover,
            .bbox-preview button:hover { border-color: #60768d; background: #1d2d3f; }
            .bbox-annotation button.active {
                color: #041018;
                background: #55c7ff;
                border-color: #9de2ff;
            }
            .bbox-body,
            .bbox-preview-body { padding: 9px 10px 10px; }
            .bbox-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; margin: 4px 0 7px; }
            .bbox-row.two { grid-template-columns: 1fr 1fr; }
            .bbox-row.four { grid-template-columns: repeat(4, 1fr); }
            .bbox-cell { min-width: 0; }
            .bbox-annotation label { display: block; color: #9fb0bf; margin-top: 6px; }
            .bbox-annotation input {
                width: 100%;
                min-width: 0;
                color: #edf2f6;
                background: #080d12;
                border: 1px solid #334658;
                border-radius: 5px;
                padding: 5px 6px;
                font: inherit;
            }
            .bbox-annotation select {
                width: 100%;
                min-width: 0;
                color: #edf2f6;
                background: #080d12;
                border: 1px solid #334658;
                border-radius: 5px;
                padding: 5px 6px;
                font: inherit;
            }
            .bbox-annotation input[type="range"],
            .bbox-preview input[type="range"] {
                padding: 0;
                margin-top: 4px;
                accent-color: #55c7ff;
            }
            .bbox-status,
            .bbox-preview-status { color: #aab8c5; margin-top: 7px; min-height: 17px; }
            .bbox-segment-status { color: #55c7ff; margin-top: 6px; min-height: 17px; }
            .bbox-check {
                display: flex;
                align-items: center;
                gap: 7px;
                min-height: 29px;
                color: #ccd6df;
            }
            .bbox-check input {
                width: auto;
                margin: 0;
                accent-color: #55c7ff;
            }
            .bbox-preview-controls {
                display: grid;
                grid-template-columns: auto 1fr auto;
                align-items: center;
                gap: 6px;
                margin-bottom: 8px;
            }
            .bbox-preview-options {
                display: grid;
                grid-template-columns: 1fr auto auto auto;
                align-items: center;
                gap: 8px;
                margin: 7px 0 8px;
            }
            .bbox-preview-options select {
                min-width: 0;
                color: #edf2f6;
                background: #080d12;
                border: 1px solid #334658;
                border-radius: 5px;
                padding: 5px 6px;
                font: inherit;
            }
            @media (max-width: 1040px) {
                .bbox-preview-options {
                    grid-template-columns: 1fr 1fr;
                }
            }
            .bbox-preview-frame {
                position: relative;
                width: 100%;
                aspect-ratio: 16 / 9;
                background: #05080c;
                border: 1px solid #283848;
                border-radius: 6px;
                overflow: hidden;
            }
            .bbox-preview.resized .bbox-preview-frame {
                flex: 1 1 auto;
                min-height: 130px;
                aspect-ratio: auto;
            }
            .bbox-preview-frame img,
            .bbox-preview-frame canvas {
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                object-fit: contain;
            }
            .bbox-preview-frame canvas { pointer-events: none; }
            .bbox-frame-label {
                color: #ccd6df;
                text-align: center;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .bbox-handle-layer {
                position: absolute;
                inset: 0;
                z-index: 25;
                width: 100%;
                height: 100%;
                pointer-events: none;
            }
            .bbox-corner-handle {
                pointer-events: all;
                cursor: grab;
                fill: #ff3b2f;
                stroke: #ffffff;
                stroke-width: 2px;
                paint-order: stroke fill;
                filter: drop-shadow(0 1px 4px rgba(0, 0, 0, 0.75));
            }
            .bbox-corner-handle:hover,
            .bbox-corner-handle.active {
                fill: #55c7ff;
                r: 7px;
            }
            .bbox-pick-hover {
                pointer-events: none;
                fill: rgba(85, 199, 255, 0.22);
                stroke: #55c7ff;
                stroke-width: 3px;
                filter: drop-shadow(0 1px 6px rgba(85, 199, 255, 0.85));
            }
            .bbox-preview-resize {
                position: absolute;
                top: 0;
                left: 0;
                width: 18px;
                height: 18px;
                cursor: nwse-resize;
                z-index: 2;
            }
            .bbox-preview-resize::after {
                content: "";
                position: absolute;
                left: 5px;
                top: 5px;
                width: 8px;
                height: 8px;
                border-left: 2px solid rgba(85, 199, 255, 0.85);
                border-top: 2px solid rgba(85, 199, 255, 0.85);
            }
            .bbox-preview.resizing {
                user-select: none;
            }
            .bbox-draw-hint {
                color: #55c7ff;
            }
            @media (max-width: 820px) {
                .bbox-preview {
                    right: 12px;
                    bottom: 260px;
                    width: calc(100vw - 24px);
                    min-width: 0;
                }
                .bbox-annotation {
                    width: calc(100vw - 24px);
                }
            }
        `;
        document.head.appendChild(style);
    }

    stopPanelEvents(el: HTMLElement) {
        ['pointerdown', 'pointerup', 'pointermove', 'wheel', 'dblclick'].forEach((eventName) => {
            el.addEventListener(eventName, (event: Event) => event.stopPropagation());
        });
    }

    sceneExtent() {
        const bound = this.scene.bound;
        const half = bound.halfExtents.length();
        return Math.max(half * 2, 1);
    }

    sliderRange(id: string) {
        const bound = this.scene.bound;
        const extent = this.sceneExtent();
        const margin = extent * 0.25;
        if (id === 'cx') return [this.sceneToMeters(bound.getMin().x - margin), this.sceneToMeters(bound.getMax().x + margin)];
        if (id === 'cy') return [this.sceneToMeters(bound.getMin().y - margin), this.sceneToMeters(bound.getMax().y + margin)];
        if (id === 'cz') return [this.sceneToMeters(bound.getMin().z - margin), this.sceneToMeters(bound.getMax().z + margin)];
        if (id.startsWith('s')) return [this.sceneToMeters(0.001), this.sceneToMeters(Math.max(extent, 0.01))];
        return [-180, 180];
    }

    isMetricInput(id: string) {
        return id[0] === 'c' || id[0] === 's';
    }

    displayValue(id: string, value: number) {
        return this.isMetricInput(id) ? this.sceneToMeters(value) : value;
    }

    sceneValue(id: string, value: number) {
        return this.isMetricInput(id) ? this.metersToScene(value) : value;
    }

    createInput(id: string, value: string, step = '0.02') {
        const input = document.createElement('input');
        input.id = `bbox-${id}`;
        input.value = value;
        input.step = step;
        input.type = id === 'id' || id === 'label' ? 'text' : 'number';
        input.addEventListener('input', () => {
            this.touched = true;
            this.readInputs();
            this.syncSliders();
            this.scene.forceRender = true;
            this.drawPreviewOverlay();
            if (id !== 'id' && id !== 'label') {
                this.segmentActive = true;
                this.scheduleSegmentUpdate();
            }
        });
        this.inputs.set(id, input);
        return input;
    }

    createSlider(id: string, value: number, step: string) {
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.step = step;
        const [min, max] = this.sliderRange(id);
        slider.min = `${min}`;
        slider.max = `${max}`;
        slider.value = `${value}`;
        slider.addEventListener('input', () => {
            const input = this.inputs.get(id);
            if (input) {
                input.value = Number(slider.value).toFixed(id.startsWith('r') ? 2 : 4);
            }
            this.touched = true;
            this.readInputs();
            this.scene.forceRender = true;
            this.drawPreviewOverlay();
            this.segmentActive = true;
            this.scheduleSegmentUpdate();
        });
        this.sliders.set(id, slider);
        return slider;
    }

    createPanel() {
        const root = document.createElement('div');
        root.className = 'bbox-annotation';
        this.stopPanelEvents(root);

        const head = document.createElement('div');
        head.className = 'bbox-head';
        const title = document.createElement('span');
        title.textContent = '3D BBox';
        const toggle = document.createElement('button');
        toggle.textContent = 'hide';
        toggle.addEventListener('click', () => {
            root.classList.toggle('collapsed');
            toggle.textContent = root.classList.contains('collapsed') ? 'show' : 'hide';
        });
        head.append(title, toggle);

        const body = document.createElement('div');
        body.className = 'bbox-body';

        const idRow = document.createElement('div');
        idRow.className = 'bbox-row two';
        idRow.append(this.createInput('id', this.box.id), this.createInput('label', this.box.label));

        body.append(this.label('Boxes'), this.boxSelectRow(), this.boxActionsRow());
        body.append(this.label('ID / label'), idRow);
        body.append(this.label('Center XYZ meters'), this.row(['cx', 'cy', 'cz'], this.box.center, '0.02'));
        body.append(this.label('Size XYZ meters'), this.row(['sx', 'sy', 'sz'], this.box.size, '0.02'));
        body.append(this.label('Euler XYZ degrees'), this.row(['rx', 'ry', 'rz'], this.box.euler, '1'));
        body.append(this.label('Metric scale'), this.metricRow());
        body.append(this.label('Segmentation'), this.segmentRow());
        body.append(this.label('View mask'), this.maskRow());
        body.append(this.label('Auto capture'), this.captureRow(), this.captureActionsRow());

        const nudge = document.createElement('div');
        nudge.className = 'bbox-row four';
        [
            ['-X', [-1, 0, 0]], ['+X', [1, 0, 0]],
            ['-Y', [0, -1, 0]], ['+Y', [0, 1, 0]],
            ['-Z', [0, 0, -1]], ['+Z', [0, 0, 1]],
            ['small', 0.9], ['large', 1.1]
        ].forEach(([name, value]) => {
            const button = document.createElement('button');
            button.textContent = name as string;
            button.addEventListener('click', () => {
                if (Array.isArray(value)) {
                    this.nudge(value as number[]);
                } else {
                    this.scale(value as number);
                }
            });
            nudge.appendChild(button);
        });
        body.append(this.label('Adjust'), nudge);

        const actions = document.createElement('div');
        actions.className = 'bbox-row two';
        actions.append(
            this.button('Reset scene', () => this.resetToScene(true)),
            this.button('Focus', () => this.focus())
        );

        const drawRow = document.createElement('div');
        drawRow.className = 'bbox-row two';
        this.drawButton = this.button('Draw BBox', () => this.toggleDrawMode());
        this.cubeButton = this.button('Place 10cm Cube', () => this.toggleCubeMode());
        drawRow.append(
            this.drawButton,
            this.cubeButton
        );

        const modeRow = document.createElement('div');
        modeRow.className = 'bbox-row two';
        modeRow.append(
            this.button('Stop Mode', () => this.clearPlacementModes()),
            this.button('10cm Size', () => this.setCubeSize())
        );

        const exportRow = document.createElement('div');
        exportRow.className = 'bbox-row two';
        exportRow.append(
            this.button('Export 3D', () => this.export3D()),
            this.button('Export 2D', () => this.export2D())
        );
        const segmentExportRow = document.createElement('div');
        segmentExportRow.className = 'bbox-row two';
        segmentExportRow.append(
            this.button('Segment Now', () => {
                this.segmentActive = true;
                this.updateSegmentSelection().catch(err => this.segmentMessage(`segment error: ${err.message ?? err}`));
            }),
            this.button('Export Segment PLY', () => {
                this.exportSegmentPly().catch(err => this.segmentMessage(`segment export error: ${err.message ?? err}`));
            })
        );
        const viewExportRow = document.createElement('div');
        viewExportRow.className = 'bbox-row two';
        viewExportRow.append(
            this.button('Export View RGB+Mask', () => {
                this.exportCurrentViewSet().catch(err => this.segmentMessage(`view export error: ${err.message ?? err}`));
            }),
            this.button('Export Mask JSON', () => {
                try {
                    this.exportCurrentViewJson();
                } catch (err) {
                    this.segmentMessage(`view json error: ${err instanceof Error ? err.message : String(err)}`);
                }
            })
        );
        body.append(actions, drawRow, modeRow, exportRow, segmentExportRow, viewExportRow);

        const status = document.createElement('div');
        status.className = 'bbox-status';
        status.id = 'bbox-status';
        status.textContent = 'SuperSplat native renderer';
        body.appendChild(status);

        this.segmentStatus = document.createElement('div');
        this.segmentStatus.className = 'bbox-segment-status';
        this.segmentStatus.textContent = 'segment: waiting for bbox placement';
        body.appendChild(this.segmentStatus);

        root.append(head, body);
        this.syncBoxSelect();
        this.writeInputs();
        return root;
    }

    createPreviewPanel() {
        const root = document.createElement('div');
        root.className = 'bbox-preview';
        this.stopPanelEvents(root);

        const head = document.createElement('div');
        head.className = 'bbox-preview-head';
        const title = document.createElement('span');
        title.textContent = '2D Projection';
        const toggle = document.createElement('button');
        toggle.textContent = 'hide';
        toggle.addEventListener('click', () => {
            root.classList.toggle('collapsed');
            toggle.textContent = root.classList.contains('collapsed') ? 'show' : 'hide';
        });
        head.append(title, toggle);

        const body = document.createElement('div');
        body.className = 'bbox-preview-body';

        const controls = document.createElement('div');
        controls.className = 'bbox-preview-controls';
        const prev = this.button('Prev', () => this.stepFrame(-1));
        this.frameLabel = document.createElement('span');
        this.frameLabel.className = 'bbox-frame-label';
        this.frameLabel.textContent = 'no frames';
        const next = this.button('Next', () => this.stepFrame(1));
        controls.append(prev, this.frameLabel, next);

        this.frameSlider = document.createElement('input');
        this.frameSlider.type = 'range';
        this.frameSlider.min = '0';
        this.frameSlider.max = '0';
        this.frameSlider.step = '1';
        this.frameSlider.value = '0';
        this.frameSlider.addEventListener('input', () => {
            this.frameIndex = Number(this.frameSlider.value);
            this.updatePreview();
        });

        const options = document.createElement('div');
        options.className = 'bbox-preview-options';

        this.projectionModeSelect = document.createElement('select');
        [
            ['train-to-data', 'Projection: train -> data'],
            ['splat-local', 'Projection: splat local'],
            ['splat-world', 'Projection: splat world'],
            ['local-applied', 'Projection: local + applied'],
            ['local-applied-inverse', 'Projection: local + inv applied']
        ].forEach(([value, text]) => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = text;
            this.projectionModeSelect.appendChild(option);
        });
        this.projectionModeSelect.value = this.projectionMode;
        this.projectionModeSelect.addEventListener('change', () => {
            this.projectionMode = this.projectionModeSelect.value as ProjectionMode;
            this.drawPreviewOverlay();
        });

        const useTransform = document.createElement('label');
        useTransform.className = 'bbox-check';
        this.useSplatTransformInput = document.createElement('input');
        this.useSplatTransformInput.type = 'checkbox';
        this.useSplatTransformInput.checked = true;
        this.useSplatTransformInput.addEventListener('change', () => this.drawPreviewOverlay());
        useTransform.append(this.useSplatTransformInput, document.createTextNode('use TRANSFORM'));

        const showPoints = document.createElement('label');
        showPoints.className = 'bbox-check';
        this.showSegmentProjectionInput = document.createElement('input');
        this.showSegmentProjectionInput.type = 'checkbox';
        this.showSegmentProjectionInput.checked = true;
        this.showSegmentProjectionInput.addEventListener('change', () => this.drawPreviewOverlay());
        showPoints.append(this.showSegmentProjectionInput, document.createTextNode('segment pts'));

        const refresh = this.button('Refresh', () => this.drawPreviewOverlay());
        options.append(this.projectionModeSelect, useTransform, showPoints, refresh);

        const frame = document.createElement('div');
        frame.className = 'bbox-preview-frame';
        this.previewImage = document.createElement('img');
        this.previewCanvas = document.createElement('canvas');
        this.previewImage.addEventListener('load', () => this.drawPreviewOverlay());
        frame.append(this.previewImage, this.previewCanvas);

        this.previewStatus = document.createElement('div');
        this.previewStatus.className = 'bbox-preview-status';
        this.previewStatus.textContent = 'load transforms to preview 2D labels';

        body.append(controls, this.frameSlider, options, frame, this.previewStatus);
        const resize = document.createElement('div');
        resize.className = 'bbox-preview-resize';
        this.installPreviewResize(root, resize);

        root.append(head, body, resize);
        return root;
    }

    installPreviewResize(root: HTMLDivElement, handle: HTMLDivElement) {
        let pointerId: number | null = null;
        let startX = 0;
        let startY = 0;
        let startWidth = 0;
        let startHeight = 0;

        const resize = (event: PointerEvent) => {
            if (pointerId !== event.pointerId) return;
            event.preventDefault();
            event.stopPropagation();

            const maxWidth = Math.max(320, window.innerWidth - 24);
            const maxHeight = Math.max(260, window.innerHeight - 24);
            const width = clamp(startWidth - (event.clientX - startX), 300, maxWidth);
            const height = clamp(startHeight - (event.clientY - startY), 250, maxHeight);
            root.style.width = `${width}px`;
            root.style.height = `${height}px`;
            root.classList.add('resized');
            this.drawPreviewOverlay();
        };

        const stop = (event: PointerEvent) => {
            if (pointerId !== event.pointerId) return;
            event.preventDefault();
            event.stopPropagation();
            root.classList.remove('resizing');
            try {
                handle.releasePointerCapture(event.pointerId);
            } catch (err) {
                // Pointer capture may already be released by the browser.
            }
            pointerId = null;
        };

        handle.addEventListener('pointerdown', (event) => {
            if (event.pointerType === 'mouse' && event.button !== 0) return;
            event.preventDefault();
            event.stopPropagation();
            pointerId = event.pointerId;
            startX = event.clientX;
            startY = event.clientY;
            const rect = root.getBoundingClientRect();
            startWidth = rect.width;
            startHeight = rect.height;
            root.classList.add('resizing');
            handle.setPointerCapture(event.pointerId);
        });
        handle.addEventListener('pointermove', resize);
        handle.addEventListener('pointerup', stop);
        handle.addEventListener('pointercancel', stop);
    }

    createHandleOverlay() {
        if (!this.canvasContainer) return null;

        const ns = 'http://www.w3.org/2000/svg';
        const svg = document.createElementNS(ns, 'svg');
        svg.classList.add('bbox-handle-layer');
        svg.setAttribute('aria-hidden', 'true');

        this.hoverNode = document.createElementNS(ns, 'circle');
        this.hoverNode.classList.add('bbox-pick-hover');
        this.hoverNode.setAttribute('r', '14');
        this.hoverNode.setAttribute('visibility', 'hidden');
        svg.appendChild(this.hoverNode);

        for (let i = 0; i < 8; i++) {
            const circle = document.createElementNS(ns, 'circle');
            circle.classList.add('bbox-corner-handle');
            circle.setAttribute('r', '5');
            circle.setAttribute('data-index', `${i}`);
            circle.addEventListener('pointerdown', event => this.startHandleDrag(event, i), true);
            svg.appendChild(circle);
            this.handleNodes.push(circle);
        }

        window.addEventListener('pointermove', event => this.moveHandleDrag(event), true);
        window.addEventListener('pointerup', event => this.endHandleDrag(event), true);
        window.addEventListener('pointercancel', event => this.endHandleDrag(event), true);

        return svg;
    }

    label(text: string) {
        const label = document.createElement('label');
        label.textContent = text;
        return label;
    }

    row(ids: string[], values: number[], step: string) {
        const row = document.createElement('div');
        row.className = 'bbox-row';
        ids.forEach((id, i) => {
            const cell = document.createElement('div');
            cell.className = 'bbox-cell';
            const value = this.displayValue(id, values[i]);
            cell.append(
                this.createInput(id, `${value}`, step),
                this.createSlider(id, value, step)
            );
            row.appendChild(cell);
        });
        return row;
    }

    boxSelectRow() {
        const row = document.createElement('div');
        row.className = 'bbox-row two';

        this.boxSelect = document.createElement('select');
        this.boxSelect.style.width = '100%';
        this.boxSelect.addEventListener('change', () => {
            this.readInputs();
            this.activeBoxIndex = clamp(Number(this.boxSelect.value), 0, this.boxes.length - 1);
            this.box = cloneBox(this.boxes[this.activeBoxIndex]);
            this.writeInputs();
            this.scene.forceRender = true;
            this.drawPreviewOverlay();
            this.updateHandles();
            this.segmentActive = false;
            this.segmentMessage(`active bbox: ${this.box.id}`);
        });

        const focus = this.button('Focus Box', () => this.focus());
        row.append(this.boxSelect, focus);
        return row;
    }

    boxActionsRow() {
        const row = document.createElement('div');
        row.className = 'bbox-row two';
        row.append(
            this.button('Add BBox', () => this.addBox()),
            this.button('Remove BBox', () => this.removeBox())
        );
        return row;
    }

    captureRow() {
        const row = document.createElement('div');
        row.className = 'bbox-row four';

        const center = this.box.center;
        [
            ['cap-cx', center[0]],
            ['cap-cy', center[1]],
            ['cap-cz', center[2]]
        ].forEach(([id, value]) => {
            const input = document.createElement('input');
            input.type = 'number';
            input.step = '0.01';
            input.value = `${this.sceneToMeters(Number(value)).toFixed(4)}`;
            input.title = `${String(id)} meters`;
            this.spiralCenterInputs.set(String(id), input);
            row.appendChild(input);
        });

        this.spiralThetaInput = document.createElement('input');
        this.spiralThetaInput.type = 'number';
        this.spiralThetaInput.step = '1';
        this.spiralThetaInput.min = '1';
        this.spiralThetaInput.max = '180';
        this.spiralThetaInput.value = '30';
        this.spiralThetaInput.title = 'd theta degrees';
        row.appendChild(this.spiralThetaInput);
        return row;
    }

    captureActionsRow() {
        const wrap = document.createElement('div');

        const radiiRow = document.createElement('div');
        radiiRow.className = 'bbox-row two';
        this.spiralRadiiInput = document.createElement('input');
        this.spiralRadiiInput.type = 'text';
        this.spiralRadiiInput.value = '1,2,3,4,5,6,7,8,9,10';
        this.spiralRadiiInput.title = 'spiral radii in meters';
        const useCenter = this.button('Use BBox Center', () => this.setSpiralCenterFromBox());
        radiiRow.append(this.spiralRadiiInput, useCenter);

        const actions = document.createElement('div');
        actions.className = 'bbox-row two';
        actions.append(
            this.button('Choose Output Folder', () => {
                this.chooseOutputFolder().catch(err => this.segmentMessage(`output folder error: ${err.message ?? err}`));
            }),
            this.button('Run Spiral Export', () => {
                this.runSpiralExport().catch(err => this.segmentMessage(`spiral export error: ${err.message ?? err}`));
            })
        );

        this.outputStatus = document.createElement('div');
        this.outputStatus.className = 'bbox-segment-status';
        this.outputStatus.textContent = 'output: browser downloads until a folder is chosen';

        wrap.append(radiiRow, actions, this.outputStatus);
        return wrap;
    }

    metricRow() {
        const wrap = document.createElement('div');

        const row = document.createElement('div');
        row.className = 'bbox-row two';

        this.metricInput = document.createElement('input');
        this.metricInput.type = 'number';
        this.metricInput.step = '0.001';
        this.metricInput.min = '0.000001';
        this.metricInput.title = 'meters per scene unit';
        this.metricInput.placeholder = 'meters/unit';
        this.metricInput.value = `${this.metersPerSceneUnit}`;
        this.metricInput.addEventListener('change', () => {
            const value = Number(this.metricInput.value);
            if (Number.isFinite(value) && value > 0) {
                this.setMetersPerSceneUnit(value, 'manual input');
            }
        });

        this.cubeMetersInput = document.createElement('input');
        this.cubeMetersInput.type = 'number';
        this.cubeMetersInput.step = '0.05';
        this.cubeMetersInput.min = '0.001';
        this.cubeMetersInput.title = 'default cube size in meters';
        this.cubeMetersInput.placeholder = 'cube meters';
        this.cubeMetersInput.value = `${defaultCubeMeters}`;

        row.append(this.metricInput, this.cubeMetersInput);

        const calibration = document.createElement('div');
        calibration.className = 'bbox-row two';
        this.calibrationMeasuredInput = document.createElement('input');
        this.calibrationMeasuredInput.type = 'number';
        this.calibrationMeasuredInput.step = '0.0001';
        this.calibrationMeasuredInput.min = '0.000001';
        this.calibrationMeasuredInput.placeholder = 'measured scene';
        this.calibrationMeasuredInput.title = 'Raw scene length from the measure tool';

        this.calibrationActualInput = document.createElement('input');
        this.calibrationActualInput.type = 'number';
        this.calibrationActualInput.step = '0.01';
        this.calibrationActualInput.min = '0.000001';
        this.calibrationActualInput.placeholder = 'actual meters';
        this.calibrationActualInput.title = 'Real-world measured length in meters';
        calibration.append(this.calibrationMeasuredInput, this.calibrationActualInput);

        const actions = document.createElement('div');
        actions.className = 'bbox-row two';
        actions.append(
            this.button('Use Last Measure', () => {
                if (this.lastMeasuredSceneLength > 0) {
                    this.calibrationMeasuredInput.value = this.lastMeasuredSceneLength.toFixed(6);
                    this.calibrationActualInput.value = this.sceneToMeters(this.lastMeasuredSceneLength).toFixed(4);
                } else {
                    this.status('measure a length first');
                }
            }),
            this.button('Apply Scale', () => this.applyScaleCalibration())
        );

        this.metricSummary = document.createElement('div');
        this.metricSummary.className = 'bbox-status';

        wrap.append(row, calibration, actions, this.metricSummary);
        this.writeMetricInputs();
        return wrap;
    }

    segmentRow() {
        const row = document.createElement('div');
        row.className = 'bbox-row two';

        const label = document.createElement('label');
        label.className = 'bbox-check';
        this.autoSegmentInput = document.createElement('input');
        this.autoSegmentInput.type = 'checkbox';
        this.autoSegmentInput.checked = true;
        this.autoSegmentInput.addEventListener('change', () => {
            if (this.autoSegmentInput.checked) {
                this.segmentActive = true;
                this.scheduleSegmentUpdate(0);
            } else if (this.segmentStatus) {
                this.segmentStatus.textContent = 'segment: auto off';
            }
        });
        label.append(this.autoSegmentInput, document.createTextNode('Auto segment'));

        const clear = this.button('Clear Segment', () => {
            this.clearSegmentSelection().catch(err => this.segmentMessage(`segment clear error: ${err.message ?? err}`));
        });

        row.append(label, clear);
        return row;
    }

    maskRow() {
        const row = document.createElement('div');
        row.className = 'bbox-row two';

        this.maskRadiusInput = document.createElement('input');
        this.maskRadiusInput.type = 'number';
        this.maskRadiusInput.step = '1';
        this.maskRadiusInput.min = '1';
        this.maskRadiusInput.max = '32';
        this.maskRadiusInput.title = 'selected center mask radius in screen pixels';
        this.maskRadiusInput.placeholder = 'radius px';
        this.maskRadiusInput.value = `${defaultMaskRadiusPx}`;

        const reset = this.button('2px radius', () => {
            this.maskRadiusInput.value = `${defaultMaskRadiusPx}`;
        });

        row.append(this.maskRadiusInput, reset);
        return row;
    }

    button(text: string, fn: () => void) {
        const button = document.createElement('button');
        button.textContent = text;
        button.addEventListener('click', fn);
        return button;
    }

    status(text: string) {
        const el = document.getElementById('bbox-status');
        if (el) el.textContent = text;
    }

    segmentMessage(text: string) {
        if (this.segmentStatus) {
            this.segmentStatus.textContent = text;
        }
    }

    activeSplat() {
        const selected = this.events.invoke('selection') as Splat | null;
        if (selected) return selected;

        const splats = this.scene.getElementsByType(ElementType.splat) as Splat[];
        const splat = splats[0] ?? null;
        if (splat) {
            this.events.fire('selection', splat);
        }
        return splat;
    }

    scheduleSegmentUpdate(delay = segmentDebounceMs) {
        if (!this.autoSegmentInput?.checked || !this.segmentActive) return;
        if (this.segmentTimer) {
            clearTimeout(this.segmentTimer);
        }
        this.segmentTimer = setTimeout(() => {
            this.segmentTimer = null;
            this.updateSegmentSelection().catch(err => this.segmentMessage(`segment error: ${err.message ?? err}`));
        }, delay);
    }

    scheduleProjectionRefresh(delay = segmentDebounceMs) {
        if (this.projectionTimer) {
            clearTimeout(this.projectionTimer);
        }
        this.projectionTimer = setTimeout(() => {
            this.projectionTimer = null;
            this.drawPreviewOverlay();
        }, delay);
    }

    async updateSegmentSelection() {
        if (this.segmentRunning) {
            this.segmentDirty = true;
            return;
        }

        this.segmentRunning = true;
        try {
            do {
                this.segmentDirty = false;
                await this.applySegmentSelection();
            } while (this.segmentDirty);
        } finally {
            this.segmentRunning = false;
        }
    }

    async applySegmentSelection() {
        const splat = this.activeSplat();
        if (!splat) {
            this.segmentMessage('segment: no splat loaded');
            return;
        }

        this.readInputs();
        const total = splat.splatData.numSplats;
        const centers = splat.entity.gsplat.instance.sorter.centers as Float32Array;
        const state = splat.splatData.getProp('state') as Uint8Array;
        const world = splat.worldTransform;
        const cx = this.box.center[0];
        const cy = this.box.center[1];
        const cz = this.box.center[2];
        const hx = Math.max(this.box.size[0], 0.001) * 0.5;
        const hy = Math.max(this.box.size[1], 0.001) * 0.5;
        const hz = Math.max(this.box.size[2], 0.001) * 0.5;
        const r = rotationMatrix(this.box.euler[0], this.box.euler[1], this.box.euler[2]);
        const point = this.segmentPoint;

        this.segmentMessage(`segmenting ${formatCount(total)} points...`);

        const count = await this.scene.commandQueue.enqueue(async () => {
            const selectedCount = splat.state.replaceSelectedByPredicate((i) => {
                const offset = i * 3;
                point.set(centers[offset], centers[offset + 1], centers[offset + 2]);
                world.transformPoint(point, point);

                const dx = point.x - cx;
                const dy = point.y - cy;
                const dz = point.z - cz;
                if (!Number.isFinite(dx) || !Number.isFinite(dy) || !Number.isFinite(dz)) {
                    return false;
                }

                const lx = r[0] * dx + r[3] * dy + r[6] * dz;
                const ly = r[1] * dx + r[4] * dy + r[7] * dz;
                const lz = r[2] * dx + r[5] * dy + r[8] * dz;
                return Math.abs(lx) <= hx && Math.abs(ly) <= hy && Math.abs(lz) <= hz;
            });
            await splat.updateState();
            return selectedCount;
        });

        this.segmentLastCount = count;
        this.segmentMessage(`segment: ${formatCount(count)} / ${formatCount(total)} points in bbox`);
    }

    async clearSegmentSelection() {
        const splat = this.activeSplat();
        if (!splat) {
            this.segmentMessage('segment: no splat loaded');
            return;
        }

        await this.scene.commandQueue.enqueue(async () => {
            splat.state.replaceSelectedByPredicate(() => false);
            await splat.updateState();
        });
        this.segmentLastCount = 0;
        this.segmentActive = false;
        this.segmentMessage('segment: cleared');
    }

    async exportSegmentPly() {
        const splat = this.activeSplat();
        if (!splat) {
            this.segmentMessage('segment: no splat loaded');
            return;
        }
        if (this.autoSegmentInput?.checked) {
            this.segmentActive = true;
            await this.updateSegmentSelection();
        }

        const state = splat.splatData.getProp('state') as Uint8Array;
        const total = splat.splatData.numSplats;
        const centers = splat.entity.gsplat.instance.sorter.centers as Float32Array;
        const world = splat.worldTransform;
        const point = this.segmentPoint;
        const rows: string[] = [];

        for (let i = 0; i < total; i++) {
            if ((state[i] & State.selected) === 0 || (state[i] & (State.locked | State.deleted)) !== 0) {
                continue;
            }
            const offset = i * 3;
            point.set(centers[offset], centers[offset + 1], centers[offset + 2]);
            world.transformPoint(point, point);
            rows.push(`${this.sceneToMeters(point.x)} ${this.sceneToMeters(point.y)} ${this.sceneToMeters(point.z)}`);
        }

        const header = [
            'ply',
            'format ascii 1.0',
            'comment coordinates are calibrated meters',
            `comment meters_per_scene_unit ${this.metersPerSceneUnit}`,
            `element vertex ${rows.length}`,
            'property float x',
            'property float y',
            'property float z',
            'end_header'
        ];
        downloadText(`${this.box.id || 'bbox'}_segment_points.ply`, `${header.concat(rows).join('\n')}\n`);
        this.segmentMessage(`segment ply exported: ${formatCount(rows.length)} points`);
    }

    cubeMeters() {
        const value = Number(this.cubeMetersInput?.value ?? defaultCubeMeters);
        return Number.isFinite(value) && value > 0 ? value : defaultCubeMeters;
    }

    cubeSceneSize() {
        return this.cubeMeters() / Math.max(this.metersPerSceneUnit, 1e-9);
    }

    writeMetricInputs() {
        if (this.metricInput) {
            this.metricInput.value = `${this.metersPerSceneUnit.toFixed(8)}`;
        }
        if (this.cubeMetersInput && !this.cubeMetersInput.value) {
            this.cubeMetersInput.value = `${defaultCubeMeters}`;
        }
        if (this.metricSummary) {
            this.metricSummary.textContent = `1 scene unit = ${this.metersPerSceneUnit.toFixed(6)} m, 1 m = ${(1 / Math.max(this.metersPerSceneUnit, 1e-9)).toFixed(4)} scene units`;
        }
    }

    readInputs(sync = true) {
        const value = (id: string) => this.inputs.get(id).value;
        const num = (id: string) => parseFloat(value(id));
        this.box = {
            id: value('id') || 'box_001',
            label: value('label') || 'object',
            center: [this.sceneValue('cx', num('cx')), this.sceneValue('cy', num('cy')), this.sceneValue('cz', num('cz'))],
            size: [
                Math.max(0.001, this.sceneValue('sx', num('sx'))),
                Math.max(0.001, this.sceneValue('sy', num('sy'))),
                Math.max(0.001, this.sceneValue('sz', num('sz')))
            ],
            euler: [num('rx'), num('ry'), num('rz')]
        };
        this.boxes[this.activeBoxIndex] = cloneBox(this.box);
        this.drawDepth = this.box.size[2];
        if (sync) {
            this.syncBoxSelect();
        }
    }

    writeInputs() {
        const set = (id: string, value: string | number) => {
            const input = this.inputs.get(id);
            if (input) {
                input.value = typeof value === 'number' ?
                    this.displayValue(id, value).toFixed(id.startsWith('r') ? 2 : 4) :
                    value;
            }
        };
        set('id', this.box.id);
        set('label', this.box.label);
        ['cx', 'cy', 'cz'].forEach((id, i) => set(id, this.box.center[i]));
        ['sx', 'sy', 'sz'].forEach((id, i) => set(id, this.box.size[i]));
        ['rx', 'ry', 'rz'].forEach((id, i) => set(id, this.box.euler[i]));
        if (this.boxes.length) {
            this.boxes[this.activeBoxIndex] = cloneBox(this.box);
        }
        this.syncBoxSelect();
        this.syncSliders();
    }

    syncBoxSelect() {
        if (!this.boxSelect) return;
        const current = `${this.activeBoxIndex}`;
        this.boxSelect.replaceChildren();
        this.boxes.forEach((box, index) => {
            const option = document.createElement('option');
            option.value = `${index}`;
            option.textContent = `${index + 1}: ${box.id || 'bbox'} (${box.label || 'object'})`;
            this.boxSelect.appendChild(option);
        });
        this.boxSelect.value = current;
    }

    addBox() {
        this.readInputs(false);
        const next = cloneBox(this.box);
        next.id = `branch_${String(this.boxes.length + 1).padStart(3, '0')}`;
        next.center = [
            next.center[0] + Math.max(next.size[0] * 0.2, 0.02),
            next.center[1],
            next.center[2]
        ];
        this.boxes.push(next);
        this.activeBoxIndex = this.boxes.length - 1;
        this.box = cloneBox(next);
        this.writeInputs();
        this.syncBoxSelect();
        this.scene.forceRender = true;
        this.drawPreviewOverlay();
        this.updateHandles();
        this.segmentActive = false;
        this.segmentMessage(`bbox added: ${this.box.id}`);
    }

    removeBox() {
        if (this.boxes.length <= 1) {
            this.segmentMessage('remove bbox: keep at least one bbox');
            return;
        }

        const removed = this.boxes.splice(this.activeBoxIndex, 1)[0];
        this.activeBoxIndex = clamp(this.activeBoxIndex, 0, this.boxes.length - 1);
        this.box = cloneBox(this.boxes[this.activeBoxIndex]);
        this.writeInputs();
        this.syncBoxSelect();
        this.scene.forceRender = true;
        this.drawPreviewOverlay();
        this.updateHandles();
        this.segmentActive = false;
        this.segmentMessage(`bbox removed: ${removed.id}`);
    }

    syncSliders() {
        const values: Record<string, number> = {
            cx: this.sceneToMeters(this.box.center[0]),
            cy: this.sceneToMeters(this.box.center[1]),
            cz: this.sceneToMeters(this.box.center[2]),
            sx: this.sceneToMeters(this.box.size[0]),
            sy: this.sceneToMeters(this.box.size[1]),
            sz: this.sceneToMeters(this.box.size[2]),
            rx: this.box.euler[0],
            ry: this.box.euler[1],
            rz: this.box.euler[2]
        };

        Object.entries(values).forEach(([id, value]) => {
            const slider = this.sliders.get(id);
            if (!slider) return;
            const [min, max] = this.sliderRange(id);
            slider.min = `${min}`;
            slider.max = `${max}`;
            slider.value = `${clamp(value, min, max)}`;
        });
    }

    resetToScene(markTouched = false) {
        const bound = this.scene.bound;
        const min = bound.getMin();
        const max = bound.getMax();
        this.box.center = [
            (min.x + max.x) * 0.5,
            (min.y + max.y) * 0.5,
            (min.z + max.z) * 0.5
        ];
        this.box.size = [
            Math.max((max.x - min.x) * 0.25, 0.2),
            Math.max((max.y - min.y) * 0.25, 0.2),
            Math.max((max.z - min.z) * 0.25, 0.2)
        ];
        this.box.euler = [0, 0, 0];
        this.drawDepth = this.box.size[2];
        this.touched = markTouched;
        this.writeInputs();
        this.scene.forceRender = true;
        this.drawPreviewOverlay();
        if (markTouched) {
            this.segmentActive = true;
            this.scheduleSegmentUpdate();
        }
    }

    focus() {
        this.readInputs();
        this.scene.camera.focus({
            focalPoint: new Vec3(this.box.center[0], this.box.center[1], this.box.center[2]),
            radius: Math.max(...this.box.size),
            speed: 1
        });
        this.scene.forceRender = true;
    }

    nudge(axis: number[]) {
        this.readInputs();
        const step = Math.max(...this.box.size) * 0.02;
        this.box.center = [
            this.box.center[0] + axis[0] * step,
            this.box.center[1] + axis[1] * step,
            this.box.center[2] + axis[2] * step
        ];
        this.touched = true;
        this.writeInputs();
        this.scene.forceRender = true;
        this.drawPreviewOverlay();
        this.segmentActive = true;
        this.scheduleSegmentUpdate();
    }

    scale(factor: number) {
        this.readInputs();
        this.box.size = [
            this.box.size[0] * factor,
            this.box.size[1] * factor,
            this.box.size[2] * factor
        ];
        this.drawDepth = this.box.size[2];
        this.touched = true;
        this.writeInputs();
        this.scene.forceRender = true;
        this.drawPreviewOverlay();
        this.segmentActive = true;
        this.scheduleSegmentUpdate();
    }

    toggleDrawMode() {
        this.setDrawMode(!this.drawMode);
    }

    setDrawMode(active: boolean) {
        this.drawMode = active;
        if (active) {
            this.setCubeMode(false);
        }
        this.drawing = false;
        this.drawPointerId = null;
        this.drawStart = null;
        if (!active) {
            this.hoverPoint = null;
            this.hoverPending = null;
        }
        this.drawButton.classList[active ? 'add' : 'remove']('active');
        if (this.canvasContainer) {
            this.canvasContainer.style.cursor = active || this.cubeMode ? 'crosshair' : '';
        }
        this.status(active ? 'Draw mode: hover a visible splat point, then click to place a 10cm cube' : 'Draw mode off');
    }

    toggleCubeMode() {
        this.setCubeMode(!this.cubeMode);
    }

    setCubeMode(active: boolean) {
        this.cubeMode = active;
        if (active) {
            this.setDrawMode(false);
        }
        if (!active) {
            this.hoverPoint = null;
            this.hoverPending = null;
        }
        this.cubeButton.classList[active ? 'add' : 'remove']('active');
        if (this.canvasContainer) {
            this.canvasContainer.style.cursor = active || this.drawMode ? 'crosshair' : '';
        }
        this.status(active ? 'Place cube mode: hover a visible splat point, then click to place a 10cm cube' : 'Place cube mode off');
    }

    clearPlacementModes() {
        this.setDrawMode(false);
        this.setCubeMode(false);
        this.hoverPoint = null;
        this.hoverPending = null;
        this.status('Placement modes off');
    }

    setCubeSize() {
        this.readInputs();
        const size = this.cubeSceneSize();
        this.box.size = [size, size, size];
        this.drawDepth = size;
        this.touched = true;
        this.writeInputs();
        this.scene.forceRender = true;
        this.drawPreviewOverlay();
        this.segmentActive = true;
        this.scheduleSegmentUpdate();
    }

    placeCubeAt(position: Vec3) {
        const size = this.cubeSceneSize();
        this.box.center = [position.x, position.y, position.z];
        this.box.size = [size, size, size];
        this.box.euler = [0, 0, 0];
        this.drawDepth = size;
        this.touched = true;
        this.writeInputs();
        this.scene.forceRender = true;
        this.drawPreviewOverlay();
        this.updateHandles();
        this.segmentActive = true;
        this.scheduleSegmentUpdate(0);
        this.status(`${(this.cubeMeters() * 100).toFixed(0)}cm cube placed at clicked splat point`);
    }

    async updateHoverPick(nx: number, ny: number) {
        this.hoverPending = { nx, ny };
        if (this.hoverPicking) return;

        this.hoverPicking = true;
        try {
            while (this.hoverPending) {
                const pending = this.hoverPending;
                this.hoverPending = null;

                const result = await this.scene.camera.intersect(pending.nx, pending.ny);
                if (!this.drawMode && !this.cubeMode) return;

                this.hoverPoint = result ? result.position.clone() : null;
                this.scene.forceRender = true;
                this.updateHandles();
            }
        } finally {
            this.hoverPicking = false;
        }
    }

    installDrawHandlers() {
        const target = this.canvasContainer;
        if (!target) return;

        const isSceneTarget = (eventTarget: EventTarget | null) => {
            return eventTarget === target || eventTarget === this.scene.canvas;
        };
        const isPrimary = (event: PointerEvent) => (event.pointerType === 'mouse' ? event.button === 0 : event.isPrimary);
        const point = (event: PointerEvent) => {
            const rect = target.getBoundingClientRect();
            return {
                x: event.clientX - rect.left,
                y: event.clientY - rect.top,
                nx: (event.clientX - rect.left) / rect.width,
                ny: (event.clientY - rect.top) / rect.height
            };
        };
        const swallow = (event: PointerEvent) => {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
        };

        const pointerdown = async (event: PointerEvent) => {
            if ((!this.drawMode && !this.cubeMode) || !isPrimary(event) || !isSceneTarget(event.target)) return;
            swallow(event);

            const p = point(event);
            this.status('Picking splat point for 10cm cube...');
            const picked = await this.scene.camera.intersect(p.nx, p.ny);
            const result = picked ?? (this.hoverPoint ? { position: this.hoverPoint } : null);
            if (!result) {
                this.status('No selectable splat point under cursor');
                return;
            }
            this.hoverPoint = result.position.clone();
            this.placeCubeAt(result.position);
        };

        const pointermove = (event: PointerEvent) => {
            if ((!this.drawMode && !this.cubeMode) || !isSceneTarget(event.target)) return;
            swallow(event);
            const p = point(event);
            this.updateHoverPick(p.nx, p.ny);
        };

        const finish = (event: PointerEvent) => {
            if (!this.drawMode || !this.drawing || event.pointerId !== this.drawPointerId) return;
            swallow(event);
            if (this.drawStart) {
                const p = point(event);
                this.applyDrawPointer(p.x, p.y);
            }
            this.drawing = false;
            this.drawPointerId = null;
            this.drawStart = null;
            try {
                target.releasePointerCapture(event.pointerId);
            } catch (err) {
                // Some browsers auto-release capture before pointercancel.
            }
            this.status('BBox drawn. Use sliders for precise adjustment.');
        };

        target.addEventListener('pointerdown', pointerdown, true);
        target.addEventListener('pointermove', pointermove, true);
        target.addEventListener('pointerup', finish, true);
        target.addEventListener('pointercancel', finish, true);
        target.addEventListener('pointerleave', () => {
            if (this.drawMode || this.cubeMode) {
                this.hoverPoint = null;
                this.hoverPending = null;
                this.scene.forceRender = true;
                this.updateHandles();
            }
        }, true);
    }

    pointerPoint(event: PointerEvent) {
        const target = this.canvasContainer;
        if (!target) return null;
        const rect = target.getBoundingClientRect();
        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top,
            nx: (event.clientX - rect.left) / rect.width,
            ny: (event.clientY - rect.top) / rect.height
        };
    }

    stopPointerEvent(event: PointerEvent) {
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
    }

    intersectScreenPlane(x: number, y: number, planePoint: Vec3, planeNormal: Vec3, ray: Ray) {
        this.scene.camera.getRay(x, y, ray);
        const denom = planeNormal.dot(ray.direction);
        if (Math.abs(denom) < 1e-6) return null;
        const diff = planePoint.clone().sub(ray.origin);
        const t = planeNormal.dot(diff) / denom;
        if (t <= 0) return null;
        return ray.origin.clone().add(ray.direction.clone().mulScalar(t));
    }

    startHandleDrag(event: PointerEvent, index: number) {
        if (event.pointerType === 'mouse' && event.button !== 0) return;
        this.stopPointerEvent(event);
        this.readInputs();

        const corners = this.corners();
        const opposite = index ^ 7;
        const handle = corners[index];
        const fixed = corners[opposite];
        const target = event.currentTarget as SVGCircleElement;

        target.setPointerCapture(event.pointerId);
        this.handleDrag = {
            index,
            pointerId: event.pointerId,
            oppositeWorld: new Vec3(fixed[0], fixed[1], fixed[2]),
            planePoint: new Vec3(handle[0], handle[1], handle[2]),
            planeNormal: this.scene.camera.forward.clone().normalize(),
            rotation: rotationMatrix(this.box.euler[0], this.box.euler[1], this.box.euler[2])
        };

        this.handleNodes.forEach((node, i) => {
            node.classList[i === index ? 'add' : 'remove']('active');
        });
        this.status(`Dragging corner ${index + 1}. Opposite corner is fixed.`);
    }

    moveHandleDrag(event: PointerEvent) {
        if (!this.handleDrag || event.pointerId !== this.handleDrag.pointerId) return;
        this.stopPointerEvent(event);
        const p = this.pointerPoint(event);
        if (!p) return;

        const hit = this.intersectScreenPlane(
            p.x,
            p.y,
            this.handleDrag.planePoint,
            this.handleDrag.planeNormal,
            this.handleRay
        );
        if (!hit) return;

        this.applyHandlePoint(hit);
    }

    endHandleDrag(event: PointerEvent) {
        if (!this.handleDrag || event.pointerId !== this.handleDrag.pointerId) return;
        this.stopPointerEvent(event);
        this.handleNodes.forEach((node) => {
            node.classList.remove('active');
            try {
                node.releasePointerCapture(event.pointerId);
            } catch (err) {
                // Pointer capture may already be released by the browser.
            }
        });
        this.handleDrag = null;
        this.status('Corner moved. Fine tune with handles, sliders, or numeric inputs.');
    }

    applyHandlePoint(point: Vec3) {
        if (!this.handleDrag) return;

        const delta = point.clone().sub(this.handleDrag.oppositeWorld);
        const localDelta = mat3TransposeVec(this.handleDrag.rotation, [delta.x, delta.y, delta.z]);
        const minSize = Math.max(this.sceneExtent() * 0.002, 0.01);
        const size = [
            Math.max(Math.abs(localDelta[0]), minSize),
            Math.max(Math.abs(localDelta[1]), minSize),
            Math.max(Math.abs(localDelta[2]), minSize)
        ];
        const centerOffset = mat3Vec(this.handleDrag.rotation, [
            localDelta[0] * 0.5,
            localDelta[1] * 0.5,
            localDelta[2] * 0.5
        ]);
        const center = this.handleDrag.oppositeWorld.clone().add(new Vec3(centerOffset[0], centerOffset[1], centerOffset[2]));

        this.box.center = [center.x, center.y, center.z];
        this.box.size = [size[0], size[1], size[2]];
        this.drawDepth = this.box.size[2];
        this.touched = true;
        this.writeInputs();
        this.scene.forceRender = true;
        this.drawPreviewOverlay();
        this.updateHandles();
        this.segmentActive = true;
        this.scheduleSegmentUpdate();
    }

    screenToDrawPlane(x: number, y: number) {
        if (!this.drawStart) return null;
        return this.intersectScreenPlane(x, y, this.drawStart, this.drawForward, this.drawRay);
    }

    applyDrawPointer(x: number, y: number) {
        const hit = this.screenToDrawPlane(x, y);
        if (!hit || !this.drawStart) return;

        const delta = hit.clone().sub(this.drawStart);
        const dx = delta.dot(this.drawRight);
        const dy = delta.dot(this.drawUp);
        const minSize = Math.max(this.sceneExtent() * 0.002, 0.01);
        const depth = Math.max(this.drawDepth, minSize);
        const center = this.drawStart.clone()
        .add(this.drawRight.clone().mulScalar(dx * 0.5))
        .add(this.drawUp.clone().mulScalar(dy * 0.5))
        .add(this.drawForward.clone().mulScalar(depth * 0.5));

        this.box.center = [center.x, center.y, center.z];
        this.box.size = [
            Math.max(Math.abs(dx), minSize),
            Math.max(Math.abs(dy), minSize),
            depth
        ];
        this.box.euler = this.drawEuler;
        this.touched = true;
        this.writeInputs();
        this.scene.forceRender = true;
        this.drawPreviewOverlay();
        this.segmentActive = true;
        this.scheduleSegmentUpdate();
    }

    boxCorners(box: Box) {
        const h = box.size.map(v => Math.max(0.001, v) * 0.5);
        const local = cornerSigns.map(sign => [sign[0] * h[0], sign[1] * h[1], sign[2] * h[2]]);
        const R = rotationMatrix(box.euler[0], box.euler[1], box.euler[2]);
        return local.map((p) => {
            const r = mat3Vec(R, p);
            return [
                box.center[0] + r[0],
                box.center[1] + r[1],
                box.center[2] + r[2]
            ];
        });
    }

    corners() {
        this.readInputs();
        return this.boxCorners(this.box);
    }

    projectionCorners() {
        const corners = this.corners();
        return corners.map(p => this.toProjectionPoint(p));
    }

    toProjectionPoint(p: number[]) {
        const splat = this.activeSplat();
        const useSplatTransform = this.useSplatTransformInput?.checked ?? true;
        if (useSplatTransform && this.projectionMode !== 'splat-world' && splat) {
            this.projectionWorldToSplat.invert(splat.worldTransform);
            this.projectionPoint.set(p[0], p[1], p[2]);
            this.projectionWorldToSplat.transformPoint(this.projectionPoint, this.projectionPoint);
            p = [this.projectionPoint.x, this.projectionPoint.y, this.projectionPoint.z];
        }

        if (this.projectionMode === 'local-applied' && this.appliedTransform) {
            return transformRowMajor(this.appliedTransform, p);
        }
        if (this.projectionMode === 'local-applied-inverse' && this.appliedTransformInverse) {
            return transformRowMajor(this.appliedTransformInverse, p);
        }
        if (this.projectionMode === 'train-to-data' && this.dataparserTransformInverse) {
            const scale = Math.max(this.dataparserScale, 1e-9);
            return transformRowMajor(this.dataparserTransformInverse, [
                p[0] / scale,
                p[1] / scale,
                p[2] / scale
            ]);
        }
        return p;
    }

    draw() {
        if (!this.visible) return;
        this.readInputs(false);
        this.boxes.forEach((box, index) => {
            const corners = this.boxCorners(box).map(p => new Vec3(p[0], p[1], p[2]));
            const color = index === this.activeBoxIndex ? activeLineColor : lineColor;
            edges.forEach(([a, b]) => {
                this.scene.app.drawLine(corners[a], corners[b], this.drawing && index === this.activeBoxIndex ? activeLineColor : color, false, this.scene.gizmoLayer);
            });
        });

        if ((this.drawMode || this.cubeMode) && this.hoverPoint) {
            const size = Math.max(this.cubeSceneSize() * 0.5, this.sceneExtent() * 0.006);
            const transform = this.scene.camera.worldTransform;
            const axes = [
                transform.getX().normalize(),
                transform.getY().normalize(),
                this.scene.camera.forward.clone().normalize()
            ];
            axes.forEach((axis) => {
                const a = this.hoverPoint.clone().add(axis.clone().mulScalar(size));
                const b = this.hoverPoint.clone().sub(axis.clone().mulScalar(size));
                this.scene.app.drawLine(a, b, activeLineColor, false, this.scene.gizmoLayer);
            });
        }
    }

    updateHandles() {
        if (!this.handleSvg || !this.canvasContainer) return;

        const width = this.canvasContainer.clientWidth;
        const height = this.canvasContainer.clientHeight;
        this.handleSvg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        this.handleSvg.setAttribute('width', `${width}`);
        this.handleSvg.setAttribute('height', `${height}`);

        const projected = new Vec3();
        if (this.hoverNode) {
            if ((this.drawMode || this.cubeMode) && this.hoverPoint) {
                this.scene.camera.worldToScreen(this.hoverPoint, projected);
                const x = projected.x * width;
                const y = projected.y * height;
                const visible = Number.isFinite(x) &&
                    Number.isFinite(y) &&
                    x >= -24 &&
                    y >= -24 &&
                    x <= width + 24 &&
                    y <= height + 24;
                this.hoverNode.setAttribute('cx', `${x}`);
                this.hoverNode.setAttribute('cy', `${y}`);
                this.hoverNode.setAttribute('visibility', visible ? 'visible' : 'hidden');
            } else {
                this.hoverNode.setAttribute('visibility', 'hidden');
            }
        }

        const corners = this.corners();
        corners.forEach((corner, i) => {
            const node = this.handleNodes[i];
            this.scene.camera.worldToScreen(new Vec3(corner[0], corner[1], corner[2]), projected);
            const x = projected.x * width;
            const y = projected.y * height;
            const visible = Number.isFinite(x) &&
                Number.isFinite(y) &&
                x >= -20 &&
                y >= -20 &&
                x <= width + 20 &&
                y <= height + 20;
            node.setAttribute('cx', `${x}`);
            node.setAttribute('cy', `${y}`);
            node.setAttribute('r', this.handleDrag?.index === i ? '7' : '5');
            node.setAttribute('visibility', visible ? 'visible' : 'hidden');
        });
    }

    payload() {
        this.readInputs();
        return {
            description: '3D box labels authored in SuperSplat native app',
            coordinate_system: 'Nerfstudio / exported Gaussian splat world coordinates',
            units: 'meters',
            metric_scale: {
                meters_per_scene_unit: this.metersPerSceneUnit,
                scene_units_per_meter: 1 / Math.max(this.metersPerSceneUnit, 1e-9)
            },
            boxes: this.boxes.map((box, index) => {
                const metric = this.boxMetric(box);
                return {
                    id: box.id,
                    label: box.label,
                    active: index === this.activeBoxIndex,
                    center: metric.center,
                    size: metric.size,
                    center_scene: box.center,
                    size_scene: box.size,
                    euler_xyz_degrees: box.euler
                };
            })
        };
    }

    export3D() {
        downloadText('boxes_3d.json', JSON.stringify(this.payload(), null, 2));
    }

    async loadTransforms(url: string) {
        try {
            this.transformsUrl = new URL(url, window.location.href);
            const meta = await (await fetch(this.transformsUrl)).json();
            const width = Number(meta.w);
            const height = Number(meta.h);
            const fx = Number(meta.fl_x);
            const fy = Number(meta.fl_y);
            const cx = Number(meta.cx);
            const cy = Number(meta.cy);
            const k1 = Number(meta.k1 ?? 0);
            const k2 = Number(meta.k2 ?? 0);
            const k3 = Number(meta.k3 ?? 0);
            const p1 = Number(meta.p1 ?? 0);
            const p2 = Number(meta.p2 ?? 0);
            if (Array.isArray(meta.applied_transform)) {
                this.appliedTransform = meta.applied_transform.flat();
                this.appliedTransformInverse = invertRigid(this.appliedTransform);
            }
            this.frames = meta.frames.map((frame: any) => ({
                file_path: frame.file_path,
                width: Number(frame.w ?? width),
                height: Number(frame.h ?? height),
                fx: Number(frame.fl_x ?? fx),
                fy: Number(frame.fl_y ?? fy),
                cx: Number(frame.cx ?? cx),
                cy: Number(frame.cy ?? cy),
                k1: Number(frame.k1 ?? k1),
                k2: Number(frame.k2 ?? k2),
                k3: Number(frame.k3 ?? k3),
                p1: Number(frame.p1 ?? p1),
                p2: Number(frame.p2 ?? p2),
                transform_matrix: frame.transform_matrix
            }));
            this.frameIndex = 0;
            this.frameSlider.min = '0';
            this.frameSlider.max = `${Math.max(this.frames.length - 1, 0)}`;
            this.frameSlider.value = '0';
            this.status(`transforms loaded: ${this.frames.length} frames`);
            this.updatePreview();
        } catch (err) {
            this.status(`transforms error: ${err instanceof Error ? err.message : String(err)}`);
            this.previewStatus.textContent = `transforms error: ${err instanceof Error ? err.message : String(err)}`;
        }
    }

    async loadMetricScale(url: string) {
        try {
            const metricUrl = new URL(url, window.location.href);
            const meta = await (await fetch(metricUrl)).json();
            const metersPerSceneUnit = Number(meta.meters_per_scene_unit ?? meta.metersPerSceneUnit);
            const sceneUnitsPerMeter = Number(meta.scene_units_per_meter ?? meta.sceneUnitsPerMeter);
            if (Number.isFinite(metersPerSceneUnit) && metersPerSceneUnit > 0) {
                this.metersPerSceneUnit = metersPerSceneUnit;
            } else if (Number.isFinite(sceneUnitsPerMeter) && sceneUnitsPerMeter > 0) {
                this.metersPerSceneUnit = 1 / sceneUnitsPerMeter;
            } else {
                throw new Error('missing meters_per_scene_unit or scene_units_per_meter');
            }
            this.writeMetricInputs();
            this.status(`metric scale loaded: 1 scene unit = ${this.metersPerSceneUnit.toFixed(4)}m`);
        } catch (err) {
            this.status(`metric scale error: ${err instanceof Error ? err.message : String(err)}`);
        }
    }

    async loadDataparserTransforms(url: string) {
        try {
            const dataparserUrl = new URL(url, window.location.href);
            const meta = await (await fetch(dataparserUrl)).json();
            if (!Array.isArray(meta.transform)) {
                throw new Error('missing transform');
            }

            this.dataparserTransform = meta.transform.flat();
            this.dataparserTransformInverse = invertRigid(this.dataparserTransform);
            const scale = Number(meta.scale ?? 1);
            this.dataparserScale = Number.isFinite(scale) && scale > 0 ? scale : 1;
            this.projectionMode = 'train-to-data';
            if (this.projectionModeSelect) {
                this.projectionModeSelect.value = this.projectionMode;
            }
            this.drawPreviewOverlay();
            this.status(`dataparser transform loaded: scale ${this.dataparserScale.toFixed(6)}`);
        } catch (err) {
            this.status(`dataparser transform error: ${err instanceof Error ? err.message : String(err)}`);
        }
    }

    frameUrl(frame: Frame) {
        if (!this.transformsUrl) return frame.file_path;
        return new URL(frame.file_path, this.transformsUrl).toString();
    }

    stepFrame(delta: number) {
        if (!this.frames.length) return;
        this.frameIndex = clamp(this.frameIndex + delta, 0, this.frames.length - 1);
        this.frameSlider.value = `${this.frameIndex}`;
        this.updatePreview();
    }

    updatePreview() {
        if (!this.frames.length) {
            this.frameLabel.textContent = 'no frames';
            this.previewStatus.textContent = 'load transforms to preview 2D labels';
            return;
        }

        const frame = this.frames[this.frameIndex];
        this.frameLabel.textContent = `${this.frameIndex + 1} / ${this.frames.length}`;
        this.frameSlider.value = `${this.frameIndex}`;
        const src = this.frameUrl(frame);
        if (this.previewImage.src !== src) {
            this.previewImage.src = src;
        }
        this.drawPreviewOverlay();
    }

    drawPreviewOverlay() {
        if (!this.frames.length || !this.previewCanvas) return;
        const frame = this.frames[this.frameIndex];
        const canvas = this.previewCanvas;
        canvas.width = frame.width;
        canvas.height = frame.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const projected = this.project(frame);
        if (!projected) {
            this.previewStatus.textContent = `${frame.file_path}: bbox not visible`;
            return;
        }

        const segmentPoints = this.drawProjectedSegmentPoints(ctx, frame);
        const [x, y, w, h] = projected.bbox;
        ctx.lineWidth = Math.max(3, Math.round(frame.width / 500));
        ctx.strokeStyle = 'rgba(255, 30, 20, 0.95)';
        ctx.fillStyle = 'rgba(255, 30, 20, 0.12)';
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(x, Math.max(0, y - 24), Math.min(220, frame.width - x), 22);
        ctx.fillStyle = '#ffffff';
        ctx.font = `${Math.max(14, Math.round(frame.width / 90))}px sans-serif`;
        ctx.fillText(`${this.box.label} ${this.box.id}`, x + 5, Math.max(16, y - 8));
        const transformStatus = this.useSplatTransformInput?.checked ? 'TRANSFORM on' : 'TRANSFORM off';
        this.previewStatus.textContent = `${frame.file_path}: ${projected.visibleCorners}/8 corners, ${formatCount(segmentPoints)} segment pts, ${this.projectionMode}, ${transformStatus}`;
    }

    projectPoint(frame: Frame, p: number[]) {
        const w2c = invertRigid(frame.transform_matrix.flat());
        const cam = transformRowMajor(w2c, p);
        const depth = -cam[2];
        if (depth <= 1e-6) return null;

        const x = cam[0] / depth;
        const y = -cam[1] / depth;
        const r2 = x * x + y * y;
        const radial = 1 + frame.k1 * r2 + frame.k2 * r2 * r2 + frame.k3 * r2 * r2 * r2;
        const xd = x * radial + 2 * frame.p1 * x * y + frame.p2 * (r2 + 2 * x * x);
        const yd = y * radial + frame.p1 * (r2 + 2 * y * y) + 2 * frame.p2 * x * y;
        return [
            frame.fx * xd + frame.cx,
            frame.fy * yd + frame.cy
        ];
    }

    drawProjectedSegmentPoints(ctx: CanvasRenderingContext2D, frame: Frame) {
        if (!this.showSegmentProjectionInput?.checked) return 0;

        const splat = this.activeSplat();
        if (!splat) return 0;

        const state = splat.splatData.getProp('state') as Uint8Array;
        const centers = splat.entity.gsplat.instance.sorter.centers as Float32Array;
        const world = splat.worldTransform;
        const point = this.segmentPoint;
        const total = splat.splatData.numSplats;
        const maxPoints = 3000;
        const stride = Math.max(1, Math.floor(Math.max(this.segmentLastCount, 1) / maxPoints));
        let selectedSeen = 0;
        let drawn = 0;

        ctx.save();
        ctx.fillStyle = 'rgba(255, 235, 0, 0.82)';
        for (let i = 0; i < total; i++) {
            if ((state[i] & State.selected) === 0 || (state[i] & (State.locked | State.deleted)) !== 0) {
                continue;
            }

            selectedSeen++;
            if (selectedSeen % stride !== 0) {
                continue;
            }

            const offset = i * 3;
            point.set(centers[offset], centers[offset + 1], centers[offset + 2]);
            world.transformPoint(point, point);
            const projected = this.projectPoint(frame, this.toProjectionPoint([point.x, point.y, point.z]));
            if (!projected) continue;

            const [x, y] = projected;
            if (x < 0 || y < 0 || x >= frame.width || y >= frame.height) continue;
            ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
            drawn++;
            if (drawn >= maxPoints) break;
        }
        ctx.restore();
        return drawn;
    }

    project(frame: Frame): ProjectedBox | null {
        const corners = this.projectionCorners();
        const uv = corners.map((p) => {
            return this.projectPoint(frame, p);
        });
        const valid = uv.filter(Boolean) as number[][];
        if (!valid.length) return null;
        let x0 = Math.min(...valid.map(p => p[0]));
        let y0 = Math.min(...valid.map(p => p[1]));
        let x1 = Math.max(...valid.map(p => p[0]));
        let y1 = Math.max(...valid.map(p => p[1]));
        x0 = clamp(x0, 0, frame.width - 1);
        y0 = clamp(y0, 0, frame.height - 1);
        x1 = clamp(x1, 0, frame.width - 1);
        y1 = clamp(y1, 0, frame.height - 1);
        if (x1 <= x0 || y1 <= y0) return null;
        return {
            bbox: [x0, y0, x1 - x0, y1 - y0],
            visibleCorners: valid.length
        };
    }

    currentCaptureSize() {
        const width = Math.max(1, Math.round(this.scene.canvas.width || this.scene.canvas.clientWidth * window.devicePixelRatio));
        const height = Math.max(1, Math.round(this.scene.canvas.height || this.scene.canvas.clientHeight * window.devicePixelRatio));
        return { width, height };
    }

    currentViewBox(width: number, height: number): ProjectedBox | null {
        return this.currentViewBoxForBox(width, height, this.box);
    }

    currentViewBoxForBox(width: number, height: number, box: Box): ProjectedBox | null {
        const projected = new Vec3();
        const uv = this.boxCorners(box).map((corner) => {
            this.scene.camera.worldToScreen(new Vec3(corner[0], corner[1], corner[2]), projected);
            const x = projected.x * width;
            const y = projected.y * height;
            if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(projected.z)) {
                return null;
            }
            return [x, y];
        }).filter(Boolean) as number[][];

        if (!uv.length) return null;
        let x0 = Math.min(...uv.map(p => p[0]));
        let y0 = Math.min(...uv.map(p => p[1]));
        let x1 = Math.max(...uv.map(p => p[0]));
        let y1 = Math.max(...uv.map(p => p[1]));
        x0 = clamp(x0, 0, width - 1);
        y0 = clamp(y0, 0, height - 1);
        x1 = clamp(x1, 0, width - 1);
        y1 = clamp(y1, 0, height - 1);
        if (x1 <= x0 || y1 <= y0) return null;
        return {
            bbox: [x0, y0, x1 - x0, y1 - y0],
            visibleCorners: uv.length
        };
    }

    binaryMaskFromSelectedTarget(rawMask: Uint8Array, width: number, height: number) {
        const binary = new Uint8ClampedArray(width * height * 4);
        let pixels = 0;
        let x0 = width;
        let y0 = height;
        let x1 = -1;
        let y1 = -1;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const offset = (y * width + x) * 4;
                const hit = rawMask[offset + 3] > 0;
                binary[offset + 0] = hit ? 255 : 0;
                binary[offset + 1] = hit ? 255 : 0;
                binary[offset + 2] = hit ? 255 : 0;
                binary[offset + 3] = 255;
                if (hit) {
                    pixels++;
                    x0 = Math.min(x0, x);
                    y0 = Math.min(y0, y);
                    x1 = Math.max(x1, x);
                    y1 = Math.max(y1, y);
                }
            }
        }

        return {
            pixels: binary,
            area: pixels,
            bbox: pixels > 0 ? [x0, y0, x1 - x0 + 1, y1 - y0 + 1] : null
        };
    }

    maskRadiusPx() {
        const value = Number(this.maskRadiusInput?.value ?? defaultMaskRadiusPx);
        return clamp(Number.isFinite(value) ? Math.round(value) : defaultMaskRadiusPx, 1, 32);
    }

    pointInBox(point: Vec3, box: Box) {
        const hx = Math.max(box.size[0], 0.001) * 0.5;
        const hy = Math.max(box.size[1], 0.001) * 0.5;
        const hz = Math.max(box.size[2], 0.001) * 0.5;
        const r = rotationMatrix(box.euler[0], box.euler[1], box.euler[2]);
        const dx = point.x - box.center[0];
        const dy = point.y - box.center[1];
        const dz = point.z - box.center[2];
        if (!Number.isFinite(dx) || !Number.isFinite(dy) || !Number.isFinite(dz)) {
            return false;
        }

        const lx = r[0] * dx + r[3] * dy + r[6] * dz;
        const ly = r[1] * dx + r[4] * dy + r[7] * dz;
        const lz = r[2] * dx + r[5] * dy + r[8] * dz;
        return Math.abs(lx) <= hx && Math.abs(ly) <= hy && Math.abs(lz) <= hz;
    }

    boxCenterMask(width: number, height: number, radius: number, box: Box) {
        const splat = this.activeSplat();
        const binary = new Uint8ClampedArray(width * height * 4);
        if (!splat) {
            return {
                pixels: binary,
                area: 0,
                bbox: null as number[] | null,
                projectedPoints: 0
            };
        }

        const centers = splat.entity.gsplat.instance.sorter.centers as Float32Array;
        const state = splat.splatData.getProp('state') as Uint8Array;
        const world = splat.worldTransform;
        const point = this.segmentPoint;
        const projected = new Vec3();
        const r2 = radius * radius;
        let projectedPoints = 0;
        let pixels = 0;
        let x0 = width;
        let y0 = height;
        let x1 = -1;
        let y1 = -1;

        const mark = (x: number, y: number) => {
            const offset = (y * width + x) * 4;
            if (binary[offset] !== 0) return;
            binary[offset + 0] = 255;
            binary[offset + 1] = 255;
            binary[offset + 2] = 255;
            binary[offset + 3] = 255;
            pixels++;
            x0 = Math.min(x0, x);
            y0 = Math.min(y0, y);
            x1 = Math.max(x1, x);
            y1 = Math.max(y1, y);
        };

        for (let i = 0; i < splat.splatData.numSplats; i++) {
            if (state && (state[i] & (State.locked | State.deleted)) !== 0) {
                continue;
            }
            const offset = i * 3;
            point.set(centers[offset], centers[offset + 1], centers[offset + 2]);
            world.transformPoint(point, point);
            if (!this.pointInBox(point, box)) {
                continue;
            }

            this.scene.camera.worldToScreen(point, projected);
            const cx = Math.round(projected.x * width);
            const cy = Math.round(projected.y * height);
            if (!Number.isFinite(cx) || !Number.isFinite(cy) || cx < -radius || cy < -radius || cx >= width + radius || cy >= height + radius) {
                continue;
            }

            projectedPoints++;
            const minX = Math.max(0, cx - radius);
            const maxX = Math.min(width - 1, cx + radius);
            const minY = Math.max(0, cy - radius);
            const maxY = Math.min(height - 1, cy + radius);
            for (let y = minY; y <= maxY; y++) {
                for (let x = minX; x <= maxX; x++) {
                    const dx = x - cx;
                    const dy = y - cy;
                    if (dx * dx + dy * dy <= r2) {
                        mark(x, y);
                    }
                }
            }
        }

        for (let i = 3; i < binary.length; i += 4) {
            if (binary[i] === 0) binary[i] = 255;
        }

        return {
            pixels: binary,
            area: pixels,
            bbox: pixels > 0 ? [x0, y0, x1 - x0 + 1, y1 - y0 + 1] : null,
            projectedPoints
        };
    }

    selectedCenterMask(width: number, height: number, radius: number) {
        return this.boxCenterMask(width, height, radius, this.box);
    }

    overlayMask(rgb: Uint8Array, maskPixels: Uint8Array | Uint8ClampedArray, width: number, height: number) {
        const overlay = new Uint8ClampedArray(width * height * 4);
        for (let i = 0; i < width * height; i++) {
            const offset = i * 4;
            const alpha = maskPixels[offset] > 0 ? 0.58 : 0;
            overlay[offset + 0] = Math.round(rgb[offset + 0] * (1 - alpha) + 255 * alpha);
            overlay[offset + 1] = Math.round(rgb[offset + 1] * (1 - alpha) + 230 * alpha);
            overlay[offset + 2] = Math.round(rgb[offset + 2] * (1 - alpha));
            overlay[offset + 3] = 255;
        }
        return overlay;
    }

    selectedCount() {
        const splat = this.activeSplat();
        if (!splat) return 0;
        const state = splat.splatData.getProp('state') as Uint8Array;
        let count = 0;
        for (let i = 0; i < state.length; i++) {
            if ((state[i] & State.selected) !== 0 && (state[i] & (State.locked | State.deleted)) === 0) {
                count++;
            }
        }
        return count;
    }

    async writeBlobFile(filename: string, blob: Blob, dirHandle: any = this.outputDirHandle) {
        if (dirHandle?.getFileHandle) {
            const fileHandle = await dirHandle.getFileHandle(filename, { create: true });
            const writable = await fileHandle.createWritable();
            await writable.write(blob);
            await writable.close();
            return;
        }
        downloadBlob(filename, blob);
    }

    async writeTextFile(filename: string, text: string, dirHandle: any = this.outputDirHandle) {
        const blob = new Blob([text], { type: 'application/json' });
        await this.writeBlobFile(filename, blob, dirHandle);
    }

    async chooseOutputFolder() {
        const picker = (window as any).showDirectoryPicker;
        if (!picker) {
            this.outputDirHandle = null;
            this.outputStatus.textContent = 'output: folder picker unsupported, using downloads';
            return null;
        }
        this.outputDirHandle = await picker.call(window, { mode: 'readwrite' });
        this.outputStatus.textContent = `output: ${this.outputDirHandle.name}`;
        return this.outputDirHandle;
    }

    setSpiralCenterFromBox() {
        this.readInputs();
        const values = {
            'cap-cx': this.sceneToMeters(this.box.center[0]),
            'cap-cy': this.sceneToMeters(this.box.center[1]),
            'cap-cz': this.sceneToMeters(this.box.center[2])
        };
        Object.entries(values).forEach(([id, value]) => {
            const input = this.spiralCenterInputs.get(id);
            if (input) input.value = value.toFixed(4);
        });
        this.segmentMessage(`capture center set from ${this.box.id}`);
    }

    writeSpiralInputsFromSceneValues(center: Vec3 | null, radii: number[] | null) {
        if (center) {
            const current = {
                'cap-cx': center.x,
                'cap-cy': center.y,
                'cap-cz': center.z
            };
            Object.entries(current).forEach(([id, sceneValue]) => {
                const input = this.spiralCenterInputs.get(id);
                if (!input || !Number.isFinite(sceneValue)) return;
                input.value = this.sceneToMeters(sceneValue).toFixed(4);
            });
        }
        if (radii?.length && this.spiralRadiiInput) {
            this.spiralRadiiInput.value = radii.map(radius => this.sceneToMeters(radius).toFixed(4)).join(',');
        }
    }

    spiralCenter() {
        const get = (id: string) => Number(this.spiralCenterInputs.get(id)?.value ?? 0);
        return new Vec3(
            this.metersToScene(get('cap-cx')),
            this.metersToScene(get('cap-cy')),
            this.metersToScene(get('cap-cz'))
        );
    }

    spiralRadii() {
        const values = this.spiralRadiiInput.value
        .split(/[,\s]+/)
        .map(value => this.metersToScene(Number(value)))
        .filter(value => Number.isFinite(value) && value > 0);
        if (!values.length) {
            throw new Error('missing radii list');
        }
        return values;
    }

    spiralThetaStep() {
        const value = Number(this.spiralThetaInput.value);
        if (!Number.isFinite(value) || value <= 0 || value > 180) {
            throw new Error('d theta must be in (0, 180]');
        }
        return value;
    }

    spiralPrefix(viewIndex: number, radius: number, theta: number) {
        const fixed = (value: number) => value.toFixed(3).replace('-', 'm').replace('.', 'p');
        return `spiral_${String(viewIndex).padStart(4, '0')}_r${fixed(radius)}_t${fixed(theta)}`;
    }

    async runSpiralExport() {
        if (this.spiralRunning) {
            this.spiralRunning = false;
            this.segmentMessage('spiral export: stopping after current view...');
            return;
        }

        this.readInputs();
        if (!this.outputDirHandle && (window as any).showDirectoryPicker) {
            await this.chooseOutputFolder();
        }

        const dirHandle = this.outputDirHandle;
        const center = this.spiralCenter();
        const radii = this.spiralRadii();
        const centerMeters = [this.sceneToMeters(center.x), this.sceneToMeters(center.y), this.sceneToMeters(center.z)];
        const radiiMeters = radii.map(radius => this.sceneToMeters(radius));
        const dTheta = this.spiralThetaStep();
        const cameraPos = this.scene.camera.position.clone();
        const heightOffset = cameraPos.y - center.y;
        const startAngle = Math.atan2(cameraPos.z - center.z, cameraPos.x - center.x);
        const manifest: any = {
            description: 'Spiral renderer-view RGB/mask export from SuperSplat labeler',
            units: 'meters',
            metric_scale: {
                meters_per_scene_unit: this.metersPerSceneUnit,
                scene_units_per_meter: 1 / Math.max(this.metersPerSceneUnit, 1e-9)
            },
            center: centerMeters,
            center_scene: [center.x, center.y, center.z],
            radii: radiiMeters,
            radii_scene: radii,
            dtheta_degrees: dTheta,
            height_offset: this.sceneToMeters(heightOffset),
            height_offset_scene: heightOffset,
            boxes: this.payload().boxes,
            views: []
        };

        this.spiralRunning = true;
        let viewIndex = 0;
        const total = radii.length * Math.ceil(360 / dTheta);
        try {
            for (const radius of radii) {
                for (let theta = 0; theta < 360 - 1e-6; theta += dTheta) {
                    if (!this.spiralRunning) break;

                    const angle = startAngle + deg(theta);
                    const position = new Vec3(
                        center.x + radius * Math.cos(angle),
                        center.y + heightOffset,
                        center.z + radius * Math.sin(angle)
                    );
                    this.scene.camera.setPose(position, center, 0);
                    this.scene.forceRender = true;

                    const radiusMeters = this.sceneToMeters(radius);
                    const prefix = this.spiralPrefix(viewIndex, radiusMeters, theta);
                    this.segmentMessage(`spiral export: ${viewIndex + 1}/${total}, r=${radiusMeters.toFixed(3)}m, theta=${theta.toFixed(1)}`);
                    const result = await this.captureViewSet(prefix, dirHandle);
                    manifest.views.push({
                        index: viewIndex,
                        prefix,
                        radius: radiusMeters,
                        radius_scene: radius,
                        theta_degrees: theta,
                        camera_position: [this.sceneToMeters(position.x), this.sceneToMeters(position.y), this.sceneToMeters(position.z)],
                        camera_position_scene: [position.x, position.y, position.z],
                        target: centerMeters,
                        target_scene: [center.x, center.y, center.z],
                        result
                    });
                    viewIndex++;

                    await new Promise((resolve) => {
                        setTimeout(resolve, 0);
                    });
                }
                if (!this.spiralRunning) break;
            }
        } finally {
            this.spiralRunning = false;
            await this.writeTextFile('spiral_manifest.json', JSON.stringify(manifest, null, 2), dirHandle);
            this.segmentMessage(`spiral export done: ${formatCount(viewIndex)} views`);
        }
    }

    currentViewPayload(width: number, height: number, masks: ViewMask[] = [], maskRadius = this.maskRadiusPx(), prefix = 'current') {
        this.readInputs();
        const camera = this.scene.camera.camera;
        const cameraTransform = this.scene.camera.mainCamera.getWorldTransform();
        return {
            description: 'Current SuperSplat renderer-view annotation. Coordinates are screen pixels in this captured view.',
            units: 'meters',
            metric_scale: {
                meters_per_scene_unit: this.metersPerSceneUnit,
                scene_units_per_meter: 1 / Math.max(this.metersPerSceneUnit, 1e-9)
            },
            image: {
                width,
                height,
                rgb_file: `${prefix}_rgb.png`,
                overlay_file: `${prefix}_overlay.png`
            },
            camera: {
                world_matrix: Array.from(cameraTransform.data),
                view_matrix: Array.from(camera.viewMatrix.data),
                projection_matrix: Array.from(camera.projectionMatrix.data),
                fov: camera.fov,
                near: camera.nearClip,
                far: camera.farClip,
                orthographic: this.scene.camera.ortho
            },
            annotations: this.boxes.map((box, index) => {
                const active = index === this.activeBoxIndex;
                const mask = masks[index] ?? null;
                const viewBox = this.currentViewBoxForBox(width, height, box);
                const metric = this.boxMetric(box);
                return {
                    id: box.id,
                    label: box.label,
                    active,
                    type: 'selected_gaussian_center_disk_mask',
                    mask_file: mask?.maskFile ?? `${prefix}_${safeFilePart(box.id)}_mask.png`,
                    mask_radius_px: maskRadius,
                    mask_area_pixels: mask?.area ?? 0,
                    mask_bbox_xywh: mask?.bbox ?? null,
                    projected_box_xywh: viewBox?.bbox ?? null,
                    projected_box_visible_corners: viewBox?.visibleCorners ?? 0,
                    projected_selected_centers: mask?.projectedPoints ?? 0,
                    box3d: {
                        center: metric.center,
                        size: metric.size,
                        center_scene: box.center,
                        size_scene: box.size,
                        euler_xyz_degrees: box.euler
                    }
                };
            })
        };
    }

    exportCurrentViewJson() {
        const { width, height } = this.currentCaptureSize();
        downloadText('current_view_annotation.json', JSON.stringify(this.currentViewPayload(width, height), null, 2));
        this.segmentMessage(`view json exported: ${width}x${height}`);
    }

    async captureViewSet(prefix: string, dirHandle: any = this.outputDirHandle) {
        const splat = this.activeSplat();
        if (!splat) {
            this.segmentMessage('view export: no splat loaded');
            return;
        }

        this.readInputs();
        this.segmentActive = true;

        const selectedClr = this.events.invoke('selectedClr') as Color;
        const savedSelectedClr = new Color(selectedClr.r, selectedClr.g, selectedClr.b, selectedClr.a);
        const captureSelectedClr = new Color(savedSelectedClr.r, savedSelectedClr.g, savedSelectedClr.b, 0);
        const { width, height } = this.currentCaptureSize();

        this.segmentMessage(`view export: rendering ${width}x${height} RGB and mask...`);
        try {
            this.events.fire('setSelectedClr', captureSelectedClr);
            const rgb = await this.events.invoke('render.offscreen', width, height) as Uint8Array;
            const radius = this.maskRadiusPx();
            const masks: ViewMask[] = this.boxes.map((box, index) => {
                const mask = this.boxCenterMask(width, height, radius, box);
                return {
                    box,
                    maskFile: `${prefix}_${String(index + 1).padStart(2, '0')}_${safeFilePart(box.id)}_mask.png`,
                    area: mask.area,
                    bbox: mask.bbox,
                    projectedPoints: mask.projectedPoints,
                    pixels: mask.pixels
                };
            });
            const unionMask = new Uint8ClampedArray(width * height * 4);
            masks.forEach((mask) => {
                for (let i = 0; i < unionMask.length; i += 4) {
                    if (mask.pixels[i] > 0) {
                        unionMask[i + 0] = 255;
                        unionMask[i + 1] = 255;
                        unionMask[i + 2] = 255;
                    }
                    unionMask[i + 3] = 255;
                }
            });
            const overlay = this.overlayMask(rgb, unionMask, width, height);

            const [rgbBlob, overlayBlob, ...maskBlobs] = await Promise.all([
                rgbaToPngBlob(rgb, width, height),
                rgbaToPngBlob(overlay, width, height),
                ...masks.map(mask => rgbaToPngBlob(mask.pixels, width, height))
            ]);

            await this.writeBlobFile(`${prefix}_rgb.png`, rgbBlob, dirHandle);
            await this.writeBlobFile(`${prefix}_overlay.png`, overlayBlob, dirHandle);
            for (let i = 0; i < masks.length; i++) {
                await this.writeBlobFile(masks[i].maskFile, maskBlobs[i], dirHandle);
            }
            await this.writeTextFile(`${prefix}_annotation.json`, JSON.stringify(this.currentViewPayload(width, height, masks, radius, prefix), null, 2), dirHandle);

            const totalCenters = masks.reduce((sum, mask) => sum + mask.projectedPoints, 0);
            this.segmentMessage(`view export: ${masks.length} masks saved, ${formatCount(totalCenters)} centers`);
            return {
                prefix,
                width,
                height,
                masks: masks.map(mask => ({
                    id: mask.box.id,
                    maskFile: mask.maskFile,
                    area: mask.area,
                    bbox: mask.bbox,
                    projectedPoints: mask.projectedPoints
                }))
            };
        } finally {
            this.events.fire('setSelectedClr', savedSelectedClr);
            this.scene.forceRender = true;
        }
    }

    async exportCurrentViewSet() {
        await this.captureViewSet('current');
    }

    export2D() {
        if (!this.frames.length) {
            this.status('no transforms loaded');
            return;
        }

        this.readInputs();
        const lines = this.frames.map((frame) => {
            const projected = this.project(frame);
            const metric = this.boxMetric(this.box);
            return JSON.stringify({
                file_name: frame.file_path,
                width: frame.width,
                height: frame.height,
                annotations: projected ? [{
                    box3d_id: this.box.id,
                    label: this.box.label,
                    bbox: projected.bbox,
                    visible_corners: projected.visibleCorners,
                    box3d: {
                        center: metric.center,
                        size: metric.size,
                        center_scene: this.box.center,
                        size_scene: this.box.size,
                        euler_xyz_degrees: this.box.euler
                    },
                    metric_scale: {
                        meters_per_scene_unit: this.metersPerSceneUnit,
                        scene_units_per_meter: 1 / Math.max(this.metersPerSceneUnit, 1e-9)
                    }
                }] : []
            });
        });
        downloadText('annotations_2d.jsonl', `${lines.join('\n')}\n`);
    }
}

export { BBoxAnnotation };
