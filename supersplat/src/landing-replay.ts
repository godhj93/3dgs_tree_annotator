import { Vec3 } from 'playcanvas';

import { ElementType } from './element';
import { drawLandingOutline, replayGrid, saveLandingFigures } from './landing-replay-figures';
import type { ReplayBox } from './landing-replay-figures';
import type { LiveView } from './live-view';
import type { Scene } from './scene';
import type { Splat } from './splat';

export const landingReplayModel = '/exports/dji_0967_supersplat/dji_0967_splat.ply';
const modelName = landingReplayModel.split('/').at(-1);
const resultPath = '/outputs/single_frame_perch_261122/da2_bayes_sequence/segment_tracking/landing_replay_5cm.json';
const poseRoot = '/data/tree_seg_3dgs_approach_6m_10000/test';
const defaultFrame = 'approach_test_003_0056';
const boxColor = '#55f28e';

type Point = [number, number, number];
type ReplayFrame = { frame_id: string; stream_id: string; sequence: number; proposals: ReplayBox[] };
type ReplayResults = { minimum_diameter_cm: number; frames: ReplayFrame[] };
type FramePose = {
    frame_id: string;
    meters_per_scene_unit: number;
    position: Point;
    projection_matrix: number[][];
    approach: { target_world_m: Point };
};

const point = (p: unknown): p is Point =>
    Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);

const readResponse = async (path: string, signal?: AbortSignal) => {
    const response = await fetch(path, { signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response;
};

const button = (text: string, action: () => void) => {
    const element = document.createElement('button');
    element.type = 'button';
    element.textContent = text;
    element.addEventListener('click', action);
    return element;
};

export class LandingReplay {
    private readonly panel = document.createElement('section');
    private readonly canvas = document.createElement('canvas');
    private readonly ctx = this.canvas.getContext('2d');
    private readonly route = document.createElement('select');
    private readonly frameNumber = document.createElement('input');
    private readonly previous = button('이전', () => this.step(-1));
    private readonly next = button('다음', () => this.step(1));
    private readonly resetCamera = button('기록 시점', () => this.jumpToCamera());
    private readonly saveFigures = button('발표용 PNG 저장', () => void this.downloadFigures());
    private readonly status = document.createElement('div');
    private readonly modelLink = document.createElement('a');
    private readonly routes = new Map<string, ReplayFrame[]>();
    private model: Splat | null = null;
    private boxes: { id: string; corners: Vec3[] }[] = [];
    private pose: FramePose | null = null;
    private selectedFrame: ReplayFrame | null = null;
    private minimumDiameter = 5;
    private savingFigures = false;
    private activeRequest: AbortController | null = null;
    private liveWasEnabled = true;
    private savedView: { overlays: boolean; gizmos: boolean; fov: number };

    constructor(private readonly scene: Scene, private readonly container: HTMLElement, private readonly live: LiveView) {
        this.canvas.id = 'landing-replay-overlay';
        this.canvas.hidden = true;
        this.container.append(this.canvas);
        this.panel.className = 'landing-replay';
        this.panel.hidden = true;
        this.panel.setAttribute('aria-label', '3DGS 착륙 결과');

        const heading = document.createElement('div');
        heading.className = 'landing-replay-heading';
        const title = document.createElement('strong');
        title.textContent = '착륙 결과';
        heading.append(title, button('닫기', () => this.close()));

        const controls = document.createElement('div');
        controls.className = 'landing-replay-controls';
        this.route.setAttribute('aria-label', '접근 경로');
        this.route.addEventListener('change', () => void this.select(Number(this.frameNumber.value) - 1));
        this.frameNumber.type = 'number';
        this.frameNumber.min = '1';
        this.frameNumber.step = '1';
        this.frameNumber.value = '1';
        this.frameNumber.setAttribute('aria-label', '프레임 번호');
        this.frameNumber.addEventListener('change', () => void this.select(Number(this.frameNumber.value) - 1));
        this.saveFigures.disabled = true;
        this.saveFigures.title = '선택한 프레임의 기록 시점에서 세그멘테이션과 박스 PNG를 ZIP으로 저장';
        this.saveFigures.dataset.action = 'save-landing-figures';
        controls.append(this.route, this.previous, this.frameNumber, this.next, this.resetCamera, this.saveFigures);

        this.status.className = 'landing-replay-status';
        this.status.setAttribute('role', 'status');
        this.modelLink.href = '?landingReplay=1';
        this.modelLink.textContent = 'DJI 3DGS 장면 열기';
        this.modelLink.hidden = true;
        this.panel.append(heading, controls, this.status, this.modelLink);
        document.body.append(this.panel);
        scene.events.on('postrender', () => this.draw());
    }

    async open(initialFrame = defaultFrame) {
        if (!this.panel.hidden) return;
        this.liveWasEnabled = this.live.enabled;
        this.savedView = {
            overlays: this.scene.camera.renderOverlays,
            gizmos: this.scene.gizmoLayer.enabled,
            fov: this.scene.camera.fov
        };
        this.live.setEnabled(false);
        this.scene.camera.renderOverlays = false;
        this.scene.gizmoLayer.enabled = false;
        this.panel.hidden = false;
        document.body.classList.add('landing-replay-active');
        this.model = (this.scene.getElementsByType(ElementType.splat) as Splat[])
            .find(splat => typeof splat.filename === 'string' && splat.filename.split(/[?#]/)[0].split('/').at(-1) === modelName) ?? null;
        this.modelLink.hidden = this.model !== null;
        this.updateControls();
        if (!this.model) {
            this.status.textContent = '저장된 박스를 표시하려면 DJI 장면이 필요합니다.';
            return;
        }
        this.canvas.hidden = false;
        const request = new AbortController();
        this.activeRequest = request;
        this.status.textContent = '저장된 착륙 결과 불러오는 중…';
        try {
            if (!this.routes.size) {
                const response = await readResponse(resultPath, request.signal);
                const results = await response.json() as ReplayResults;
                if (request.signal.aborted) return;
                if (!(Number.isFinite(results.minimum_diameter_cm) && results.minimum_diameter_cm > 0) ||
                    !Array.isArray(results.frames) || !results.frames.length) {
                    throw new Error('Invalid diameter-filtered results');
                }
                const routes = new Map<string, ReplayFrame[]>();
                for (const row of results.frames) {
                    if (!/^approach_test_\d{3}$/.test(row.stream_id) || !Number.isInteger(row.sequence) ||
                        row.frame_id !== `${row.stream_id}_${String(row.sequence).padStart(4, '0')}` ||
                        !Array.isArray(row.proposals) || row.proposals.some(box =>
                            typeof box.box_id !== 'string' || box.box_corners_world_m?.length !== 8 ||
                            !box.box_corners_world_m.every(point) ||
                            !(box.diameter_cm >= results.minimum_diameter_cm) ||
                            box.box_corners_image_px?.length !== 8 || !box.box_corners_image_px.every(p =>
                                Array.isArray(p) && p.length === 2 && p.every(Number.isFinite)) ||
                            !box.mask_runs?.length || box.mask_runs.some(([start, count]) =>
                                !Number.isInteger(start) || start < 0 || start >= replayGrid * replayGrid ||
                                !Number.isInteger(count) || count < 1 || start % replayGrid + count > replayGrid))) {
                        throw new Error('Invalid landing result');
                    }
                    if (!routes.has(row.stream_id)) routes.set(row.stream_id, []);
                    routes.get(row.stream_id).push(row);
                }
                this.minimumDiameter = results.minimum_diameter_cm;
                for (const [name, frames] of routes) {
                    this.routes.set(name, frames.sort((a, b) => a.sequence - b.sequence));
                    this.route.add(new Option(`경로 ${Number(name.split('_').at(-1)) + 1}`, name));
                }
            }
            const matchingRoute = [...this.routes].find(([, frames]) => frames.some(frame => frame.frame_id === initialFrame));
            const route = matchingRoute ?? [...this.routes][0];
            this.route.value = route[0];
            const first = route[1].findIndex(row => row.frame_id === initialFrame);
            await this.select(first < 0 ? route[1].findIndex(row => row.proposals.length > 0) : first);
        } catch (err) {
            if (!request.signal.aborted) this.status.textContent = `착륙 결과를 불러올 수 없습니다. ${err.message}`;
        }
    }

    close() {
        this.activeRequest?.abort();
        this.panel.hidden = true;
        this.canvas.hidden = true;
        this.boxes = [];
        this.pose = null;
        this.selectedFrame = null;
        this.model = null;
        document.body.classList.remove('landing-replay-active');
        this.scene.camera.renderOverlays = this.savedView.overlays;
        this.scene.gizmoLayer.enabled = this.savedView.gizmos;
        this.live.setEnabled(this.liveWasEnabled);
        if (!this.liveWasEnabled) this.scene.camera.fov = this.savedView.fov;
    }

    private step(delta: number) {
        void this.select(Number(this.frameNumber.value) - 1 + delta);
    }

    private updateControls() {
        const count = this.routes.get(this.route.value)?.length ?? 0;
        const index = Number(this.frameNumber.value);
        this.route.disabled = !this.model || !this.routes.size;
        this.frameNumber.disabled = !this.model || !count;
        this.frameNumber.max = String(count);
        this.previous.disabled = !count || index <= 1;
        this.next.disabled = !count || index >= count;
        this.resetCamera.disabled = !this.pose;
        this.saveFigures.disabled = this.savingFigures || !this.selectedFrame?.proposals.length;
    }

    private async select(index: number) {
        const frames = this.routes.get(this.route.value);
        if (!frames?.length || this.panel.hidden) return;
        const selected = Math.min(frames.length - 1, Math.max(0, Number.isFinite(index) ? Math.trunc(index) : 0));
        const frame = frames[selected];
        this.activeRequest?.abort();
        const request = new AbortController();
        this.activeRequest = request;
        this.frameNumber.value = String(selected + 1);
        this.pose = null;
        this.selectedFrame = null;
        this.boxes = [];
        delete this.canvas.dataset.frameId;
        delete this.canvas.dataset.boxCount;
        this.draw();
        this.updateControls();
        this.status.textContent = '기록 시점 불러오는 중…';
        try {
            const response = await readResponse(`${poseRoot}/${frame.stream_id}/Pose/${frame.frame_id}.json`, request.signal);
            const pose = await response.json() as FramePose;
            if (request.signal.aborted) return;
            const scale = pose.meters_per_scene_unit;
            if (pose.frame_id !== frame.frame_id || !(Number.isFinite(scale) && scale > 0) ||
                !point(pose.position) || !point(pose.approach?.target_world_m) ||
                ![pose.projection_matrix?.[0]?.[0], pose.projection_matrix?.[1]?.[1]]
                    .every(value => Number.isFinite(value) && value > 0)) {
                throw new Error('Invalid recorded camera pose');
            }
            this.pose = pose;
            this.selectedFrame = frame;
            this.boxes = frame.proposals.map(box => ({
                id: box.box_id.split(':').at(-1),
                corners: box.box_corners_world_m.map(p => new Vec3(...p).mulScalar(1 / scale))
            }));
            this.canvas.dataset.frameId = frame.frame_id;
            this.canvas.dataset.boxCount = String(this.boxes.length);
            this.status.textContent = `프레임 ${selected + 1}/${frames.length} · 착륙 가능 ${this.boxes.length}개 · 추정 지름 ≥ ${this.minimumDiameter}cm`;
            this.updateControls();
            this.jumpToCamera();
        } catch (err) {
            if (!request.signal.aborted) this.status.textContent = `기록 시점을 불러올 수 없습니다. ${err.message}`;
        }
    }

    private async downloadFigures() {
        const frame = this.selectedFrame;
        if (!frame?.proposals.length || this.savingFigures) return;
        this.savingFigures = true;
        this.saveFigures.textContent = '저장 중…';
        this.updateControls();
        try {
            await saveLandingFigures(frame.frame_id, this.minimumDiameter, frame.proposals);
        } catch (err) {
            if (this.selectedFrame === frame) this.status.textContent = `PNG 저장 실패 · ${err.message}`;
        } finally {
            this.savingFigures = false;
            this.saveFigures.textContent = '발표용 PNG 저장';
            this.updateControls();
        }
    }

    private jumpToCamera() {
        if (!this.pose) return;
        const camera = this.scene.camera;
        const position = new Vec3(...this.pose.position);
        const target = new Vec3(...this.pose.approach.target_world_m).mulScalar(1 / this.pose.meters_per_scene_unit);
        const horizontal = 1 / this.pose.projection_matrix[0][0];
        const vertical = 1 / this.pose.projection_matrix[1][1];
        const aspect = this.container.clientWidth / Math.max(1, this.container.clientHeight);
        // Fit the recorded frustum into the current viewport, including portrait windows.
        const halfFov = aspect >= 1 ? Math.max(horizontal, vertical * aspect) :
            Math.max(vertical, horizontal / Math.max(.1, aspect));
        camera.fov = Math.min(170, 2 * Math.atan(halfFov) * 180 / Math.PI);
        camera.setPose(position, target, 0);
        camera.onUpdate(0);
        this.scene.forceRender = true;
    }

    private draw() {
        if (this.canvas.hidden || !this.ctx) return;
        const { ctx } = this;
        const width = this.container.clientWidth, height = this.container.clientHeight;
        if (width <= 0 || height <= 0) return;
        const ratio = window.devicePixelRatio || 1;
        const w = Math.round(width * ratio), h = Math.round(height * ratio);
        if (this.canvas.width !== w) this.canvas.width = w;
        if (this.canvas.height !== h) this.canvas.height = h;
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.clearRect(0, 0, width, height);
        if (!this.model?.visible || !this.scene.getElementsByType(ElementType.splat).includes(this.model)) return;

        const camera = this.scene.camera;
        const projected = new Vec3();
        const delta = new Vec3();
        const position = camera.mainCamera.getPosition(), forward = camera.mainCamera.forward;
        ctx.lineJoin = 'round';
        ctx.font = '600 12px system-ui, sans-serif';
        for (const { id, corners } of this.boxes) {
            const screen = corners.map(corner => {
                if (delta.sub2(corner, position).dot(forward) <= camera.camera.nearClip) return null;
                camera.worldToScreen(corner, projected);
                return [projected.x * width, projected.y * height];
            });
            drawLandingOutline(ctx, screen);
            const visible = screen.filter(p => p && p[0] >= 0 && p[0] < width && p[1] >= 0 && p[1] < height);
            if (!visible.length) continue;
            const label = `#${id}`;
            const x = Math.min(width - ctx.measureText(label).width - 9, Math.max(7, Math.min(...visible.map(p => p[0]))));
            const y = Math.max(16, Math.min(...visible.map(p => p[1])) - 5);
            ctx.setLineDash([]);
            ctx.strokeStyle = '#05130b';
            ctx.lineWidth = 4;
            ctx.strokeText(label, x, y);
            ctx.fillStyle = boxColor;
            ctx.fillText(label, x, y);
        }
    }
}
