import type { SpaceAssessment } from '../static/lib/branch-detector/landing-space-analysis.mjs';

const reasonLabels: Record<string, string> = {
    approach_end_on: '가지가 접근 방향으로 뻗어 있음',
    perch_space_blocked: '위쪽 꼭짓점 충돌',
    approach_space_blocked: '접근 통로에 장애물',
    axis_uncertain: '가지 방향 추정 불안정',
    metric_geometry_missing: '유효한 3D 깊이 부족',
    space_unobserved: '가림·화면 밖·깊이 누락',
    inspection_budget: '검사 범위가 너무 큼',
    space_pending: '공간 검사 중'
};

export const describeSpace = (space?: SpaceAssessment) => {
    if (!space) return '';
    const status = space.status === 'clear' ? '공간 확보 추정' : space.status === 'blocked' ? '충돌 / 접근 불가' : '확인 불가 · 경고';
    const corners = space.top_corners ?? [];
    return `${status}${corners.length ? ` · 위쪽 꼭짓점 ${corners.filter(c => c.status === 'clear').length}/4 통과` : ''}` +
        `${space.approach_angle_deg == null ? '' : ` · 접근각 ${space.approach_angle_deg.toFixed(1)}°`}${
            space.reasons.length ? ` · ${space.reasons.map(reason => reasonLabels[reason] ?? (reason.startsWith('space_error') ? '공간 검사 오류' : reason)).join(', ')}` : ''}`;
};

const rejectionLabels = [
    ['approach_end_on', '접근각'],
    ['approach_space_blocked', '접근 통로 충돌'],
    ['perch_space_blocked', '위쪽 꼭짓점 충돌']
] as const;

export const summarizeSpaceRejections = (assessments: (SpaceAssessment | undefined)[]) => {
    const blocked = assessments.filter(space => space?.status === 'blocked');
    return rejectionLabels.flatMap(([reason, label]) => {
        const count = blocked.filter(space => space.reasons.includes(reason)).length;
        return count ? [`${label} ${count}`] : [];
    }).join(' · ');
};

export const projectSpacePoint = (p: number[], intrinsics: number[][]) => (p[2] < -0.001 ? [
    intrinsics[0][2] + intrinsics[0][0] * p[0] / -p[2],
    intrinsics[1][2] - intrinsics[1][1] * p[1] / -p[2]
] : null);

export const drawSpaceBox = (ctx: CanvasRenderingContext2D, space: SpaceAssessment | undefined,
    intrinsics: number[][], crop = { x: 0, y: 0, size: 640 }) => {
    if (space?.status !== 'clear' || !space.geometry) return;
    const project = (p: number[]) => {
        const pixel = projectSpacePoint(p, intrinsics);
        return pixel && [(pixel[0] - crop.x) / crop.size * ctx.canvas.width, (pixel[1] - crop.y) / crop.size * ctx.canvas.height];
    };
    const end = space.geometry.corners_camera_m.map(project);
    const segment = (a: number[] | null, b: number[] | null) => {
        if (!a || !b) return;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    };
    ctx.save();
    ctx.strokeStyle = '#45ed9b';
    ctx.lineWidth = 2;
    for (let i = 0; i < 8; i++) for (let k = 0; k < 3; k++) if (!(i & (1 << k))) segment(end[i], end[i | (1 << k)]);
    if (space.geometry.travelLength > 0) {
        const start = space.geometry.start_corners_camera_m.map(project);
        ctx.setLineDash([5, 4]);
        for (let i = 0; i < 8; i++) {
            segment(start[i], end[i]);
            for (let k = 0; k < 3; k++) if (!(i & (1 << k))) segment(start[i], start[i | (1 << k)]);
        }
    }
    ctx.restore();
};
