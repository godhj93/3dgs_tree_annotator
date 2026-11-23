import { encodeFigurePng, figureArchive } from './figure-export';

export const replayGrid = 640;
const imageRoot = '/data/tree_seg_3dgs_approach_6m_10000_hd_261120/test/RGB';
const boxColor = '#55f28e';
const edges = Array.from({ length: 8 }, (_, i) => [1, 2, 4].filter(bit => !(i & bit))
    .map(bit => [i, i | bit])).flat();

export type ReplayBox = {
    box_id: string;
    diameter_cm: number;
    box_corners_world_m: [number, number, number][];
    box_corners_image_px: [number, number][];
    mask_runs: [number, number][];
};

const makeCanvas = (width: number, height: number) => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
};

export const drawLandingOutline = (ctx: CanvasRenderingContext2D, screen: (number[] | null)[], scale = 1) => {
    ctx.save();
    ctx.beginPath();
    for (const [a, b] of edges) {
        if (!screen[a] || !screen[b]) continue;
        ctx.moveTo(screen[a][0], screen[a][1]);
        ctx.lineTo(screen[b][0], screen[b][1]);
    }
    ctx.lineJoin = 'round';
    ctx.setLineDash([5 * scale, 4 * scale]);
    ctx.strokeStyle = '#05130b';
    ctx.lineWidth = 4 * scale;
    ctx.stroke();
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 2 * scale;
    ctx.stroke();
    ctx.restore();
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const frameCrop = (boxes: ReplayBox[], width: number, height: number) => {
    const points: [number, number][] = boxes.flatMap(box => [
        ...box.box_corners_image_px,
        ...box.mask_runs.flatMap(([start, length]): [number, number][] => [
            [start % replayGrid, Math.floor(start / replayGrid)],
            [start % replayGrid + length, Math.floor(start / replayGrid) + 1]
        ])
    ]);
    const xs = points.map(([x]) => width * clamp(x, 0, replayGrid) / replayGrid);
    const ys = points.map(([, y]) => height * clamp(y, 0, replayGrid) / replayGrid);
    const [left, right, top, bottom] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const cropWidth = Math.min(width, Math.max(width * 0.5, (right - left) * 1.55, (bottom - top) * 1.7 * width / height));
    const cropHeight = cropWidth * height / width;
    return {
        x: clamp((left + right - cropWidth) / 2, 0, width - cropWidth),
        y: clamp((top + bottom - cropHeight) / 2, 0, height - cropHeight),
        width: cropWidth, height: cropHeight
    };
};

export const saveLandingFigures = async (frameId: string, minimumDiameter: number, boxes: ReplayBox[]) => {
    const response = await fetch(`${imageRoot}/${frameId}.png`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const rgb = await createImageBitmap(await response.blob());
    try {
        const crop = frameCrop(boxes, rgb.width, rgb.height);
        const width = 1280, height = 720;
        const segmentation = makeCanvas(width, height);
        const geometry = makeCanvas(width, height);
        const mask = makeCanvas(replayGrid, replayGrid);
        const maskContext = mask.getContext('2d');
        maskContext.fillStyle = '#2be2f5';
        for (const { mask_runs } of boxes) for (const [start, length] of mask_runs) {
            maskContext.fillRect(start % replayGrid, Math.floor(start / replayGrid), length, 1);
        }
        const segContext = segmentation.getContext('2d');
        const boxContext = geometry.getContext('2d');
        for (const ctx of [segContext, boxContext]) {
            ctx.imageSmoothingQuality = 'high';
            ctx.drawImage(rgb, crop.x, crop.y, crop.width, crop.height, 0, 0, width, height);
        }
        segContext.imageSmoothingEnabled = false;
        segContext.globalAlpha = 0.8;
        segContext.shadowColor = '#05130b';
        segContext.shadowBlur = 3;
        segContext.drawImage(mask, crop.x * replayGrid / rgb.width, crop.y * replayGrid / rgb.height,
            crop.width * replayGrid / rgb.width, crop.height * replayGrid / rgb.height, 0, 0, width, height);
        for (const box of boxes) {
            drawLandingOutline(boxContext, box.box_corners_image_px.map(([x, y]) => [
                (x * rgb.width / replayGrid - crop.x) * width / crop.width,
                (y * rgb.height / replayGrid - crop.y) * height / crop.height
            ]), 1.55);
        }
        const [segPng, boxPng] = await Promise.all([encodeFigurePng(segmentation), encodeFigurePng(geometry)]);
        const zip = await figureArchive([
            { name: `${frameId}_01-landing-branch-segmentation.png`, blob: segPng },
            { name: `${frameId}_02-landing-bboxes.png`, blob: boxPng }
        ]);
        const link = document.createElement('a');
        const url = URL.createObjectURL(zip);
        link.href = url;
        link.download = `${frameId}_landing_min-${minimumDiameter}cm.zip`;
        document.body.append(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 30000);
    } finally {
        rgb.close();
    }
};
