export type DepthSource = 'da2' | '3dgs';
export type ComparisonStage = 'boundary' | 'skeleton' | 'links' | 'pca' | 'merged' | 'gravity' | 'final';
export type ComparisonSourceView = {
    status: string;
    images?: Record<ComparisonStage, HTMLCanvasElement>;
    captions?: Partial<Record<ComparisonStage, string>>;
    inspect?: (x: number, y: number) => string;
};

export const comparisonStages: [ComparisonStage, string][] = [
    ['boundary', 'Depth support · orange boundary / gray missing depth'],
    ['skeleton', 'Depth-aware skeleton'], ['links', 'Separated branch regions'],
    ['pca', '3D PCA axes'], ['merged', 'PCA merged regions'],
    ['gravity', 'Gravity / length / trunk filter'], ['final', 'Final landing branches']
];

export class DepthComparisonView {
    root = document.createElement('section');
    tiles = new Map<string, { canvas: HTMLCanvasElement; caption: HTMLDivElement }>();
    statuses: Record<DepthSource, HTMLDivElement>;
    summary = document.createElement('div');
    difference = document.createElement('canvas');
    overlaySource: DepthSource | 'difference' = '3dgs';
    zoom = true;
    crop = { x: 0, y: 0, size: 640 };
    views: Partial<Record<DepthSource, ComparisonSourceView>> = {};

    constructor(host: HTMLElement, sharedControls: HTMLElement[],
        gapChanged: (value: number) => void, redraw: () => void) {
        this.root.className = 'depth-comparison';
        this.root.style.cssText = 'grid-column:1/-1;min-width:0';
        const title = document.createElement('h3');
        title.textContent = 'DA2 vs 3DGS · same RGB / annotation';
        title.style.cssText = 'font-size:12px;color:#a9e7ff;margin:8px 0';
        const controls = document.createElement('div'); controls.className = 'live-pca-controls';
        const gap = document.createElement('input');
        let lastGap = 0.15;
        gap.type = 'number'; gap.min = '0'; gap.max = '1'; gap.step = '0.01'; gap.value = '0.15';
        gap.setAttribute('aria-label', 'Shared skeleton depth gap');
        gap.addEventListener('change', () => {
            const value = Number(gap.value);
            if (!Number.isFinite(value) || value < 0 || value > 1) {
                gap.value = String(lastGap); return;
            }
            lastGap = value;
            gapChanged(value);
        });
        const source = document.createElement('select');
        source.setAttribute('aria-label', 'Main overlay depth source');
        for (const [value, label] of [['da2', 'DA2'], ['3dgs', '3DGS'], ['difference', 'Final mask difference']]) {
            const option = document.createElement('option'); option.value = value; option.textContent = label; source.append(option);
        }
        source.value = this.overlaySource;
        source.addEventListener('change', () => {
            this.overlaySource = source.value as typeof this.overlaySource; redraw();
        });
        const zoom = document.createElement('label');
        const check = document.createElement('input'); check.type = 'checkbox'; check.checked = true;
        check.addEventListener('change', () => {
            this.zoom = check.checked; redraw();
        });
        zoom.append(check, 'Zoom to annotation');
        controls.append('Depth gap / robust range', gap, 'Main overlay', source, zoom);
        const filters = document.createElement('details');
        const heading = document.createElement('summary'); heading.textContent = 'PCA, gravity and landing filters · length: DA2 px / 3DGS cm';
        filters.append(heading, ...sharedControls);
        const header = document.createElement('div'); header.className = 'live-junction-grid';
        this.statuses = { da2: document.createElement('div'), '3dgs': document.createElement('div') };
        for (const key of ['da2', '3dgs'] as const) {
            this.statuses[key].className = 'live-junction-label'; header.append(this.statuses[key]);
        }
        this.root.append(title, controls, filters, header);
        for (const [stage, label] of comparisonStages) {
            const row = document.createElement('div'); row.className = `live-tile live-junction-tile compare-${stage}`;
            const heading = document.createElement('div'); heading.className = 'live-tile-title'; heading.textContent = label;
            const grid = document.createElement('div'); grid.className = 'live-junction-grid';
            for (const key of ['da2', '3dgs'] as const) {
                const item = document.createElement('div'); item.className = 'live-junction-view';
                const canvas = document.createElement('canvas'); canvas.width = canvas.height = 640;
                canvas.dataset.depthSource = key; canvas.dataset.stage = stage;
                const caption = document.createElement('div'); caption.className = 'live-junction-label';
                item.append(canvas, caption); grid.append(item); this.tiles.set(`${key}:${stage}`, { canvas, caption });
                if (['skeleton', 'links', 'pca', 'merged', 'gravity', 'final'].includes(stage)) {
                    canvas.addEventListener('mousemove', (event) => {
                        const rect = canvas.getBoundingClientRect();
                        const x = this.crop.x + (event.clientX - rect.left) / rect.width * this.crop.size;
                        const y = this.crop.y + (event.clientY - rect.top) / rect.height * this.crop.size;
                        caption.textContent = this.views[key]?.inspect?.(Math.floor(x), Math.floor(y)) ?? 'Waiting for analysis';
                    });
                    canvas.addEventListener('mouseleave', () => {
                        caption.textContent = this.views[key]?.captions?.[stage] ?? '';
                    });
                }
            }
            row.append(heading, grid); this.root.append(row);
        }
        this.difference.width = this.difference.height = 640;
        this.difference.style.cssText = 'display:block;width:50%;height:auto;margin:4px auto';
        this.summary.className = 'live-junction-summary';
        this.root.append(this.difference, this.summary);
        host.append(this.root);
    }

    render(views: Partial<Record<DepthSource, ComparisonSourceView>>, crop: typeof this.crop,
        difference: HTMLCanvasElement, summary: string) {
        this.views = views; this.crop = this.zoom ? crop : { x: 0, y: 0, size: 640 };
        for (const source of ['da2', '3dgs'] as const) {
            const view = views[source];
            this.statuses[source].textContent = `${source === 'da2' ? 'DA2 · relative pseudo-3D' : '3DGS · calibrated geometry'} · ${view?.status ?? 'waiting'}`;
            for (const [stage] of comparisonStages) {
                const tile = this.tiles.get(`${source}:${stage}`), ctx = tile.canvas.getContext('2d');
                ctx.clearRect(0, 0, 640, 640);
                const image = view?.images?.[stage];
                if (image) ctx.drawImage(image, this.crop.x, this.crop.y, this.crop.size, this.crop.size, 0, 0, 640, 640);
                tile.caption.textContent = `${source === 'da2' ? 'DA2' : '3DGS'} · ${view?.captions?.[stage] ?? view?.status ?? 'waiting'}`;
            }
        }
        const ctx = this.difference.getContext('2d'); ctx.clearRect(0, 0, 640, 640);
        ctx.drawImage(difference, this.crop.x, this.crop.y, this.crop.size, this.crop.size, 0, 0, 640, 640);
        this.summary.textContent = summary;
    }
}
