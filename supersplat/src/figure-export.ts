export type Figure = { name: string; title: string; caption: string; canvas: HTMLCanvasElement };
export type PngFile = { name: string; blob: Blob };

export const freezeFigures = (figures: Figure[]): Figure[] => figures.map((figure) => {
    const canvas = document.createElement('canvas');
    canvas.width = figure.canvas.width; canvas.height = figure.canvas.height;
    canvas.getContext('2d').drawImage(figure.canvas, 0, 0);
    return { ...figure, canvas };
});

const lines = (context: CanvasRenderingContext2D, value: string, width: number) => {
    const result: string[] = [];
    for (const paragraph of value.split('\n')) {
        let line = '';
        for (const word of paragraph.split(/\s+/).filter(Boolean)) {
            const next = line ? `${line} ${word}` : word;
            if (line && context.measureText(next).width > width) {
                result.push(line); line = word;
            } else line = next;
        }
        if (line) result.push(line);
    }
    return result;
};

// Individual PNGs keep the exact displayed canvas at its native resolution.
// The overview adds panel titles/captions without changing those images.
export const figureOverview = (figures: Figure[], title: string, description: string) => {
    const tile = 640, margin = 24, width = tile * 2 + margin * 3;
    const canvas = document.createElement('canvas'), measure = canvas.getContext('2d');
    measure.font = '18px sans-serif';
    const cards = figures.map(figure => ({ ...figure, captions: lines(measure, figure.caption, tile) }));
    const header = lines(measure, description, width - margin * 2);
    const headerHeight = 72 + header.length * 24;
    const rows: number[] = [];
    for (let i = 0; i < cards.length; i += 2) {
        rows.push(42 + tile + Math.max(cards[i].captions.length, cards[i + 1]?.captions.length ?? 0) * 23 + margin);
    }
    canvas.width = width; canvas.height = headerHeight + rows.reduce((sum, value) => sum + value, 0) + margin;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#071018'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#e7f3fa'; ctx.font = 'bold 24px sans-serif'; ctx.fillText(title, margin, 34);
    ctx.fillStyle = '#adc4d5'; ctx.font = '18px sans-serif';
    header.forEach((line, index) => ctx.fillText(line, margin, 64 + index * 24));
    let y = headerHeight;
    for (let row = 0; row < rows.length; row++) {
        for (let column = 0; column < 2; column++) {
            const card = cards[row * 2 + column];
            if (!card) continue;
            const x = margin + column * (tile + margin);
            ctx.fillStyle = '#8cdbff'; ctx.font = 'bold 20px sans-serif'; ctx.fillText(card.title, x, y + 24);
            ctx.drawImage(card.canvas, x, y + 38, tile, tile);
            ctx.strokeStyle = '#2b4252'; ctx.strokeRect(x, y + 38, tile, tile);
            ctx.fillStyle = '#cbdde8'; ctx.font = '18px sans-serif';
            for (let index = 0; index < card.captions.length; index++) {
                ctx.fillText(card.captions[index], x, y + 38 + tile + 24 + index * 23);
            }
        }
        y += rows[row];
    }
    return canvas;
};

const png = (canvas: HTMLCanvasElement) => new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(blob => (blob ? resolve(blob) : reject(new Error('Figure PNG encoding failed'))), 'image/png');
});

export const encodeFigures = async (figures: Figure[], title: string, description: string): Promise<PngFile[]> => {
    const files: PngFile[] = [];
    for (const figure of figures) files.push({ name: `${figure.name}.png`, blob: await png(figure.canvas) });
    files.push({ name: 'overview.png', blob: await png(figureOverview(figures, title, description)) });
    return files;
};

const crcTable = Array.from({ length: 256 }, (_, value) => {
    for (let bit = 0; bit < 8; bit++) value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    return value >>> 0;
});
const crc32 = (bytes: Uint8Array) => {
    let crc = 0xffffffff;
    for (const byte of bytes) crc = crcTable[(crc ^ byte) & 255] ^ (crc >>> 8);
    return (crc ^ 0xffffffff) >>> 0;
};

// PNGs are already compressed. Store them unchanged in one ZIP when browser
// directory access is unavailable, avoiding blocked multiple downloads.
export const figureArchive = async (files: PngFile[]): Promise<Blob> => {
    if (files.length > 65535) throw new Error('Too many figure files');
    const local: ArrayBuffer[] = [], central: ArrayBuffer[] = [];
    let offset = 0, centralSize = 0;
    for (const file of files) {
        const name = new TextEncoder().encode(file.name), data = await file.blob.arrayBuffer();
        if (name.length > 65535 || data.byteLength > 0xffffffff) throw new Error('Figure exceeds ZIP size limits');
        const crc = crc32(new Uint8Array(data));
        const header = new ArrayBuffer(30 + name.length), h = new DataView(header);
        h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true);
        h.setUint16(12, 33, true); h.setUint32(14, crc, true);
        h.setUint32(18, data.byteLength, true); h.setUint32(22, data.byteLength, true);
        h.setUint16(26, name.length, true); new Uint8Array(header, 30).set(name);
        const entry = new ArrayBuffer(46 + name.length), e = new DataView(entry);
        e.setUint32(0, 0x02014b50, true); e.setUint16(4, 20, true); e.setUint16(6, 20, true);
        e.setUint16(8, 0x0800, true); e.setUint16(14, 33, true); e.setUint32(16, crc, true);
        e.setUint32(20, data.byteLength, true); e.setUint32(24, data.byteLength, true);
        e.setUint16(28, name.length, true); e.setUint32(42, offset, true); new Uint8Array(entry, 46).set(name);
        local.push(header, data); central.push(entry);
        offset += header.byteLength + data.byteLength; centralSize += entry.byteLength;
        if (offset + centralSize > 0xffffffff) throw new Error('Figures exceed ZIP archive size limits');
    }
    const end = new ArrayBuffer(22), e = new DataView(end);
    e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true);
    e.setUint32(12, centralSize, true); e.setUint32(16, offset, true);
    return new Blob([...local, ...central, end], { type: 'application/zip' });
};
