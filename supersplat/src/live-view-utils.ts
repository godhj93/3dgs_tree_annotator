const SENSOR_PROFILE = Object.freeze({
    name: 'LI-IMX678-MIPI-HL-118H',
    native_width: 3856,
    native_height: 2176,
    horizontal_fov_degrees: 118,
    vertical_fov_degrees: 2 * Math.atan(2176 / 3856 * Math.tan(118 * Math.PI / 360)) * 180 / Math.PI,
    nominal_focal_length_mm: 4.063,
    projection_model: 'nominal_fov_pinhole_approximation',
    distortion_applied: false,
    framing: 'full_sensor_fov_resized_to_square',
    render_width: 963,
    render_height: 544,
    output_width: 640,
    output_height: 640,
    source: 'https://leopardimaging.com/wp-content/uploads/2024/05/LI-IMX678-MIPI-HL-118H_Datasheet.pdf'
});

const sensorIntrinsics = () => {
    const fx = 640 / (2 * Math.tan(118 * Math.PI / 360));
    const fy = fx * SENSOR_PROFILE.native_width / SENSOR_PROFILE.native_height;
    return [[fx, 0, 320], [0, fy, 320], [0, 0, 1]];
};

// RT1 uses premultiplied blending: red is visible annotated contribution,
// alpha is total scene opacity. Non-annotated foreground attenuates red.
const visibleMask = (contribution: Uint8Array) => {
    const mask = new Uint8ClampedArray(contribution.length);
    for (let i = 0; i < contribution.length; i += 4) {
        const alpha = contribution[i + 3] / 255;
        const value = alpha >= 0.05 && contribution[i] >= contribution[i + 3] * 0.5 ? 255 : 0;
        mask[i] = mask[i + 1] = mask[i + 2] = value;
        mask[i + 3] = 255;
    }
    return mask;
};

const rowMajorMatrix = (columnMajor: ArrayLike<number>) => Array.from({ length: 4 }, (_, r) => Array.from({ length: 4 }, (_, c) => columnMajor[c * 4 + r])
);

const numpyFloat32 = (depth: Float32Array, width: number, height: number) => {
    if (depth.length !== width * height) throw new Error('Depth shape mismatch');
    const description = `{'descr': '<f4', 'fortran_order': False, 'shape': (${height}, ${width}), }`;
    const padding = (64 - ((10 + description.length + 1) % 64)) % 64;
    const header = new TextEncoder().encode(`${description}${' '.repeat(padding)}\n`);
    const result = new Uint8Array(10 + header.length + depth.length * 4);
    result.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
    const view = new DataView(result.buffer);
    view.setUint16(8, header.length, true);
    result.set(header, 10);
    for (let i = 0; i < depth.length; i++) view.setFloat32(10 + header.length + i * 4, depth[i], true);
    return result;
};

export { SENSOR_PROFILE, sensorIntrinsics, visibleMask, rowMajorMatrix, numpyFloat32 };
