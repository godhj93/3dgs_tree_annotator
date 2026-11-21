export type BirdSpaceSettings = { enabled: boolean; widthM: number; lengthM: number; heightM: number;
    marginM: number; checkApproachPath: boolean; approachM: number; checkApproachAngle: boolean; minApproachAngleDeg: number };
export type SpaceGeometry = { contact: number[]; center: number[]; basis: number[][]; approach: number[];
    start: number[]; travel: number[]; travelLength: number; low: number[]; high: number[];
    corners_camera_m: number[][]; start_corners_camera_m: number[][] };
export type Da2SpaceCalibration = {
    method: 'metric_depth_ratio' | 'metric_inverse_affine' | 'assumed_median';
    inverse_scale: number; inverse_offset: number; samples?: number; region?: 'annotation' | 'full_frame';
    median_reference_m?: number;
};
export type SpaceAssessment = { id: number; status: 'clear' | 'blocked' | 'unknown'; reasons: string[];
    approach_angle_deg: number | null; min_clearance_m: number | null; observed_fraction: number;
    top_corners?: {index: number; status: 'clear' | 'blocked' | 'unknown'; hit_pixels: number; free_pixels: number; camera_m: number[]}[];
    geometry_units?: 'approximate_meters' | 'da2_assumed_meters' | 'da2_calibrated_meters';
    contact_camera_m: number[] | null; geometry: SpaceGeometry | null; blocker_camera_m?: number[] | null;
    axis_camera?: number[]; axis_linearity?: number; contact_world_m?: number[] };
export const defaultBirdSpace: Readonly<BirdSpaceSettings>;
export const cornerProbeRadiusM: number;
export const cornerMinHitPixels: number;
export function birdBoxBounds(settings: BirdSpaceSettings): { low: number[]; high: number[] };
export function validateBirdSpace(settings: Partial<BirdSpaceSettings>): BirdSpaceSettings;
export type SpaceScene = { depth: Float32Array; intrinsics: number[]; width: number; height: number;
    points: Float32Array; pixels: Uint32Array; geometryUnits: SpaceAssessment['geometry_units']; calibration: Da2SpaceCalibration | null };
export function buildSpaceScene(depth: Float32Array, intrinsics: number[], width?: number, height?: number,
    da2ReferenceM?: number | null, referenceMask?: Uint8Array | null, metricDepth?: Float32Array | null): SpaceScene;
export function evaluateLandingSpace(input: { scene: SpaceScene | null; candidates: { id: number }[];
    supportPixels: Map<number, number[]>; up: number[]; settings: BirdSpaceSettings }): SpaceAssessment[];
export function selectSpaceCandidates<T>(base: T, assessments: SpaceAssessment[], topK: number | null): T;
export function projectedHeight(a: number[], b: number[], up: number[]): number;
