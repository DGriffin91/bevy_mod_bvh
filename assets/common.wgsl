

const TAU: f32 = 6.28318530717958647692528676655900577;
const INV_TAU: f32 = 0.159154943;
const PHI: f32 = 1.61803398875;
const PHIMINUS1: f32 = 0.61803398875;

const F32_EPSILON: f32 = 1.1920929E-7;
const F32_MAX: f32 = 3.402823466E+38;
const U32_MAX: u32 = 0xFFFFFFFFu;


const PROBE_RES_SCALE: i32 = 8;
const PROBE_RES_SCALEF: f32 = 8.0;



fn rand(co: f32) -> f32 { 
    return fract(sin(co*(91.3458)) * 47453.5453); 
}

fn gold_noise(xy: vec2<f32>, seed: f32) -> f32 {
    //return fract(tan(distance(xy*PHI, xy)*seed)*xy.x);
    return fract(tan(distance(xy*10.0*PHI, xy)*seed)*xy.x);
}

fn uhash(a: u32, b: u32) -> u32 { 
    var x = ((a * 1597334673u) ^ (b * 3812015801u));
    // from https://nullprogram.com/blog/2018/07/31/
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn unormf(n: u32) -> f32 { 
    return f32(n) * (1.0 / f32(0xffffffffu)); 
}

fn hash_noise(ifrag_coord: vec2<i32>, frame: u32) -> f32 {
    let urnd = uhash(u32(ifrag_coord.x), (u32(ifrag_coord.y) << 11u) + frame);
    return unormf(urnd);
}

fn sphericalFibonacci(i: f32, n: f32) -> vec3<f32> {
    let phi = TAU * fract(i * PHIMINUS1);
    let cosTheta = 1.0 - (2.0 * i + 1.0) * (1.0 / n);
    let sinTheta = sqrt(saturate(1.0 - cosTheta * cosTheta));

    return vec3(
        cos(phi) * sinTheta,
        sin(phi) * sinTheta,
        cosTheta);
}

fn previous_uv(data_tex: texture_1d<f32>, ihit_pos: vec3<f32>) -> vec2<f32> {
    let prev_model = mat4x4<f32>(
        textureLoad(data_tex, 1, 0),
        textureLoad(data_tex, 2, 0),
        textureLoad(data_tex, 3, 0),
        textureLoad(data_tex, 4, 0),
    );    
    let clipSpace = view.projection * ( prev_model * vec4(ihit_pos, 1.0) );
    let ndc = clipSpace.xyz / clipSpace.w;
    var prev_uv = (ndc.xy * .5 + .5);
    prev_uv.y = 1.0 - prev_uv.y;
    return prev_uv;
}

fn position_from_uv(uv: vec2<f32>, linear_depth: f32) -> vec3<f32> {
    var clip = uv * 2.0 - 1.0;
    var eye = view.inverse_projection * vec4(clip.x, -clip.y, -1.0, 1.0);
    eye.w = 0.0;
    let eye_dir = view.view * eye;
    return view.world_position.xyz + normalize(eye_dir.xyz)* linear_depth;
}


// https://github.com/kayru/Probulator/blob/9929231fd973fa678e67ab7a56838bd597fce83a/Source/Probulator/SphericalHarmonics.h#L136
fn shEvaluateDiffuseL1Geomerics(sh: vec4<f32>, n: vec3<f32>) -> f32 {
	// http://www.geomerics.com/wp-content/uploads/2015/08/CEDEC_Geomerics_ReconstructingDiffuseLighting1.pdf

	let R0 = sh[0];

	let R1 = 0.5 * vec3(sh[3], sh[1], sh[2]);
	let lenR1 = length(R1);

	let q = 0.5 * (1.0 + dot(R1 / lenR1, n));

	let p = 1.0 + 2.0 * lenR1 / R0;
	let a = (1.0 - lenR1 / R0) / (1.0 + lenR1 / R0);

	return R0 * (a + (1.0 - a) * (p + 1.0) * pow(q, p));
}