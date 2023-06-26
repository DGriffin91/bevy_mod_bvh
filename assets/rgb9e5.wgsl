const _RGB9E5_EXPONENT_BITS        = 5u;
const _RGB9E5_MANTISSA_BITS        = 9;
const _RGB9E5_MANTISSA_BITSU       = 9u;
const _RGB9E5_EXP_BIAS             = 15;
const _RGB9E5_MAX_VALID_BIASED_EXP = 31u;

//#define _MAX_RGB9E5_EXP               (_RGB9E5_MAX_VALID_BIASED_EXP - _RGB9E5_EXP_BIAS)
//#define _RGB9E5_MANTISSA_VALUES       (1<<_RGB9E5_MANTISSA_BITS)
//#define _MAX_RGB9E5_MANTISSA          (_RGB9E5_MANTISSA_VALUES-1)
//#define _MAX_RGB9E5                   ((f32(_MAX_RGB9E5_MANTISSA))/_RGB9E5_MANTISSA_VALUES * (1<<_MAX_RGB9E5_EXP))
//#define _EPSILON_RGB9E5               ((1.0/_RGB9E5_MANTISSA_VALUES) / (1<<_RGB9E5_EXP_BIAS))

const _MAX_RGB9E5_EXP              = 16u;
const _RGB9E5_MANTISSA_VALUES      = 512u;
const _MAX_RGB9E5_MANTISSA         = 511;
const _MAX_RGB9E5_MANTISSAU        = 511u;
const _MAX_RGB9E5                  = 65408.0;
const _EPSILON_RGB9E5              = 0.000000059604645;



fn _floor_log2(x: f32) -> i32 {
    let f = bitcast<u32>(x);
    let biasedexponent = (f & 0x7F800000u) >> 23u;
    return i32(biasedexponent) - 127;
}

// https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_shared_exponent.txt
fn _vec3_to_rgb9e5(rgb: vec3<f32>) -> u32 {
    let rc = clamp(rgb.x, 0.0, _MAX_RGB9E5);
    let gc = clamp(rgb.y, 0.0, _MAX_RGB9E5);
    let bc = clamp(rgb.z, 0.0, _MAX_RGB9E5);

    let maxrgb = max(rc, max(gc, bc));
    var exp_shared = max(-_RGB9E5_EXP_BIAS - 1, _floor_log2(maxrgb)) + 1 + _RGB9E5_EXP_BIAS;
    var denom = exp2(f32(exp_shared - _RGB9E5_EXP_BIAS - _RGB9E5_MANTISSA_BITS));

    let maxm = i32(floor(maxrgb / denom + 0.5));
    if (maxm == _MAX_RGB9E5_MANTISSA + 1) {
        denom *= 2.0;
        exp_shared += 1;
    }

    let rm = u32(floor(rc / denom + 0.5));
    let gm = u32(floor(gc / denom + 0.5));
    let bm = u32(floor(bc / denom + 0.5));
    
    return (u32(exp_shared) << 27u) | (bm << 18u) | (gm << 9u) | (rm << 0u);
}

fn _bitfield_extract(value: u32, offset: u32, bits: u32) -> u32 {
    let mask = (1u << bits) - 1u;
    return (value >> offset) & mask;
}

fn _rgb9e5_to_vec3(v: u32) -> vec3<f32> {
    let exponent = i32(_bitfield_extract(v, 27u, _RGB9E5_EXPONENT_BITS)) - _RGB9E5_EXP_BIAS - _RGB9E5_MANTISSA_BITS;
    let scale = exp2(f32(exponent));

    return vec3(
        f32(_bitfield_extract(v, 0u, _RGB9E5_MANTISSA_BITSU)) * scale,
        f32(_bitfield_extract(v, 9u, _RGB9E5_MANTISSA_BITSU)) * scale,
        f32(_bitfield_extract(v, 18u, _RGB9E5_MANTISSA_BITSU)) * scale
    );
}