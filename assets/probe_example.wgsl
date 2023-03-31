#import "printing.wgsl"
#import "common.wgsl"

#import bevy_pbr::mesh_types
#import bevy_pbr::mesh_view_bindings
#import bevy_pbr::utils
#import bevy_core_pipeline::fullscreen_vertex_shader


struct TraceSettings {
    frame: u32,
    fps: f32,
}
@group(0) @binding(1)
var<uniform> settings: TraceSettings;
@group(0) @binding(2)
var prev_sh_tex: texture_storage_2d<rgba16float,read>;
@group(0) @binding(3)
var next_sh_tex: texture_storage_2d<rgba16float,write>;
@group(0) @binding(4)
var vert_indices: texture_2d<i32>;
@group(0) @binding(5)
var vert_pos: texture_2d<f32>;
@group(0) @binding(6)
var vert_nor: texture_2d<f32>;
@group(0) @binding(7)
var tri_nor: texture_2d<f32>;
@group(0) @binding(8)
var blas: texture_2d<f32>;
@group(0) @binding(9)
var static_tlas_data: texture_2d<f32>;
@group(0) @binding(10)
var dynamic_tlas_data: texture_2d<f32>;
@group(0) @binding(11)
var static_instance_data: texture_2d<i32>;
@group(0) @binding(12)
var dynamic_instance_data: texture_2d<i32>;
@group(0) @binding(13)
var static_instance_mat: texture_2d<f32>;
@group(0) @binding(14)
var dynamic_instance_mat: texture_2d<f32>;

#import "tracing.wgsl"

fn get_screen_ray(uv: vec2<f32>) -> Ray {
    var clip = uv * 2.0 - 1.0;
    var eye = view.inverse_projection * vec4(clip.x, -clip.y, -1.0, 1.0);
    eye.w = 0.0;
    let eye_dir = view.view * eye;

    var ray: Ray;
    ray.origin = view.world_position.xyz;
    ray.direction = normalize(eye_dir.xyz);
    ray.inv_direction = 1.0 / ray.direction;

    return ray;
}

fn signNotZero(k: f32) -> f32 {
    return select(-1.0, 1.0, k >= 0.0);
}

fn signNotZeroV2(v: vec2<f32>) -> vec2<f32> {
    return vec2(signNotZero(v.x), signNotZero(v.y));
}

// http://jcgt.org/published/0003/02/01/
// https://github.com/RomkoSI/G3D/blob/master/data-files/shader/octahedral.glsl

/** Assumes that v is a unit vector. The result is an octahedral vector on the [-1, +1] square. */
fn oct_encode(v: vec3<f32>) -> vec2<f32> {
    let l1norm = abs(v.x) + abs(v.y) + abs(v.z);
    var result = v.xy * (1.0 / l1norm);
    if (v.z < 0.0) {
        result = (1.0 - abs(result.yx)) * signNotZeroV2(result.xy);
    }
    return result;
}


/** Returns a unit vector. Argument o is an octahedral vector packed via octEncode,
    on the [-1, +1] square*/
fn oct_decode(o: vec2<f32>) -> vec3<f32> {
    var v = vec3(o.x, o.y, 1.0 - abs(o.x) - abs(o.y));
    if (v.z < 0.0) {
        let tmp = (1.0 - abs(v.yx)) * signNotZeroV2(v.xy);
        v.x = tmp.x;
        v.y = tmp.y;
    }
    return normalize(v);
}

// input uv 0 .. 1, tot size in px (inc border), border 1.0 would be 1 px border on all sides
fn oct_decode_n(o: vec2<f32>, size: f32, border: f32) -> vec3<f32> {
    let border_factor = 1.0 + ((border * 2.0) / size);
    return oct_decode((o * 2.0 - 1.0) * border_factor);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let coord = in.position.xy;
    let icoord = vec2<i32>(in.position.xy);
    let frame = settings.frame;
    let iframe = i32(frame);
    let odd_frame = iframe / 2; 
    
//    let prev_dist = textureLoad(prev_sh_tex, icoord).x;

    let rand_vec = vec3(
        hash_noise(icoord + 1, frame * 100u + 1123u), 
        hash_noise(icoord + 2, frame * 200u + 11234u),  
        hash_noise(icoord + 3, frame * 300u + 111345u)) * 2.0 - 1.0;

    let probe_spacing = 0.5;
    let offset = vec3(-4.0, 0.0, -8.0);

    let cx = 8;
    let cy = 8;
    let cz = 8;
    let size = 12;

    let id = icoord.x + icoord.y * cx * cy * size;
    let xy = vec2<i32>(icoord.x % size, icoord.y % size);

    let probe_ls = vec3<i32>((icoord.x / size) % cx, icoord.x / (cx * size), icoord.y / size);

    let probe_ws = (vec3<f32>(probe_ls) + offset) * probe_spacing;


    

    let fsize = f32(size);

    //let x = odd_frame % size;
    //let y = (odd_frame / size) % size;



    let fpos = vec2<f32>(xy);
    let v = fpos / fsize;
    let dir = oct_decode_n(v, fsize, 1.0);

    var ray: Ray;
    ray.origin = probe_ws;
    ray.direction = dir;
    ray.inv_direction = 1.0 / ray.direction;

    let query = scene_query(ray);


    textureStore(next_sh_tex, icoord, vec4(vec3(1.0 - 1.0/query.hit.distance), 1.0));

    //let ch = print_value_custom(fpos, vec2(0.0), vec2(6.0, 8.0), f32(probe_id), 3.0, 0.0);
    //textureStore(next_sh_tex, icoord, vec4(vec3(ch), 1.0));
    //textureStore(next_sh_tex, icoord, vec4(fpos / fsize, 0.0, 1.0));

    

    return vec4(0.0);
}