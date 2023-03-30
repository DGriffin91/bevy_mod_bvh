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

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let coord = in.position.xy;
    let icoord = vec2<i32>(in.position.xy);
    let frame = settings.frame;
    
    let prev_dist = textureLoad(prev_sh_tex, icoord).x;

    let rand_vec = vec3(
        hash_noise(icoord + 1, frame * 100u + 1123u), 
        hash_noise(icoord + 2, frame * 200u + 11234u),  
        hash_noise(icoord + 3, frame * 300u + 111345u)) * 2.0 - 1.0;

    let probe_spacing = 0.5;
    let offset = vec3(-8.0, 0.0, -16.0);

    let probe_idx = vec3<i32>(icoord.x % 16, icoord.y, icoord.x / 16);

    let ws_pos = (vec3<f32>(probe_idx) + offset) * probe_spacing;

    
    var ray: Ray;
    ray.origin = ws_pos;
    ray.direction = normalize(rand_vec);
    ray.inv_direction = 1.0 / ray.direction;

    let query = scene_query(ray);

    let new_dist = mix(query.hit.distance, prev_dist, 0.9);
    
    textureStore(next_sh_tex, vec2(probe_idx.x + probe_idx.z * 16, probe_idx.y), vec4(new_dist));

    return vec4(0.0);
}