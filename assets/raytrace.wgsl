#import "printing.wgsl"
#import "common.wgsl"

#import bevy_pbr::mesh_types
#import bevy_pbr::mesh_view_bindings
#import bevy_pbr::utils
#import bevy_core_pipeline::fullscreen_vertex_shader

@group(0) @binding(1)
var screen_tex: texture_2d<f32>;
@group(0) @binding(2)
var texture_sampler: sampler;
struct PostProcessSettings {
    intensity: f32,
}
@group(0) @binding(3)
var<uniform> settings: PostProcessSettings;
@group(0) @binding(4)
var gpu_static_tlas_data: texture_2d<f32>;
@group(0) @binding(5)
var gpu_dynamic_tlas_data: texture_2d<f32>;
@group(0) @binding(6)
var mesh_data: texture_2d<u32>;
@group(0) @binding(7)
var vert_indices: texture_2d<u32>;
@group(0) @binding(8)
var vert_pos_nor: texture_2d<f32>;
@group(0) @binding(9)
var blas: texture_2d<f32>;
@group(0) @binding(10)
var gpu_static_instance_data: texture_2d<f32>;
@group(0) @binding(11)
var gpu_dynamic_instance_data: texture_2d<f32>;

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


//--------------------------
//--------------------------
//--------------------------

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let coord = in.position.xy;
    let icoord = vec2<i32>(in.position.xy);
    let uv = in.position.xy/view.viewport.zw;
    let offset_strength = settings.intensity;
    
    var col = textureSample(screen_tex, texture_sampler, in.uv);

    var ray = get_screen_ray(uv);

    let hit_static = traverse_tlas(gpu_static_tlas_data, gpu_static_instance_data, ray);
    let hit_dynamic = traverse_tlas(gpu_dynamic_tlas_data, gpu_dynamic_instance_data, ray);

    let static_closer = hit_static.distance < hit_dynamic.distance;

    var hit: Hit;

    var diffuse = vec4(0.0);

    if static_closer {
        hit = hit_static;
        diffuse = get_instance_diffuse(gpu_static_instance_data, hit.instance_idx);
    } else {
        hit = hit_dynamic;
        diffuse = get_instance_diffuse(gpu_dynamic_instance_data, hit.instance_idx);
    }

    if hit.distance == F32_MAX {
        diffuse = vec4(0.0);
    }

    let mist = pow(hit.distance * 0.1, 5.0);
    let front_light = dot(hit.normal, -ray.direction);
    let color = diffuse;

    col = diffuse * front_light + mist;

    col = print_value(coord, col, 0, f32(get_tlas_max_length(gpu_static_tlas_data)));
    col = print_value(coord, col, 1, f32(get_tlas_max_length(gpu_dynamic_tlas_data)));

    return col;

}