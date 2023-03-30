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
struct TraceSettings {
    frame: u32,
    fps: f32,
}
@group(0) @binding(3)
var<uniform> settings: TraceSettings;
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
    let uv = in.position.xy/view.viewport.zw;
    let frame = settings.frame;
    
    var col = textureSample(screen_tex, texture_sampler, in.uv);

    let ray = get_screen_ray(uv);

    let query = scene_query(ray);

    if query.hit.distance != F32_MAX {
        var normal = vec3(0.0);

        // could use query.hit.instance_idx to look up into your own material data
        // given a texture that has references ordered the same way as tlas.aabbs
        // see create_instance_mesh_data

        if query.static_tlas {
            normal = get_surface_normal(static_instance_data, static_instance_mat, query.hit);
        } else {
            normal = get_surface_normal(dynamic_instance_data, dynamic_instance_mat, query.hit);
        }

        col = vec4(vec3(normal), 1.0);
    } else {
        col = vec4(0.0);    
    }

    col = print_value(coord, col, 0, f32(settings.fps));
    col = print_value(coord, col, 1, f32(frame));
    col = print_value(coord, col, 2, f32(get_tlas_max_length(static_tlas_data)));
    col = print_value(coord, col, 3, f32(get_tlas_max_length(dynamic_tlas_data)));

    return col;
}