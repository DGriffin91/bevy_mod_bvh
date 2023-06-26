#import "rgb9e5.wgsl"
#import "printing.wgsl"
#import "common.wgsl"
#import "trace_gpu_types.wgsl"

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
var<storage> vertex_buffer: array<VertexDataPacked>;
@group(0) @binding(5)
var<storage> index_buffer: array<VertexIndices>;
@group(0) @binding(6)
var<storage> blas_buffer: array<BLASBVHData>;
@group(0) @binding(7)
var<storage> static_tlas_buffer: array<TLASBVHData>;
@group(0) @binding(8)
var<storage> dynamic_tlas_buffer: array<TLASBVHData>;
@group(0) @binding(9)
var<storage> static_mesh_instance_buffer: array<InstanceData>;
@group(0) @binding(10)
var<storage> dynamic_mesh_instance_buffer: array<InstanceData>;

#define RT_STATS

#import "traverse_tlas.wgsl"
#import "tracing.wgsl"

fn get_screen_ray(uv: vec2<f32>) -> Ray {
    var ndc = uv * 2.0 - 1.0;
    var eye = view.inverse_view_proj * vec4(ndc.x, -ndc.y, 0.0, 1.0);

    var ray: Ray;
    ray.origin = view.world_position.xyz;
    ray.direction = normalize(eye.xyz);
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

    let query = scene_query(ray, F32_MAX);

    if query.hit.distance != F32_MAX {
        var normal = vec3(0.0);

        // could use query.hit.instance_idx to look up into your own material data
        // given a texture that has references ordered the same way as tlas.aabbs
        // see create_instance_mesh_data
        
        // another option would be to tack on a material idx to static_instance_data
        // then order the materials by that idx, this would allow the material layout
        // to not need to be updated when instances change
        var instance: InstanceData;
        if query.static_tlas {
            instance = static_mesh_instance_buffer[query.hit.instance_idx];
        } else {
            instance = dynamic_mesh_instance_buffer[query.hit.instance_idx];
        }
        normal = get_surface_normal(query);

        col = vec4(vec3(normal), 1.0);
    } else {
        col = vec4(0.0);    
    }

    //col = vec4(temperature(f32(query.stats.aabb_hit_blas + query.stats.aabb_hit_tlas), 50.0), 1.0);

    col = print_value(coord, col, 0, f32(settings.fps));
    col = print_value(coord, col, 1, f32(frame));
    col = print_value(coord, col, 2, f32(arrayLength(&dynamic_tlas_buffer)));
    col = print_value(coord, col, 3, f32(arrayLength(&static_tlas_buffer)));

    return col;
}