#define RT_STATS

#import bevy_core_pipeline::fullscreen_vertex_shader  FullscreenVertexOutput
#import "raytrace_bindings_types.wgsl" as bindings
#import "tracing.wgsl" as tracing
#import bevy_pbr::mesh_view_bindings view, globals
#import "trace_gpu_types.wgsl" as gputypes
#import "printing.wgsl" as printing
#import bevy_pbr::utils PI, HALF_PI

fn get_screen_ray(uv: vec2<f32>) -> tracing::Ray {
    var ndc = uv * 2.0 - 1.0;
    var eye = view.inverse_view_proj * vec4(ndc.x, -ndc.y, 0.0, 1.0);

    var ray: tracing::Ray;
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
    let frame = bindings::settings.frame;
    
    var col = textureSample(bindings::screen_tex, bindings::texture_sampler, in.uv);

    let ray = get_screen_ray(uv);

    let query = tracing::scene_query(ray, tracing::F32_MAX);

    if query.hit.distance != tracing::F32_MAX {
        var normal = vec3(0.0);

        // could use query.hit.instance_idx to look up into your own material data
        // given a texture that has references ordered the same way as tlas.aabbs
        // see create_instance_mesh_data
        
        // another option would be to tack on a material idx to static_instance_data
        // then order the materials by that idx, this would allow the material layout
        // to not need to be updated when instances change
        var instance: gputypes::InstanceData;
        if query.static_tlas {
            instance = bindings::static_mesh_instance_buffer[query.hit.instance_idx];
        } else {
            instance = bindings::dynamic_mesh_instance_buffer[query.hit.instance_idx];
        }
        normal = tracing::get_surface_normal(query);

        col = vec4(vec3(normal), 1.0);
    } else {
        col = vec4(0.0);    
    }

    //col = vec4(temperature(f32(query.stats.aabb_hit_blas + query.stats.aabb_hit_tlas), 50.0), 1.0);

    col = printing::print_value(coord, col, 0, f32(bindings::settings.fps));
    col = printing::print_value(coord, col, 1, f32(frame));
    col = printing::print_value(coord, col, 2, f32(arrayLength(&bindings::dynamic_tlas_buffer)));
    col = printing::print_value(coord, col, 3, f32(arrayLength(&bindings::static_tlas_buffer)));

    return col;
}