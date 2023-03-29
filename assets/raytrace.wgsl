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
var gpu_static_tlas_data: texture_2d<f32>;
@group(0) @binding(5)
var gpu_dynamic_tlas_data: texture_2d<f32>;
@group(0) @binding(6)
var mesh_data: texture_2d<i32>;
@group(0) @binding(7)
var vert_indices: texture_2d<i32>;
@group(0) @binding(8)
var vert_pos: texture_2d<f32>;
@group(0) @binding(9)
var vert_nor: texture_2d<f32>;
@group(0) @binding(10)
var blas: texture_2d<f32>;
@group(0) @binding(11)
var gpu_static_instance_data: texture_2d<f32>;
@group(0) @binding(12)
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


@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let coord = in.position.xy;
    let icoord = vec2<i32>(in.position.xy);
    let uv = in.position.xy/view.viewport.zw;
    let frame = settings.frame;
    
    var col = textureSample(screen_tex, texture_sampler, in.uv);

    let ray = get_screen_ray(uv);

    let query = scene_query(ray);

    //if query.hit.distance != F32_MAX {
        var diffuse = vec4(0.0);
        var normal = vec3(0.0);

        if query.static_tlas {
            diffuse = get_instance_diffuse(gpu_static_instance_data, query.hit.instance_idx);
            normal = get_tri_normal(gpu_static_instance_data, query.hit);
        } else {
            diffuse = get_instance_diffuse(gpu_dynamic_instance_data, query.hit.instance_idx);
            normal = get_tri_normal(gpu_dynamic_instance_data, query.hit);
        }


        col = vec4(vec3(normal), 1.0);// * diffuse * front_light + mist; //

    //} else {
    //    col = vec4(0.0);    
    //}

    col = print_value(coord, col, 0, f32(settings.fps));
    col = print_value(coord, col, 1, f32(frame));
    col = print_value(coord, col, 2, f32(get_tlas_max_length(gpu_static_tlas_data)));
    col = print_value(coord, col, 3, f32(get_tlas_max_length(gpu_dynamic_tlas_data)));

    return col;
}


//        let mist = pow(query.hit.distance * 0.1, 5.0);
//        let front_light = dot(normal, -ray.direction);
//        var ao = 0.0;
//
//        // AO
//        // TODO Shouldn't need to shorten with 0.993, need to address self occlusion
//        var hit_p = ray.origin + ray.direction * query.hit.distance * 0.993; 
//        hit_p += normal * 0.00001; // bias away from surface
//        for (var i = 1u; i < 1u + 1u; i+=1u) {
//            var rand_vec = vec3(
//                hash_noise(icoord, i * frame + 1432u),
//                hash_noise(icoord, i * frame + 13456u),
//                hash_noise(icoord, i * frame + 187654u)
//            ) * 2.0 - 1.0;
//
//            rand_vec *= dot(normal, rand_vec); // hemisphere
//
//            var ray: Ray;
//            ray.origin = hit_p;
//            ray.direction = rand_vec;
//            ray.inv_direction = 1.0 / ray.direction;
//
//            let query2 = scene_query(ray);
//
//            ao += saturate(query2.hit.distance);
//
//        }
//        ao = saturate(ao / 1.0);
//