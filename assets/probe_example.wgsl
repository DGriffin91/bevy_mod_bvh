#import "printing.wgsl"
#import "common.wgsl"

#import bevy_pbr::mesh_types
#import bevy_pbr::mesh_view_bindings
#import bevy_pbr::utils
#import bevy_core_pipeline::fullscreen_vertex_shader


struct TraceSettings {
    sun_direction: vec3<f32>,
    frame: u32,
    fps: f32,
    render_depth_this_frame: u32,
}
@group(0) @binding(1)
var<uniform> settings: TraceSettings;
@group(0) @binding(2)
var vert_indices: texture_2d<i32>;
@group(0) @binding(3)
var vert_pos: texture_2d<f32>;
@group(0) @binding(4)
var vert_nor: texture_2d<f32>;
@group(0) @binding(5)
var tri_nor: texture_2d<f32>;
@group(0) @binding(6)
var blas: texture_2d<f32>;
@group(0) @binding(7)
var static_tlas_data: texture_2d<f32>;
@group(0) @binding(8)
var dynamic_tlas_data: texture_2d<f32>;
@group(0) @binding(9)
var static_instance_data: texture_2d<i32>;
@group(0) @binding(10)
var dynamic_instance_data: texture_2d<i32>;
@group(0) @binding(11)
var static_instance_mat: texture_2d<f32>;
@group(0) @binding(12)
var dynamic_instance_mat: texture_2d<f32>;
@group(0) @binding(13)
var prev_probe_tex: texture_storage_2d<rgba16float,read>;
@group(0) @binding(14)
var next_probe_tex: texture_storage_2d<rgba16float,write>;

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

const CAS = vec3<i32>(18, 18, 18);
const CASF = vec3<f32>(18.0, 18.0, 18.0);
const SIZE = 6;
const FSIZE = 6.0;


fn rand_dir(icoord: vec2<i32>, frame: u32) -> vec3<f32> {
    return vec3(hash_noise(icoord + 1, frame * 100u + 1123u), 
                hash_noise(icoord + 2, frame * 200u + 11234u),  
                hash_noise(icoord + 3, frame * 300u + 111345u)) * 2.0 - 1.0;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let coord = in.position.xy;
    let icoord = vec2<i32>(in.position.xy);
    let frame = settings.frame;
    let prev_frame = frame - 1u;
    let iframe = i32(frame);
    let odd_frame = iframe / 2; 
    let render_depth_this_frame = settings.render_depth_this_frame;
    let hysteresis = 0.995;
    let rnd_focus = 0.5;

    let sun_dir = settings.sun_direction;
    let sun_color = vec3(1.0,1.0,0.9) * 4.0;
    let sky_color = vec3(0.2,0.2,1.0) * 4.0;
    
//    let prev_dist = textureLoad(prev_sh_tex, icoord).x;


    let probe_spacing = 1.5;
    var ws_offset = vec3(probe_spacing * -4.0, 0.5, probe_spacing * -4.0);


    let probe_data_height = CAS.z * SIZE;

    let id = icoord.x + icoord.y * CAS.x * CAS.y * SIZE;
    let xy = vec2<i32>(icoord.x % SIZE, icoord.y % SIZE);

    let probe_ls = vec3<i32>((icoord.x / SIZE) % CAS.x, icoord.x / (CAS.x * SIZE), icoord.y / SIZE);

    let probe_ws = vec3<f32>(probe_ls) * probe_spacing + ws_offset;


    let fpos = vec2<f32>(xy);
    let oct_uv = fpos / FSIZE;
    let col_section = vec2(0, probe_data_height);

    if render_depth_this_frame == 0u {
        let prev_dist = textureLoad(prev_probe_tex, icoord);
        var next_dist = prev_dist;
        let prev_avg_dist = prev_dist.x;
        let prev_avg_dist_sq = prev_dist.y;

        let prev_color = textureLoad(prev_probe_tex, icoord + col_section);
        var next_color = prev_color;

        var init_dir = oct_decode_n(oct_uv, FSIZE, 1.0);

        var rand_vec = rand_dir(icoord, frame);
        var dir = init_dir * dot(rand_vec, init_dir);
        dir = mix(init_dir, dir, rnd_focus);

        var ray: Ray;
        ray.origin = probe_ws;
        ray.direction = normalize(dir);
        ray.inv_direction = 1.0 / ray.direction;

        let query = scene_query(ray);

        let hit_dist = query.hit.distance;

        next_dist.x = mix(hit_dist, prev_avg_dist, hysteresis);
        next_dist.y = mix(clamp(hit_dist * hit_dist, 0.0, F32_MAX), prev_avg_dist_sq, hysteresis);
        next_dist.w = hit_dist;
        textureStore(next_probe_tex, icoord, next_dist);

        if hit_dist == F32_MAX {
            //hit sky
            let mixed_color = mix(sky_color, prev_color.rgb, hysteresis);
            textureStore(next_probe_tex, icoord + col_section, vec4(mixed_color, prev_color.w));
        } else {
            // get reflected color from hit
            // find the closest probe
            let hitp = ray.origin + ray.direction * hit_dist;
            let ls_hit_probe = clamp(floor((hitp - ws_offset) / probe_spacing), vec3(0.0), CASF);
            let probe_tex_x = ls_hit_probe.x * FSIZE + ls_hit_probe.y * FSIZE * f32(CAS.x);
            let probe_tex_y = ls_hit_probe.z * FSIZE;

            let probe_uv = oct_encode_n(-ray.direction, FSIZE, 1.0);

            let hit_probe_coord = vec2(i32(probe_uv.x * FSIZE + probe_tex_x),
                                       i32(probe_uv.y * FSIZE + probe_tex_y));

            var refl_color = textureLoad(prev_probe_tex, hit_probe_coord + col_section).rgb;
            let surface_color = vec3(0.8); // replace with material
            refl_color *= surface_color;

            let mixed_color = mix(refl_color, prev_color.rgb, hysteresis);

            textureStore(next_probe_tex, icoord + col_section, vec4(mixed_color, prev_color.w));
        }

    } else {
        let prev_dist = textureLoad(prev_probe_tex, icoord);
        var next_dist = prev_dist;
        let prev_avg_dist = clamp(prev_dist.x, 0.0, F32_MAX);
        let prev_avg_dist_sq = clamp(prev_dist.y, 0.0, F32_MAX);
        let prev_exact_dist = prev_dist.w;


        let prev_color = textureLoad(prev_probe_tex, icoord + col_section);
        var next_color = prev_color;

        let init_dir = oct_decode_n(oct_uv, FSIZE, 1.0);

        //rnd_amount is how much we randomize the direction
        var rand_vec = rand_dir(icoord, prev_frame);
        var dir = init_dir * dot(rand_vec, init_dir);
        dir = mix(init_dir, dir, rnd_focus);


        var ray: Ray;
        ray.origin = probe_ws + dir * prev_exact_dist * 0.995;
        ray.direction = normalize(-sun_dir);
        ray.inv_direction = 1.0 / ray.direction;

        let query = scene_query(ray);


        var new_col = select(vec3(0.0), sun_color, query.hit.distance == F32_MAX);
        let surface_color = vec3(0.8); // replace with material
        new_col *= surface_color;

        
        let mixed_color = mix(new_col, prev_color.rgb, hysteresis);
        textureStore(next_probe_tex, icoord, next_dist);
        textureStore(next_probe_tex, icoord + col_section, vec4(mixed_color, next_color.w));

        /*
            tex dist fields:
                avg dist
                avg dist sq
                spare
                last frame exact dist
            tex color fields: avg rgb spare
        */
    }




    //let ch = print_value_custom(fpos, vec2(0.0), vec2(6.0, 8.0), f32(probe_id), 3.0, 0.0);
    //textureStore(next_sh_tex, icoord, vec4(vec3(ch), 1.0));
    //textureStore(next_sh_tex, icoord, vec4(fpos / FSIZE, 0.0, 1.0));

    

    var neighbors = vec4(0.0);
    neighbors += textureLoad(prev_probe_tex, icoord + col_section + vec2(-1, -1));
    neighbors += textureLoad(prev_probe_tex, icoord + col_section + vec2( 0, -1));
    neighbors += textureLoad(prev_probe_tex, icoord + col_section + vec2( 1, -1));
    neighbors += textureLoad(prev_probe_tex, icoord + col_section + vec2(-1,  0));
    neighbors += textureLoad(prev_probe_tex, icoord + col_section + vec2( 1,  0));
    neighbors += textureLoad(prev_probe_tex, icoord + col_section + vec2(-1,  1));
    neighbors += textureLoad(prev_probe_tex, icoord + col_section + vec2( 0,  1));
    neighbors += textureLoad(prev_probe_tex, icoord + col_section + vec2( 1,  1));
    neighbors /= 8.0;

    let center = textureLoad(prev_probe_tex, icoord + col_section + vec2( 0,  0));

    let col_blur_section = vec2(0, probe_data_height * 2);
    textureStore(next_probe_tex, icoord + col_blur_section, vec4(mix(neighbors.rgb, center.rgb, vec3(0.25)), 1.0));

    return vec4(0.0);
}