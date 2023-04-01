
#import "common.wgsl"

#import bevy_pbr::mesh_view_bindings
#import bevy_pbr::mesh_bindings

#import bevy_pbr::pbr_types
#import bevy_pbr::utils

@group(1) @binding(0)
var ddgi_texture: texture_2d<f32>;
@group(1) @binding(1)
var ddgi_texture_sampler: sampler;
@group(1) @binding(2)
var texture2: texture_2d<f32>;
@group(1) @binding(3)
var texture_sampler2: sampler;

struct FragmentInput {
    @builtin(front_facing) is_front: bool,
    @builtin(position) position: vec4<f32>,
    #import bevy_pbr::mesh_vertex_output
};

const CAS = vec3<i32>(12, 12, 12);
const CASF = vec3<f32>(12.0, 12.0, 12.0);
const SIZE = 8;
const FSIZE = 8.0;

#import "ddgi.wgsl"

fn basic_sample(N: vec3<f32>, ws_pos: vec3<f32>) -> vec3<f32> {
    let probe_spacing = 1.5;
    var ws_offset = vec3(probe_spacing * -4.0, 0.5, probe_spacing * -4.0);
    ws_offset -= probe_spacing * 0.5;

    let ls_probe_no_clamp = floor((ws_pos - ws_offset) / probe_spacing);
    let ls_hit_probe = clamp(ls_probe_no_clamp, vec3(0.0), CASF);
    let probe_tex_x = ls_hit_probe.x * FSIZE + ls_hit_probe.y * FSIZE * f32(CAS.x);
    let probe_tex_y = ls_hit_probe.z * FSIZE;

    let probe_uv = oct_encode_n(N, FSIZE, 1.0);

    let hit_probe_coord = vec2(probe_uv.x * FSIZE + probe_tex_x,
                               probe_uv.y * FSIZE + probe_tex_y);
    let tex_size = vec2<f32>(textureDimensions(ddgi_texture).xy);

    let probe_data_height = CAS.z * SIZE;
    let tuv = vec2<f32>(hit_probe_coord + vec2(0.0, f32(probe_data_height))) / tex_size;
    let refl_color = textureSampleLevel(ddgi_texture, ddgi_texture_sampler, tuv, 0.0);
    return refl_color.rgb;
}

@fragment
fn fragment(in: FragmentInput) -> @location(0) vec4<f32> {
    let coord = in.position.xy;
    let uv = coord / view.viewport.zw;
    let V = normalize(view.world_position.xyz - in.world_position.xyz);
    let N = normalize(in.world_normal);



//    let col = textureSample(ddgi_texture, ddgi_texture_sampler, uv);




    let ir = irradiance_DDGI(N, V, in.world_position.xyz);

    //if ls_probe_no_clamp.x != ls_hit_probe.x ||
    //   ls_probe_no_clamp.y != ls_hit_probe.y ||
    //   ls_probe_no_clamp.z != ls_hit_probe.z {
    //    return vec4(vec3(1.0,0.0,1.0), 1.0);
    //}
    //return vec4(1.0 - 1.0/vec3(col), 1.0);
    return vec4(ir, 1.0);


}