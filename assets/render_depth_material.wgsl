#import bevy_pbr::mesh_view_bindings
#import bevy_pbr::mesh_bindings

#import bevy_pbr::pbr_types
#import bevy_pbr::utils

@group(1) @binding(0)
var texture: texture_2d<f32>;
@group(1) @binding(1)
var texture_sampler: sampler;

struct FragmentInput {
    @builtin(front_facing) is_front: bool,
    @builtin(position) frag_coord: vec4<f32>,
    #import bevy_pbr::mesh_vertex_output
};

@fragment
fn fragment(in: FragmentInput) -> @location(0) vec4<f32> {
    let uv = vec2(in.frag_coord.x/view.viewport.z, in.frag_coord.y/view.viewport.w);
    let tr = textureSampleLevel(texture, texture_sampler, uv, 0.0);
    return vec4(vec3(1.0 / tr.x), 1.0);
}