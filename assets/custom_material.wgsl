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
    @builtin(position) position: vec4<f32>,
    #import bevy_pbr::mesh_vertex_output
};


@fragment
fn fragment(in: FragmentInput) -> @location(0) vec4<f32> {
    let coord = in.position.xy;
    let uv = coord / view.viewport.zw;

    let col = textureSample(texture, texture_sampler, uv).rgb;

    return vec4(col, 1.0);
}