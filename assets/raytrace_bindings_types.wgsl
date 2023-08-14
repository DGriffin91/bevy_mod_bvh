#import "trace_gpu_types.wgsl" as gputypes
#import bevy_pbr::mesh_view_bindings view, globals

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
var<storage> vertex_buffer: array<gputypes::VertexDataPacked>;
@group(0) @binding(5)
var<storage> index_buffer: array<gputypes::VertexIndices>;
@group(0) @binding(6)
var<storage> blas_buffer: array<gputypes::BLASBVHData>;
@group(0) @binding(7)
var<storage> static_tlas_buffer: array<gputypes::TLASBVHData>;
@group(0) @binding(8)
var<storage> dynamic_tlas_buffer: array<gputypes::TLASBVHData>;
@group(0) @binding(9)
var<storage> static_mesh_instance_buffer: array<gputypes::InstanceData>;
@group(0) @binding(10)
var<storage> dynamic_mesh_instance_buffer: array<gputypes::InstanceData>;