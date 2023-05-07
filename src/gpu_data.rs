use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy::math::vec3;
use bevy::pbr::{MAX_CASCADES_PER_LIGHT, MAX_DIRECTIONAL_LIGHTS};
use bevy::prelude::*;

use bevy::render::render_resource::encase::private::WriteInto;
use bevy::render::render_resource::{
    BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingType, BufferBindingType,
    CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState, MultisampleState,
    PipelineCache, PrimitiveState, RenderPipelineDescriptor, SamplerBindingType, ShaderDefVal,
    ShaderSize, ShaderStages, ShaderType, StorageBuffer, TextureFormat,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::texture::BevyDefault;
use bevy::render::view::ViewUniform;
use bevy::render::{Extract, RenderApp};

use bevy::utils::HashMap;
use bevy_mod_mesh_tools::{mesh_normals, mesh_positions};

use bvh::flat_bvh::FlatNode;

use crate::{DynamicTLASData, StaticTLASData, BLAS};

pub struct GPUDataPlugin;
impl Plugin for GPUDataPlugin {
    fn build(&self, app: &mut App) {
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<GPUBuffers>()
            .init_resource::<StaticInstanceOrder>()
            .init_resource::<DynamicInstanceOrder>()
            .add_systems(ExtractSchedule, extract_gpu_data);
    }
}

#[macro_export]
macro_rules! bind_group_layout_entry {
    () => {
        pub fn bind_group_layout_entry(
            binding: u32,
        ) -> bevy::render::render_resource::BindGroupLayoutEntry {
            bevy::render::render_resource::BindGroupLayoutEntry {
                binding,
                visibility: bevy::render::render_resource::ShaderStages::FRAGMENT
                    | bevy::render::render_resource::ShaderStages::COMPUTE,
                ty: bevy::render::render_resource::BindingType::Buffer {
                    ty: bevy::render::render_resource::BufferBindingType::Storage {
                        read_only: true,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: Some(Self::min_size()),
                },
                count: None,
            }
        }
    };
}

#[derive(ShaderType)]
pub struct VertexData {
    pub position: Vec3,
    pub normal: Vec3,
}

impl VertexData {
    bind_group_layout_entry!();
}

#[derive(ShaderType, Clone, Copy)]
pub struct MeshData {
    // all i32 so conversion doesn't need to happen on gpu
    pub vert_idx_start: i32,
    pub vert_data_start: i32,
    pub blas_start: i32,
    pub blas_count: i32,
}

#[derive(ShaderType)]
pub struct InstanceData {
    pub local_to_world: Mat4,
    pub world_to_local: Mat4,
    pub mesh_data: MeshData,
}

impl InstanceData {
    bind_group_layout_entry!();
}

#[derive(ShaderType)]
pub struct VertexIndices {
    pub idx: u32,
}

impl VertexIndices {
    bind_group_layout_entry!();
}

#[derive(ShaderType)]
pub struct BVHData {
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    // precomputed triangle normals
    pub tri_nor: Vec3, //TODO use smaller, less precise format
    //if positive: entry_idx if negative: -shape_idx
    pub entry_or_shape_idx: i32,
    pub exit_idx: i32,
}

impl BVHData {
    bind_group_layout_entry!();
}

#[derive(Resource, Default)]
pub struct GPUBuffers {
    pub vertex_data_buffer: StorageBuffer<Vec<VertexData>>,
    pub index_data_buffer: StorageBuffer<Vec<VertexIndices>>,
    pub blas_data_buffer: StorageBuffer<Vec<BVHData>>,
    pub static_tlas_data_buffer: StorageBuffer<Vec<BVHData>>,
    pub dynamic_tlas_data_buffer: StorageBuffer<Vec<BVHData>>,
    pub static_mesh_instance_data_buffer: StorageBuffer<Vec<InstanceData>>,
    pub dynamic_mesh_instance_data_buffer: StorageBuffer<Vec<InstanceData>>,
}

macro_rules! some_or_return_none {
    ($buffer:expr) => {{
        let Some(r) = $buffer.binding() else {return None};
        r
    }};
}

impl GPUBuffers {
    pub fn bind_group_layout_entry(bindings: [u32; 7]) -> [BindGroupLayoutEntry; 7] {
        [
            VertexData::bind_group_layout_entry(bindings[0]),
            VertexIndices::bind_group_layout_entry(bindings[1]),
            BVHData::bind_group_layout_entry(bindings[2]),
            BVHData::bind_group_layout_entry(bindings[3]),
            BVHData::bind_group_layout_entry(bindings[4]),
            InstanceData::bind_group_layout_entry(bindings[5]),
            InstanceData::bind_group_layout_entry(bindings[6]),
        ]
    }

    pub fn bind_group_entries<'a>(&'a self, bindings: [u32; 7]) -> Option<[BindGroupEntry<'a>; 7]> {
        Some([
            BindGroupEntry {
                binding: bindings[0],
                resource: some_or_return_none!(self.vertex_data_buffer),
            },
            BindGroupEntry {
                binding: bindings[1],
                resource: some_or_return_none!(self.index_data_buffer),
            },
            BindGroupEntry {
                binding: bindings[2],
                resource: some_or_return_none!(self.blas_data_buffer),
            },
            BindGroupEntry {
                binding: bindings[3],
                resource: some_or_return_none!(self.static_tlas_data_buffer),
            },
            BindGroupEntry {
                binding: bindings[4],
                resource: some_or_return_none!(self.dynamic_tlas_data_buffer),
            },
            BindGroupEntry {
                binding: bindings[5],
                resource: some_or_return_none!(self.static_mesh_instance_data_buffer),
            },
            BindGroupEntry {
                binding: bindings[6],
                resource: some_or_return_none!(self.dynamic_mesh_instance_data_buffer),
            },
        ])
    }
}

pub fn new_storage_buffer<T: ShaderSize + WriteInto>(
    vec: Vec<T>,
    label: &'static str,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
) -> StorageBuffer<Vec<T>> {
    let mut buffer = StorageBuffer::default();
    buffer.set(vec);
    buffer.set_label(Some(label));
    buffer.write_buffer(render_device, render_queue);
    buffer
}

#[derive(Resource, Default)]
pub struct StaticInstanceOrder(pub Vec<Entity>);

#[derive(Resource, Default)]
pub struct DynamicInstanceOrder(pub Vec<Entity>);

pub fn extract_gpu_data(
    meshes: Extract<Res<Assets<Mesh>>>,
    blas: Extract<Res<BLAS>>,
    static_tlas: Extract<Res<StaticTLASData>>,
    dynamic_tlas: Extract<Res<DynamicTLASData>>,
    mut instance_mesh_data: Local<Vec<MeshData>>,
    mut mesh_data_reverse_map: Local<HashMap<Handle<Mesh>, usize>>,
    mesh_entities: Extract<Query<(Entity, &Handle<Mesh>, &GlobalTransform)>>,
    mut gpu_buffers: ResMut<GPUBuffers>,
    mut static_instance_order: ResMut<StaticInstanceOrder>,
    mut dynamic_instance_order: ResMut<DynamicInstanceOrder>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    // any time a mesh is added/removed/modified run get all the vertices and put them into a texture
    // put buffer mesh start and length into resource
    if blas.is_changed() {
        let mut index_data = Vec::new();
        let mut vertex_data = Vec::new();
        let mut blas_data = Vec::new();
        *instance_mesh_data = Vec::new();
        *mesh_data_reverse_map = HashMap::new();
        for (mesh_h, mesh_blas) in blas.0.iter() {
            let mesh = meshes.get(mesh_h).unwrap();
            let blas_start = blas_data.len();
            let index_start = index_data.len();
            let vertex_start = vertex_data.len();

            match mesh.indices().unwrap() {
                bevy::render::mesh::Indices::U16(indices) => {
                    for index in indices {
                        index_data.push(VertexIndices { idx: *index as u32 });
                    }
                }
                bevy::render::mesh::Indices::U32(indices) => {
                    for index in indices {
                        index_data.push(VertexIndices { idx: *index as u32 });
                    }
                }
            };

            for (p, n) in mesh_positions(mesh).zip(mesh_normals(mesh)) {
                vertex_data.push(VertexData {
                    position: *p,
                    normal: *n,
                })
            }

            let flat_bvh = mesh_blas.bvh.flatten();
            // TODO keep binary mesh data around
            blas_data.append(&mut BVHData::from_flat_node_blas(
                &flat_bvh,
                vertex_start,
                &vertex_data,
                &index_data,
                index_start,
            ));

            let mesh_data_idx = instance_mesh_data.len();
            instance_mesh_data.push(MeshData {
                blas_start: blas_start as i32,
                blas_count: flat_bvh.len() as i32,
                vert_idx_start: index_start as i32,
                vert_data_start: vertex_start as i32,
            });
            mesh_data_reverse_map.insert(mesh_h.clone(), mesh_data_idx);
        }
        gpu_buffers.vertex_data_buffer =
            new_storage_buffer(vertex_data, "vertex_data", &render_device, &render_queue);
        gpu_buffers.index_data_buffer =
            new_storage_buffer(index_data, "index_data", &render_device, &render_queue);
        gpu_buffers.blas_data_buffer =
            new_storage_buffer(blas_data, "blas_data", &render_device, &render_queue);
    }
    if static_tlas.is_changed() {
        if let Some(bvh) = &static_tlas.0.bvh {
            gpu_buffers.static_tlas_data_buffer = new_storage_buffer(
                BVHData::from_flat_node_tlas(&bvh.flatten()),
                "tlas_data",
                &render_device,
                &render_queue,
            );
        }
    }
    if dynamic_tlas.is_changed() {
        if let Some(bvh) = &dynamic_tlas.0.bvh {
            gpu_buffers.dynamic_tlas_data_buffer = new_storage_buffer(
                BVHData::from_flat_node_tlas(&bvh.flatten()),
                "tlas_data",
                &render_device,
                &render_queue,
            );
        }
    }

    if static_tlas.is_changed() || blas.is_changed() {
        let mut gpu_mesh_instance_data = Vec::new();
        static_instance_order.0 = Vec::new();
        for item in &static_tlas.0.aabbs {
            let (entity, mesh_h, trans) = mesh_entities.get(item.entity).unwrap();
            if let Some(mesh_idx) = mesh_data_reverse_map.get(mesh_h) {
                static_instance_order.0.push(entity.clone());
                let mesh_data = instance_mesh_data[*mesh_idx];
                let local_to_world = trans.compute_matrix();
                gpu_mesh_instance_data.push(InstanceData {
                    local_to_world,
                    world_to_local: local_to_world.inverse(),
                    mesh_data,
                });
            }
        }
        gpu_buffers.static_mesh_instance_data_buffer = new_storage_buffer(
            gpu_mesh_instance_data,
            "gpu_mesh_instance_data",
            &render_device,
            &render_queue,
        );
    }
    if dynamic_tlas.is_changed() || blas.is_changed() {
        let mut gpu_mesh_instance_data = Vec::new();
        dynamic_instance_order.0 = Vec::new();
        for item in &dynamic_tlas.0.aabbs {
            let (entity, mesh_h, trans) = mesh_entities.get(item.entity).unwrap();
            if let Some(mesh_idx) = mesh_data_reverse_map.get(mesh_h) {
                dynamic_instance_order.0.push(entity.clone());
                let mesh_data = instance_mesh_data[*mesh_idx];
                let local_to_world = trans.compute_matrix();
                gpu_mesh_instance_data.push(InstanceData {
                    local_to_world,
                    world_to_local: local_to_world.inverse(),
                    mesh_data,
                });
            }
        }
        gpu_buffers.dynamic_mesh_instance_data_buffer = new_storage_buffer(
            gpu_mesh_instance_data,
            "gpu_mesh_instance_data",
            &render_device,
            &render_queue,
        );
    }
}

impl BVHData {
    fn from_flat_node_blas(
        flat_bvh: &[FlatNode<f32, 3>],
        vertex_start: usize,
        vertex_data: &[VertexData],
        index_data: &[VertexIndices],
        index_start: usize,
    ) -> Vec<Self> {
        let mut bvh_data = Vec::new();
        for f in flat_bvh.iter() {
            let (entry_or_shape_idx, tri_nor) = if f.entry_index == u32::MAX {
                let ind = f.shape_index as usize * 3;
                let a = vertex_data[vertex_start + index_data[index_start + ind + 0].idx as usize]
                    .position;
                let b = vertex_data[vertex_start + index_data[index_start + ind + 1].idx as usize]
                    .position;
                let c = vertex_data[vertex_start + index_data[index_start + ind + 2].idx as usize]
                    .position;

                let v1 = b - a;
                let v2 = c - a;

                let normal = v1.cross(v2).normalize();

                // If the entry_index is negative, then it's a leaf node.
                // Shape index in this case is the index into the vertex indices
                (-(f.shape_index as i32 + 1), normal)
            } else {
                (f.entry_index as i32, Vec3::ZERO)
            };

            bvh_data.push(BVHData {
                //TODO is there something faster? Transmute?
                aabb_min: vec3(f.aabb.min[0], f.aabb.min[1], f.aabb.min[2]),
                aabb_max: vec3(f.aabb.max[0], f.aabb.max[1], f.aabb.max[2]),
                tri_nor,
                entry_or_shape_idx,
                exit_idx: f.exit_index as i32,
            })
        }
        bvh_data
    }

    fn from_flat_node_tlas(flat_bvh: &[FlatNode<f32, 3>]) -> Vec<Self> {
        let mut bvh_data = Vec::new();
        for f in flat_bvh.iter() {
            let (entry_or_shape_idx, tri_nor) = if f.entry_index == u32::MAX {
                // If the entry_index is negative, then it's a leaf node.
                // Shape index in this case is the index into the vertex indices
                (-(f.shape_index as i32 + 1), Vec3::ZERO)
            } else {
                (f.entry_index as i32, Vec3::ZERO)
            };

            bvh_data.push(BVHData {
                aabb_min: vec3(f.aabb.min[0], f.aabb.min[1], f.aabb.min[2]),
                aabb_max: vec3(f.aabb.max[0], f.aabb.max[1], f.aabb.max[2]),
                tri_nor,
                entry_or_shape_idx,
                exit_idx: f.exit_index as i32,
            });
        }
        bvh_data
    }
}

pub fn sampler_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
        count: None,
    }
}

pub fn view_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: Some(ViewUniform::min_size()),
        },
        count: None,
    }
}

pub fn get_default_pipeline_desc(
    mut shader_defs: Vec<ShaderDefVal>,
    layout: BindGroupLayout,
    pipeline_cache: &mut PipelineCache,
    shader: Handle<Shader>,
    hdr: bool,
) -> CachedRenderPipelineId {
    shader_defs.push(ShaderDefVal::UInt(
        "MAX_DIRECTIONAL_LIGHTS".to_string(),
        MAX_DIRECTIONAL_LIGHTS as u32,
    ));
    shader_defs.push(ShaderDefVal::UInt(
        "MAX_CASCADES_PER_LIGHT".to_string(),
        MAX_CASCADES_PER_LIGHT as u32,
    ));

    pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("post_process_pipeline".into()),
        layout: vec![layout],
        vertex: fullscreen_shader_vertex_state(),
        fragment: Some(FragmentState {
            shader,
            shader_defs,
            entry_point: "fragment".into(),
            targets: vec![Some(ColorTargetState {
                format: if hdr {
                    TextureFormat::Rgba16Float
                } else {
                    TextureFormat::bevy_default()
                },
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        push_constant_ranges: vec![],
    })
}
