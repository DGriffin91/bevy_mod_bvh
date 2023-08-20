use bevy::math::{vec2, vec3};

use bevy::prelude::*;

use bevy::render::render_resource::encase::private::WriteInto;
use bevy::render::render_resource::{
    BindGroupEntry, BindGroupLayoutEntry, ShaderSize, ShaderType, StorageBuffer,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{Extract, RenderApp};

use bevy::utils::HashMap;
use bevy_mod_mesh_tools::{mesh_normals, mesh_positions};

use bvh::flat_bvh::FlatNode;
use half::f16;

use crate::packing::{octa_decode, octa_encode};
use crate::{
    bind_group_layout_entry, some_binding_or_return_none, DynamicTLASData, StaticTLASData,
    TraceMesh, BLAS,
};

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

#[derive(ShaderType)]
pub struct VertexDataPacked {
    pub data1: u32,
    pub data2: u32,
}

impl VertexDataPacked {
    pub fn pack(position: Vec3, normal: Vec3) -> Self {
        let mut data1 = f16::from_f32(position.x).to_bits() as u32;
        data1 |= (f16::from_f32(position.y).to_bits() as u32) << 16;
        let mut data2 = f16::from_f32(position.z).to_bits() as u32;
        let oct = octa_encode(normal);
        data2 |= ((oct.x.clamp(0.0, 1.0) * 255.0 + 0.5) as u32) << 16;
        data2 |= ((oct.y.clamp(0.0, 1.0) * 255.0 + 0.5) as u32) << 24;
        Self { data1, data2 }
    }
    pub fn unpack(&self) -> (Vec3, Vec3) {
        let pos = vec3(
            f16::from_bits((self.data1 & 0xffff) as u16).to_f32(),
            f16::from_bits((self.data1 >> 16 & 0xffff) as u16).to_f32(),
            f16::from_bits((self.data2 & 0xffff) as u16).to_f32(),
        );
        let octn = vec2(
            (self.data2 >> 16 & 0xff) as f32,
            (self.data2 >> 24 & 0xff) as f32,
        ) / 255.0;
        let n = octa_decode(octn);
        (pos, n)
    }
}

#[test]
fn vertex_data_pack_test() {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..10000 {
        let pos = vec3(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>()) * 100.0;
        let nor = vec3(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>()).normalize();
        let data = VertexDataPacked::pack(pos, nor);
        let (b_pos, b_nor) = data.unpack();
        assert!(pos.distance(b_pos) < 0.06);
        assert!(b_nor.dot(nor) > 0.999);
    }
}

impl VertexDataPacked {
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
pub struct TLASBVHData {
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    //if positive: entry_idx, if negative: -shape_idx
    pub entry_or_shape_idx: i32,
    pub exit_idx: i32,
}

impl TLASBVHData {
    bind_group_layout_entry!();
}

#[derive(ShaderType)]
pub struct BLASBVHData {
    pub aabb_minxy: u32, // f16 min x, y
    pub aabb_maxxy: u32, // f16 max x, y
    pub aabb_z: u32,     // f16 min z, max z
    //if positive: entry_idx, if negative: -shape_idx
    pub entry_or_shape_idx: i32,
    pub exit_idx: i32,
}

impl BLASBVHData {
    bind_group_layout_entry!();
}

#[derive(Resource, Default)]
pub struct GPUBuffers {
    pub vertex_data_buffer: StorageBuffer<Vec<VertexDataPacked>>,
    pub index_data_buffer: StorageBuffer<Vec<VertexIndices>>,
    pub blas_data_buffer: StorageBuffer<Vec<BLASBVHData>>,
    pub static_tlas_data_buffer: StorageBuffer<Vec<TLASBVHData>>,
    pub dynamic_tlas_data_buffer: StorageBuffer<Vec<TLASBVHData>>,
    pub static_mesh_instance_data_buffer: StorageBuffer<Vec<InstanceData>>,
    pub dynamic_mesh_instance_data_buffer: StorageBuffer<Vec<InstanceData>>,
}

impl GPUBuffers {
    pub fn bind_group_layout_entry(bindings: [u32; 7]) -> [BindGroupLayoutEntry; 7] {
        [
            VertexDataPacked::bind_group_layout_entry(bindings[0]),
            VertexIndices::bind_group_layout_entry(bindings[1]),
            BLASBVHData::bind_group_layout_entry(bindings[2]),
            TLASBVHData::bind_group_layout_entry(bindings[3]),
            TLASBVHData::bind_group_layout_entry(bindings[4]),
            InstanceData::bind_group_layout_entry(bindings[5]),
            InstanceData::bind_group_layout_entry(bindings[6]),
        ]
    }

    pub fn bind_group_entries<'a>(&'a self, bindings: [u32; 7]) -> Option<[BindGroupEntry<'a>; 7]> {
        Some([
            BindGroupEntry {
                binding: bindings[0],
                resource: some_binding_or_return_none!(self.vertex_data_buffer),
            },
            BindGroupEntry {
                binding: bindings[1],
                resource: some_binding_or_return_none!(self.index_data_buffer),
            },
            BindGroupEntry {
                binding: bindings[2],
                resource: some_binding_or_return_none!(self.blas_data_buffer),
            },
            BindGroupEntry {
                binding: bindings[3],
                resource: some_binding_or_return_none!(self.static_tlas_data_buffer),
            },
            BindGroupEntry {
                binding: bindings[4],
                resource: some_binding_or_return_none!(self.dynamic_tlas_data_buffer),
            },
            BindGroupEntry {
                binding: bindings[5],
                resource: some_binding_or_return_none!(self.static_mesh_instance_data_buffer),
            },
            BindGroupEntry {
                binding: bindings[6],
                resource: some_binding_or_return_none!(self.dynamic_mesh_instance_data_buffer),
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
    mesh_entities: Extract<Query<(Entity, &TraceMesh, &GlobalTransform)>>,
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
            let mesh = if let Some(mesh) = meshes.get(mesh_h) {
                mesh
            } else {
                continue;
            };

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
                vertex_data.push(VertexDataPacked::pack(*p, *n));
            }

            let flat_bvh = mesh_blas.bvh.flatten();
            // TODO keep binary mesh data around
            blas_data.append(&mut BLASBVHData::from_flat_node(
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
                TLASBVHData::from_flat_node(&bvh.flatten()),
                "tlas_data",
                &render_device,
                &render_queue,
            );
        }
    }
    if dynamic_tlas.is_changed() {
        if let Some(bvh) = &dynamic_tlas.0.bvh {
            gpu_buffers.dynamic_tlas_data_buffer = new_storage_buffer(
                TLASBVHData::from_flat_node(&bvh.flatten()),
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
            let (entity, tmesh, trans) = mesh_entities.get(item.entity).unwrap();
            if let Some(mesh_idx) = mesh_data_reverse_map.get(&tmesh.mesh_h) {
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
            "static_mesh_instance_data_buffer",
            &render_device,
            &render_queue,
        );
    }
    if dynamic_tlas.is_changed() || blas.is_changed() {
        let mut gpu_mesh_instance_data = Vec::new();
        dynamic_instance_order.0 = Vec::new();
        for item in &dynamic_tlas.0.aabbs {
            let (entity, tmesh, trans) = mesh_entities.get(item.entity).unwrap();
            if let Some(mesh_idx) = mesh_data_reverse_map.get(&tmesh.mesh_h) {
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
            "dynamic_mesh_instance_data_buffer",
            &render_device,
            &render_queue,
        );
    }
}

impl BLASBVHData {
    fn from_flat_node(
        flat_bvh: &[FlatNode<f32, 3>],
        vertex_start: usize,
        vertex_data: &[VertexDataPacked],
        index_data: &[VertexIndices],
        index_start: usize,
    ) -> Vec<Self> {
        let mut bvh_data = Vec::new();
        for f in flat_bvh.iter() {
            let (entry_or_shape_idx, tri_nor, tri_center) = if f.entry_index == u32::MAX {
                let ind = f.shape_index as usize * 3;
                let a = vertex_data[vertex_start + index_data[index_start + ind + 0].idx as usize]
                    .unpack()
                    .0;
                let b = vertex_data[vertex_start + index_data[index_start + ind + 1].idx as usize]
                    .unpack()
                    .0;
                let c = vertex_data[vertex_start + index_data[index_start + ind + 2].idx as usize]
                    .unpack()
                    .0;

                let v1 = b - a;
                let v2 = c - a;

                let normal = v1.cross(v2).normalize();

                let center = (a + b + c) / 3.0;

                // If the entry_index is negative, then it's a leaf node.
                // Shape index in this case is the index into the vertex indices
                (
                    -(f.shape_index as i32 + 1),
                    normal
                        * a.distance(center)
                            .max(b.distance(center))
                            .max(c.distance(center)),
                    center,
                )
            } else {
                (f.entry_index as i32, Vec3::ZERO, Vec3::ZERO)
            };

            let (mut aabb_min, mut aabb_max) = if f32::is_finite(f.aabb.min[0]) {
                let aabb_min = vec3(f.aabb.min[0], f.aabb.min[1], f.aabb.min[2]);
                let aabb_max = vec3(f.aabb.max[0], f.aabb.max[1], f.aabb.max[2]);
                (aabb_min, aabb_max)
            } else {
                (Vec3::ZERO, Vec3::ZERO)
            };

            if f.entry_index == u32::MAX {
                aabb_min = tri_nor;
                aabb_max = tri_center;
            }

            let mut aabb_minxy = f16::from_f32(aabb_min.x).to_bits() as u32;
            aabb_minxy |= (f16::from_f32(aabb_min.y).to_bits() as u32) << 16;

            let mut aabb_maxxy = f16::from_f32(aabb_max.x).to_bits() as u32;
            aabb_maxxy |= (f16::from_f32(aabb_max.y).to_bits() as u32) << 16;

            let mut aabb_z = f16::from_f32(aabb_min.z).to_bits() as u32;
            aabb_z |= (f16::from_f32(aabb_max.z).to_bits() as u32) << 16;

            bvh_data.push(BLASBVHData {
                aabb_minxy,
                aabb_maxxy,
                aabb_z,
                entry_or_shape_idx,
                exit_idx: f.exit_index as i32,
            })
        }
        bvh_data
    }
}

impl TLASBVHData {
    fn from_flat_node(flat_bvh: &[FlatNode<f32, 3>]) -> Vec<Self> {
        let mut bvh_data = Vec::new();
        for f in flat_bvh.iter() {
            let (entry_or_shape_idx, _tri_nor) = if f.entry_index == u32::MAX {
                // If the entry_index is negative, then it's a leaf node.
                // Shape index in this case is the index into the vertex indices
                (-(f.shape_index as i32 + 1), Vec3::ZERO)
            } else {
                (f.entry_index as i32, Vec3::ZERO)
            };

            let (aabb_min, aabb_max) = if f32::is_finite(f.aabb.min[0]) {
                let aabb_min = vec3(f.aabb.min[0], f.aabb.min[1], f.aabb.min[2]);
                let aabb_max = vec3(f.aabb.max[0], f.aabb.max[1], f.aabb.max[2]);
                (aabb_min, aabb_max)
            } else {
                (Vec3::ZERO, Vec3::ZERO)
            };

            bvh_data.push(TLASBVHData {
                aabb_min,
                aabb_max,
                entry_or_shape_idx,
                exit_idx: f.exit_index as i32,
            });
        }
        bvh_data
    }
}
