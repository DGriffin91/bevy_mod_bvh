use bevy::core::cast_slice;

use bevy::prelude::*;

use bevy::render::render_resource::{
    Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};

use bevy::utils::HashMap;
use bevy_mod_mesh_tools::mesh_positions;

use bvh::bvh::BVH;

use crate::{BVHSet, DynamicTLASData, StaticTLASData, BLAS, TLAS};

use bytemuck::{cast, NoUninit};

use num_integer::Roots;

pub struct GPUDataPlugin;
impl Plugin for GPUDataPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            (
                update_vertices_indices_blas_data,
                apply_system_buffers,
                update_tlas_data,
                update_instance_data,
            )
                .chain()
                .in_set(BVHSet::GpuData)
                .after(BVHSet::BlasTlas),
        );
    }
}

#[derive(NoUninit, Clone, Copy)]
#[repr(C)]
pub struct MeshData {
    pub index_start: u32,
    pub index_len: u32,
    pub pos_start: u32,
    pub pos_len: u32,
    pub blas_start: u32,
    pub blas_len: u32,
    pub blas_count: u32,
}

//-shape_idx * instance_data_len is index into static/dynamic instance data
#[derive(Resource)]
pub struct GpuStaticTlasData(Handle<Image>);
#[derive(Resource)]
pub struct GpuDynamicTlasData(Handle<Image>);

#[derive(Resource)]
pub struct GpuMeshData {
    // Vec4 with w unused
    pub vert_positions: Handle<Image>,

    // Uint idx flat as tri abcabcabc...
    pub vert_indices: Handle<Image>,

    // All BVH data is:
    // Vec4Vec4 aabb min/max = xyz,
    // 1st w (bitcast to i32) is entry_index  or -shape_idx
    // 2nd w (bitcast to i32) is exit_index

    //-shape_idx * 3 is index into vert_indices
    pub blas: Handle<Image>,

    // all u32
    // index_start, index_len, pos_start, pos_len, blas_start, blas_len, blas_count
    pub mesh_data: Handle<Image>,

    // use idx * len(MeshData)
    pub mesh_data_reverse_map: HashMap<Handle<Mesh>, usize>,
}

/*
    TODO the user should provide these in the order of tlas.0.aabbs
    So that BVH traversal will point to the correct instance index
    InstanceData {
        index into mesh data
        index into material data
        transform/model (probably trans.compute_matrix().inverse())
    }
*/
// currently diffuse rgba, emit rgba, u32 index into mesh_data, inv mat: Vec4Vec4Vec4Vec4
#[derive(Resource)]
pub struct GpuStaticInstanceData(Handle<Image>);
#[derive(Resource)]
pub struct GpuDynamicInstanceData(Handle<Image>);

// TODO don't include all meshes
fn update_vertices_indices_blas_data(
    mut commands: Commands,
    mesh_events: EventReader<AssetEvent<Mesh>>,
    meshes: Res<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    blas: Res<BLAS>,
) {
    // any time a mesh is added/removed/modified run get all the vertices and put them into a texture
    // put buffer mesh start and length into resource
    if mesh_events.is_empty() {
        return;
    }
    dbg!("update_vertices_indices_blas_data");
    let mut mesh_data = Vec::new();
    let mut index_data = Vec::new();
    let mut pos_data = Vec::new();
    let mut blas_data = Vec::new();
    let mut mesh_data_reverse_map = HashMap::new();

    for (_mesh_idx, (id, mesh)) in meshes.iter().enumerate() {
        let mesh_h = meshes.get_handle(id);
        let blas_start = blas_data.len();
        let index_start = index_data.len();
        let pos_start = pos_data.len();
        let indices = match mesh.indices().unwrap() {
            bevy::render::mesh::Indices::U16(_) => panic!("U16 Indices not supported"),
            bevy::render::mesh::Indices::U32(a) => a,
        };
        for index in indices {
            index_data.push(*index);
        }

        for p in mesh_positions(&mesh) {
            pos_data.push(p.extend(0.0));
        }

        let mesh_blas = blas.0.get(&mesh_h).unwrap();

        let flat_bvh = mesh_blas.bvh.flatten();

        // TODO keep binary mesh data around
        for f in flat_bvh.iter() {
            let entry_index = if f.entry_index == u32::MAX {
                // If the entry_index is negative, then it's a leaf node.
                // Shape index in this case is the index into the vertex indices
                cast(-(f.shape_index as i32 + 1))
            } else {
                cast(f.entry_index as i32)
            };
            blas_data.push(f.aabb.min.extend(entry_index));
            blas_data.push(f.aabb.max.extend(cast(f.exit_index as i32)));
            // To retrieve triangle data:
            /*
                let aabb_min = get_bvh(next_idx * 2 + 0);
                let aabb_max = get_bvh(next_idx * 2 + 1);
                let entry_idx = bitcast<i32>(aabb_min.w);
                let exit_idx = bitcast<i32>(aabb_max.w);
                if entry_idx < 0 {
                    var shape_idx = (entry_idx + 1) * -3;
                    shape_idx += mesh_index_start; // offset to where this mesh starts in the buffer
                    let ind1 = get_ind(shape_idx+0); // read from ind buffer
                    let ind2 = get_ind(shape_idx+1);
                    let ind3 = get_ind(shape_idx+2);
                    let p1 = get_vert(ind1 * 2); // retrieve vertex positions
                    let p2 = get_vert(ind2 * 2);
                    let p3 = get_vert(ind3 * 2);
                    let intr = intersects_triangle(ray, p1, p2, p3);
            */
        }

        let mesh_data_idx = mesh_data.len();
        mesh_data_reverse_map.insert(mesh_h.clone(), mesh_data_idx);

        mesh_data.push(MeshData {
            index_start: index_start as u32,
            index_len: (index_data.len() - index_start) as u32,
            pos_start: pos_start as u32,
            pos_len: (index_data.len() - index_start) as u32,
            blas_start: blas_start as u32,
            blas_len: (blas_data.len() - blas_start) as u32,
            blas_count: flat_bvh.len() as u32,
        });
    }

    commands.insert_resource(GpuMeshData {
        vert_indices: images.add(u32_image(&index_data)),
        vert_positions: images.add(f32rgba_image(&pos_data)),
        blas: images.add(f32rgba_image(&blas_data)),
        mesh_data: images.add(u32_image(&cast_slice(&mesh_data))),
        mesh_data_reverse_map,
    })
}

fn update_tlas_data(
    mut commands: Commands,
    static_tlas: Res<StaticTLASData>,
    dynamic_tlas: Res<DynamicTLASData>,
    mut images: ResMut<Assets<Image>>,
) {
    // when the static or dynamic tlas is modified remake flattened version into texture
    if static_tlas.is_changed() {
        if let Some(bvh) = &static_tlas.0.bvh {
            dbg!("update_static_tlas");
            commands.insert_resource(GpuStaticTlasData(images.add(create_gpu_tlas_data(bvh))));
        }
    }
    if dynamic_tlas.is_changed() {
        if let Some(bvh) = &dynamic_tlas.0.bvh {
            commands.insert_resource(GpuDynamicTlasData(images.add(create_gpu_tlas_data(bvh))));
        }
    }
}

fn create_gpu_tlas_data(bvh: &BVH) -> Image {
    let mut tlas_data = Vec::new();
    let flat_bvh = bvh.flatten();
    for f in flat_bvh.iter() {
        let entry_index = if f.entry_index == u32::MAX {
            // If the entry_index is negative, then it's a leaf node.
            // Shape index in this case is the mesh entity instance index
            // Look up the equivalent info as: static_tlas.0.aabbs[shape_index].entity
            // Order entity instances on the GPU per static_tlas.0.aabbs
            cast(-(f.shape_index as i32 + 1))
        } else {
            cast(f.entry_index as i32)
        };
        tlas_data.push(f.aabb.min.extend(entry_index));
        tlas_data.push(f.aabb.max.extend(cast(f.exit_index as i32)));
    }
    f32rgba_image(&tlas_data)
}

// TODO temporary, user needs to implement
fn update_instance_data(
    mut commands: Commands,
    static_tlas: Res<StaticTLASData>,
    dynamic_tlas: Res<DynamicTLASData>,
    gpu_data: Option<Res<GpuMeshData>>,
    mesh_entities: Query<(&Handle<Mesh>, &GlobalTransform, &Handle<StandardMaterial>)>,
    materials: Res<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(gpu_data) = gpu_data else {
        return;
    };

    if static_tlas.is_changed() {
        dbg!("GpuStaticInstanceData");
        commands.insert_resource(GpuStaticInstanceData(images.add(create_instance_data(
            &static_tlas.0,
            &mesh_entities,
            &materials,
            &gpu_data,
        ))));
    }
    if dynamic_tlas.is_changed() {
        commands.insert_resource(GpuDynamicInstanceData(images.add(create_instance_data(
            &dynamic_tlas.0,
            &mesh_entities,
            &materials,
            &gpu_data,
        ))));
    }
}

fn create_instance_data(
    tlas: &TLAS,
    mesh_entities: &Query<(&Handle<Mesh>, &GlobalTransform, &Handle<StandardMaterial>)>,
    materials: &Assets<StandardMaterial>,
    gpu_data: &GpuMeshData,
) -> Image {
    let mut instance_gpu_data = Vec::new();
    for item in &tlas.aabbs {
        let (mesh_h, trans, material_h) = mesh_entities.get(item.entity).unwrap();
        let material = materials.get(material_h).unwrap();
        // currently diffuse rgba, emit rgba, index into mesh_data, inv mat: Vec4Vec4Vec4Vec4
        instance_gpu_data.push(cast(material.base_color.as_linear_rgba_f32()));
        instance_gpu_data.push(cast(material.emissive.as_linear_rgba_f32()));
        let mesh_data_idx = gpu_data.mesh_data_reverse_map[mesh_h];
        instance_gpu_data.push(Vec4::new(cast(mesh_data_idx as u32), 0.0, 0.0, 0.0));
        let inv_mat = trans.compute_matrix().inverse();
        instance_gpu_data.push(inv_mat.x_axis);
        instance_gpu_data.push(inv_mat.y_axis);
        instance_gpu_data.push(inv_mat.z_axis);
        instance_gpu_data.push(inv_mat.w_axis);
    }
    f32rgba_image(&instance_gpu_data)
}

pub fn u32_image(u32data: &[u32]) -> Image {
    let dimension = (u32data.len().sqrt() + 1).next_power_of_two();
    let mut img = Image {
        //dimension * dimension image with 1 u32 which are 4 bytes
        data: vec![0u8; dimension * dimension * 1 * 4],
        texture_descriptor: TextureDescriptor {
            label: None,
            size: Extent3d {
                width: dimension as u32,
                height: dimension as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Uint,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        ..default()
    };
    let data = cast_slice::<u32, u8>(&u32data).to_vec();
    img.data.splice(0..data.len(), data);
    img
}

pub fn f32rgba_image(vec4data: &[Vec4]) -> Image {
    let dimension = (vec4data.len().sqrt() + 1).next_power_of_two();
    let mut img = Image {
        //dimension * dimension image with 4 f32s which are 4 bytes
        data: vec![0u8; dimension * dimension * 4 * 4],
        texture_descriptor: TextureDescriptor {
            label: None,
            size: Extent3d {
                width: dimension as u32,
                height: dimension as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        ..default()
    };
    let data = cast_slice::<Vec4, u8>(&vec4data).to_vec();
    img.data.splice(0..data.len(), data);
    img
}
