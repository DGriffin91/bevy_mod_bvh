use bevy::core::cast_slice;

use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy::math::{vec4, Vec4Swizzles};
use bevy::pbr::{MAX_CASCADES_PER_LIGHT, MAX_DIRECTIONAL_LIGHTS};
use bevy::prelude::*;

use bevy::render::primitives::Aabb;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType,
    BufferBindingType, CachedRenderPipelineId, ColorTargetState, ColorWrites, Extent3d,
    FragmentState, MultisampleState, PipelineCache, PrimitiveState, RenderPipelineDescriptor,
    SamplerBindingType, ShaderDefVal, ShaderStages, ShaderType, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureViewDimension,
};
use bevy::render::texture::BevyDefault;
use bevy::render::view::ViewUniform;
use bevy::render::{Extract, RenderApp};

use bevy::utils::HashMap;
use bevy_mod_mesh_tools::{mesh_normals, mesh_positions};

use bvh::aabb::AABB;
use bvh::bvh::BVH;

use crate::{BVHSet, DynamicTLASData, StaticTLASData, BLAS, TLAS};

use bytemuck::{cast, NoUninit};

use num_integer::Roots;

pub struct GPUDataPlugin;
impl Plugin for GPUDataPlugin {
    fn build(&self, app: &mut App) {
        app.add_startup_system(init_gpu_data).add_systems(
            (
                update_vertices_indices_blas_data,
                update_tlas_data,
                update_instance_data,
            )
                .chain()
                .in_set(BVHSet::GpuData)
                .after(BVHSet::BlasTlas),
        );

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.add_system(extract_gpu_data.in_schedule(ExtractSchedule));
    }
}

#[derive(NoUninit, Clone, Copy)]
#[repr(C)]
pub struct MeshData {
    pub index_start: i32,
    pub pos_start: i32,
    pub blas_start: i32,
    pub blas_count: i32,
}

// Data is currently all stored as i32 to avoid conversion
// in hot paths on the GPU. Need to evaluate if this makes sense.
#[derive(Resource, Clone)]
pub struct GpuData {
    //-shape_idx * instance_data_len is index into static/dynamic instance data
    pub gpu_static_tlas_data: Handle<Image>,
    pub gpu_dynamic_tlas_data: Handle<Image>,

    // Vec4 with w unused
    pub vert_pos: Handle<Image>,
    pub vert_nor: Handle<Image>,
    // precomputed triangle normals vec4 with w as distance to center of AABB
    pub tri_nor: Handle<Image>,

    // Uint idx flat as tri abcabcabc...
    pub vert_indices: Handle<Image>,

    // All BVH data is:
    // Vec4Vec4 aabb min/max = xyz,
    // 1st w (bitcast to i32) is entry_index  or -shape_idx
    // 2nd w (bitcast to i32) is exit_index

    //-shape_idx * 3 is index into vert_indices
    pub blas: Handle<Image>,

    // all i32
    // index_start, pos_start, blas_start, blas_count
    pub mesh_data: Handle<Image>,

    // use idx * len(MeshData)
    pub mesh_data_reverse_map: HashMap<Handle<Mesh>, usize>,

    /*
        TODO the user should provide these in the order of tlas.0.aabbs
        So that BVH traversal will point to the correct instance index
        InstanceData {
            index into mesh data
            index into material data
            transform/model (probably trans.compute_matrix().inverse())
        }
    */
    // currently diffuse rgba, emit rgba, i32 index into mesh_data, inv mat: Vec4Vec4Vec4Vec4
    pub gpu_static_instance_data: Handle<Image>,
    pub gpu_dynamic_instance_data: Handle<Image>,
}

fn init_gpu_data(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    commands.insert_resource(GpuData {
        gpu_static_tlas_data: images
            .add(f32rgba_image(&[Vec4::ZERO], "uninit_gpu_static_tlas_data")),
        gpu_dynamic_tlas_data: images
            .add(f32rgba_image(&[Vec4::ZERO], "uninit_gpu_dynamic_tlas_data")),
        vert_pos: images.add(f32rgba_image(&[Vec4::ZERO], "uninit_vert_positions")),
        vert_nor: images.add(f32rgba_image(&[Vec4::ZERO], "uninit_vert_normals")),
        tri_nor: images.add(f32rgba_image(&[Vec4::ZERO], "uninit_tri_normals")),
        vert_indices: images.add(i32_image(&[0], "uninit_vert_indices")),
        blas: images.add(f32rgba_image(&[Vec4::ZERO], "uninit_blas")),
        mesh_data: images.add(i32_image(&[0], "uninit_mesh_data")),
        mesh_data_reverse_map: HashMap::new(),
        gpu_static_instance_data: images.add(f32rgba_image(
            &[Vec4::ZERO],
            "uninit_gpu_static_instance_data",
        )),
        gpu_dynamic_instance_data: images.add(f32rgba_image(
            &[Vec4::ZERO],
            "uninit_gpu_dynamic_instance_data",
        )),
    })
}

// TODO don't include all meshes
fn update_vertices_indices_blas_data(
    mesh_events: EventReader<AssetEvent<Mesh>>,
    meshes: Res<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    blas: Res<BLAS>,
    mut gpu_data: ResMut<GpuData>,
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
    let mut nor_data = Vec::new();
    let mut tri_nor_data = Vec::new();
    let mut blas_data = Vec::new();
    let mut mesh_data_reverse_map = HashMap::new();

    let mut largest = 0;
    for (_mesh_idx, (id, mesh)) in meshes.iter().enumerate() {
        let mesh_h = meshes.get_handle(id);
        let blas_start = blas_data.len();
        let index_start = index_data.len();
        let pos_start = pos_data.len(); // can be reused for normals
        let indices = match mesh.indices().unwrap() {
            bevy::render::mesh::Indices::U16(_) => panic!("U16 Indices not supported"),
            bevy::render::mesh::Indices::U32(a) => a,
        };
        for index in indices {
            index_data.push(*index as i32);
        }

        for (p, n) in mesh_positions(&mesh).zip(mesh_normals(&mesh)) {
            pos_data.push(p.extend(0.0));
            nor_data.push(n.extend(0.0));
        }

        for ind in indices.chunks(3) {
            let a = pos_data[pos_start + ind[0] as usize].xyz();
            let b = pos_data[pos_start + ind[1] as usize].xyz();
            let c = pos_data[pos_start + ind[2] as usize].xyz();

            let v1 = b - a;
            let v2 = c - a;

            let normal = v1.cross(v2).normalize();

            let aabb = AABB::empty().grow(&a).grow(&b).grow(&c);

            let mid_point = (a + b + c) / 3.0;

            let dist = (aabb.center() - mid_point).dot(normal);

            tri_nor_data.push(normal.extend(dist));
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
        }

        let mesh_data_idx = mesh_data.len();
        mesh_data_reverse_map.insert(mesh_h.clone(), mesh_data_idx);

        mesh_data.push(MeshData {
            index_start: index_start as i32,
            pos_start: pos_start as i32,
            blas_start: blas_start as i32,
            blas_count: flat_bvh.len() as i32,
        });
    }

    gpu_data.vert_indices = images.add(i32_image(&index_data, "vert_indices"));
    gpu_data.vert_pos = images.add(f32rgba_image(&pos_data, "vert_positions"));
    gpu_data.vert_nor = images.add(f32rgba_image(&nor_data, "vert_normals"));
    gpu_data.tri_nor = images.add(f32rgba_image(&tri_nor_data, "tri_normals"));
    gpu_data.blas = images.add(f32rgba_image(&blas_data, "blas"));
    gpu_data.mesh_data = images.add(i32_image(&cast_slice(&mesh_data), "mesh_data"));
    gpu_data.mesh_data_reverse_map = mesh_data_reverse_map;
}

fn update_tlas_data(
    static_tlas: Res<StaticTLASData>,
    dynamic_tlas: Res<DynamicTLASData>,
    mut images: ResMut<Assets<Image>>,
    mut gpu_data: ResMut<GpuData>,
) {
    // when the static or dynamic tlas is modified remake flattened version into texture
    if static_tlas.is_changed() {
        if let Some(bvh) = &static_tlas.0.bvh {
            dbg!("update_static_tlas");
            gpu_data.gpu_static_tlas_data = images.add(create_gpu_tlas_data(bvh, "static_tlas"));
        }
    }
    if dynamic_tlas.is_changed() {
        if let Some(bvh) = &dynamic_tlas.0.bvh {
            gpu_data.gpu_dynamic_tlas_data = images.add(create_gpu_tlas_data(bvh, "dynamic_tlas"));
        }
    }
}

fn create_gpu_tlas_data(bvh: &BVH, label: &'static str) -> Image {
    let mut tlas_data = Vec::new();
    let flat_bvh = bvh.flatten();
    //first is length, todo move elsewhere
    tlas_data.push(vec4(flat_bvh.len() as f32, 0.0, 0.0, 0.0));
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
    f32rgba_image(&tlas_data, label)
}

// TODO temporary, user needs to implement
fn update_instance_data(
    static_tlas: Res<StaticTLASData>,
    dynamic_tlas: Res<DynamicTLASData>,
    mut gpu_data: ResMut<GpuData>,
    mesh_entities: Query<(&Handle<Mesh>, &GlobalTransform, &Handle<StandardMaterial>)>,
    materials: Res<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    if static_tlas.is_changed() {
        dbg!("GpuStaticInstanceData");
        gpu_data.gpu_static_instance_data = images.add(create_instance_data(
            &static_tlas.0,
            &mesh_entities,
            &materials,
            &gpu_data,
            "gpu_static_instance_data",
        ));
    }
    if dynamic_tlas.is_changed() {
        gpu_data.gpu_dynamic_instance_data = images.add(create_instance_data(
            &dynamic_tlas.0,
            &mesh_entities,
            &materials,
            &gpu_data,
            "gpu_dynamic_instance_data",
        ));
    }
}

fn create_instance_data(
    tlas: &TLAS,
    mesh_entities: &Query<(&Handle<Mesh>, &GlobalTransform, &Handle<StandardMaterial>)>,
    materials: &Assets<StandardMaterial>,
    gpu_data: &GpuData,
    label: &'static str,
) -> Image {
    let mut instance_gpu_data = Vec::new();
    for item in &tlas.aabbs {
        let (mesh_h, trans, material_h) = mesh_entities.get(item.entity).unwrap();
        let material = materials.get(material_h).unwrap();
        // currently diffuse rgba, emit rgba, index into mesh_data, inv mat: Vec4Vec4Vec4Vec4
        instance_gpu_data.push(cast(material.base_color.as_linear_rgba_f32()));
        instance_gpu_data.push(cast(material.emissive.as_linear_rgba_f32()));
        let mesh_data_idx = gpu_data.mesh_data_reverse_map[mesh_h];
        instance_gpu_data.push(Vec4::new(cast(mesh_data_idx as i32), 0.0, 0.0, 0.0));
        let inv_mat = trans.compute_matrix().inverse();
        instance_gpu_data.push(inv_mat.x_axis);
        instance_gpu_data.push(inv_mat.y_axis);
        instance_gpu_data.push(inv_mat.z_axis);
        instance_gpu_data.push(inv_mat.w_axis);
    }
    f32rgba_image(&instance_gpu_data, label)
}

pub fn i32_image(i32data: &[i32], label: &'static str) -> Image {
    let dimension = (i32data.len().sqrt() + 1).next_power_of_two();
    let mut img = Image {
        //dimension * dimension image with 1 u32 which are 4 bytes
        data: vec![0u8; dimension * dimension * 1 * 4],
        texture_descriptor: TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: dimension as u32,
                height: dimension as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Sint,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        ..default()
    };
    let data = cast_slice::<i32, u8>(&i32data).to_vec();
    img.data.splice(0..data.len(), data);
    img
}

pub fn f32rgba_image(vec4data: &[Vec4], label: &'static str) -> Image {
    let dimension = (vec4data.len().sqrt() + 1).next_power_of_two();
    let mut img = Image {
        //dimension * dimension image with 4 f32s which are 4 bytes
        data: vec![0u8; dimension * dimension * 4 * 4],
        texture_descriptor: TextureDescriptor {
            label: Some(label),
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

pub fn extract_gpu_data(mut commands: Commands, gpu_data: Extract<Res<GpuData>>) {
    commands.insert_resource(gpu_data.clone());
}

pub fn get_bindings<'a>(
    images: &'a RenderAssets<Image>,
    gpu_data: &'a GpuData,
    bindings: [u32; 10],
) -> Option<[BindGroupEntry<'a>; 10]> {
    if let (
        Some(gpu_static_tlas_data),
        Some(gpu_dynamic_tlas_data),
        Some(mesh_data),
        Some(vert_indices),
        Some(vert_positions),
        Some(vert_normals),
        Some(tri_normals),
        Some(blas),
        Some(gpu_static_instance_data),
        Some(gpu_dynamic_instance_data),
    ) = (
        images.get(&gpu_data.gpu_static_tlas_data),
        images.get(&gpu_data.gpu_dynamic_tlas_data),
        images.get(&gpu_data.mesh_data),
        images.get(&gpu_data.vert_indices),
        images.get(&gpu_data.vert_pos),
        images.get(&gpu_data.vert_nor),
        images.get(&gpu_data.tri_nor),
        images.get(&gpu_data.blas),
        images.get(&gpu_data.gpu_static_instance_data),
        images.get(&gpu_data.gpu_dynamic_instance_data),
    ) {
        Some([
            BindGroupEntry {
                binding: bindings[0],
                resource: BindingResource::TextureView(&gpu_static_tlas_data.texture_view),
            },
            BindGroupEntry {
                binding: bindings[1],
                resource: BindingResource::TextureView(&gpu_dynamic_tlas_data.texture_view),
            },
            BindGroupEntry {
                binding: bindings[2],
                resource: BindingResource::TextureView(&mesh_data.texture_view),
            },
            BindGroupEntry {
                binding: bindings[3],
                resource: BindingResource::TextureView(&vert_indices.texture_view),
            },
            BindGroupEntry {
                binding: bindings[4],
                resource: BindingResource::TextureView(&vert_positions.texture_view),
            },
            BindGroupEntry {
                binding: bindings[5],
                resource: BindingResource::TextureView(&vert_normals.texture_view),
            },
            BindGroupEntry {
                binding: bindings[6],
                resource: BindingResource::TextureView(&tri_normals.texture_view),
            },
            BindGroupEntry {
                binding: bindings[7],
                resource: BindingResource::TextureView(&blas.texture_view),
            },
            BindGroupEntry {
                binding: bindings[8],
                resource: BindingResource::TextureView(&gpu_static_instance_data.texture_view),
            },
            BindGroupEntry {
                binding: bindings[9],
                resource: BindingResource::TextureView(&gpu_dynamic_instance_data.texture_view),
            },
        ])
    } else {
        return None;
    }
}

pub fn get_bind_group_layout_entries(bindings: [u32; 10]) -> [BindGroupLayoutEntry; 10] {
    [
        layout_entry_d2(bindings[0], TextureSampleType::Float { filterable: false }),
        layout_entry_d2(bindings[1], TextureSampleType::Float { filterable: false }),
        layout_entry_d2(bindings[2], TextureSampleType::Sint),
        layout_entry_d2(bindings[3], TextureSampleType::Sint),
        layout_entry_d2(bindings[4], TextureSampleType::Float { filterable: false }),
        layout_entry_d2(bindings[5], TextureSampleType::Float { filterable: false }),
        layout_entry_d2(bindings[6], TextureSampleType::Float { filterable: false }),
        layout_entry_d2(bindings[7], TextureSampleType::Float { filterable: false }),
        layout_entry_d2(bindings[8], TextureSampleType::Float { filterable: false }),
        layout_entry_d2(bindings[9], TextureSampleType::Float { filterable: false }),
    ]
}

pub fn layout_entry_d2(binding: u32, sample_type: TextureSampleType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
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
