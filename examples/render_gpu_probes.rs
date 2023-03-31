use bevy::{
    core_pipeline::core_3d,
    diagnostic::{Diagnostics, FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    prelude::*,
    reflect::TypeUuid,
    render::{
        extract_component::{
            ComponentUniforms, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
        },
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, SlotInfo, SlotType},
        render_resource::{
            AsBindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
            BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType,
            CachedRenderPipelineId, Extent3d, Operations, PipelineCache, RenderPassColorAttachment,
            RenderPassDescriptor, ShaderRef, ShaderStages, ShaderType, StorageTextureAccess,
            TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
            TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        texture::{CachedTexture, ImageSampler},
        view::{ExtractedView, ViewTarget, ViewUniformOffset, ViewUniforms},
        Extract, RenderApp,
    },
    window::PresentMode,
};
use bevy_basic_camera::{CameraController, CameraControllerPlugin};
use bevy_mod_bvh::{
    gpu_data::{
        get_bind_group_layout_entries, get_bindings, get_default_pipeline_desc, view_entry,
        GPUDataPlugin, GpuData,
    },
    BVHPlugin, BVHSet, DynamicTLAS, StaticTLAS,
};

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    watch_for_changes: true,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: PresentMode::Immediate,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugin(PostProcessPlugin)
        .add_plugin(BVHPlugin)
        .add_plugin(GPUDataPlugin)
        .add_plugin(CameraControllerPlugin)
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(MaterialPlugin::<CustomMaterial>::default())
        .add_startup_system(setup)
        .add_systems((cube_rotator, update_settings).in_base_set(CoreSet::Update))
        .add_startup_system(load_sponza)
        .add_system(set_sponza_tlas.before(BVHSet::BlasTlas))
        .add_system(swap_custom_material)
        .run();
}

struct PostProcessPlugin;
impl Plugin for PostProcessPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(ExtractComponentPlugin::<TraceSettings>::default())
            .add_plugin(UniformComponentPlugin::<TraceSettings>::default())
            .add_startup_system(prepare_textures);

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.add_system(extract_probe_textures.in_schedule(ExtractSchedule));

        render_app.init_resource::<PostProcessPipeline>();

        let node = RayTraceNode::new(&mut render_app.world);

        let mut graph = render_app.world.resource_mut::<RenderGraph>();

        let core_3d_graph = graph.get_sub_graph_mut(core_3d::graph::NAME).unwrap();

        core_3d_graph.add_node(RayTraceNode::NAME, node);
        let id = core_3d_graph.input_node().id;

        core_3d_graph.add_slot_edge(
            id,
            core_3d::graph::input::VIEW_ENTITY,
            RayTraceNode::NAME,
            RayTraceNode::IN_VIEW,
        );

        // run before main pass
        core_3d_graph.add_node_edge(RayTraceNode::NAME, core_3d::graph::node::MAIN_PASS);
    }
}

struct RayTraceNode {
    query: QueryState<(&'static ViewUniformOffset, &'static ViewTarget), With<ExtractedView>>,
    frame: u32,
}

impl RayTraceNode {
    pub const IN_VIEW: &'static str = "view";
    pub const NAME: &str = "post_process";

    fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
            frame: 0,
        }
    }
}

impl Node for RayTraceNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(RayTraceNode::IN_VIEW, SlotType::Entity)]
    }

    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
        self.frame = self.frame.wrapping_add(1);
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
        let view_uniforms = world.resource::<ViewUniforms>();
        let view_uniforms = view_uniforms.uniforms.binding().unwrap();
        let images = world.resource::<RenderAssets<Image>>();
        let gpu_data = world.resource::<GpuData>();
        let probe_textures = world.resource::<ProbeTextures>();

        let Ok((view_uniform_offset, _view_target)) = self.query.get_manual(world, view_entity) else {
            return Ok(());
        };

        let post_process_pipeline = world.resource::<PostProcessPipeline>();

        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(pipeline) = pipeline_cache.get_render_pipeline(post_process_pipeline.pipeline_id) else {
            return Ok(());
        };

        let settings_uniforms = world.resource::<ComponentUniforms<TraceSettings>>();
        let Some(settings_binding) = settings_uniforms.uniforms().binding() else {
            return Ok(());
        };

        let prev_image = images
            .get(&probe_textures.sh_tex[probe_textures.previous(self.frame)])
            .unwrap();
        let next_image = images
            .get(&probe_textures.sh_tex[probe_textures.next(self.frame)])
            .unwrap();
        let target_tex = images.get(&probe_textures.target_tex).unwrap();

        let mut entries = vec![
            BindGroupEntry {
                binding: 0,
                resource: view_uniforms.clone(),
            },
            BindGroupEntry {
                binding: 1,
                resource: settings_binding.clone(),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(&prev_image.texture_view),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(&next_image.texture_view),
            },
        ];

        let Some(rt_bindings) = get_bindings(images, gpu_data, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) else {
            return Ok(());
        };

        entries.append(&mut rt_bindings.to_vec());

        let bind_group = render_context
            .render_device()
            .create_bind_group(&BindGroupDescriptor {
                label: Some("post_process_bind_group"),
                layout: &post_process_pipeline.layout,
                entries: &entries,
            });

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("post_process_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &target_tex.texture_view,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

#[derive(Resource)]
struct PostProcessPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
}

impl FromWorld for PostProcessPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let mut entries = vec![
            view_entry(0),
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: bevy::render::render_resource::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadOnly,
                    format: TextureFormat::Rgba16Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba16Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
        ];

        entries.append(
            &mut get_bind_group_layout_entries([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]).to_vec(),
        );

        let layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("post_process_bind_group_layout"),
            entries: &entries,
        });

        let shader = world.resource::<AssetServer>().load("probe_example.wgsl");

        let pipeline_id = get_default_pipeline_desc(
            Vec::new(),
            layout.clone(),
            &mut world.resource_mut::<PipelineCache>(),
            shader,
            false,
        );

        Self {
            layout,
            pipeline_id,
        }
    }
}

// not using cached texture so we can access on materials
#[derive(Resource, Clone)]
pub struct ProbeTextures {
    pub sh_tex: [Handle<Image>; 2],
    pub target_tex: Handle<Image>,
}

pub fn default_view(tex: &CachedTexture) -> TextureView {
    tex.texture.create_view(&TextureViewDescriptor::default())
}

impl ProbeTextures {
    pub fn previous(&self, frame: u32) -> usize {
        (frame % 2) as usize
    }
    pub fn next(&self, frame: u32) -> usize {
        (frame + 1) as usize % 2
    }
}

pub fn prepare_textures(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let values_per_probe = 9;
    let cascade_x = 8;
    let cascade_y = 8;
    let cascade_z = 8;
    let size = Extent3d {
        width: 1024,
        height: 256,
        depth_or_array_layers: 1,
    };
    let target_size = Extent3d {
        width: cascade_x * cascade_y,
        height: cascade_z,
        depth_or_array_layers: 1,
    };
    let mut sh_tex_1 = Image {
        texture_descriptor: TextureDescriptor {
            label: Some("sh_tex_1"),
            size: size.clone(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        sampler_descriptor: ImageSampler::nearest(),
        ..default()
    };
    sh_tex_1.resize(size);
    let mut sh_tex_2 = Image {
        texture_descriptor: TextureDescriptor {
            label: Some("sh_tex_2"),
            size: size.clone(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        sampler_descriptor: ImageSampler::nearest(),
        ..default()
    };
    sh_tex_2.resize(size);
    let mut target_tex = Image {
        texture_descriptor: TextureDescriptor {
            label: Some("probe_target_tex"),
            size: target_size.clone(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[TextureFormat::Rgba8UnormSrgb],
        },
        ..default()
    };
    target_tex.resize(target_size);

    commands.insert_resource(ProbeTextures {
        sh_tex: [images.add(sh_tex_1), images.add(sh_tex_2)],
        target_tex: images.add(target_tex),
    });
}

pub fn extract_probe_textures(mut commands: Commands, probe_textures: Extract<Res<ProbeTextures>>) {
    commands.insert_resource(probe_textures.clone());
}

#[derive(Component, Default, Clone, Copy, ExtractComponent, ShaderType)]
struct TraceSettings {
    frame: u32,
    fps: f32,
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // plane
    commands
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(shape::Plane::from_size(5.0).into()),
            material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
            ..default()
        })
        .insert(StaticTLAS);
    // cube
    commands
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
            transform: Transform::from_xyz(0.0, 0.5, 0.0),
            ..default()
        })
        .insert(StaticTLAS);
    // rotating cubes
    commands
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(Color::rgb(0.1, 0.1, 0.9).into()),
            transform: Transform::from_xyz(0.0, 1.5, 0.0),
            ..default()
        })
        .insert(DynamicTLAS)
        .insert(RotateCube);
    commands
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(Color::rgb(0.8, 0.1, 0.1).into()),
            transform: Transform::from_xyz(1.0, 1.5, 0.0),
            ..default()
        })
        .insert(DynamicTLAS)
        .insert(RotateCube);
    // camera
    commands
        .spawn(Camera3dBundle {
            transform: Transform::from_xyz(-10.5, 1.7, -1.0)
                .looking_at(Vec3::new(0.0, 3.5, 0.0), Vec3::Y),
            ..default()
        })
        .insert(CameraController::default())
        .insert(TraceSettings { frame: 0, fps: 0.0 });
}

impl Material for CustomMaterial {
    fn fragment_shader() -> ShaderRef {
        "custom_material.wgsl".into()
    }
}

// This is the struct that will be passed to your shader
#[derive(AsBindGroup, Debug, Clone, TypeUuid)]
#[uuid = "717f64fe-6844-4822-8926-e0ed314294c8"]
pub struct CustomMaterial {
    #[texture(0)]
    #[sampler(1)]
    pub probe_texture: Handle<Image>,
}

fn swap_custom_material(
    mut commands: Commands,
    mut material_events: EventReader<AssetEvent<StandardMaterial>>,
    entites: Query<(Entity, &Handle<StandardMaterial>)>,
    mut custom_materials: ResMut<Assets<CustomMaterial>>,
    probe_textures: Res<ProbeTextures>,
) {
    for event in material_events.iter() {
        let handle = match event {
            AssetEvent::Created { handle } => handle,
            _ => continue,
        };
        let custom_mat_h = custom_materials.add(CustomMaterial {
            probe_texture: probe_textures.sh_tex[0].clone(), //TODO use both so we update every frame
        });
        for (entity, entity_mat_h) in entites.iter() {
            if entity_mat_h == handle {
                let mut ecmds = commands.entity(entity);
                ecmds.remove::<Handle<StandardMaterial>>();
                ecmds.insert(custom_mat_h.clone());
            }
        }
    }
}

fn load_sponza(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(SceneBundle {
        scene: asset_server.load(
            "H:/dev/programming/rust/bevy/bevy_mod_bvh/sponza/NewSponza_Main_glTF_002.gltf#Scene0",
        ),
        ..default()
    });

    //commands.spawn(SceneBundle {
    //    scene: asset_server.load(
    //        "H:/dev/programming/rust/bevy/bevy_mod_bvh/sponza/NewSponza_Curtains_glTF.gltf#Scene0",
    //    ),
    //    ..default()
    //});
}

fn set_sponza_tlas(
    mut commands: Commands,
    query: Query<
        Entity,
        (
            With<Handle<Mesh>>,
            Without<StaticTLAS>,
            Without<DynamicTLAS>,
        ),
    >,
) {
    for entity in &query {
        commands.entity(entity).insert(StaticTLAS);
    }
}

#[derive(Component)]
struct RotateCube;

fn cube_rotator(
    time: Res<Time>,
    mut query: Query<&mut Transform, With<RotateCube>>,
    keys: Res<Input<KeyCode>>,
    mut pause: Local<bool>,
) {
    if keys.just_pressed(KeyCode::Space) {
        *pause = !*pause;
    }
    if !*pause {
        for mut transform in &mut query {
            transform.rotate_x(1.0 * time.delta_seconds());
            transform.rotate_y(0.7 * time.delta_seconds());
        }
    }
}

fn update_settings(mut settings: Query<&mut TraceSettings>, diagnostics: Res<Diagnostics>) {
    for mut setting in &mut settings {
        setting.frame = setting.frame.wrapping_add(1);
        if let Some(diag) = diagnostics.get(FrameTimeDiagnosticsPlugin::FPS) {
            let hysteresis = 0.9;
            let fps = hysteresis + diag.value().unwrap_or(0.0) as f32;
            setting.fps = setting.fps * hysteresis + fps * (1.0 - hysteresis);
        }
    }
}
