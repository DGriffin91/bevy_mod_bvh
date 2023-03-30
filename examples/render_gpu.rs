use bevy::{
    core_pipeline::core_3d,
    diagnostic::{Diagnostics, FrameTimeDiagnosticsPlugin},
    prelude::*,
    render::{
        extract_component::{
            ComponentUniforms, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
        },
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, SlotInfo, SlotType},
        render_resource::{
            BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntry, BindingResource, BindingType, CachedRenderPipelineId, Operations,
            PipelineCache, RenderPassColorAttachment, RenderPassDescriptor, Sampler,
            SamplerDescriptor, ShaderStages, ShaderType, TextureSampleType,
        },
        renderer::{RenderContext, RenderDevice},
        view::{ExtractedView, ViewTarget, ViewUniformOffset, ViewUniforms},
        RenderApp,
    },
    window::PresentMode,
};
use bevy_basic_camera::{CameraController, CameraControllerPlugin};
use bevy_mod_bvh::{
    gpu_data::{
        get_bind_group_layout_entries, get_bindings, get_default_pipeline_desc, layout_entry_d2,
        sampler_entry, view_entry, GPUDataPlugin, GpuData,
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
        .add_startup_system(setup)
        .add_systems((cube_rotator, update_settings).in_base_set(CoreSet::Update))
        .add_startup_system(load_sponza)
        .add_system(set_sponza_tlas.before(BVHSet::BlasTlas))
        .run();
}

struct PostProcessPlugin;
impl Plugin for PostProcessPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(ExtractComponentPlugin::<TraceSettings>::default())
            .add_plugin(UniformComponentPlugin::<TraceSettings>::default());

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

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

        core_3d_graph.add_node_edge(core_3d::graph::node::MAIN_PASS, RayTraceNode::NAME);
    }
}

struct RayTraceNode {
    query: QueryState<(&'static ViewUniformOffset, &'static ViewTarget), With<ExtractedView>>,
}

impl RayTraceNode {
    pub const IN_VIEW: &'static str = "view";
    pub const NAME: &str = "post_process";

    fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl Node for RayTraceNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(RayTraceNode::IN_VIEW, SlotType::Entity)]
    }

    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
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

        let Ok((view_uniform_offset, view_target)) = self.query.get_manual(world, view_entity) else {
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

        let post_process = view_target.post_process_write();

        let mut entries = vec![
            BindGroupEntry {
                binding: 0,
                resource: view_uniforms.clone(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(post_process.source),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(&post_process_pipeline.sampler),
            },
            BindGroupEntry {
                binding: 3,
                resource: settings_binding.clone(),
            },
        ];

        let Some(rt_bindings) = get_bindings(images, gpu_data, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]) else {
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
                view: post_process.destination,
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
    sampler: Sampler,
    pipeline_id: CachedRenderPipelineId,
}

impl FromWorld for PostProcessPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let mut entries = vec![
            view_entry(0),
            layout_entry_d2(1, TextureSampleType::Float { filterable: true }),
            sampler_entry(2),
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: bevy::render::render_resource::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        entries.append(
            &mut get_bind_group_layout_entries([4, 5, 6, 7, 8, 9, 10, 11, 12, 13]).to_vec(),
        );

        let layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("post_process_bind_group_layout"),
            entries: &entries,
        });

        let sampler = render_device.create_sampler(&SamplerDescriptor::default());

        let shader = world.resource::<AssetServer>().load("raytrace.wgsl");

        let pipeline_id = get_default_pipeline_desc(
            Vec::new(),
            layout.clone(),
            &mut world.resource_mut::<PipelineCache>(),
            shader,
            false,
        );

        Self {
            layout,
            sampler,
            pipeline_id,
        }
    }
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

fn load_sponza(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(SceneBundle {
        scene: asset_server.load(
            "H:/dev/programming/rust/bevy/bevy_mod_bvh/sponza/NewSponza_Main_glTF_002.gltf#Scene0",
        ),
        ..default()
    });
    commands.spawn(SceneBundle {
        scene: asset_server.load(
            "H:/dev/programming/rust/bevy/bevy_mod_bvh/sponza/NewSponza_Curtains_glTF.gltf#Scene0",
        ),
        ..default()
    });
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
