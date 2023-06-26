use std::time::Duration;

use bevy::{
    asset::ChangeWatcher,
    core_pipeline::core_3d,
    diagnostic::FrameTimeDiagnosticsPlugin,
    prelude::*,
    render::{
        extract_component::{
            ComponentUniforms, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
        },
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
        render_resource::{
            BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindingResource, CachedRenderPipelineId, Operations, PipelineCache,
            RenderPassColorAttachment, RenderPassDescriptor, Sampler, SamplerDescriptor,
            ShaderType, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        view::{ExtractedView, ViewTarget, ViewUniformOffset, ViewUniforms},
        RenderApp,
    },
    window::PresentMode,
};
use bevy_basic_camera::{CameraController, CameraControllerPlugin};
use bevy_mod_bvh::{
    gpu_data::{GPUBuffers, GPUDataPlugin},
    pipeline_utils::{
        get_default_pipeline_desc, image_entry, sampler_entry, uniform_entry, view_entry,
    },
    BVHPlugin, BVHSet, DynamicTLAS, StaticTLAS,
};

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    watch_for_changes: ChangeWatcher::with_delay(Duration::from_millis(200)),
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
        .add_systems(Startup, (setup, load_sponza))
        .add_systems(Update, (cube_rotator, update_settings))
        .add_systems(Update, set_sponza_tlas.before(BVHSet::BlasTlas))
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

        render_app
            .add_render_graph_node::<RayTraceNode>(core_3d::graph::NAME, RayTraceNode::NAME)
            .add_render_graph_edges(
                core_3d::graph::NAME,
                &[
                    core_3d::graph::node::BLOOM,
                    RayTraceNode::NAME,
                    core_3d::graph::node::TONEMAPPING,
                ],
            );
    }

    fn finish(&self, app: &mut App) {
        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };
        render_app.init_resource::<PostProcessPipeline>();
    }
}

struct RayTraceNode {
    query: QueryState<(&'static ViewUniformOffset, &'static ViewTarget), With<ExtractedView>>,
}

impl RayTraceNode {
    pub const NAME: &str = "post_process";
}

impl FromWorld for RayTraceNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl Node for RayTraceNode {
    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        graph_context: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph_context.view_entity();
        let view_uniforms: &ViewUniforms = world.resource::<ViewUniforms>();
        let view_uniforms = view_uniforms.uniforms.binding().unwrap();
        let gpu_buffers = world.resource::<GPUBuffers>();

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

        let Some(gpu_buffer_bind_group_entries) = gpu_buffers
                .bind_group_entries([4, 5, 6, 7, 8, 9, 10]) else {
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

        entries.append(&mut gpu_buffer_bind_group_entries.to_vec());

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
            image_entry(1, TextureViewDimension::D2),
            sampler_entry(2),
            uniform_entry(3, Some(TraceSettings::min_size())),
        ];

        entries.append(&mut GPUBuffers::bind_group_layout_entry([4, 5, 6, 7, 8, 9, 10]).to_vec());

        let layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("post_process_bind_group_layout"),
            entries: &entries,
        });

        let sampler = render_device.create_sampler(&SamplerDescriptor::default());

        let shader = world
            .resource::<AssetServer>()
            .load("raytrace_example.wgsl");

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
    frame_time: f32,
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
        .insert(TraceSettings {
            frame: 0,
            frame_time: 0.0,
        });
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

fn update_settings(mut settings: Query<&mut TraceSettings>, time: Res<Time>) {
    for mut setting in &mut settings {
        setting.frame = setting.frame.wrapping_add(1);
        let hysteresis = 0.03;
        let ms = time.delta_seconds() * 1000.0;
        setting.frame_time = ms * hysteresis + setting.frame_time * (1.0 - hysteresis);
    }
}
