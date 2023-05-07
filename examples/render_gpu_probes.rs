use std::f32::consts::PI;

use bevy::{
    core_pipeline::core_3d,
    diagnostic::{Diagnostics, FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    math::vec3,
    pbr::{
        MaterialPipeline, MaterialPipelineKey, StandardMaterialFlags, PBR_PREPASS_SHADER_HANDLE,
    },
    prelude::*,
    reflect::TypeUuid,
    render::{
        extract_component::{
            ComponentUniforms, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
        },
        mesh::MeshVertexBufferLayout,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
        render_resource::{
            AsBindGroup, AsBindGroupShaderType, BindGroupDescriptor, BindGroupEntry,
            BindGroupLayout, BindGroupLayoutDescriptor, BindingResource, CachedRenderPipelineId,
            Extent3d, Face, Operations, PipelineCache, RenderPassColorAttachment,
            RenderPassDescriptor, RenderPipelineDescriptor, ShaderRef, ShaderType,
            SpecializedMeshPipelineError, TextureDescriptor, TextureDimension, TextureFormat,
            TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
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
    gpu_data::{GPUBuffers, GPUDataPlugin},
    pipeline_utils::{
        get_default_pipeline_desc, storage_tex_read, storage_tex_write, uniform_entry, view_entry,
    },
    BVHPlugin, BVHSet, DynamicTLAS, StaticTLAS,
};

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::rgb(1.75, 1.9, 1.99)))
        .insert_resource(AmbientLight {
            color: Color::rgb(1.0, 1.0, 1.0),
            brightness: 0.0,
        })
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
        .add_plugin(MaterialPlugin::<CustomStandardMaterial>::default())
        .add_systems(Update, (cube_rotator, update_settings))
        .add_systems(Startup, (setup, load_sponza).chain())
        .add_systems(
            Update,
            (proc_sponza_scene, swap_standard_material, set_sponza_tlas)
                .chain()
                .before(BVHSet::BlasTlas),
        )
        .run();
}

struct PostProcessPlugin;
impl Plugin for PostProcessPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(ExtractComponentPlugin::<TraceSettings>::default())
            .add_plugin(UniformComponentPlugin::<TraceSettings>::default())
            .add_systems(Startup, prepare_textures);

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.add_systems(ExtractSchedule, extract_probe_textures);

        render_app
            .add_render_graph_node::<RayTraceNode>(core_3d::graph::NAME, RayTraceNode::NAME)
            .add_render_graph_edges(
                core_3d::graph::NAME,
                &[
                    core_3d::graph::node::PREPASS,
                    RayTraceNode::NAME,
                    core_3d::graph::node::START_MAIN_PASS,
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
    frame: u32,
}

impl RayTraceNode {
    pub const NAME: &str = "post_process";
}

impl FromWorld for RayTraceNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
            frame: 0,
        }
    }
}

impl Node for RayTraceNode {
    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
        self.frame = self.frame.wrapping_add(1);
    }

    fn run(
        &self,
        graph_context: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph_context.view_entity();
        let view_uniforms = world.resource::<ViewUniforms>();
        let view_uniforms = view_uniforms.uniforms.binding().unwrap();
        let images = world.resource::<RenderAssets<Image>>();
        let probe_textures = world.resource::<ProbeTextures>();
        let gpu_buffers = world.resource::<GPUBuffers>();

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

        let Some(gpu_buffer_bind_group_entries) = gpu_buffers
                .bind_group_entries([2, 3, 4, 5, 6, 7, 8]) else {
            return Ok(());
        };

        let prev_tex = images
            .get(&probe_textures.probe_tex[probe_textures.previous(self.frame)])
            .unwrap();
        let next_tex = images
            .get(&probe_textures.probe_tex[probe_textures.next(self.frame)])
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
                binding: 9,
                resource: BindingResource::TextureView(&prev_tex.texture_view),
            },
            BindGroupEntry {
                binding: 10,
                resource: BindingResource::TextureView(&next_tex.texture_view),
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
            uniform_entry(1, Some(TraceSettings::min_size())),
            storage_tex_read(9, TextureFormat::Rgba16Float, TextureViewDimension::D2),
            storage_tex_write(10, TextureFormat::Rgba16Float, TextureViewDimension::D2),
        ];

        entries.append(&mut GPUBuffers::bind_group_layout_entry([2, 3, 4, 5, 6, 7, 8]).to_vec());

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
    pub probe_tex: [Handle<Image>; 2],
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
    let cascade_x = 18;
    let cascade_y = 18;
    let cascade_z = 18;
    let size = 6;
    let output_size = Extent3d {
        width: cascade_x * cascade_y * size,
        height: cascade_z * size * 2, //*2 since the color data is on the bottom half
        depth_or_array_layers: 1,
    };
    let invocation_size = Extent3d {
        width: cascade_x * cascade_y * size,
        height: cascade_z * size,
        depth_or_array_layers: 1,
    };
    let mut probe_tex1 = Image {
        texture_descriptor: TextureDescriptor {
            label: Some("probe_tex1"),
            size: output_size.clone(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        sampler_descriptor: ImageSampler::linear(),
        ..default()
    };
    probe_tex1.resize(output_size);

    let mut probe_tex2 = probe_tex1.clone();
    probe_tex2.texture_descriptor.label = Some("probe_tex2");

    let mut target_tex = Image {
        texture_descriptor: TextureDescriptor {
            label: Some("probe_target_tex"),
            size: invocation_size.clone(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[TextureFormat::Rgba8UnormSrgb],
        },
        ..default()
    };
    target_tex.resize(invocation_size);

    commands.insert_resource(ProbeTextures {
        probe_tex: [images.add(probe_tex1), images.add(probe_tex2)],
        target_tex: images.add(target_tex),
    });
}

pub fn extract_probe_textures(mut commands: Commands, probe_textures: Extract<Res<ProbeTextures>>) {
    commands.insert_resource(probe_textures.clone());
}

#[derive(Component, Default, Clone, Copy, ExtractComponent, ShaderType)]
struct TraceSettings {
    sun_direction: Vec3,
    frame: u32,
    fps: f32,
    render_depth_this_frame: u32,
}
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // plane
    //commands
    //    .spawn(MaterialMeshBundle {
    //        mesh: meshes.add(shape::Plane::from_size(5.0).into()),
    //        material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
    //        ..default()
    //    })
    //    .insert(StaticTLAS);
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
            /*
            x = cos(-0.43*pi)*cos(-0.08*pi)
            y = sin(-0.43*pi)*cos(-0.08*pi)
            z = sin(0.0)

            for transform: Transform::from_rotation(Quat::from_euler(
                EulerRot::XYZ,
                PI * -0.43,
                PI * -0.08,
                0.0,
            )),

            idk if it's correct yet

             */
            sun_direction: vec3(0.2112898703307094, -0.9452565422770496, 0.0),
            frame: 0,
            fps: 0.0,
            render_depth_this_frame: 1,
        });
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
    #[texture(2)]
    #[sampler(3)]
    pub probe_texture2: Handle<Image>,
}

fn _swap_custom_material(
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
            probe_texture: probe_textures.probe_tex[0].clone(),
            probe_texture2: probe_textures.probe_tex[1].clone(),
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
    let _without_textures =
        "H:/dev/programming/rust/bevy/bevy_mod_bvh/sponza/NewSponza_Main_glTF_002.gltf#Scene0";
    let with_textures = "H:/dev/programming/rust/bevy/bevy_mod_bvh/sponza/main_sponza/NewSponza_Main_glTF_002_no_cameras.gltf#Scene0";
    let _low_poly =
        "H:/dev/programming/rust/bevy/bevy_mod_bvh/sponza/main_sponza/untitled.glb#Scene0";
    commands
        .spawn(SceneBundle {
            scene: asset_server.load(with_textures),
            ..default()
        })
        .insert(PostProcScene);
    // Sun
    commands
        .spawn(DirectionalLightBundle {
            transform: Transform::from_rotation(Quat::from_euler(
                EulerRot::XYZ,
                PI * -0.43,
                PI * -0.08,
                0.0,
            )),
            directional_light: DirectionalLight {
                color: Color::rgb(1.0, 1.0, 0.99),
                illuminance: 400000.0,
                shadows_enabled: true,
                shadow_depth_bias: 0.3,
                shadow_normal_bias: 0.7,
            },
            ..default()
        })
        .insert(GrifLight);
    //commands.spawn(SceneBundle {
    //    scene: asset_server.load(
    //        "H:/dev/programming/rust/bevy/bevy_mod_bvh/sponza/NewSponza_Curtains_glTF.gltf#Scene0",
    //    ),
    //    ..default()
    //}).insert(PostProcScene);
}

pub fn all_children<F: FnMut(Entity)>(
    children: &Children,
    children_query: &Query<&Children>,
    closure: &mut F,
) {
    for child in children {
        if let Ok(children) = children_query.get(*child) {
            all_children(children, children_query, closure);
        }
        closure(*child);
    }
}

#[derive(Component)]
pub struct GrifLight;

#[derive(Component)]
pub struct PostProcScene;

#[allow(clippy::type_complexity)]
pub fn proc_sponza_scene(
    mut commands: Commands,
    flip_normals_query: Query<Entity, With<PostProcScene>>,
    children_query: Query<&Children>,
    has_std_mat: Query<&Handle<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    lights: Query<
        Entity,
        (
            Or<(With<PointLight>, With<DirectionalLight>, With<SpotLight>)>,
            Without<GrifLight>,
        ),
    >,
    cameras: Query<Entity, With<Camera>>,
) {
    for entity in flip_normals_query.iter() {
        if let Ok(children) = children_query.get(entity) {
            all_children(children, &children_query, &mut |entity| {
                // Sponza needs flipped normals
                if let Ok(mat_h) = has_std_mat.get(entity) {
                    if let Some(mat) = materials.get_mut(mat_h) {
                        mat.flip_normal_map_y = true;
                    }
                }

                // Sponza has a bunch of lights by default
                if lights.get(entity).is_ok() {
                    commands.entity(entity).despawn_recursive();
                }

                // Sponza has a bunch of cameras by default
                if cameras.get(entity).is_ok() {
                    commands.entity(entity).despawn_recursive();
                }
            });
            commands.entity(entity).remove::<PostProcScene>();
        }
    }
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

fn update_settings(
    mut settings: Query<&mut TraceSettings>,
    diagnostics: Res<Diagnostics>,
    sun: Query<&Transform, With<DirectionalLight>>,
) {
    let Some(sun) = sun.iter().next() else {
        return;
    };
    for mut setting in &mut settings {
        setting.frame = setting.frame.wrapping_add(1);
        if setting.frame % 2 == 0 {
            setting.render_depth_this_frame = 1;
        } else {
            setting.render_depth_this_frame = 0;
        }
        if let Some(diag) = diagnostics.get(FrameTimeDiagnosticsPlugin::FPS) {
            let hysteresis = 0.9;
            let fps = hysteresis + diag.value().unwrap_or(0.0) as f32;
            setting.fps = setting.fps * hysteresis + fps * (1.0 - hysteresis);
            setting.sun_direction = sun.forward();
        }
    }
}

//----------------------------
//----------------------------
//----------------------------

#[derive(AsBindGroup, Debug, Clone, TypeUuid)]
#[uuid = "165799f2-923e-4548-8879-be574f9db989"]
#[bind_group_data(CustomStandardMaterialKey)]
#[uniform(0, CustomStandardMaterialUniform)]
pub struct CustomStandardMaterial {
    pub base_color: Color,
    #[texture(1)]
    #[sampler(2)]
    pub base_color_texture: Option<Handle<Image>>,
    pub emissive: Color,
    #[texture(3)]
    #[sampler(4)]
    pub emissive_texture: Option<Handle<Image>>,
    pub perceptual_roughness: f32,
    pub metallic: f32,
    #[texture(5)]
    #[sampler(6)]
    pub metallic_roughness_texture: Option<Handle<Image>>,
    #[doc(alias = "specular_intensity")]
    pub reflectance: f32,
    #[texture(9)]
    #[sampler(10)]
    pub normal_map_texture: Option<Handle<Image>>,
    pub flip_normal_map_y: bool,
    #[texture(7)]
    #[sampler(8)]
    pub occlusion_texture: Option<Handle<Image>>,
    pub double_sided: bool,
    pub cull_mode: Option<Face>,
    pub unlit: bool,
    pub fog_enabled: bool,
    pub alpha_mode: AlphaMode,
    pub depth_bias: f32,
    #[texture(11)]
    #[sampler(12)]
    pub ddgi_texture: Handle<Image>,
}

#[derive(Clone, Default, ShaderType)]
pub struct CustomStandardMaterialUniform {
    pub base_color: Vec4,
    pub emissive: Vec4,
    pub roughness: f32,
    pub metallic: f32,
    pub reflectance: f32,
    pub flags: u32,
    pub alpha_cutoff: f32,
}

impl AsBindGroupShaderType<CustomStandardMaterialUniform> for CustomStandardMaterial {
    fn as_bind_group_shader_type(
        &self,
        images: &RenderAssets<Image>,
    ) -> CustomStandardMaterialUniform {
        let mut flags = StandardMaterialFlags::NONE;
        if self.base_color_texture.is_some() {
            flags |= StandardMaterialFlags::BASE_COLOR_TEXTURE;
        }
        if self.emissive_texture.is_some() {
            flags |= StandardMaterialFlags::EMISSIVE_TEXTURE;
        }
        if self.metallic_roughness_texture.is_some() {
            flags |= StandardMaterialFlags::METALLIC_ROUGHNESS_TEXTURE;
        }
        if self.occlusion_texture.is_some() {
            flags |= StandardMaterialFlags::OCCLUSION_TEXTURE;
        }
        if self.double_sided {
            flags |= StandardMaterialFlags::DOUBLE_SIDED;
        }
        if self.unlit {
            flags |= StandardMaterialFlags::UNLIT;
        }
        if self.fog_enabled {
            flags |= StandardMaterialFlags::FOG_ENABLED;
        }
        let has_normal_map = self.normal_map_texture.is_some();
        if has_normal_map {
            if let Some(texture) = images.get(self.normal_map_texture.as_ref().unwrap()) {
                match texture.texture_format {
                    // All 2-component unorm formats
                    TextureFormat::Rg8Unorm
                    | TextureFormat::Rg16Unorm
                    | TextureFormat::Bc5RgUnorm
                    | TextureFormat::EacRg11Unorm => {
                        flags |= StandardMaterialFlags::TWO_COMPONENT_NORMAL_MAP;
                    }
                    _ => {}
                }
            }
            if self.flip_normal_map_y {
                flags |= StandardMaterialFlags::FLIP_NORMAL_MAP_Y;
            }
        }
        let mut alpha_cutoff = 0.5;
        match self.alpha_mode {
            AlphaMode::Opaque => flags |= StandardMaterialFlags::ALPHA_MODE_OPAQUE,
            AlphaMode::Mask(c) => {
                alpha_cutoff = c;
                flags |= StandardMaterialFlags::ALPHA_MODE_MASK;
            }
            AlphaMode::Blend => flags |= StandardMaterialFlags::ALPHA_MODE_BLEND,
            AlphaMode::Premultiplied => flags |= StandardMaterialFlags::ALPHA_MODE_PREMULTIPLIED,
            AlphaMode::Add => flags |= StandardMaterialFlags::ALPHA_MODE_ADD,
            AlphaMode::Multiply => flags |= StandardMaterialFlags::ALPHA_MODE_MULTIPLY,
        };

        CustomStandardMaterialUniform {
            base_color: self.base_color.as_linear_rgba_f32().into(),
            emissive: self.emissive.as_linear_rgba_f32().into(),
            roughness: self.perceptual_roughness,
            metallic: self.metallic,
            reflectance: self.reflectance,
            flags: flags.bits(),
            alpha_cutoff,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct CustomStandardMaterialKey {
    normal_map: bool,
    cull_mode: Option<Face>,
    depth_bias: i32,
}

impl From<&CustomStandardMaterial> for CustomStandardMaterialKey {
    fn from(material: &CustomStandardMaterial) -> Self {
        CustomStandardMaterialKey {
            normal_map: material.normal_map_texture.is_some(),
            cull_mode: material.cull_mode,
            depth_bias: material.depth_bias as i32,
        }
    }
}

impl Material for CustomStandardMaterial {
    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayout,
        key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if key.bind_group_data.normal_map {
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment
                    .shader_defs
                    .push("STANDARDMATERIAL_NORMAL_MAP".into());
            }
        }
        descriptor.primitive.cull_mode = key.bind_group_data.cull_mode;
        if let Some(label) = &mut descriptor.label {
            *label = format!("pbr_{}", *label).into();
        }
        if let Some(depth_stencil) = descriptor.depth_stencil.as_mut() {
            depth_stencil.bias.constant = key.bind_group_data.depth_bias;
        }
        Ok(())
    }

    fn prepass_fragment_shader() -> ShaderRef {
        PBR_PREPASS_SHADER_HANDLE.typed().into()
    }

    fn fragment_shader() -> ShaderRef {
        "pbr.wgsl".into()
    }

    #[inline]
    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    #[inline]
    fn depth_bias(&self) -> f32 {
        self.depth_bias
    }
}

fn swap_standard_material(
    mut commands: Commands,
    mut material_events: EventReader<AssetEvent<StandardMaterial>>,
    entites: Query<(Entity, &Handle<StandardMaterial>)>,
    standard_materials: Res<Assets<StandardMaterial>>,
    mut custom_materials: ResMut<Assets<CustomStandardMaterial>>,
    probe_textures: Res<ProbeTextures>,
) {
    for event in material_events.iter() {
        let handle = match event {
            AssetEvent::Created { handle } => handle,
            _ => continue,
        };
        if let Some(material) = standard_materials.get(handle) {
            let custom_mat_h = custom_materials.add(CustomStandardMaterial {
                base_color: material.base_color,
                base_color_texture: material.base_color_texture.clone(),
                emissive: material.emissive,
                emissive_texture: material.emissive_texture.clone(),
                perceptual_roughness: material.perceptual_roughness,
                metallic: material.metallic,
                metallic_roughness_texture: material.metallic_roughness_texture.clone(),
                reflectance: material.reflectance,
                normal_map_texture: material.normal_map_texture.clone(),
                flip_normal_map_y: material.flip_normal_map_y,
                occlusion_texture: material.occlusion_texture.clone(),
                double_sided: material.double_sided,
                cull_mode: material.cull_mode,
                unlit: material.unlit,
                fog_enabled: material.fog_enabled,
                alpha_mode: material.alpha_mode,
                depth_bias: material.depth_bias,
                ddgi_texture: probe_textures.probe_tex[0].clone(),
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
}
