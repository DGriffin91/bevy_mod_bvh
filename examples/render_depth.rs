use std::time::Duration;

use bevy::{
    asset::ChangeWatcher,
    math::{vec3, vec4},
    prelude::*,
    reflect::{TypePath, TypeUuid},
    render::{
        camera::CameraProjection,
        render_resource::{
            AsBindGroup, Extent3d, ShaderRef, TextureDescriptor, TextureDimension, TextureFormat,
            TextureUsages,
        },
        texture::ImageSampler,
    },
    window::PresentMode,
};
use bevy_basic_camera::{CameraController, CameraControllerPlugin};
use bevy_mod_bvh::{
    trace::trace_ray, BVHPlugin, DynamicTLAS, DynamicTLASData, StaticTLAS, StaticTLASData,
    TraceMesh, BLAS,
};

#[derive(Component)]
pub struct TestTrace {
    pub image: Handle<Image>,
    pub data: Vec<Vec4>,
    pub width: usize,
    pub height: usize,
}

impl TestTrace {
    fn set_px(&mut self, x: usize, y: usize, val: Vec4) {
        self.data[x % self.width + y * self.width] = val;
    }

    pub fn new(width: usize, height: usize, images: &mut Assets<Image>) -> TestTrace {
        let test_trace_image = Image {
            //image with 4 f32s which are 4 bytes each
            data: vec![0u8; width * height * 4 * 4],
            texture_descriptor: TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: width as u32,
                    height: height as u32,
                    depth_or_array_layers: 1,
                },
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                mip_level_count: 1,
                sample_count: 1,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            },
            sampler_descriptor: ImageSampler::nearest(),
            texture_view_descriptor: None,
        };

        let test_trace_image_h = images.add(test_trace_image);
        TestTrace {
            image: test_trace_image_h,
            data: vec![Vec4::ZERO; width * height],
            width,
            height,
        }
    }

    fn update(&self, images: &mut Assets<Image>) {
        if let Some(image) = images.get_mut(&self.image) {
            copy_vec_vec4_to_image(image, &self.data);
        }
    }
}

fn copy_vec_vec4_to_image(image: &mut Image, data: &[Vec4]) {
    if image.data.len() < data.len() * 2 {
        panic!(
            "image.data size {} only bytes, but attempt to copy {} bytes",
            image.data.len(),
            data.len() * 2
        )
    }
    unsafe {
        let im = image.data.as_mut_ptr();
        std::ptr::copy(data.as_ptr() as *const u8, im, 4 * 4 * data.len());
    }
}

pub fn camera_trace_depth(
    entity_bvh: Query<(Entity, &GlobalTransform, &TraceMesh)>,
    mut camera: Query<(&GlobalTransform, &Projection, &mut TestTrace)>,
    mut images: ResMut<Assets<Image>>,
    (static_tlas, dynamic_tlas): (Res<StaticTLASData>, Res<DynamicTLASData>),
    blas: Res<BLAS>,
    mut materials: ResMut<Assets<CustomMaterial>>,
) {
    if let Some((trans, proj, mut test_trace)) = camera.iter_mut().next() {
        if let Projection::Perspective(proj) = proj {
            let projection = proj.get_projection_matrix();
            let inverse_view_proj = trans.compute_matrix() * projection.inverse();

            let origin = trans.translation();
            for py in 0..test_trace.height {
                for px in 0..test_trace.width {
                    let x = px as f32 * 2.0 / test_trace.width as f32 - 1.0;
                    let y = py as f32 * 2.0 / test_trace.height as f32 - 1.0;

                    let ws_eye =
                        (inverse_view_proj.project_point3(vec3(x, -y, 1.0)) - origin).normalize();

                    if let Some(hit) = trace_ray(
                        origin,
                        ws_eye,
                        &entity_bvh,
                        &static_tlas.0,
                        &dynamic_tlas.0,
                        &blas,
                    ) {
                        let hit_dist = hit.hitp.distance(origin);
                        if hit.intersection.backface || hit_dist < 0.0001 {
                            // Hit backface
                            test_trace.set_px(px, py, vec4(0.0, 0.0, 0.0, hit_dist));
                        } else {
                            let dist = origin.distance(hit.hitp);
                            test_trace.set_px(px, py, vec4(dist, dist, dist, 1.0));
                        }
                    } else {
                        // Hit sky
                        let dist = f32::INFINITY;
                        test_trace.set_px(px, py, vec4(dist, dist, dist, 1.0));
                    }
                }
            }
            test_trace.update(&mut images);
            for _mat in materials.iter_mut() {
                //mat.1.texture = test_trace.image.clone();
                // just touch materials
            }
        }
    }
}

/// The Material trait is very configurable, but comes with sensible defaults for all methods.
/// You only need to implement functions for features that need non-default behavior. See the Material api docs for details!
impl Material for CustomMaterial {
    fn fragment_shader() -> ShaderRef {
        "render_depth_material.wgsl".into()
    }
}

// This is the struct that will be passed to your shader
#[derive(AsBindGroup, Debug, Clone, TypeUuid, TypePath)]
#[uuid = "717f64fe-6844-4822-8926-e0ed374294c8"]
pub struct CustomMaterial {
    #[texture(0)]
    #[sampler(1)]
    pub texture: Handle<Image>,
}

fn main() {
    let mut app = App::new();

    app.add_plugins(
        DefaultPlugins
            .set(AssetPlugin {
                watch_for_changes: ChangeWatcher::with_delay(Duration::from_millis(200)),
                ..default()
            })
            .set(WindowPlugin {
                primary_window: Some(Window {
                    present_mode: PresentMode::AutoVsync,
                    ..default()
                }),
                ..default()
            }),
    )
    .add_systems(Startup, setup)
    .add_plugins((
        BVHPlugin,
        MaterialPlugin::<CustomMaterial>::default(),
        CameraControllerPlugin,
    ))
    .add_systems(Update, (camera_trace_depth, cube_rotator))
    .run();
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CustomMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let test_trace = TestTrace::new(64, 32, &mut images);

    let material_h = materials.add(CustomMaterial {
        texture: test_trace.image.clone(),
    });

    // plane
    commands
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(shape::Plane::from_size(5.0).into()),
            material: material_h.clone(),
            ..default()
        })
        .insert(StaticTLAS);
    // cube
    commands
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: material_h.clone(),
            transform: Transform::from_xyz(0.0, 0.5, 0.0),
            ..default()
        })
        .insert(StaticTLAS);
    // rotating cube
    commands
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: material_h.clone(),
            transform: Transform::from_xyz(0.0, 2.5, 0.0),
            ..default()
        })
        .insert(DynamicTLAS)
        .insert(RotateCube);
    // camera
    commands
        .spawn(Camera3dBundle {
            transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        })
        .insert(CameraController::default())
        .insert(test_trace);
}

#[derive(Component)]
struct RotateCube;

fn cube_rotator(time: Res<Time>, mut query: Query<&mut Transform, With<RotateCube>>) {
    for mut transform in &mut query {
        transform.rotate_x(1.0 * time.delta_seconds());
        transform.rotate_y(0.7 * time.delta_seconds());
    }
}
