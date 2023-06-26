pub mod gpu_data;
#[macro_use]
pub mod pipeline_utils;
pub mod packing;
pub mod ray;
pub mod rgb9e5;
pub mod trace;

use bevy::math::{vec3, vec3a};
use bevy::prelude::*;
use bevy::render::mesh::{Indices, VertexAttributeValues};
use bevy::utils::HashMap;
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::Bvh;
use nalgebra::{Point3, Vector3};

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum BVHSet {
    BlasTlas,
    GpuData,
}

pub struct BVHPlugin;
impl Plugin for BVHPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(TLASUpdateSkip::default())
            .insert_resource(DynamicTLASData::default())
            .insert_resource(StaticTLASData::default())
            .insert_resource(BLAS::default())
            .add_systems(
                Update,
                (check_tlas_need_update, build_blas, update_tlas)
                    .chain()
                    .in_set(BVHSet::BlasTlas),
            );
    }
}

pub fn build_blas(
    mut mesh_events: EventReader<AssetEvent<Mesh>>,
    meshes: Res<Assets<Mesh>>,
    mut blas: ResMut<BLAS>,
) {
    for event in mesh_events.iter() {
        match event {
            AssetEvent::Created { handle } | AssetEvent::Modified { handle } => {
                if let Some(bvh) = MeshBVHItem::new(meshes.get(handle).unwrap()) {
                    blas.0.insert(handle.clone(), bvh);
                }
            }
            AssetEvent::Removed { handle } => {
                let _ = blas.0.remove(handle);
            }
        }
    }
}

pub fn check_tlas_need_update(
    static_entities: Query<
        Entity,
        (
            With<Handle<Mesh>>,
            With<StaticTLAS>,
            Without<DynamicTLAS>,
            Or<(
                Changed<Transform>,
                Changed<Visibility>,
                Changed<bevy::render::primitives::Aabb>,
                Changed<StaticTLAS>,
            )>,
        ),
    >,
    dynamic_entities: Query<
        Entity,
        (
            With<Handle<Mesh>>,
            With<DynamicTLAS>,
            Without<StaticTLAS>,
            Or<(
                Changed<Transform>,
                Changed<Visibility>,
                Changed<bevy::render::primitives::Aabb>,
                Changed<DynamicTLAS>,
            )>,
        ),
    >,
    mut update: ResMut<TLASUpdateSkip>,
) {
    update.0 .0 = static_entities.is_empty();
    update.0 .1 = dynamic_entities.is_empty();
}

pub fn update_tlas(
    static_entities: Query<
        (
            Entity,
            &GlobalTransform,
            &bevy::render::primitives::Aabb,
            &ComputedVisibility,
        ),
        (With<Handle<Mesh>>, With<StaticTLAS>),
    >,
    dynamic_entities: Query<
        (
            Entity,
            &GlobalTransform,
            &bevy::render::primitives::Aabb,
            &ComputedVisibility,
        ),
        (With<Handle<Mesh>>, With<DynamicTLAS>),
    >,
    (mut static_tlas, mut dynamic_tlas): (ResMut<StaticTLASData>, ResMut<DynamicTLASData>),
    update: Res<TLASUpdateSkip>,
) {
    if !update.0 .0 {
        let mut static_aabbs = Vec::new();
        for (entity, trans, aabb, visibility) in &static_entities {
            if !visibility.is_visible() {
                continue;
            }
            static_aabbs.push(TLASAABB::new(entity, aabb, trans));
        }
        if !static_aabbs.is_empty() {
            static_tlas.0.bvh = Some(Bvh::build(&mut static_aabbs));
            static_tlas.0.aabbs = static_aabbs;
        } else {
            static_tlas.0.bvh = None;
            static_tlas.0.aabbs = static_aabbs;
        }
    }
    if !update.0 .1 {
        let mut dynamic_aabbs = Vec::new();
        for (entity, trans, aabb, visibility) in &dynamic_entities {
            if !visibility.is_visible() {
                continue;
            }
            dynamic_aabbs.push(TLASAABB::new(entity, aabb, trans));
        }
        if !dynamic_aabbs.is_empty() {
            dynamic_tlas.0.bvh = Some(Bvh::build(&mut dynamic_aabbs));
            dynamic_tlas.0.aabbs = dynamic_aabbs;
        } else {
            dynamic_tlas.0.bvh = None;
            dynamic_tlas.0.aabbs = dynamic_aabbs;
        }
    }
}

// Bottom-Level Acceleration Structure
#[derive(Default, Resource)]
pub struct BLAS(pub HashMap<Handle<Mesh>, MeshBVHItem>);

pub struct MeshBVHItem {
    pub bvh: Bvh<f32, 3>,
    pub triangles: Vec<Triangle>,
}

impl MeshBVHItem {
    fn new(mesh: &Mesh) -> Option<MeshBVHItem> {
        if let VertexAttributeValues::Float32x3(vertices) =
            &mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap()
        {
            let indices = match &mesh.indices().unwrap() {
                Indices::U16(indices) => {
                    let mut u32indices = Vec::new();
                    for ind in indices.iter() {
                        u32indices.push(*ind as u32);
                    }
                    u32indices.to_vec()
                }
                Indices::U32(indices) => indices.to_vec(),
            };
            let mut triangles = Vec::new();
            for chunk in indices.chunks(3) {
                let a: Vec3 = vertices[chunk[0] as usize].into();
                let b: Vec3 = vertices[chunk[1] as usize].into();
                let c: Vec3 = vertices[chunk[2] as usize].into();
                triangles.push(Triangle::new(a, b, c, [chunk[0], chunk[1], chunk[2]]));
            }
            if !triangles.is_empty() && indices.len() >= 3 {
                let bvh_item = MeshBVHItem {
                    bvh: Bvh::build(&mut triangles),
                    triangles,
                };
                return Some(bvh_item);
            } else {
                panic!("empty");
                //return None;
            }
        }
        panic!("incompatible");
        //None
    }
}

#[derive(Component)]
pub struct StaticTLAS;

#[derive(Component)]
pub struct DynamicTLAS;

// Top-Level Acceleration Structure
#[derive(Default)]
pub struct TLAS {
    pub bvh: Option<Bvh<f32, 3>>,
    pub aabbs: Vec<TLASAABB>,
    pub skip_update: bool,
}

// There isn't a functional difference between the static and dynamic TLAS.
// Either can be updated. For many scenes most entities are static, this allows them
// to be partitioned and only update the BVH for the dynamic ones as they move.

#[derive(Default, Resource)]
pub struct DynamicTLASData(pub TLAS);

#[derive(Default, Resource)]
pub struct StaticTLASData(pub TLAS);

#[derive(Default, Resource)]
pub struct TLASUpdateSkip(pub (bool, bool));

#[derive(Debug, Clone)]
pub struct TLASAABB {
    pub entity: Entity,
    aabb: Aabb<f32, 3>,
    node_index: usize,
}

impl TLASAABB {
    pub fn new(
        entity: Entity,
        bevy_aabb: &bevy::render::primitives::Aabb,
        trans: &GlobalTransform,
    ) -> TLASAABB {
        let model = trans.compute_matrix();
        let aabb_min = bevy_aabb.center - bevy_aabb.half_extents;
        let extents = bevy_aabb.half_extents * 2.0;
        let mut aabb = Aabb::empty();
        for v in [
            vec3a(0.0, 0.0, 0.0),
            vec3a(extents.x, 0.0, 0.0),
            vec3a(0.0, extents.y, 0.0),
            vec3a(extents.x, extents.y, 0.0),
            vec3a(0.0, 0.0, extents.z),
            vec3a(0.0, extents.y, extents.z),
            vec3a(extents.x, 0.0, extents.z),
            extents,
        ] {
            let p = &model.transform_point3a(aabb_min + v);
            aabb = aabb.grow(&Point3::new(p.x, p.y, p.z));
        }

        TLASAABB {
            entity,
            aabb,
            node_index: 0,
        }
    }
}

impl Bounded<f32, 3> for TLASAABB {
    fn aabb(&self) -> Aabb<f32, 3> {
        self.aabb
    }
}

impl BHShape<f32, 3> for TLASAABB {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

/// A triangle struct. Instance of a more complex `Bounded` primitive.
#[derive(Debug, Clone)]
pub struct Triangle {
    pub a: Vec3,
    pub b: Vec3,
    pub c: Vec3,
    pub indices: [u32; 3],
    aabb: Aabb<f32, 3>,
    node_index: usize,
}

impl Triangle {
    pub fn new(a: Vec3, b: Vec3, c: Vec3, indices: [u32; 3]) -> Triangle {
        Triangle {
            a,
            b,
            c,
            indices,
            aabb: Aabb::empty()
                .grow(&vec3_to_point3(a))
                .grow(&vec3_to_point3(b))
                .grow(&vec3_to_point3(c)),
            node_index: 0,
        }
    }

    pub fn local_mesh_normal(&self) -> Vec3 {
        (self.b - self.a).cross(self.c - self.a)
    }
}

impl Bounded<f32, 3> for Triangle {
    fn aabb(&self) -> Aabb<f32, 3> {
        self.aabb
    }
}

impl BHShape<f32, 3> for Triangle {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

fn vec3_to_point3(p: Vec3) -> Point3<f32> {
    Point3::new(p.x, p.y, p.z)
}

fn point3_to_vec3(p: Point3<f32>) -> Vec3 {
    vec3(p.x, p.y, p.z)
}

fn glam_vec3_to_vector3(p: Vec3) -> Vector3<f32> {
    Vector3::new(p.x, p.y, p.z)
}
