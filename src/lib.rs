pub mod ray;
pub mod trace;

use bevy::math::vec3a;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, VertexAttributeValues};
use bevy::render::primitives::Aabb;
use bevy::utils::{HashMap, Instant};
use bvh::aabb::{Bounded, AABB};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::BVH;

pub struct BVHPlugin;
impl Plugin for BVHPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(MultiTLAS::default())
            .insert_resource(BLAS::default())
            .add_systems((check_tlas_need_update, build_blas, update_tlas).chain());
    }
}

pub fn build_blas(
    entities: Query<&Handle<Mesh>>,
    meshes: Res<Assets<Mesh>>,
    mut blas: ResMut<BLAS>,
) {
    // TODO remove items from BLAS when they are no longer needed
    let now = Instant::now();
    let mut count = 0;
    // Build BVHs as needed
    for mesh_h in entities.iter() {
        if blas.0.get(mesh_h).is_none() {
            if let Some(mesh) = meshes.get(mesh_h) {
                if let Some(bvh) = MeshBVHItem::new(mesh) {
                    blas.0.insert(mesh_h.clone(), bvh);
                    count += 1;
                }
            }
        }
    }
    if count > 0 {
        println!("Time to build {count} BVHs {}", now.elapsed().as_secs_f32());
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
                Changed<Aabb>,
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
                Changed<Aabb>,
                Changed<DynamicTLAS>,
            )>,
        ),
    >,
    mut tlas: ResMut<MultiTLAS>,
) {
    tlas.static_tlas.skip_update = static_entities.is_empty();
    tlas.dynamic_tlas.skip_update = dynamic_entities.is_empty();
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
    mut multi_tlas: ResMut<MultiTLAS>,
) {
    if !multi_tlas.static_tlas.skip_update {
        let mut static_aabbs = Vec::new();
        for (entity, trans, aabb, visibility) in &static_entities {
            if !visibility.is_visible() {
                continue;
            }
            static_aabbs.push(TLASAABB::new(entity.clone(), aabb, trans));
        }
        if !static_aabbs.is_empty() {
            multi_tlas.static_tlas.bvh = Some(BVH::build(&mut static_aabbs));
            multi_tlas.static_tlas.aabbs = static_aabbs;
        } else {
            multi_tlas.static_tlas.bvh = None;
            multi_tlas.static_tlas.aabbs = static_aabbs;
        }
    }
    if !multi_tlas.dynamic_tlas.skip_update {
        let mut dynamic_aabbs = Vec::new();
        for (entity, trans, aabb, visibility) in &dynamic_entities {
            if !visibility.is_visible() {
                continue;
            }
            dynamic_aabbs.push(TLASAABB::new(entity.clone(), aabb, trans));
        }
        if !dynamic_aabbs.is_empty() {
            multi_tlas.dynamic_tlas.bvh = Some(BVH::build(&mut dynamic_aabbs));
            multi_tlas.dynamic_tlas.aabbs = dynamic_aabbs;
        } else {
            multi_tlas.dynamic_tlas.bvh = None;
            multi_tlas.dynamic_tlas.aabbs = dynamic_aabbs;
        }
    }
}

// Bottom-Level Acceleration Structure
#[derive(Default, Resource)]
pub struct BLAS(pub HashMap<Handle<Mesh>, MeshBVHItem>);

pub struct MeshBVHItem {
    pub bvh: BVH,
    pub triangles: Vec<Triangle>,
}

impl MeshBVHItem {
    fn new(mesh: &Mesh) -> Option<MeshBVHItem> {
        if let VertexAttributeValues::Float32x3(vertices) =
            &mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap()
        {
            if let Indices::U32(indices) = &mesh.indices().unwrap() {
                let mut triangles = indices
                    .chunks(3)
                    .map(|chunk| {
                        Triangle::new(
                            vertices[chunk[0] as usize].into(),
                            vertices[chunk[1] as usize].into(),
                            vertices[chunk[2] as usize].into(),
                            [chunk[0], chunk[1], chunk[2]],
                        )
                    })
                    .collect::<Vec<Triangle>>();
                if triangles.len() > 0 {
                    return Some(MeshBVHItem {
                        bvh: BVH::build(&mut triangles),
                        triangles,
                    });
                } else {
                    return None;
                }
            }
        }
        None
    }
}

#[derive(Component)]
pub struct StaticTLAS;

#[derive(Component)]
pub struct DynamicTLAS;

// Top-Level Acceleration Structure
#[derive(Default)]
pub struct TLAS {
    pub bvh: Option<BVH>,
    pub aabbs: Vec<TLASAABB>,
    pub skip_update: bool,
}

// There isn't a functional difference between the static and dynamic TLAS.
// Either can be updated. For many scenes most entities are static, this allows them
// to be partitioned and only update the BVH for the dynamic ones as they move.
#[derive(Default, Resource)]
pub struct MultiTLAS {
    pub static_tlas: TLAS,
    pub dynamic_tlas: TLAS,
}

#[derive(Debug, Clone)]
pub struct TLASAABB {
    pub entity: Entity,
    aabb: AABB,
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
        let mut aabb = AABB::empty();
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
            aabb = aabb.grow(&model.transform_point3a(aabb_min + v).into());
        }

        TLASAABB {
            entity,
            aabb,
            node_index: 0,
        }
    }
}

impl Bounded for TLASAABB {
    fn aabb(&self) -> AABB {
        self.aabb
    }
}

impl BHShape for TLASAABB {
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
    aabb: AABB,
    node_index: usize,
}

impl Triangle {
    pub fn new(a: Vec3, b: Vec3, c: Vec3, indices: [u32; 3]) -> Triangle {
        Triangle {
            a,
            b,
            c,
            indices,
            aabb: AABB::empty().grow(&a).grow(&b).grow(&c),
            node_index: 0,
        }
    }

    pub fn local_mesh_normal(&self) -> Vec3 {
        (self.b - self.a).cross(self.c - self.a)
    }
}

impl Bounded for Triangle {
    fn aabb(&self) -> AABB {
        self.aabb
    }
}

impl BHShape for Triangle {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}
