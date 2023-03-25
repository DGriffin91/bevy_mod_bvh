use bevy::prelude::*;
use bvh::ray::Ray;

use crate::{
    ray::{intersects_triangle, Intersection},
    Triangle, BLAS, TLAS,
};

#[derive(Clone)]
pub struct HitResult {
    pub hitp: Vec3,
    pub tri: Triangle,
    pub mat: Mat4,
    pub entity: Entity,
    pub intersection: Intersection,
}

pub fn append_closest_entities_hit(
    tlas: &TLAS,
    blas: &BLAS,
    scene_ray: &Ray,
    entities: &Query<(Entity, &GlobalTransform, &Handle<Mesh>)>,
    hit_entites: &mut Vec<HitResult>,
) {
    let bvh = if let Some(bvh) = &tlas.bvh {
        bvh
    } else {
        return;
    };
    let entity_hits = bvh.traverse(&scene_ray, &tlas.aabbs);
    for entity_hit in entity_hits {
        if let Ok((entity, trans, mesh_h)) = entities.get(entity_hit.entity) {
            let binding = blas.0.get(mesh_h);
            let bvh_item = if let Some(bvh_item) = &binding {
                bvh_item
            } else {
                continue;
            };
            let mat = trans.compute_matrix();
            let inv_mat = mat.inverse();

            let t_origin = inv_mat.project_point3(scene_ray.origin);
            let t_direction = inv_mat.transform_vector3(scene_ray.direction);

            let ray = Ray::new(t_origin, t_direction);
            let hits = bvh_item.bvh.traverse(&ray, &bvh_item.triangles);

            let mut closest = Intersection::new(std::f32::INFINITY, 0.0, 0.0, false);
            let mut closest_tri = None;
            for tri in hits {
                let hit = intersects_triangle(&ray, &tri.a, &tri.b, &tri.c);

                if hit.distance < closest.distance {
                    closest = hit;
                    closest_tri = Some(tri);
                }
            }
            if let Some(tri) = closest_tri {
                hit_entites.push(HitResult {
                    hitp: mat.project_point3(closest.distance * t_direction + t_origin),
                    mat,
                    entity,
                    tri: tri.clone(),
                    intersection: closest,
                });
            }
        }
    }
}

// Return closest hit if any
pub fn trace_ray(
    origin: Vec3,
    direction: Vec3,
    entities: &Query<(Entity, &GlobalTransform, &Handle<Mesh>)>,
    static_tlas: &TLAS,
    dynamic_tlas: &TLAS,
    blas: &BLAS,
) -> Option<HitResult> {
    //Not handling non-uniform scale correctly

    let mut hit_entites: Vec<HitResult> = Vec::new();

    let scene_ray = Ray::new(origin, direction);

    append_closest_entities_hit(static_tlas, blas, &scene_ray, &entities, &mut hit_entites);

    append_closest_entities_hit(dynamic_tlas, blas, &scene_ray, &entities, &mut hit_entites);

    let mut closest_dist = std::f32::MAX;

    let mut closest_hit = None;
    for (i, hit) in hit_entites.iter().enumerate() {
        let dist = hit.hitp.distance(origin);
        if dist < closest_dist {
            closest_dist = dist;
            closest_hit = Some(i);
        }
    }

    if let Some(idx) = closest_hit {
        return Some(hit_entites[idx].clone());
    }
    None
}

/// returns true if blocked
pub fn trace_visibility_ray(
    origin: Vec3,
    dest: Vec3,
    entity_bvh: &Query<(Entity, &GlobalTransform, &Handle<Mesh>)>,
    static_tlas: &TLAS,
    dynamic_tlas: &TLAS,
    blas: &BLAS,
) -> bool {
    let mut blocked = false;
    let direction = (dest - origin).normalize();
    let statin_entity_hits = if let Some(scene_bvh) = &static_tlas.bvh {
        let scene_ray = Ray::new(origin, direction);
        scene_bvh.traverse(&scene_ray, &static_tlas.aabbs)
    } else {
        Vec::new()
    };
    let dynamic_entity_hits = if let Some(scene_bvh) = &dynamic_tlas.bvh {
        let scene_ray = Ray::new(origin, direction);
        scene_bvh.traverse(&scene_ray, &dynamic_tlas.aabbs)
    } else {
        Vec::new()
    };
    'outer: for entity_hit in dynamic_entity_hits.iter().chain(statin_entity_hits.iter()) {
        if let Ok((_entity, trans, mesh_h)) = entity_bvh.get(entity_hit.entity) {
            let binding = blas.0.get(mesh_h);
            let bvh_item = if let Some(bvh_item) = &binding {
                bvh_item
            } else {
                continue;
            };
            let mat = trans.compute_matrix();
            let inv_mat = mat.inverse();

            let t_origin = inv_mat.project_point3(origin);
            let t_direction = inv_mat.transform_vector3(direction);
            let t_dest_pos = inv_mat.project_point3(dest);
            let t_distance = t_dest_pos.distance(t_origin);

            let ray = Ray::new(t_origin, t_direction);
            let hits = bvh_item.bvh.traverse(&ray, &bvh_item.triangles);

            for tri in hits {
                let hit = intersects_triangle(&ray, &tri.a, &tri.b, &tri.c);

                if hit.distance < t_distance {
                    blocked = true;
                    break 'outer;
                }
            }
        }
    }

    blocked
}
