fn get_vert_position(idx: i32) -> vec3<f32> {
    let idx = idx * 2 + 0;
    let dimension = textureDimensions(vert_indices).x;
    return textureLoad(vert_pos_nor, vec2<i32>(idx % dimension, idx / dimension), 0).xyz;
}

fn get_vert_normal(idx: i32) -> vec3<f32> {
    let idx = idx * 2 + 1;
    let dimension = textureDimensions(vert_indices).x;
    return textureLoad(vert_pos_nor, vec2<i32>(idx % dimension, idx / dimension), 0).xyz;
}

fn get_vert_index(idx: i32) -> i32 {
    let dimension = textureDimensions(vert_indices).x;
    return i32(textureLoad(vert_indices, vec2<i32>(idx % dimension, idx / dimension), 0).x);
}

fn get_tlas_max_length(tlas_tex: texture_2d<f32>) -> i32 {
    return i32(textureLoad(tlas_tex, vec2<i32>(0, 0), 0).x);
}

fn get_tlas_bvh(tlas_tex: texture_2d<f32>, idx: i32) -> vec4<f32> {
    let idx = idx + 1; //first is length, todo move elsewhere
    let dimension = textureDimensions(tlas_tex).x;
    return textureLoad(tlas_tex, vec2<i32>(idx % dimension, idx / dimension), 0);
}

fn get_blas_bvh(idx: i32) -> vec4<f32> {
    let dimension = textureDimensions(blas).x;
    return textureLoad(blas, vec2<i32>(idx % dimension, idx / dimension), 0);
}

fn get_instance_diffuse(instance_tex: texture_2d<f32>, idx: i32) -> vec4<f32> {
    let idx = idx * 7 + 0;
    let dimension = textureDimensions(instance_tex).x;
    return textureLoad(instance_tex, vec2<i32>(idx % dimension, idx / dimension), 0);
}

fn get_instance_emit(instance_tex: texture_2d<f32>, idx: i32) -> vec4<f32> {
    let idx = idx * 7 + 1;
    let dimension = textureDimensions(instance_tex).x;
    return textureLoad(instance_tex, vec2<i32>(idx % dimension, idx / dimension), 0);
}

fn get_instance_mesh_data_idx(instance_tex: texture_2d<f32>, idx: i32) -> u32 {
    let idx = idx * 7 + 2;
    let dimension = textureDimensions(instance_tex).x;
    return bitcast<u32>(textureLoad(instance_tex, vec2<i32>(idx % dimension, idx / dimension), 0).x);
}

fn get_instance_model(instance_tex: texture_2d<f32>, idx: i32) -> mat4x4<f32> {
    let idx = idx * 7 + 3;
    let dimension = textureDimensions(instance_tex).x;
    return mat4x4<f32>(
        textureLoad(instance_tex, vec2<i32>((idx+0) % dimension, (idx+0) / dimension), 0),
        textureLoad(instance_tex, vec2<i32>((idx+1) % dimension, (idx+1) / dimension), 0),
        textureLoad(instance_tex, vec2<i32>((idx+2) % dimension, (idx+2) / dimension), 0),
        textureLoad(instance_tex, vec2<i32>((idx+3) % dimension, (idx+3) / dimension), 0),
    );
}

fn mesh_index_start(idx: i32) -> u32 {
    let idx = idx * 4 + 0;
    let dimension = textureDimensions(mesh_data).x;
    return textureLoad(mesh_data, vec2<i32>(idx % dimension, idx / dimension), 0).x;
}

fn mesh_pos_start(idx: i32) -> u32 {
    let idx = idx * 4 + 1;
    let dimension = textureDimensions(mesh_data).x;
    return textureLoad(mesh_data, vec2<i32>(idx % dimension, idx / dimension), 0).x;
}

fn mesh_blas_start(idx: i32) -> u32 {
    let idx = idx * 4 + 2;
    let dimension = textureDimensions(mesh_data).x;
    return textureLoad(mesh_data, vec2<i32>(idx % dimension, idx / dimension), 0).x;
}

fn mesh_blas_count(idx: i32) -> u32 {
    let idx = idx * 4 + 3;
    let dimension = textureDimensions(mesh_data).x;
    return textureLoad(mesh_data, vec2<i32>(idx % dimension, idx / dimension), 0).x;
}

//--------------------------
//--------------------------
//--------------------------

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    inv_direction: vec3<f32>,
};

struct Hit {
    normal: vec3<f32>,
    distance: f32,
    instance_idx: i32,
    backface: bool,
};

struct Aabb {
    min: vec3<f32>,
    max: vec3<f32>,
};

struct Intersection {
    uv: vec2<f32>,
    distance: f32,
    backface: bool,
};

fn inside_aabb(p: vec3<f32>, minv: vec3<f32>, maxv: vec3<f32>) -> bool {
    return all(p > minv) && all(p < maxv);
}

fn intersects_aabb(ray: Ray, minv: vec3<f32>, maxv: vec3<f32>) -> f32 {
    let t1 = (minv - ray.origin) * ray.inv_direction;
    let t2 = (maxv - ray.origin) * ray.inv_direction;

    var t_min = min(t1.x, t2.x);
    var t_max = max(t1.x, t2.x);

    t_min = max(t_min, min(t1.y, t2.y));
    t_max = min(t_max, max(t1.y, t2.y));

    t_min = max(t_min, min(t1.z, t2.z));
    t_max = min(t_max, max(t1.z, t2.z));

    return select(F32_MAX, t_min, t_max >= t_min && t_max >= 0.0);
}

fn intersects_triangle(ray: Ray, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>) -> Intersection {
    var result: Intersection;
    result.distance = F32_MAX;
    result.backface = false;

    let ab = p1 - p2;
    let ac = p3 - p2;

    let u_vec = cross(ray.direction, ac);
    let det = dot(ab, u_vec);
    if abs(det) < F32_EPSILON {
        return result;
    }

    let inv_det = 1.0 / det;
    let ao = ray.origin - p2;
    let u = dot(ao, u_vec) * inv_det;
    if u < 0.0 || u > 1.0 {
        result.uv = vec2<f32>(u, 0.0);
        return result;
    }

    let v_vec = cross(ao, ab);
    let v = dot(ray.direction, v_vec) * inv_det;
    result.uv = vec2<f32>(u, v);
    if v < 0.0 || u + v > 1.0 {
        return result;
    }

    let distance = dot(ac, v_vec) * inv_det;
    result.distance = select(result.distance, distance, distance > F32_EPSILON);

    result.backface = distance > F32_EPSILON && det < F32_EPSILON;

    return result;
}

fn traverse_blas(ray: Ray, mesh_idx: i32) -> Hit {
    let blas_start = i32(mesh_blas_start(mesh_idx));
    let blas_count = i32(mesh_blas_count(mesh_idx));
    let mesh_index_start = i32(mesh_index_start(mesh_idx));
    let mesh_pos_start = i32(mesh_pos_start(mesh_idx));

    var next_idx = 0;
    var min_dist = F32_MAX;
    var hit: Hit;
    hit.normal = vec3(0.0, 0.0, 0.0);
    hit.distance = F32_MAX;
    while (next_idx < blas_count) {
        let aabb_min = get_blas_bvh(next_idx * 2 + 0 + blas_start); 
        let aabb_max = get_blas_bvh(next_idx * 2 + 1 + blas_start); 
        let entry_idx = bitcast<i32>(aabb_min.w);
        let exit_idx = bitcast<i32>(aabb_max.w);
        if entry_idx < 0 {
            let shape_idx = (entry_idx + 1) * -3;
            // If the entry_index is negative, then it's a leaf node.
            let ind1 = get_vert_index(shape_idx + 0 + mesh_index_start);
            let ind2 = get_vert_index(shape_idx + 1 + mesh_index_start);
            let ind3 = get_vert_index(shape_idx + 2 + mesh_index_start);
            let p1 = get_vert_position(ind1 + mesh_pos_start);
            let p2 = get_vert_position(ind2 + mesh_pos_start);
            let p3 = get_vert_position(ind3 + mesh_pos_start);
            

            let intr = intersects_triangle(ray, p1, p2, p3);
            if intr.distance < hit.distance {
                // vert order is acb?
                let a = get_vert_normal(ind1 + mesh_pos_start).xyz;
                let c = get_vert_normal(ind2 + mesh_pos_start).xyz;
                let b = get_vert_normal(ind3 + mesh_pos_start).xyz;
                // Barycentric Coordinates
                let u = intr.uv.x;
                let v = intr.uv.y;
                hit.normal = u * a + v * b + (1.0 - u - v) * c;
                hit.distance = intr.distance;
                hit.backface = intr.backface;
            }
            //min_dist = min(min_dist, intr.distance);
            // Exit the current node.
            next_idx = exit_idx;
        } else {
            // If entry_index is not -1 and the AABB test passes, then
            // proceed to the node in entry_index (which goes down the bvh branch).

            // If entry_index is not -1 and the AABB test fails, then
            // proceed to the node in exit_index (which defines the next untested partition).
            next_idx = select(exit_idx, 
                              entry_idx, 
                              intersects_aabb(ray, aabb_min.xyz, aabb_max.xyz) < min_dist);
        }
    }
    return hit;
}

fn traverse_tlas(tlas_tex: texture_2d<f32>, instance_tex: texture_2d<f32>, ray: Ray) -> Hit {
    var next_idx = 0;
    var min_dist = F32_MAX;
    var temp_return = vec4(0.0);
    var hit: Hit;
    hit.normal = vec3(0.0, 0.0, 0.0);
    hit.distance = F32_MAX;
    while (next_idx < get_tlas_max_length(tlas_tex)) {
        let aabb_min = get_tlas_bvh(tlas_tex, next_idx * 2 + 0); 
        let aabb_max = get_tlas_bvh(tlas_tex, next_idx * 2 + 1); 
        let entry_idx = bitcast<i32>(aabb_min.w);
        let exit_idx = bitcast<i32>(aabb_max.w);
        if entry_idx < 0 {
            // If the entry_index is negative, then it's a leaf node.
            // Shape index in this case is the mesh entity instance index
            // Look up the equivalent info as: static_tlas.0.aabbs[shape_index].entity
            let instance_idx = (entry_idx + 1) * -1;

            let mesh_idx = i32(get_instance_mesh_data_idx(instance_tex, instance_idx));

            let model = get_instance_model(instance_tex, instance_idx);

            // Transform ray into local instance space
            var local_ray: Ray;
            local_ray.origin = (model * vec4(ray.origin, 1.0)).xyz;
            local_ray.direction = normalize((model * vec4(ray.direction, 0.0)).xyz);
            local_ray.inv_direction = 1.0 / local_ray.direction;

            var new_hit = traverse_blas(local_ray, i32(mesh_idx));

            if new_hit.distance < hit.distance {
                hit = new_hit;
                hit.instance_idx = instance_idx;
                // transform local space normal into world space
                hit.normal = normalize(transpose(model) * vec4(new_hit.normal, 0.0)).xyz;
            }

            // TODO lookup mesh BVH from instance and traverse_blas()
            // Exit the current node.
            next_idx = exit_idx;
        } else {
            // If entry_index is not -1 and the AABB test passes, then
            // proceed to the node in entry_index (which goes down the bvh branch).

            // If entry_index is not -1 and the AABB test fails, then
            // proceed to the node in exit_index (which defines the next untested partition).
            next_idx = select(exit_idx, 
                              entry_idx, 
                              intersects_aabb(ray, aabb_min.xyz, aabb_max.xyz) < min_dist);
        }
    }
    return hit;
}