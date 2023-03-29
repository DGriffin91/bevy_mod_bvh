fn get_vert_position(idx: i32) -> vec3<f32> {
    let dimension = textureDimensions(vert_pos).x;
    return textureLoad(vert_pos, vec2<i32>(idx % dimension, idx / dimension), 0).xyz;
}

fn get_vert_normal(idx: i32) -> vec3<f32> {
    let dimension = textureDimensions(vert_nor).x;
    return textureLoad(vert_nor, vec2<i32>(idx % dimension, idx / dimension), 0).xyz;
}

fn get_vert_index(idx: i32) -> i32 {
    let dimension = textureDimensions(vert_indices).x;
    return textureLoad(vert_indices, vec2<i32>(idx % dimension, idx / dimension), 0).x;
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

fn get_instance_mesh_data_idx(instance_tex: texture_2d<f32>, idx: i32) -> i32 {
    let idx = idx * 7 + 2;
    let dimension = textureDimensions(instance_tex).x;
    return bitcast<i32>(textureLoad(instance_tex, vec2<i32>(idx % dimension, idx / dimension), 0).x);
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

fn mesh_index_start(idx: i32) -> i32 {
    let idx = idx * 4 + 0;
    let dimension = textureDimensions(mesh_data).x;
    return textureLoad(mesh_data, vec2<i32>(idx % dimension, idx / dimension), 0).x;
}

fn mesh_pos_start(idx: i32) -> i32 {
    let idx = idx * 4 + 1;
    let dimension = textureDimensions(mesh_data).x;
    return textureLoad(mesh_data, vec2<i32>(idx % dimension, idx / dimension), 0).x;
}

fn mesh_blas_start(idx: i32) -> i32 {
    let idx = idx * 4 + 2;
    let dimension = textureDimensions(mesh_data).x;
    return textureLoad(mesh_data, vec2<i32>(idx % dimension, idx / dimension), 0).x;
}

fn mesh_blas_count(idx: i32) -> i32 {
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
    uv: vec2<f32>,
    distance: f32,
    instance_idx: i32,
    triangle_idx: i32,
};

struct SceneQuery {
    hit: Hit,
    static_tlas: bool,
};

struct Aabb {
    min: vec3<f32>,
    max: vec3<f32>,
};

struct Intersection {
    uv: vec2<f32>,
    distance: f32,
};

fn inside_aabb(p: vec3<f32>, minv: vec3<f32>, maxv: vec3<f32>) -> bool {
    return all(p > minv) && all(p < maxv);
}

// returns distance to intersection
fn intersects_aabb(ray: Ray, minv: vec3<f32>, maxv: vec3<f32>) -> f32 {
    let t1 = (minv - ray.origin) * ray.inv_direction;
    let t2 = (maxv - ray.origin) * ray.inv_direction;

    let tmin = min(t1, t2);
    let tmax = max(t1, t2);

    let tmin_n = max(tmin.x, max(tmin.y, tmin.z));
    let tmax_n = min(tmax.x, min(tmax.y, tmax.z));

    return select(F32_MAX, tmin_n, tmax_n >= tmin_n && tmax_n >= 0.0);
}

// A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel Rendering
// Alexander Majercik, Cyril Crassin, Peter Shirley, and Morgan McGuire
fn slabs(ray: Ray, minv: vec3<f32>, maxv: vec3<f32>) -> bool {
    let t0 = (minv - ray.origin) * ray.inv_direction;
    let t1 = (maxv - ray.origin) * ray.inv_direction;

    let tmin = min(t0, t1);
    let tmax = max(t0, t1);

    return max(tmin.x, max(tmin.y, tmin.z)) <= min(tmax.x, min(tmax.y, tmax.z));
}



fn intersects_triangle(ray: Ray, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>) -> Intersection {
    var result: Intersection;
    result.distance = F32_MAX;

    let ab = p1 - p2;
    let ac = p3 - p2;

    let u_vec = cross(ray.direction, ac);
    let det = dot(ab, u_vec);

    // If backface culling on: det, off: abs(det)
    if abs(det) < F32_EPSILON  {
        return result;
    }

    let inv_det = 1.0 / det;
    let ao = ray.origin - p2;
    let u = dot(ao, u_vec) * inv_det;
    if u < 0.0 || u > 1.0 {
        return result;
    }

    let v_vec = cross(ao, ab);
    let v = dot(ray.direction, v_vec) * inv_det;
    result.uv = vec2(u, v);
    if v < 0.0 || u + v > 1.0 {
        return result;
    }

    let distance = dot(ac, v_vec) * inv_det;
    result.distance = select(result.distance, distance, distance > F32_EPSILON);


    return result;
}

fn traverse_blas(ray: Ray, mesh_idx: i32, min_dist: f32) -> Hit {
    let blas_start = mesh_blas_start(mesh_idx);
    let blas_count = mesh_blas_count(mesh_idx);
    let mesh_index_start = mesh_index_start(mesh_idx);
    let mesh_pos_start = mesh_pos_start(mesh_idx);
    
    //TODO Should we start at 1 since we already tested aginst the first AABB in the TLAS?
    var next_idx = 0; 
    var hit: Hit;
    hit.distance = F32_MAX;
    var min_dist = min_dist;
    while (next_idx < blas_count) {
        let aabb_min = get_blas_bvh(next_idx * 2 + 0 + blas_start); 
        let aabb_max = get_blas_bvh(next_idx * 2 + 1 + blas_start); 
        let entry_idx = bitcast<i32>(aabb_min.w);
        let exit_idx = bitcast<i32>(aabb_max.w);
        if entry_idx < 0 {
            let triangle_idx = (entry_idx + 1) * -3;
            // If the entry_index is negative, then it's a leaf node.
            let ind1 = get_vert_index(triangle_idx + 0 + mesh_index_start);
            let ind2 = get_vert_index(triangle_idx + 1 + mesh_index_start);
            let ind3 = get_vert_index(triangle_idx + 2 + mesh_index_start);
            let p1 = get_vert_position(ind1 + mesh_pos_start);
            let p2 = get_vert_position(ind2 + mesh_pos_start);
            let p3 = get_vert_position(ind3 + mesh_pos_start);

            // vert order is acb?
            let intr = intersects_triangle(ray, p1, p3, p2);
            if intr.distance < hit.distance {
                hit.distance = intr.distance;
                hit.triangle_idx = triangle_idx;
                hit.uv = intr.uv;
            }
            min_dist = min(min_dist, hit.distance);
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

fn traverse_tlas(tlas_tex: texture_2d<f32>, instance_tex: texture_2d<f32>, ray: Ray, min_dist: f32) -> Hit {
    var next_idx = 0;
    var temp_return = vec4(0.0);
    var hit: Hit;
    hit.distance = F32_MAX;
    var min_dist = min_dist;
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

            let mesh_idx = get_instance_mesh_data_idx(instance_tex, instance_idx);

            let model = get_instance_model(instance_tex, instance_idx);

            // Transform ray into local instance space
            var local_ray: Ray;
            local_ray.origin = (model * vec4(ray.origin, 1.0)).xyz;
            local_ray.direction = normalize((model * vec4(ray.direction, 0.0)).xyz);
            local_ray.inv_direction = 1.0 / local_ray.direction;

            var new_hit = traverse_blas(local_ray, mesh_idx, min_dist);

            if new_hit.distance < hit.distance {
                hit = new_hit;
                hit.instance_idx = instance_idx;
            }
            min_dist = min(min_dist, hit.distance);

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

fn scene_query(ray: Ray) -> SceneQuery {
    let hit_static = traverse_tlas(gpu_static_tlas_data, gpu_static_instance_data, ray, F32_MAX);
    let hit_dynamic = traverse_tlas(gpu_dynamic_tlas_data, gpu_dynamic_instance_data, ray, hit_static.distance);

    if hit_static.distance < hit_dynamic.distance {
        var query: SceneQuery;
        query.hit = hit_static;
        query.static_tlas = true;
        return query;
    } else {
        var query: SceneQuery;
        query.hit = hit_dynamic;
        query.static_tlas = false;
        return query;
    }
}

// Inefficient, don't use this if getting more than normal.
fn get_tri_normal(instance_tex: texture_2d<f32>, hit: Hit) -> vec3<f32> {
    let mesh_idx = get_instance_mesh_data_idx(instance_tex, hit.instance_idx);
    let mesh_index_start = mesh_index_start(mesh_idx);
    let mesh_pos_start = mesh_pos_start(mesh_idx);
    let model = get_instance_model(instance_tex, hit.instance_idx);

    let ind1 = get_vert_index(hit.triangle_idx + 0 + mesh_index_start);
    let ind2 = get_vert_index(hit.triangle_idx + 1 + mesh_index_start);
    let ind3 = get_vert_index(hit.triangle_idx + 2 + mesh_index_start);
    
    let a = get_vert_normal(ind1 + mesh_pos_start).xyz;
    let b = get_vert_normal(ind2 + mesh_pos_start).xyz;
    let c = get_vert_normal(ind3 + mesh_pos_start).xyz;

    // Barycentric Coordinates
    let u = hit.uv.x;
    let v = hit.uv.y;
    var normal = u * a + v * b + (1.0 - u - v) * c;
    
    // transform local space normal into world space
    normal = normalize(transpose(model) * vec4(normal, 0.0)).xyz;

    return normal;
}

fn get_surface_normal(instance_tex: texture_2d<f32>, hit: Hit) -> vec3<f32> {
    let mesh_idx = get_instance_mesh_data_idx(instance_tex, hit.instance_idx);
    let mesh_index_start = mesh_index_start(mesh_idx);
    let mesh_pos_start = mesh_pos_start(mesh_idx);
    let model = get_instance_model(instance_tex, hit.instance_idx);    
    
    let ind1 = get_vert_index(hit.triangle_idx + 0 + mesh_index_start);
    let ind2 = get_vert_index(hit.triangle_idx + 1 + mesh_index_start);
    let ind3 = get_vert_index(hit.triangle_idx + 2 + mesh_index_start);
    
    let a = get_vert_position(ind1 + mesh_pos_start).xyz;
    let b = get_vert_position(ind2 + mesh_pos_start).xyz;
    let c = get_vert_position(ind3 + mesh_pos_start).xyz;

    let v1 = b - a;
    let v2 = c - a;
    var normal = normalize(cross(v1, v2));

    // transform local space normal into world space
    normal = normalize(transpose(model) * vec4(normal, 0.0)).xyz;

    return normal; 
}