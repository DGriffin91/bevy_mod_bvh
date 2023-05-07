// instance_data: ptr<storage, array<InstanceData>>
// Argument 'instance_data' at index 0 is a pointer of space Storage { access: LOAD }, which can't be passed into functions.
// need to figure out a why to not have to duplicate these functions
// seems like storage arrays cant be passed
// bindless would probably work, but isn't supported in webgpu

// IF ONE OF THESE FUNCTIONS ARE UPDATED, THEY ALL SHOULD BE

fn static_traverse_tlas(ray: Ray, min_dist: f32) -> Hit {
    var next_idx = 0;
    var temp_return = vec4(0.0);
    var hit: Hit;
    hit.distance = F32_MAX;
    var min_dist = min_dist;
    while (next_idx < i32(arrayLength(&static_tlas_buffer))) {
        let tlas = static_tlas_buffer[next_idx];
        let aabb_min = tlas.aabb_min;
        let aabb_max = tlas.aabb_max;
        let entry_idx = tlas.entry_or_shape_idx;
        let exit_idx = tlas.exit_idx;
        if entry_idx < 0 {
            // If the entry_index is negative, then it's a leaf node.
            // Shape index in this case is the mesh entity instance index
            // Look up the equivalent info as: static_tlas.0.aabbs[shape_index].entity
            let instance_idx = (entry_idx + 1) * -1;

            let instance = static_mesh_instance_buffer[instance_idx];

            let model = instance.model;

            // Transform ray into local instance space
            var local_ray: Ray;
            local_ray.origin = (model * vec4(ray.origin, 1.0)).xyz;
            local_ray.direction = normalize((model * vec4(ray.direction, 0.0)).xyz);
            local_ray.inv_direction = 1.0 / local_ray.direction;

            var new_hit = traverse_blas(instance.mesh_data, local_ray, min_dist);

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

fn dynamic_traverse_tlas(ray: Ray, min_dist: f32) -> Hit {
    var next_idx = 0;
    var temp_return = vec4(0.0);
    var hit: Hit;
    hit.distance = F32_MAX;
    var min_dist = min_dist;
    while (next_idx < i32(arrayLength(&dynamic_tlas_buffer))) {
        let tlas = dynamic_tlas_buffer[next_idx];
        let aabb_min = tlas.aabb_min;
        let aabb_max = tlas.aabb_max;
        let entry_idx = tlas.entry_or_shape_idx;
        let exit_idx = tlas.exit_idx;
        if entry_idx < 0 {
            // If the entry_index is negative, then it's a leaf node.
            // Shape index in this case is the mesh entity instance index
            // Look up the equivalent info as: dynamic_tlas.0.aabbs[shape_index].entity
            let instance_idx = (entry_idx + 1) * -1;

            let instance = dynamic_mesh_instance_buffer[instance_idx];

            let model = instance.model;

            // Transform ray into local instance space
            var local_ray: Ray;
            local_ray.origin = (model * vec4(ray.origin, 1.0)).xyz;
            local_ray.direction = normalize((model * vec4(ray.direction, 0.0)).xyz);
            local_ray.inv_direction = 1.0 / local_ray.direction;

            var new_hit = traverse_blas(instance.mesh_data, local_ray, min_dist);

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

