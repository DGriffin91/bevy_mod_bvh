
struct VertexData {
    position: vec3<f32>,
    normal: vec3<f32>,
}

struct MeshData {
    vert_idx_start: u32,
    vert_data_start: u32,
    blas_start: u32,
    blas_count: u32,
}

struct InstanceData {
    model: mat4x4<f32>,
    mesh_data: MeshData,
}

struct VertexIndices {
    idx: u32,
}

struct BVHData {
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    // precomputed triangle normals
    tri_nor: vec3<f32>, //TODO use smaller, less precise format
    //if positive: entry_idx if negative: -shape_idx
    entry_or_shape_idx: i32,
    exit_idx: i32,
}