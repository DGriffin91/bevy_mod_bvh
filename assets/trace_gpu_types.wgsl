
fn octa_wrap(v: vec2<f32>) -> vec2<f32> {
    return (1.0 - abs(v.yx)) * sign(v.xy);
}

fn octa_encode(n_in: vec3<f32>) -> vec2<f32> {
    var n = n_in / (abs(n_in.x) + abs(n_in.y) + abs(n_in.z));
    if (n.z < 0.0) {
        n = vec3(octa_wrap(n.xy), n.z);
    }
    return n.xy * 0.5 + 0.5;
}

fn octa_decode(f_in: vec2<f32>) -> vec3<f32> {
    var f = f_in * 2.0 - 1.0;
    var n = vec3( f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) {
        n = vec3(octa_wrap(n.xy), n.z);
    }
    return normalize(n);
}

struct VertexDataPacked {
    data1_: u32,
    data2_: u32,
}

struct VertexData {
    position: vec3<f32>,
    normal: vec3<f32>,
}

fn VertexData_unpack(in: VertexDataPacked) -> VertexData {
    var vertex_data: VertexData;
    let pos_xy = unpack2x16float(in.data1_);
    let pos_z = unpack2x16float(in.data2_).x;
    vertex_data.position = vec3(pos_xy, pos_z);
    let octn = vec2(
        f32(in.data2_ >> 16u & 0xffu),
        f32(in.data2_ >> 24u & 0xffu),
    ) / 255.0;
    vertex_data.normal = octa_decode(octn);
    return vertex_data;
}

fn VertexData_unpack_pos(in: VertexDataPacked) -> vec3<f32> {
    var vertex_data: VertexData;
    let pos_xy = unpack2x16float(in.data1_);
    let pos_z = unpack2x16float(in.data2_).x;
    return vec3(pos_xy, pos_z);
}

struct MeshData {
    vert_idx_start: i32,
    vert_data_start: i32,
    blas_start: i32,
    blas_count: i32,
}

struct InstanceData {
    local_to_world: mat4x4<f32>,
    world_to_local: mat4x4<f32>,
    mesh_data: MeshData,
}

struct VertexIndices {
    idx: u32,
}

struct TLASBVHData {
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    //if positive: entry_idx if negative: -shape_idx
    entry_or_shape_idx: i32,
    exit_idx: i32,
}

struct BLASBVHData {
    // In the leaf nodes the aabbs are unused, could store the verts here. 
    // (maybe 10bit unorm inside the aabb?)
    aabb_minxy: u32, // f16 min x, y
    aabb_maxxy: u32, // f16 max x, y
    aabb_z: u32,     // f16 min z, max z
    //if positive: entry_idx if negative: -shape_idx
    entry_or_shape_idx: i32,
    exit_idx: i32,
}
