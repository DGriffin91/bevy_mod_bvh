use bevy::math::*;

pub fn octa_wrap(v: Vec2) -> Vec2 {
    return (1.0 - v.yx().abs()) * v.signum();
}

pub fn octa_encode(n: Vec3) -> Vec2 {
    let mut n = n / (n.x.abs() + n.y.abs() + n.z.abs());
    if n.z < 0.0 {
        let w = octa_wrap(n.xy());
        n.x = w.x;
        n.y = w.y;
    }
    return n.xy() * 0.5 + 0.5;
}

pub fn octa_decode(f: Vec2) -> Vec3 {
    let f = f * 2.0 - 1.0;
    let mut n = vec3(f.x, f.y, 1.0 - f.x.abs() - f.y.abs());
    if n.z < 0.0 {
        let w = octa_wrap(n.xy());
        n.x = w.x;
        n.y = w.y;
    }
    return n.normalize();
}
