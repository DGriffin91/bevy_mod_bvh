pub const EPSILON: f32 = 0.00001;
use bvh::ray::Ray;
use nalgebra::Point3;
use std::f32::INFINITY;

#[derive(Clone)]
pub struct Intersection {
    /// Distance from the ray origin to the intersection point.
    pub distance: f32,

    /// U coordinate of the intersection.
    pub u: f32,

    /// V coordinate of the intersection.
    pub v: f32,

    /// Hit backface of triangle
    pub backface: bool,
}

impl Intersection {
    /// Constructs an `Intersection`. `distance` should be set to positive infinity,
    /// if the intersection does not occur.
    pub fn new(distance: f32, u: f32, v: f32, backface: bool) -> Intersection {
        Intersection {
            distance,
            u,
            v,
            backface,
        }
    }
}

pub fn intersects_triangle(
    ray: &Ray<f32, 3>,
    a: &Point3<f32>,
    b: &Point3<f32>,
    c: &Point3<f32>,
) -> Intersection {
    let a_to_b = *b - *a;
    let a_to_c = *c - *a;

    // Begin calculating determinant - also used to calculate u parameter
    // u_vec lies in view plane
    // length of a_to_c in view_plane = |u_vec| = |a_to_c|*sin(a_to_c, dir)
    let u_vec = ray.direction.cross(&a_to_c);

    // If determinant is near zero, ray lies in plane of triangle
    // The determinant corresponds to the parallelepiped volume:
    // det = 0 => [dir, a_to_b, a_to_c] not linearly independant
    let det = a_to_b.dot(&u_vec);

    // Only testing positive bound, thus enabling backface culling
    // If backface culling is not desired write:
    // det < EPSILON && det > -EPSILON
    // For backface culling on: det < EPSILON
    if det < EPSILON && det > -EPSILON {
        return Intersection::new(INFINITY, 0.0, 0.0, false);
    }

    let inv_det = 1.0 / det;

    // Vector from point a to ray origin
    let a_to_origin = ray.origin - *a;

    // Calculate u parameter
    let u = a_to_origin.dot(&u_vec) * inv_det;

    // Test bounds: u < 0 || u > 1 => outside of triangle
    if !(0.0..=1.0).contains(&u) {
        return Intersection::new(INFINITY, u, 0.0, false);
    }

    // Prepare to test v parameter
    let v_vec = a_to_origin.cross(&a_to_b);

    // Calculate v parameter and test bound
    let v = ray.direction.dot(&v_vec) * inv_det;
    // The intersection lies outside of the triangle
    if v < 0.0 || u + v > 1.0 {
        return Intersection::new(INFINITY, u, v, false);
    }

    let dist = a_to_c.dot(&v_vec) * inv_det;

    if dist > EPSILON {
        Intersection::new(dist, u, v, det < EPSILON)
    } else {
        Intersection::new(INFINITY, u, v, false)
    }
}
