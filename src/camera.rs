use glam::{Mat4, Vec3};

#[derive(Debug, Copy, Clone)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,

    pub fov_y: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn new(eye: Vec3, target: Vec3) -> Self {
        Self {
            eye,
            target,
            up: Vec3::Y,
            fov_y: 45_f32.to_radians(),
            znear: 0.1,
            zfar: 100.0,
        }
    }

    fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    fn proj(&self, width: f32, height: f32) -> Mat4 {
        let aspect = width / height.max(1.0);
        Mat4::perspective_rh(self.fov_y, aspect, self.znear, self.zfar)
    }

    pub fn view_proj(&self, width: f32, height: f32) -> Mat4 {
        self.proj(width, height) * self.view()
    }

    pub fn fit_to_aabb(
        bounds_min: Vec3,
        bounds_max: Vec3,
        aspect: f32,
        fov_y: f32,
        padding: f32,
    ) -> Self {
        let center = (bounds_min + bounds_max) * 0.5;
        let radius = ((bounds_max - bounds_min) * 0.5).length().max(1e-4);

        // horizontal fov derived from vertical fov + aspect
        let half_fov_y = fov_y * 0.5;
        let half_fov_x = (half_fov_y.tan() * aspect).atan();

        // fit in both directions
        let limiting_half_fov = half_fov_y.min(half_fov_x);
        let dist = (radius / limiting_half_fov.tan()) * padding;

        // choose a viewing direction
        let view_dir = Vec3::new(0.0, 0.35, 1.0).normalize();
        let eye = center + view_dir * dist;

        // near/far based on distance + radius
        let znear = (dist - radius * 2.0).max(0.01);
        let zfar = dist + radius * 4.0;

        Self {
            eye,
            target: center,
            up: Vec3::Y,
            fov_y,
            znear,
            zfar,
        }
    }
}
