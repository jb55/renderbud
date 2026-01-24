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

/// Arcball camera controller for orbital navigation around a target point.
#[derive(Debug, Clone)]
pub struct ArcballController {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,   // radians, around Y axis
    pub pitch: f32, // radians, up/down
    pub sensitivity: f32,
    pub zoom_sensitivity: f32,
    pub min_distance: f32,
    pub max_distance: f32,
}

impl Default for ArcballController {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 5.0,
            yaw: 0.0,
            pitch: 0.3,
            sensitivity: 0.005,
            zoom_sensitivity: 0.1,
            min_distance: 0.1,
            max_distance: 1000.0,
        }
    }
}

impl ArcballController {
    /// Initialize from an existing camera.
    pub fn from_camera(camera: &Camera) -> Self {
        let offset = camera.eye - camera.target;
        let distance = offset.length();

        // Compute yaw (rotation around Y) and pitch (elevation)
        let yaw = offset.x.atan2(offset.z);
        let pitch = (offset.y / distance).asin();

        Self {
            target: camera.target,
            distance,
            yaw,
            pitch,
            ..Default::default()
        }
    }

    /// Handle mouse drag delta (in pixels).
    pub fn on_drag(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw -= delta_x * self.sensitivity;
        self.pitch -= delta_y * self.sensitivity;

        // Clamp pitch to avoid gimbal lock
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-limit, limit);
    }

    /// Handle scroll for zoom (positive = zoom in).
    pub fn on_scroll(&mut self, delta: f32) {
        self.distance *= 1.0 - delta * self.zoom_sensitivity;
        self.distance = self.distance.clamp(self.min_distance, self.max_distance);
    }

    /// Compute the camera eye position from current orbit state.
    pub fn eye(&self) -> Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.target + Vec3::new(x, y, z)
    }

    /// Update a camera with the current arcball state.
    pub fn update_camera(&self, camera: &mut Camera) {
        camera.eye = self.eye();
        camera.target = self.target;
    }
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
