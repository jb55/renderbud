use crate::camera::Camera;
use crate::model::Model;

pub struct World {
    pub camera: Camera,
    pub selected_model: Option<Model>,
}
