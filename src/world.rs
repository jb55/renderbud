use glam::{Mat4, Quat, Vec3};

use crate::camera::Camera;
use crate::model::Model;

/// A unique handle for a placed object in the scene
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ObjectId(pub u64);

/// Transform for a placed object
#[derive(Clone, Debug)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn from_translation(t: Vec3) -> Self {
        Self {
            translation: t,
            ..Default::default()
        }
    }

    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

/// A placed instance of a model in the scene
pub struct SceneObject {
    pub model: Model,
    pub transform: Transform,
}

pub struct World {
    pub camera: Camera,
    objects: Vec<(ObjectId, SceneObject)>,
    next_id: u64,
    pub selected_object: Option<ObjectId>,
}

impl World {
    pub fn new(camera: Camera) -> Self {
        Self {
            camera,
            objects: Vec::new(),
            next_id: 0,
            selected_object: None,
        }
    }

    pub fn add_object(&mut self, model: Model, transform: Transform) -> ObjectId {
        let id = ObjectId(self.next_id);
        self.next_id += 1;
        self.objects.push((id, SceneObject { model, transform }));
        id
    }

    pub fn remove_object(&mut self, id: ObjectId) -> bool {
        let len = self.objects.len();
        self.objects.retain(|(oid, _)| *oid != id);
        self.objects.len() != len
    }

    pub fn update_transform(&mut self, id: ObjectId, transform: Transform) -> bool {
        if let Some((_, obj)) = self.objects.iter_mut().find(|(oid, _)| *oid == id) {
            obj.transform = transform;
            true
        } else {
            false
        }
    }

    pub fn get_object(&self, id: ObjectId) -> Option<&SceneObject> {
        self.objects
            .iter()
            .find(|(oid, _)| *oid == id)
            .map(|(_, o)| o)
    }

    pub fn objects(&self) -> &[(ObjectId, SceneObject)] {
        &self.objects
    }

    pub fn num_objects(&self) -> usize {
        self.objects.len()
    }
}
