use glam::{Mat4, Quat, Vec3};

use crate::camera::Camera;
use crate::model::Model;

/// A unique handle for a node in the scene graph.
/// Uses arena index + generation to prevent stale handle reuse.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct NodeId {
    pub index: u32,
    pub generation: u32,
}

/// Backward-compatible alias for existing code that uses ObjectId.
pub type ObjectId = NodeId;

/// Transform for a scene node (position, rotation, scale).
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

/// A node in the scene graph.
pub struct Node {
    /// Local transform relative to parent (or world if root).
    pub local: Transform,

    /// Cached world-space matrix. Valid when `dirty == false`.
    world_matrix: Mat4,

    /// When true, world_matrix needs recomputation.
    dirty: bool,

    /// Generation for this slot (matches NodeId.generation when alive).
    generation: u32,

    /// Parent node. None means this is a root node.
    parent: Option<NodeId>,

    /// First child (intrusive linked list through siblings).
    first_child: Option<NodeId>,

    /// Next sibling in parent's child list.
    next_sibling: Option<NodeId>,

    /// If Some, this node is renderable with the given Model handle.
    /// If None, this is a grouping/transform-only node.
    pub model: Option<Model>,

    /// Whether this slot is occupied.
    alive: bool,
}

impl Node {
    /// Get the cached world-space matrix.
    /// Only valid after `update_world_transforms()`.
    pub fn world_matrix(&self) -> Mat4 {
        self.world_matrix
    }
}

pub struct World {
    pub camera: Camera,

    /// Arena of all nodes.
    nodes: Vec<Node>,

    /// Free slot indices for reuse.
    free_list: Vec<u32>,

    /// Cached list of NodeIds that have a Model (renderable).
    /// Rebuilt when renderables_dirty is true.
    renderables: Vec<NodeId>,

    /// True when renderables list needs rebuilding.
    renderables_dirty: bool,

    pub selected_object: Option<NodeId>,
}

impl World {
    pub fn new(camera: Camera) -> Self {
        Self {
            camera,
            nodes: Vec::new(),
            free_list: Vec::new(),
            renderables: Vec::new(),
            renderables_dirty: false,
            selected_object: None,
        }
    }

    // ── Arena internals ──────────────────────────────────────────

    fn alloc_slot(&mut self) -> (u32, u32) {
        if let Some(index) = self.free_list.pop() {
            let node = &mut self.nodes[index as usize];
            node.generation += 1;
            node.alive = true;
            node.dirty = true;
            node.parent = None;
            node.first_child = None;
            node.next_sibling = None;
            node.model = None;
            node.world_matrix = Mat4::IDENTITY;
            (index, node.generation)
        } else {
            let index = self.nodes.len() as u32;
            self.nodes.push(Node {
                local: Transform::default(),
                world_matrix: Mat4::IDENTITY,
                dirty: true,
                generation: 0,
                parent: None,
                first_child: None,
                next_sibling: None,
                model: None,
                alive: true,
            });
            (index, 0)
        }
    }

    fn is_valid(&self, id: NodeId) -> bool {
        let idx = id.index as usize;
        idx < self.nodes.len()
            && self.nodes[idx].alive
            && self.nodes[idx].generation == id.generation
    }

    fn mark_dirty(&mut self, id: NodeId) {
        let mut stack = vec![id];
        while let Some(nid) = stack.pop() {
            let node = &mut self.nodes[nid.index as usize];
            if node.dirty {
                continue;
            }
            node.dirty = true;
            let mut child = node.first_child;
            while let Some(c) = child {
                stack.push(c);
                child = self.nodes[c.index as usize].next_sibling;
            }
        }
    }

    fn attach_child(&mut self, parent: NodeId, child: NodeId) {
        let old_first = self.nodes[parent.index as usize].first_child;
        self.nodes[child.index as usize].next_sibling = old_first;
        self.nodes[parent.index as usize].first_child = Some(child);
    }

    fn detach_child(&mut self, parent: NodeId, child: NodeId) {
        let first = self.nodes[parent.index as usize].first_child;
        if first == Some(child) {
            self.nodes[parent.index as usize].first_child =
                self.nodes[child.index as usize].next_sibling;
        } else {
            let mut prev = first;
            while let Some(p) = prev {
                let next = self.nodes[p.index as usize].next_sibling;
                if next == Some(child) {
                    self.nodes[p.index as usize].next_sibling =
                        self.nodes[child.index as usize].next_sibling;
                    break;
                }
                prev = next;
            }
        }
        self.nodes[child.index as usize].next_sibling = None;
    }

    fn is_ancestor(&self, ancestor: NodeId, node: NodeId) -> bool {
        let mut cur = Some(node);
        while let Some(c) = cur {
            if c == ancestor {
                return true;
            }
            cur = self.nodes[c.index as usize].parent;
        }
        false
    }

    // ── Public scene graph API ───────────────────────────────────

    /// Create a grouping node (no model) with an optional parent.
    pub fn create_node(&mut self, local: Transform, parent: Option<NodeId>) -> NodeId {
        let (index, generation) = self.alloc_slot();
        self.nodes[index as usize].local = local;

        let id = NodeId { index, generation };

        if let Some(p) = parent {
            if self.is_valid(p) {
                self.nodes[index as usize].parent = Some(p);
                self.attach_child(p, id);
            }
        }

        id
    }

    /// Create a renderable node with a Model and optional parent.
    pub fn create_renderable(
        &mut self,
        model: Model,
        local: Transform,
        parent: Option<NodeId>,
    ) -> NodeId {
        let id = self.create_node(local, parent);
        self.nodes[id.index as usize].model = Some(model);
        self.renderables_dirty = true;
        id
    }

    /// Remove a node and all its descendants.
    pub fn remove_node(&mut self, id: NodeId) -> bool {
        if !self.is_valid(id) {
            return false;
        }

        // Collect all nodes in the subtree
        let mut to_remove = Vec::new();
        let mut stack = vec![id];
        while let Some(nid) = stack.pop() {
            to_remove.push(nid);
            let mut child = self.nodes[nid.index as usize].first_child;
            while let Some(c) = child {
                stack.push(c);
                child = self.nodes[c.index as usize].next_sibling;
            }
        }

        // Detach root of subtree from its parent
        if let Some(parent_id) = self.nodes[id.index as usize].parent {
            self.detach_child(parent_id, id);
        }

        // Free all collected nodes
        for nid in &to_remove {
            let node = &mut self.nodes[nid.index as usize];
            node.alive = false;
            node.first_child = None;
            node.next_sibling = None;
            node.parent = None;
            node.model = None;
            self.free_list.push(nid.index);
        }

        self.renderables_dirty = true;
        true
    }

    /// Set a node's local transform. Marks it and descendants dirty.
    pub fn set_local_transform(&mut self, id: NodeId, local: Transform) -> bool {
        if !self.is_valid(id) {
            return false;
        }
        self.nodes[id.index as usize].local = local;
        self.mark_dirty(id);
        true
    }

    /// Reparent a node. Pass None to make it a root node.
    pub fn set_parent(&mut self, id: NodeId, new_parent: Option<NodeId>) -> bool {
        if !self.is_valid(id) {
            return false;
        }
        if let Some(p) = new_parent {
            if !self.is_valid(p) {
                return false;
            }
            if self.is_ancestor(id, p) {
                return false;
            }
        }

        // Detach from old parent
        if let Some(old_parent) = self.nodes[id.index as usize].parent {
            self.detach_child(old_parent, id);
        }

        // Attach to new parent
        self.nodes[id.index as usize].parent = new_parent;
        if let Some(p) = new_parent {
            self.attach_child(p, id);
        }

        self.mark_dirty(id);
        true
    }

    /// Attach or detach a Model on an existing node.
    pub fn set_model(&mut self, id: NodeId, model: Option<Model>) -> bool {
        if !self.is_valid(id) {
            return false;
        }
        self.nodes[id.index as usize].model = model;
        self.renderables_dirty = true;
        true
    }

    /// Get the cached world matrix for a node.
    pub fn world_matrix(&self, id: NodeId) -> Option<Mat4> {
        if !self.is_valid(id) {
            return None;
        }
        Some(self.nodes[id.index as usize].world_matrix)
    }

    /// Get a node's local transform.
    pub fn local_transform(&self, id: NodeId) -> Option<&Transform> {
        if !self.is_valid(id) {
            return None;
        }
        Some(&self.nodes[id.index as usize].local)
    }

    /// Get a node by id.
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        if !self.is_valid(id) {
            return None;
        }
        Some(&self.nodes[id.index as usize])
    }

    /// Iterate renderable node ids (nodes with a Model).
    pub fn renderables(&self) -> &[NodeId] {
        &self.renderables
    }

    /// Recompute world matrices for all dirty nodes. Call once per frame.
    pub fn update_world_transforms(&mut self) {
        // Rebuild renderables list if needed
        if self.renderables_dirty {
            self.renderables.clear();
            for (i, node) in self.nodes.iter().enumerate() {
                if node.alive && node.model.is_some() {
                    self.renderables.push(NodeId {
                        index: i as u32,
                        generation: node.generation,
                    });
                }
            }
            self.renderables_dirty = false;
        }

        // Process root nodes (no parent) and recurse into children
        for i in 0..self.nodes.len() {
            let node = &self.nodes[i];
            if !node.alive || !node.dirty || node.parent.is_some() {
                continue;
            }
            self.nodes[i].world_matrix = self.nodes[i].local.to_matrix();
            self.nodes[i].dirty = false;
            self.update_children(i);
        }

        // Second pass: catch any remaining dirty nodes (reparented mid-frame)
        for i in 0..self.nodes.len() {
            if self.nodes[i].alive && self.nodes[i].dirty {
                self.recompute_world_matrix(i);
            }
        }
    }

    fn update_children(&mut self, parent_idx: usize) {
        let parent_world = self.nodes[parent_idx].world_matrix;
        let mut child_id = self.nodes[parent_idx].first_child;
        while let Some(cid) = child_id {
            let ci = cid.index as usize;
            if self.nodes[ci].alive {
                let local = self.nodes[ci].local.to_matrix();
                self.nodes[ci].world_matrix = parent_world * local;
                self.nodes[ci].dirty = false;
                self.update_children(ci);
            }
            child_id = self.nodes[ci].next_sibling;
        }
    }

    fn recompute_world_matrix(&mut self, index: usize) {
        // Build chain from this node up to root
        let mut chain = Vec::with_capacity(8);
        let mut cur = index;
        loop {
            chain.push(cur);
            match self.nodes[cur].parent {
                Some(p) if self.nodes[p.index as usize].alive => {
                    cur = p.index as usize;
                }
                _ => break,
            }
        }

        // Walk from root down to target
        chain.reverse();
        let mut parent_world = Mat4::IDENTITY;
        for &idx in &chain {
            let node = &self.nodes[idx];
            if !node.dirty {
                parent_world = node.world_matrix;
                continue;
            }
            let world = parent_world * node.local.to_matrix();
            self.nodes[idx].world_matrix = world;
            self.nodes[idx].dirty = false;
            parent_world = world;
        }
    }

    // ── Backward-compatible API ──────────────────────────────────

    /// Legacy: place a renderable object as a root node.
    pub fn add_object(&mut self, model: Model, transform: Transform) -> ObjectId {
        self.create_renderable(model, transform, None)
    }

    /// Legacy: remove an object.
    pub fn remove_object(&mut self, id: ObjectId) -> bool {
        self.remove_node(id)
    }

    /// Legacy: update an object's transform.
    pub fn update_transform(&mut self, id: ObjectId, transform: Transform) -> bool {
        self.set_local_transform(id, transform)
    }

    /// Legacy: get a node by object id.
    pub fn get_object(&self, id: ObjectId) -> Option<&Node> {
        self.get_node(id)
    }

    /// Number of renderable objects in the scene.
    pub fn num_objects(&self) -> usize {
        self.renderables.len()
    }
}
