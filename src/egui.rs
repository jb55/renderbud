use std::sync::Arc;
use std::sync::Mutex;

use crate::Model;
use crate::Renderer;

#[derive(Clone)]
pub struct EguiRenderer {
    pub renderer: Arc<Mutex<Renderer>>,
}

#[cfg(feature = "egui")]
impl EguiRenderer {
    pub fn new(rs: &egui_wgpu::RenderState, size: (u32, u32)) -> Self {
        let renderer = Renderer::new(&rs.device, &rs.queue, rs.target_format, size);
        let egui_renderer = Self {
            renderer: Arc::new(Mutex::new(renderer)),
        };

        rs.renderer
            .write()
            .callback_resources
            .insert(egui_renderer.clone());

        egui_renderer
    }
}

// TODO(jb55): eventually this should be just a generic renderable handle
// instead of hardcoding it to model
#[cfg(feature = "egui")]
impl egui_wgpu::CallbackTrait for Model {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let egui_renderer: &EguiRenderer = resources.get().unwrap();

        let mut renderer = egui_renderer.renderer.lock().unwrap();

        renderer.resize(device);
        renderer.prepare(queue);

        Vec::with_capacity(0)
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let egui_renderer: &EguiRenderer = resources.get().unwrap();

        egui_renderer
            .renderer
            .lock()
            .unwrap()
            .render_pass(render_pass, *self)
    }
}
