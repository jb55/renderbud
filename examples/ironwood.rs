use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

struct App {
    model: Option<renderbud::Model>,
    renderbud: Option<renderbud::Renderbud>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            renderbud: None,
            model: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        // Create the window *after* the event loop is running (winit 0.30+).
        let window: Window = el
            .create_window(WindowAttributes::default())
            .expect("create_window failed");

        let mut renderbud = pollster::block_on(renderbud::Renderbud::new(window));

        // pick a path relative to crate root
        let model_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/ironwood.glb");
        self.model = Some(renderbud.load_gltf_model(model_path).unwrap());

        self.renderbud = Some(renderbud);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(renderbud) = self.renderbud.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => el.exit(),

            WindowEvent::Resized(sz) => renderbud.resize((sz.width, sz.height)),

            WindowEvent::KeyboardInput { event, .. } => {
                if let KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state: ElementState::Pressed,
                    ..
                } = event
                {
                    match code {
                        KeyCode::Space => {
                            // do something
                        }
                        _ => {}
                    }
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        let Some(renderbud) = self.renderbud.as_mut() else {
            return;
        };

        let Some(model) = self.model else {
            return;
        };

        // Continuous rendering.
        renderbud.update();
        renderbud.prepare();
        match renderbud.render(model) {
            Ok(_) => {}
            Err(wgpu::SurfaceError::Lost) => renderbud.resize(renderbud.size()),
            Err(wgpu::SurfaceError::OutOfMemory) => el.exit(),
            Err(_) => {}
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;

    // Equivalent to your `elwt.set_control_flow(ControlFlow::Poll);`
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}
