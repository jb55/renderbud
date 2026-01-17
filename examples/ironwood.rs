// Cargo.toml deps (recent-ish):

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut renderbud = pollster::block_on(renderbud::Renderbud::new(window));

    // pick a path relative to crate root

    let model_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/ironwood.glb");
    let model = renderbud.load_gltf_model(model_path).unwrap();
    renderbud.set_model(model);

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(sz) => renderbud.resize(sz),
                    WindowEvent::KeyboardInput { event, .. } => {
                        if let KeyEvent {
                            physical_key: winit::keyboard::PhysicalKey::Code(code),
                            state: ElementState::Pressed,
                            ..
                        } = event
                        {
                            match code {
                                winit::keyboard::KeyCode::Space => {}
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                },
                Event::AboutToWait => {
                    renderbud.update();
                    match renderbud.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => renderbud.resize(renderbud.size()),
                        Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                        Err(_) => {}
                    }
                }
                _ => {}
            }
        })
        .unwrap();
}
