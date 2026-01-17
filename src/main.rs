// Cargo.toml deps (recent-ish):

use encase::ShaderType;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use glam::{Mat4, Vec2, Vec3};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    nrm: [f32; 3],
    uv: [f32; 2],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // normal
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as u64,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // uv
                wgpu::VertexAttribute {
                    offset: (mem::size_of::<[f32; 3]>() + mem::size_of::<[f32; 3]>()) as u64,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

#[derive(Debug, Copy, Clone, ShaderType)]
struct Globals {
    // 0..16
    time: f32,
    _pad0: f32,
    resolution: Vec2, // 8 bytes, finishes first 16-byte slot

    // 16..32
    cam_pos: Vec3, // takes 12, but aligned to 16
    _pad3: f32,    // fills the last 4 bytes of this 16-byte slot nicely

    // 32..48
    light_dir: Vec3,
    _pad1: f32,

    // 48..64
    light_color: Vec3,
    _pad2: f32,

    view_proj: Mat4,
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    depth_tex: wgpu::Texture,
    depth_view: wgpu::TextureView,

    pipeline: wgpu::RenderPipeline,
    globals: Globals,
    globals_buf: wgpu::Buffer,
    globals_bg: wgpu::BindGroup,
    globals_staging: Vec<u8>,

    test_mesh: Mesh,

    start: std::time::Instant,
}

struct Mesh {
    num_indices: u32,
    vert_buf: wgpu::Buffer,
    ind_buf: wgpu::Buffer,
}

impl State {
    async fn new(window: winit::window::Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Prefer an sRGB format for correct output
        let format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let eye = Vec3::new(0.0, 0.8, 2.5);
        let globals = Globals {
            time: 0.0,
            _pad0: 0.0,
            _pad3: 0.0,
            resolution: Vec2::new(config.width as f32, config.height as f32),
            _pad1: 0.0,
            cam_pos: eye,
            _pad2: 0.0,
            light_dir: Vec3::new(0.4, 0.7, 0.2),
            light_color: Vec3::new(1.0, 0.98, 0.92),
            view_proj: Self::calc_view_proj(eye, config.width as f32, config.height as f32),
        };

        println!("Globals size = {}", std::mem::size_of::<Globals>());

        let ubo_size = Globals::min_size().get() as u64;

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: ubo_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals_bg"),
            layout: &globals_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&globals_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let globals_staging = Vec::with_capacity(256);

        let test_mesh = {
            let vertices: Vec<Vertex> = vec![
                Vertex {
                    pos: [-0.8, -0.8, 0.0],
                    nrm: [0.0, 0.0, 1.0],
                    uv: [0.0, 1.0],
                },
                Vertex {
                    pos: [0.8, -0.8, 0.0],
                    nrm: [0.0, 0.0, 1.0],
                    uv: [1.0, 1.0],
                },
                Vertex {
                    pos: [0.8, 0.8, 0.0],
                    nrm: [0.0, 0.0, 1.0],
                    uv: [1.0, 0.0],
                },
                Vertex {
                    pos: [-0.8, 0.8, 0.0],
                    nrm: [0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                },
            ];
            let indices: Vec<u16> = vec![0, 1, 2, 0, 2, 3];
            let num_indices = indices.len() as u32;

            let vert_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vert_buf"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ind_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ind_buf"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            Mesh {
                vert_buf,
                ind_buf,
                num_indices,
            }
        };

        let (depth_tex, depth_view) = create_depth(&device, &config);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            pipeline,
            globals,
            globals_buf,
            globals_bg,
            globals_staging,
            test_mesh,
            depth_tex,
            depth_view,
            start: std::time::Instant::now(),
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        let width = new_size.width.max(1);
        let height = new_size.height.max(1);

        self.size = new_size;
        self.config.width = width;
        self.config.height = height;
        self.globals.resolution = Vec2::new(width as f32, height as f32);
        self.surface.configure(&self.device, &self.config);

        self.globals.view_proj =
            Self::calc_view_proj(self.globals.cam_pos, width as f32, height as f32);

        let (depth_tex, depth_view) = create_depth(&self.device, &self.config);
        self.depth_tex = depth_tex;
        self.depth_view = depth_view;
    }

    fn calc_view_proj(eye: Vec3, width: f32, height: f32) -> Mat4 {
        let target = Vec3::new(0.0, 0.0, 0.0);
        let up = Vec3::Y;

        let view = Mat4::look_at_rh(eye, target, up);
        let proj = Mat4::perspective_rh_gl(45f32.to_radians(), width / height, 0.1, 100.0);
        proj * view
    }

    fn update(&mut self) {
        self.globals.time = self.start.elapsed().as_secs_f32();
        // animate light a bit
        let t = self.globals.time * 0.3;
        self.globals.light_dir = Vec3::new(t.cos() * 0.6, 0.7, t.sin() * 0.6);
        write_globals(
            &self.queue,
            &self.globals_buf,
            &mut self.globals_staging,
            &self.globals,
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rpass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.06,
                            b: 0.07,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.globals_bg, &[]);
            rpass.set_vertex_buffer(0, self.test_mesh.vert_buf.slice(..));
            rpass.set_index_buffer(self.test_mesh.ind_buf.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..self.test_mesh.num_indices, 0, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

fn write_globals(
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    staging: &mut Vec<u8>,
    globals: &Globals,
) {
    staging.clear();
    let mut storage = encase::UniformBuffer::new(staging);
    storage.write(globals).unwrap();
    queue.write_buffer(buf, 0, storage.as_ref());
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("wgpu hello (ACNH-ish / Overwatch-ish)")
        .build(&event_loop)
        .unwrap();

    let mut state = pollster::block_on(State::new(window));

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(sz) => state.resize(sz),
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
                    state.update();
                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                        Err(_) => {}
                    }
                }
                _ => {}
            }
        })
        .unwrap();
}

fn create_depth(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24Plus,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}
