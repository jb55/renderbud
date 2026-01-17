// Cargo.toml deps (recent-ish):

use encase::{ShaderType, internal::WriteInto};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use glam::{Mat4, Vec2, Vec3};

use crate::material::{MaterialUniform, make_material_gpudata};
use crate::model::Model;
use crate::model::Vertex;
use crate::model::load_gltf_model;

mod material;
mod model;
mod texture;

#[derive(Debug, Copy, Clone, ShaderType)]
struct ObjectUniform {
    model: Mat4,
    normal: Mat4, // inverse-transpose(model)
}

impl ObjectUniform {
    fn from_model(model: Mat4) -> Self {
        Self {
            model,
            normal: model.inverse().transpose(),
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

struct GpuData<R> {
    data: R,
    buffer: wgpu::Buffer,
    bindgroup: wgpu::BindGroup,
    staging: Vec<u8>,
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

    globals: GpuData<Globals>,
    object: GpuData<ObjectUniform>,
    material: GpuData<MaterialUniform>,

    model: Model,

    start: std::time::Instant,
}

fn make_global_gpudata(
    device: &wgpu::Device,
    width: f32,
    height: f32,
) -> (GpuData<Globals>, wgpu::BindGroupLayout) {
    let eye = Vec3::new(0.0, 16.0, 24.0);
    let view_proj = calc_view_proj(eye, width, height);
    let globals = Globals {
        time: 0.0,
        _pad0: 0.0,
        resolution: Vec2::new(width, height),
        cam_pos: eye,
        _pad3: 0.0,
        light_dir: Vec3::new(0.8, 0.8, 0.8),
        _pad1: 0.0,
        light_color: Vec3::new(1.0, 0.98, 0.92),
        _pad2: 0.0,
        view_proj,
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

    (
        GpuData::<Globals> {
            data: globals,
            staging: Vec::with_capacity(256),
            buffer: globals_buf,
            bindgroup: globals_bg,
        },
        globals_bgl,
    )
}

fn make_object_gpudata(device: &wgpu::Device) -> (GpuData<ObjectUniform>, wgpu::BindGroupLayout) {
    let object_uniform = ObjectUniform::from_model(Mat4::IDENTITY);
    let object_ubo_size = ObjectUniform::min_size().get() as u64;

    let object_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("object"),
        size: object_ubo_size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let object_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("object_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let object_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("object_bg"),
        layout: &object_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: object_buf.as_entire_binding(),
        }],
    });

    (
        GpuData::<ObjectUniform> {
            data: object_uniform,
            staging: Vec::with_capacity(128),
            buffer: object_buf,
            bindgroup: object_bg,
        },
        object_bgl,
    )
}

async fn _check_adapters(instance: &wgpu::Instance) {
    // What adapters exist at all?
    for (i, a) in instance
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .enumerate()
    {
        let info = a.get_info();
        eprintln!(
            "Adapter #{i}: {:?} {} {} (backend: {:?})",
            info.device_type, info.vendor, info.name, info.backend
        );
    }

    // Ask without a surface constraint (diagnostic)
    let any_adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: None,
            ..Default::default()
        })
        .await;

    eprintln!("adapter without surface? {}", any_adapter.is_some());
}

impl State {
    async fn new(window: winit::window::Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).unwrap();

        //check_adapters(&instance).await;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                //power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let (globals, globals_bgl) =
            make_global_gpudata(&device, config.width as f32, config.height as f32);
        let (object, object_bgl) = make_object_gpudata(&device);
        let (material, material_bgl) = make_material_gpudata(&device, &queue);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&globals_bgl, &object_bgl, &material_bgl],
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

        /*
        let test_mesh = {
            let vertices: Vec<Vertex> = vec![
                Vertex {
                    pos: [-0.8, -0.8, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0, 1.0],
                },
                Vertex {
                    pos: [0.8, -0.8, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [1.0, 1.0],
                },
                Vertex {
                    pos: [0.8, 0.8, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [1.0, 0.0],
                },
                Vertex {
                    pos: [-0.8, 0.8, 0.0],
                    normal: [0.0, 0.0, 1.0],
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
        */

        let (depth_tex, depth_view) = create_depth(&device, &config);

        let model = load_gltf_model(
            &device,
            &queue,
            &material_bgl,
            "/home/jb55/var/models/ironwood/ironwood.glb",
        )
        .unwrap();

        Self {
            surface,
            device,
            queue,
            config,
            size,
            pipeline,
            globals,
            object,
            material,
            model,
            depth_tex,
            depth_view,
            start: std::time::Instant::now(),
        }
    }

    fn globals_mut(&mut self) -> &mut Globals {
        &mut self.globals.data
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        let width = new_size.width.max(1);
        let height = new_size.height.max(1);

        self.size = new_size;
        self.config.width = width;
        self.config.height = height;
        self.globals_mut().resolution = Vec2::new(width as f32, height as f32);
        self.surface.configure(&self.device, &self.config);

        self.globals_mut().view_proj =
            calc_view_proj(self.globals_mut().cam_pos, width as f32, height as f32);

        let (depth_tex, depth_view) = create_depth(&self.device, &self.config);
        self.depth_tex = depth_tex;
        self.depth_view = depth_view;
    }

    fn update(&mut self) {
        self.globals_mut().time = self.start.elapsed().as_secs_f32();

        let t = self.globals_mut().time * 0.1;
        //self.globals_mut().light_dir = Vec3::new(t_slow.cos() * 0.6, 0.7, t_slow.sin() * 0.6);

        // Example: slowly rotate the test mesh so you can verify transforms
        let model = Mat4::from_rotation_y(self.globals.data.time * 0.6);
        self.object.data = ObjectUniform::from_model(model);

        write_gpu_data(&self.queue, &mut self.globals);
        write_gpu_data(&self.queue, &mut self.object);
        write_gpu_data(&self.queue, &mut self.material);
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
                            r: 0.00,
                            g: 0.00,
                            b: 0.00,
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
            rpass.set_bind_group(0, &self.globals.bindgroup, &[]);
            rpass.set_bind_group(1, &self.object.bindgroup, &[]); // <-- add this

            for d in &self.model.draws {
                rpass.set_bind_group(2, &self.model.materials[d.material_index].bindgroup, &[]);
                rpass.set_vertex_buffer(0, d.mesh.vert_buf.slice(..));
                rpass.set_index_buffer(d.mesh.ind_buf.slice(..), wgpu::IndexFormat::Uint16);
                rpass.draw_indexed(0..d.mesh.num_indices, 0, 0..1);
            }

            //rpass.set_vertex_buffer(0, self.test_mesh.vert_buf.slice(..));
            //rpass.set_index_buffer(self.test_mesh.ind_buf.slice(..), wgpu::IndexFormat::Uint16);
            //rpass.draw_indexed(0..self.test_mesh.num_indices, 0, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

fn write_gpu_data<R: ShaderType + WriteInto>(queue: &wgpu::Queue, state: &mut GpuData<R>) {
    state.staging.clear();
    let mut storage = encase::UniformBuffer::new(&mut state.staging);
    storage.write(&state.data).unwrap();
    queue.write_buffer(&state.buffer, 0, storage.as_ref());
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

pub fn calc_view_proj(eye: Vec3, width: f32, height: f32) -> Mat4 {
    let target = Vec3::new(0.0, 0.0, 0.0);
    let up = Vec3::Y;

    let view = Mat4::look_at_rh(eye, target, up);
    let proj = Mat4::perspective_rh(45f32.to_radians(), width / height, 0.1, 100.0);
    proj * view
}
