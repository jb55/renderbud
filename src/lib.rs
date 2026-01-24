use glam::{Mat4, Vec2, Vec3};

use crate::material::{MaterialUniform, make_material_gpudata};
use crate::model::ModelData;
use crate::model::Vertex;
use std::collections::HashMap;

mod camera;
mod ibl;
mod material;
mod model;
mod texture;
mod world;

#[cfg(feature = "egui")]
pub mod egui;

pub use camera::{ArcballController, Camera};
pub use model::Model;
pub use world::World;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::NoUninit, bytemuck::Zeroable)]
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

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
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

impl Globals {
    fn set_camera(&mut self, w: f32, h: f32, camera: &Camera) {
        self.cam_pos = camera.eye;
        self.view_proj = camera.view_proj(w, h);
    }
}

struct GpuData<R> {
    data: R,
    buffer: wgpu::Buffer,
    bindgroup: wgpu::BindGroup,
}

pub struct Renderbud {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
}

pub struct Renderer {
    size: (u32, u32),

    /// To propery resize we need a device. Provide a target size so
    /// we can dynamically resize next time get one.
    target_size: (u32, u32),

    model_ids: u64,

    depth_tex: wgpu::Texture,
    depth_view: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,

    world: World,
    arcball: camera::ArcballController,

    globals: GpuData<Globals>,
    object: GpuData<ObjectUniform>,
    material: GpuData<MaterialUniform>,

    material_bgl: wgpu::BindGroupLayout,

    ibl: ibl::IblData,

    models: HashMap<Model, ModelData>,

    start: std::time::Instant,
}

fn make_global_gpudata(
    device: &wgpu::Device,
    width: f32,
    height: f32,
    camera: &Camera,
) -> (GpuData<Globals>, wgpu::BindGroupLayout) {
    let globals = Globals {
        time: 0.0,
        _pad0: 0.0,
        resolution: Vec2::new(width, height),
        cam_pos: camera.eye,
        _pad3: 0.0,
        light_dir: Vec3::new(0.9, 0.4, 0.4),
        _pad1: 0.0,
        light_color: Vec3::new(1.0, 0.98, 0.92),
        _pad2: 0.0,
        view_proj: camera.view_proj(width, height),
    };

    println!("Globals size = {}", std::mem::size_of::<Globals>());

    let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("globals"),
        size: std::mem::size_of::<Globals>() as u64,
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
            buffer: globals_buf,
            bindgroup: globals_bg,
        },
        globals_bgl,
    )
}

fn make_object_gpudata(device: &wgpu::Device) -> (GpuData<ObjectUniform>, wgpu::BindGroupLayout) {
    let object_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("object"),
        size: std::mem::size_of::<ObjectUniform>() as u64,
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
            data: ObjectUniform::from_model(Mat4::IDENTITY),
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

impl Renderbud {
    pub async fn new(window: winit::window::Window) -> Self {
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
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
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

        let renderer = Renderer::new(&device, &queue, format, (config.width, config.height));

        Self {
            config,
            surface,
            queue,
            device,
            renderer,
        }
    }

    pub fn update(&mut self) {
        self.renderer.update();
    }

    pub fn prepare(&self) {
        self.renderer.prepare(&self.queue);
    }

    pub fn resize(&mut self, new_size: (u32, u32)) {
        let width = new_size.0.max(1);
        let height = new_size.1.max(1);

        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);

        self.renderer.set_target_size((width, height));
        self.renderer.resize(&self.device)
    }

    pub fn size(&self) -> (u32, u32) {
        self.renderer.size
    }

    /// Handle mouse drag for arcball rotation.
    pub fn on_mouse_drag(&mut self, delta_x: f32, delta_y: f32) {
        self.renderer.on_mouse_drag(delta_x, delta_y);
    }

    /// Handle scroll for arcball zoom.
    pub fn on_scroll(&mut self, delta: f32) {
        self.renderer.on_scroll(delta);
    }

    pub fn load_gltf_model(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Model, gltf::Error> {
        self.renderer
            .load_gltf_model(&self.device, &self.queue, path)
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.renderer.render(&view, &mut encoder);
        self.queue.submit(Some(encoder.finish()));
        frame.present();

        Ok(())
    }
}

impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        size: (u32, u32),
    ) -> Self {
        let (width, height) = size;

        let eye = Vec3::new(0.0, 16.0, 24.0);
        let target = Vec3::new(0.0, 0.0, 0.0);
        let camera = Camera::new(eye, target);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let (globals, globals_bgl) =
            make_global_gpudata(device, width as f32, height as f32, &camera);
        let (object, object_bgl) = make_object_gpudata(device);
        let (material, material_bgl) = make_material_gpudata(device, queue);

        let ibl_bgl = ibl::create_ibl_bind_group_layout(device);
        let ibl = ibl::load_hdr_ibl(device, queue, &ibl_bgl, "assets/venice_sunset_1k.hdr")
            .expect("failed to load HDR environment map");

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&globals_bgl, &object_bgl, &material_bgl, &ibl_bgl],
            push_constant_ranges: &[],
        });

        /*
        let pipeline_cache = unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("pipeline_cache"),
                data: None,
                fallback: true,
            })
        };
        */
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"),
            //cache: Some(&pipeline_cache),
            cache: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
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

        let (depth_tex, depth_view) = create_depth(device, width, height);

        /* TODO: move to example
        let model = load_gltf_model(
            &device,
            &queue,
            &material_bgl,
            "/home/jb55/var/models/ironwood/ironwood.glb",
        )
        .unwrap();
        */

        let model_ids = 0;

        let world = World {
            camera,
            selected_model: None,
        };

        let arcball = camera::ArcballController::from_camera(&world.camera);

        Self {
            world,
            arcball,
            target_size: size,
            model_ids,
            size,
            pipeline,
            globals,
            object,
            material,
            material_bgl,
            ibl,
            models: HashMap::new(),
            depth_tex,
            depth_view,
            start: std::time::Instant::now(),
        }
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    fn globals_mut(&mut self) -> &mut Globals {
        &mut self.globals.data
    }

    pub fn load_gltf_model(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Model, gltf::Error> {
        let model_data = crate::model::load_gltf_model(device, queue, &self.material_bgl, path)?;

        self.model_ids += 1;
        let id = Model { id: self.model_ids };

        self.models.insert(id, model_data);

        // TODO(jb55): we probably want to separate these
        // pick it
        self.world.selected_model = Some(id);

        // point camera at it
        self.focus_model(id);

        Ok(id)
    }

    /// Perform a resize if the target size is not the same as size
    pub fn set_target_size(&mut self, size: (u32, u32)) {
        self.target_size = size;
    }

    pub fn resize(&mut self, device: &wgpu::Device) {
        if self.target_size == self.size {
            return;
        }

        let (width, height) = self.target_size;
        let w = width as f32;
        let h = height as f32;

        self.size = self.target_size;

        self.globals.data.resolution = Vec2::new(w, h);
        self.globals.data.set_camera(w, h, &self.world.camera);

        let (depth_tex, depth_view) = create_depth(device, width, height);
        self.depth_tex = depth_tex;
        self.depth_view = depth_view;
    }

    pub fn focus_model(&mut self, model: Model) {
        let Some(md) = self.models.get(&model) else {
            return;
        };

        let (w, h) = self.size;
        let w = w as f32;
        let h = h as f32;

        let aspect = w / h.max(1.0);

        self.world.camera = Camera::fit_to_aabb(
            md.bounds.min,
            md.bounds.max,
            aspect,
            45_f32.to_radians(),
            1.2,
        );

        // Sync arcball to new camera position
        self.arcball = camera::ArcballController::from_camera(&self.world.camera);

        self.globals.data.set_camera(w, h, &self.world.camera);
    }

    /// Handle mouse drag for arcball rotation.
    pub fn on_mouse_drag(&mut self, delta_x: f32, delta_y: f32) {
        self.arcball.on_drag(delta_x, delta_y);
    }

    /// Handle scroll for arcball zoom.
    pub fn on_scroll(&mut self, delta: f32) {
        self.arcball.on_scroll(delta);
    }

    pub fn update(&mut self) {
        self.globals_mut().time = self.start.elapsed().as_secs_f32();

        // Update camera from arcball controller
        self.arcball.update_camera(&mut self.world.camera);
        let (w, h) = self.size;
        self.globals.data.set_camera(w as f32, h as f32, &self.world.camera);

        //let t = self.globals_mut().time * 0.3;
        //self.globals_mut().light_dir = Vec3::new(t_slow.cos() * 0.6, 0.7, t_slow.sin() * 0.6);

        // Example: slowly rotate the test mesh so you can verify transforms
        let model = Mat4::from_rotation_y(self.globals.data.time * 0.6);
        self.object.data = ObjectUniform::from_model(model);
    }

    pub fn prepare(&self, queue: &wgpu::Queue) {
        write_gpu_data(queue, &self.globals);
        write_gpu_data(queue, &self.object);
        write_gpu_data(queue, &self.material);
    }

    pub fn render(&self, frame: &wgpu::TextureView, encoder: &mut wgpu::CommandEncoder) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("rpass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame,
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

        self.render_pass(&mut rpass);
    }

    pub fn render_pass(&self, rpass: &mut wgpu::RenderPass<'_>) {
        let Some(model) = self.world.selected_model else {
            return;
        };

        let Some(model) = self.models.get(&model) else {
            return;
        };

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.globals.bindgroup, &[]);
        rpass.set_bind_group(1, &self.object.bindgroup, &[]);
        rpass.set_bind_group(3, &self.ibl.bindgroup, &[]);

        for d in &model.draws {
            rpass.set_bind_group(2, &model.materials[d.material_index].bindgroup, &[]);
            rpass.set_vertex_buffer(0, d.mesh.vert_buf.slice(..));
            rpass.set_index_buffer(d.mesh.ind_buf.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..d.mesh.num_indices, 0, 0..1);
        }
    }
}

fn write_gpu_data<R: bytemuck::NoUninit>(queue: &wgpu::Queue, state: &GpuData<R>) {
    //state.staging.clear();
    //let mut storage = encase::UniformBuffer::new(&mut state.staging);
    //storage.write(&state.data).unwrap();
    queue.write_buffer(&state.buffer, 0, bytemuck::bytes_of(&state.data));
}

fn create_depth(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    assert!(width < 8192);
    assert!(height < 8192);
    let size = wgpu::Extent3d {
        width,
        height,
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
