pub struct IblData {
    pub irradiance_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub bindgroup: wgpu::BindGroup,
}

pub fn create_ibl_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ibl_bgl"),
        entries: &[
            // irradiance cubemap
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            // sampler
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Create IBL data with a procedural gradient cubemap for testing.
/// Replace this with a real irradiance map later.
pub fn create_test_ibl(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> IblData {
    let size = 32u32; // small for testing
    let irradiance_view = create_gradient_cubemap(device, queue, size);

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("ibl_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ibl_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&irradiance_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    IblData {
        irradiance_view,
        sampler,
        bindgroup,
    }
}

/// Creates a simple gradient cubemap for testing IBL pipeline.
/// Sky-ish blue on top, ground-ish brown on bottom, neutral sides.
fn create_gradient_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: u32,
) -> wgpu::TextureView {
    let extent = wgpu::Extent3d {
        width: size,
        height: size,
        depth_or_array_layers: 6,
    };

    // Use Rgba16Float for HDR values > 1.0
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("irradiance_cubemap"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Face order: +X, -X, +Y, -Y, +Z, -Z
    // HDR values - will be tonemapped in shader
    let face_colors: [[f32; 3]; 6] = [
        [0.4, 0.38, 0.35], // +X (right) - warm neutral
        [0.35, 0.38, 0.4], // -X (left) - cool neutral
        [0.5, 0.6, 0.8],   // +Y (up/sky) - blue sky
        [0.25, 0.2, 0.15], // -Y (down/ground) - brown ground
        [0.4, 0.4, 0.4],   // +Z (front) - neutral
        [0.38, 0.38, 0.42], // -Z (back) - slightly cool
    ];

    let bytes_per_pixel = 8usize; // 4 x f16 = 8 bytes
    let unpadded_row = size as usize * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let padded_row = unpadded_row.div_ceil(align) * align;

    for (face_idx, color) in face_colors.iter().enumerate() {
        let mut data = vec![0u8; padded_row * size as usize];

        for y in 0..size {
            for x in 0..size {
                let offset = (y as usize * padded_row) + (x as usize * bytes_per_pixel);
                let r = half::f16::from_f32(color[0]);
                let g = half::f16::from_f32(color[1]);
                let b = half::f16::from_f32(color[2]);
                let a = half::f16::from_f32(1.0);

                data[offset..offset + 2].copy_from_slice(&r.to_le_bytes());
                data[offset + 2..offset + 4].copy_from_slice(&g.to_le_bytes());
                data[offset + 4..offset + 6].copy_from_slice(&b.to_le_bytes());
                data[offset + 6..offset + 8].copy_from_slice(&a.to_le_bytes());
            }
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: face_idx as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row as u32),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
    }

    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("irradiance_cubemap_view"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    })
}
