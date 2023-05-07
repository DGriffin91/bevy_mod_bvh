use std::num::NonZeroU64;

use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;

use bevy::pbr::{MAX_CASCADES_PER_LIGHT, MAX_DIRECTIONAL_LIGHTS};
use bevy::prelude::*;

use bevy::render::globals::GlobalsUniform;
use bevy::render::render_resource::{
    BindGroupLayout, BindGroupLayoutEntry, BindingType, BufferBindingType, CachedRenderPipelineId,
    ColorTargetState, ColorWrites, FragmentState, MultisampleState, PipelineCache, PrimitiveState,
    RenderPipelineDescriptor, SamplerBindingType, ShaderDefVal, ShaderStages, ShaderType,
    StorageTextureAccess, TextureFormat, TextureSampleType, TextureViewDimension,
};
use bevy::render::texture::BevyDefault;
use bevy::render::view::ViewUniform;

pub fn sampler_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
        count: None,
    }
}

pub fn image_entry(binding: u32, dim: TextureViewDimension) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: dim,
            multisampled: false,
        },
        count: None,
    }
}

pub fn storage_tex_read(
    binding: u32,
    format: TextureFormat,
    dim: TextureViewDimension,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::ReadOnly,
            format,
            view_dimension: dim,
        },
        count: None,
    }
}

pub fn storage_tex_write(
    binding: u32,
    format: TextureFormat,
    dim: TextureViewDimension,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format,
            view_dimension: dim,
        },
        count: None,
    }
}

pub fn view_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: Some(ViewUniform::min_size()),
        },
        count: None,
    }
}

pub fn globals_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(GlobalsUniform::min_size()),
        },
        count: None,
    }
}

pub fn uniform_entry(binding: u32, min_binding_size: Option<NonZeroU64>) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: bevy::render::render_resource::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size,
        },
        count: None,
    }
}

pub fn get_default_pipeline_desc(
    mut shader_defs: Vec<ShaderDefVal>,
    layout: BindGroupLayout,
    pipeline_cache: &mut PipelineCache,
    shader: Handle<Shader>,
    hdr: bool,
) -> CachedRenderPipelineId {
    shader_defs.push(ShaderDefVal::UInt(
        "MAX_DIRECTIONAL_LIGHTS".to_string(),
        MAX_DIRECTIONAL_LIGHTS as u32,
    ));
    shader_defs.push(ShaderDefVal::UInt(
        "MAX_CASCADES_PER_LIGHT".to_string(),
        MAX_CASCADES_PER_LIGHT as u32,
    ));

    pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("post_process_pipeline".into()),
        layout: vec![layout],
        vertex: fullscreen_shader_vertex_state(),
        fragment: Some(FragmentState {
            shader,
            shader_defs,
            entry_point: "fragment".into(),
            targets: vec![Some(ColorTargetState {
                format: if hdr {
                    TextureFormat::Rgba16Float
                } else {
                    TextureFormat::bevy_default()
                },
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        push_constant_ranges: vec![],
    })
}

#[macro_export]
macro_rules! some_binding_or_return_none {
    ($buffer:expr) => {{
        let Some(r) = $buffer.binding() else {return None};
        r
    }};
}

#[macro_export]
macro_rules! bind_group_layout_entry {
    () => {
        pub fn bind_group_layout_entry(
            binding: u32,
        ) -> bevy::render::render_resource::BindGroupLayoutEntry {
            bevy::render::render_resource::BindGroupLayoutEntry {
                binding,
                visibility: bevy::render::render_resource::ShaderStages::FRAGMENT
                    | bevy::render::render_resource::ShaderStages::COMPUTE,
                ty: bevy::render::render_resource::BindingType::Buffer {
                    ty: bevy::render::render_resource::BufferBindingType::Storage {
                        read_only: true,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: Some(Self::min_size()),
                },
                count: None,
            }
        }
    };
}
