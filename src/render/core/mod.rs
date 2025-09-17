//! Render component that performs actual low-level rendering (manages meshes, materials, instances, etc.)

use ash::vk;

// STD imports
use std::{
    cell::{Cell, RefCell}, collections::HashSet, ffi::{CStr, CString}, hash::Hash, sync::Arc
};

// ANIM imports
use crate::{
    math::{self, FMat, FVec},
    render::core::{
        device_context::DeviceContext,
        memory::{Allocator, FlushContext},
        swapchain::{Swapchain, SwapchainHandle},
        util::DropGuard
    }
};

mod device_context;
mod memory;
mod swapchain;
mod util;

/// Standard vertex format
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Vertex {
    /// Vertex position
    pub position: math::Mat<f32, 3>,

    /// Vertex texture coordinate
    pub tex_coord: math::Mat<f32, 2>,

    /// Octmapped vertex normal (r16g16_snorm)
    pub normal: u32,

    /// Octmapped vertex tangent (r16g16_snorm)
    pub tangent: u32,

    /// Misc data (bitangent sign, ...)
    pub misc: u32,
}

impl Vertex {
    /// Pack normal vector
    pub fn pack_direction_octmap(dx: f32, dy: f32, dz: f32) -> u32 {
        // Calculate normalized (by manhattan distance) direction vector
        let inv_length = (dx.abs() + dy.abs() + dz.abs()).recip();
        let (mut x, mut y) = (dx * inv_length, dy * inv_length);

        if dz.is_sign_negative() {
            (x, y) = (
                (1.0 - y.abs()) * x.signum(),
                (1.0 - x.abs()) * y.signum()
            );
        }

        let (x, y) = (
            unsafe { (x * 32767.0).to_int_unchecked::<i16>() }.cast_unsigned() as u32,
            unsafe { (y * 32767.0).to_int_unchecked::<i16>() }.cast_unsigned() as u32,
        );

        y << 16 | x
    }

    /// Unpack normal vector
    pub fn unpack_direction_octmap(packed: u32) -> (f32, f32, f32) {
        let (x, y) = (
            ((packed & 0xFFFF) as u16).cast_signed(),
            ((packed >> 16) as u16).cast_signed()
        );

        let (x, y) = ((x as f32) / 32767.0, (y as f32) / 32767.0);
        let z = 1.0 - x.abs() - y.abs();
        let t = (-z).clamp(0.0, 1.0);

        (x - x.copysign(t), y - y.copysign(t), z)
    }
}

/// Material descriptor
pub struct Material {
    /// RGB color
    pub base_color: [f32; 3],

    /// Model metalness
    pub metallic: f32,

    /// Model roughness
    pub roughness: f32,
}

/// Mesh - structure that holds some vertex data
pub struct Mesh {
    /// Allocator reference
    allocator: Arc<Allocator>,

    /// Device context holder
    _dc: Arc<DeviceContext>,

    /// Mesh memory allocation
    buffer_allocation: vk_mem::Allocation,

    /// Mesh bufffer
    buffer: vk::Buffer,

    /// Vertex buffer memory region
    vertex_span: std::ops::Range<usize>,

    /// Index buffer memory region
    index_span: std::ops::Range<usize>,

    /// Count of the mesh indices
    index_count: usize,
}

impl Drop for Mesh {
    fn drop(&mut self) {
        unsafe { self.allocator.destroy_buffer(self.buffer, &mut self.buffer_allocation) };
    }
}

/// Arc structure that provides basic functions based on pointer on T (instead of T's value)
#[derive(Clone)]
pub struct BlindArc<T>(pub Arc<T>);

impl<T> PartialEq for BlindArc<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<T> Eq for BlindArc<T> {}

impl<T> Hash for BlindArc<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(self.0.as_ref() as *const _, state)
    }
}

// /// Weak arc that provides basic functions on T
// pub struct BlindWeak<T>(pub std::sync::Weak<T>);

// impl<T> PartialEq for BlindWeak<T> {
//     fn eq(&self, other: &Self) -> bool {
//         std::sync::Weak::ptr_eq(&self.0, &other.0)
//     }
// }

// impl<T> Eq for BlindWeak<T> {}

// impl<T> Hash for BlindWeak<T> {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         self.0.as_ptr().hash(state)
//     }
// }

/// Structure that manages set of rendered objects
pub struct RenderSet {
    /// Render set internals
    data: RefCell<HashSet<BlindArc<Instance>>>,
}

impl RenderSet {
    /// Create new render set
    pub fn new() -> Self {
        Self {
            data: RefCell::new(HashSet::new()),
        }
    }

    /// Manually insert item to the render set
    pub fn insert(&self, instance: Arc<Instance>) {
        self.data.borrow_mut().insert(BlindArc(instance));
    }

    /// Manually remove item from render set
    pub fn remove(&self, instance: Arc<Instance>) {
        self.data.borrow_mut().remove(&BlindArc(instance));
    }

    /// Take render set snapshot
    pub fn snapshot(&self) -> Vec<Arc<Instance>> {
        self.data.borrow()
            .iter()
            .map(|bweak| bweak.0.clone())
            .collect()
    }
}

/// Rendered instance representation structure
pub struct Instance {
    /// Underlying mesh
    mesh: Arc<Mesh>,

    /// Instance transformation matrix
    transform: Cell<math::FMat>,

    /// Render set (weak) reference
    render_set: std::sync::Weak<RenderSet>,

    /// Material structure
    _material: Material,
}

impl Instance {
    /// Set transform matrix of the mesh instance
    pub fn set_transform(&self, transform: math::FMat) {
        self.transform.set(transform);
    }

    /// Get transform matrix of the mesh instance
    pub fn get_transform(&self) -> math::FMat {
        self.transform.get()
    }

    /// Enable instance
    pub fn enable(self: &Arc<Self>) {
        // Try to get access set
        if let Some(set) = self.render_set.upgrade() {
            set.insert(self.clone());
        }
    }

    /// Disable instance
    pub fn disable(self: &Arc<Self>) {
        // Try to get access set
        if let Some(set) = self.render_set.upgrade() {
            set.remove(self.clone());
        }
    }
}

/// Vulkan-compatible surface
pub trait WindowContext {
    /// Enumerate required instance extensions
    fn get_instance_extensions(&self) -> Result<Vec<CString>, String>;

    /// Create surface from VkInstance value
    fn create_surface(&self, instance: usize) -> Result<usize, String>;
}

/// Render core initialization error
#[derive(Debug)]
pub enum CoreInitError {
    /// Ash entry loading error
    EntryLoadingError(ash::LoadingError),

    /// Vulkan-level error
    VulkanError(vk::Result),

    /// Invalid CStr bytes got
    InvalidCStr(std::ffi::FromBytesUntilNulError),

    /// Required instance layer is missing
    InstanceLayerNotPresent(CString),

    /// Instance extension is not present
    InstanceExtensionNotPresent(CString),

    /// Device extension is missing
    DeviceExtensionNotPresent(CString),

    /// No suitable physical device
    SuitablePhysicalDeviceMissing,

    /// Suitable surface format is missing
    SuitableSurfaceFormatMissing,

    /// Suitable presentation mode is missing
    SuitablePresentModeMissing,

    /// Error happened somewhere in the window context implementation
    WindowContextError(String),

    /// Some undefined error
    Unknown(String),
}

impl From<vk::Result> for CoreInitError {
    fn from(value: vk::Result) -> Self {
        Self::VulkanError(value)
    }
}

/// Framebuffer image
#[derive(Default)]
struct FramebufferImage {
    /// Image allocation (may be null)
    allocation: Option<vk_mem::Allocation>,

    /// Image itself
    image: vk::Image,

    /// Image view
    view: vk::ImageView,
}

/// Framebuffer
struct Framebuffer {
    /// Guard of the swapchain structure
    swapchain_handle: Arc<SwapchainHandle>,

    /// Image
    swapchain_image: FramebufferImage,

    /// Depth buffer
    depth_image: FramebufferImage,

    /// Framebuffer itself
    framebuffer: vk::Framebuffer,
}

/// In-flight frame representation
#[derive(Default)]
struct FrameContext {
    /// Frame acquision semaphore (for frame output start)
    frame_acquired_semaphore: Cell<vk::Semaphore>,

    /// Semaphore that indicates rendering process end (for presentation)
    render_finished_semaphore: vk::Semaphore,

    /// Transfer operation finish semaphore
    transfer_finished_semaphore: vk::Semaphore,

    /// Fence to wait then frame is reused
    fence: vk::Fence,

    /// Command buffer used for frame contents
    command_buffer: vk::CommandBuffer,

    /// New framebuffer structure
    framebuffer: Option<Framebuffer>,

    /// Context of the flush operations
    flush_context: Option<FlushContext>,

    /// Current frame render set
    render_set: Vec<Arc<Instance>>,
}

/// Core object of the renderer
pub struct Core {
    /// Context of the device
    dc: Arc<DeviceContext>,

    /// Memory allocator handle
    allocator: Arc<Allocator>,

    /// Swapchain
    swapchain: Swapchain,

    /// Command pool for per-frame allocation command buffer
    frame_command_pool: vk::CommandPool,

    /// Unsignaled semaphore to use as a new frame semaphore
    buffered_semaphore: Cell<vk::Semaphore>,

    /// Main render pass
    render_pass: vk::RenderPass,

    /// Temp pipeline layout storage
    pipeline_layout: vk::PipelineLayout,

    /// Temp pipeline storage
    pipeline: vk::Pipeline,

    /// Set of frames
    frames: Vec<FrameContext>,

    /// Set of rendered objects
    render_set: Arc<RenderSet>,

    /// View * Projection matrix
    matrix_view_projection: FMat,
}

impl Core {
    /// Create main render pass
    pub fn create_render_pass(dc: &DeviceContext, swapchain: &Swapchain) -> Result<vk::RenderPass, vk::Result> {
        let target_attachment = vk::AttachmentDescription::default()
            // pub flags: AttachmentDescriptionFlags,
            .format(swapchain.image_format())
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let depth_attachment = vk::AttachmentDescription::default()
            // pub flags: AttachmentDescriptionFlags,
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let attachments = [target_attachment, depth_attachment];

        let target_att_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_att_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::array::from_ref(&target_att_ref))
            .depth_stencil_attachment(&depth_att_ref)
        ;

        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::array::from_ref(&subpass));

        unsafe { dc.device.create_render_pass(&create_info, None) }
    }

    /// Create (the) pipeline
    unsafe fn create_pipeline(
        dc: &DeviceContext,
        render_pass: vk::RenderPass
    ) -> Result<(vk::PipelineLayout, vk::Pipeline), vk::Result> {
        let layout = {
            let push_constant_range = vk::PushConstantRange::default()
                .offset(0)
                .size(std::mem::size_of::<FMat>() as u32)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                ;

            let create_info = vk::PipelineLayoutCreateInfo::default()
                .push_constant_ranges(std::array::from_ref(&push_constant_range))
                ;

            DropGuard::new(
                unsafe { dc.device.create_pipeline_layout(&create_info, None) }?,
                |l| unsafe { dc.device.destroy_pipeline_layout(*l, None) }
            )
        };

        let shader_modules = {
            let compile = |main_fn_name: &str, shader_model: &str| {
                let spirv_bytes = hassle_rs::compile_hlsl(
                    "static/model.hlsl",
                    include_str!("static/model.hlsl"),
                    main_fn_name,
                    shader_model,
                    &["-spirv"],
                    &[]
                ).unwrap();

                // Repack SPIR-V as bytes into [u32]
                let spirv = spirv_bytes
                    .chunks(4)
                    .map(|v| {
                        let mut bytes = [0u8; 4];
                        bytes.copy_from_slice(v);
                        u32::from_ne_bytes(bytes)
                    })
                    .collect::<Vec<u32>>();

                let module_create_info = vk::ShaderModuleCreateInfo::default().code(&spirv);

                let module = unsafe { dc.device.create_shader_module(&module_create_info, None) }?;

                Ok(module)
            };

            let drop = |m: &mut vk::ShaderModule| unsafe { dc.device.destroy_shader_module(*m, None) };

            DropGuard::zip(
                DropGuard::new(compile("vs_main", "vs_5_1")?, drop),
                DropGuard::new(compile("fs_main", "ps_5_1")?, drop)
            )
        };

        let stage_create_infos = {
            let stage = |sm, sn, st| vk::PipelineShaderStageCreateInfo::default()
                .module(sm).stage(st).name(sn);

            [
                stage(shader_modules.0, c"vs_main", vk::ShaderStageFlags::VERTEX),
                stage(shader_modules.1, c"fs_main", vk::ShaderStageFlags::FRAGMENT)
            ]
        };

        let vertex_input_binding_desc = vk::VertexInputBindingDescription::default()
            .binding(0)
            .input_rate(vk::VertexInputRate::VERTEX)
            .stride(std::mem::size_of::<Vertex>() as u32)
            ;

        let vertex_input_attr_descs = {
            // Attribute generator function
            let attr = |bind, loc, fmt, off| vk::VertexInputAttributeDescription::default()
                .binding(bind).format(fmt).location(loc).offset(off as u32);

            [
                attr(0, 0, vk::Format::R32G32B32_SFLOAT, std::mem::offset_of!(Vertex, position)),
                attr(0, 1, vk::Format::R32G32_SFLOAT,    std::mem::offset_of!(Vertex, tex_coord)),
                attr(0, 2, vk::Format::R16G16_SNORM,     std::mem::offset_of!(Vertex, normal)),
                attr(0, 3, vk::Format::R16G16_SNORM,     std::mem::offset_of!(Vertex, tangent)),
                attr(0, 4, vk::Format::R8G8B8A8_SSCALED, std::mem::offset_of!(Vertex, misc)),
            ]
        };

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::array::from_ref(&vertex_input_binding_desc))
            .vertex_attribute_descriptions(&vertex_input_attr_descs);

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .primitive_restart_enable(false)
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1)
            ;

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .line_width(1.0)
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            // pub flags: PipelineDepthStencilStateCreateFlags,
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::GREATER)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            // pub front: StencilOpState,
            // pub back: StencilOpState,
            // pub min_depth_bounds: f32,
            // pub max_depth_bounds: f32,
            ;

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            ;

        let target_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO);

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::array::from_ref(&target_attachment));

        let dynamic_states = [
            vk::DynamicState::SCISSOR,
            vk::DynamicState::VIEWPORT,
        ];

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states)
            ;

        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stage_create_infos)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            // pub p_tessellation_state: *const PipelineTessellationStateCreateInfo<'a>,
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(*layout)
            .render_pass(render_pass)
            .subpass(0)
            ;

        let pipeline = unsafe {
            dc.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::array::from_ref(&create_info),
                None
            ).map_err(|(_, err)| err)?[0]
        };

        Ok((layout.into_inner(), pipeline))
    }

    /// Construct instance
    pub fn new(
        window_context: Arc<dyn WindowContext>,
        application_name: Option<&CStr>
    ) -> Result<Self, CoreInitError> {
        let dc = Arc::new(DeviceContext::new(window_context, application_name)?);
        let swapchain = Swapchain::new(dc.clone(), true)?;

        // Create render pass
        let render_pass = DropGuard::new(
            Self::create_render_pass(dc.as_ref(), &swapchain)?,
            |pass| unsafe { dc.device.destroy_render_pass(*pass, None) }
        );

        // Create command pool for per-frame command buffer allocation
        let frame_command_pool = {
            let create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            DropGuard::new(
                unsafe { dc.device.create_command_pool(&create_info, None) }?,
                |pool| unsafe { dc.device.destroy_command_pool(*pool, None) }
            )
        };

        let acquision_semaphore = DropGuard::new(
            unsafe { dc.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? },
            |semaphore| unsafe { dc.device.destroy_semaphore(*semaphore, None) }
        );

        let allocator = Arc::new(Allocator::new(dc.clone())?);

        let (pipeline_layout, pipeline) = unsafe { Self::create_pipeline(dc.as_ref(), *render_pass) }?;

        Ok(Self {
            frame_command_pool: frame_command_pool.into_inner(),
            render_pass: render_pass.into_inner(),
            buffered_semaphore: Cell::new(acquision_semaphore.into_inner()),
            frames: Vec::new(),
            render_set: Arc::new(RenderSet::new()),
            matrix_view_projection: {
                let view = FMat::view(
                    FVec::new3(4.0, 4.0, 4.0),
                    FVec::new3(0.0, 0.0, 0.0),
                    FVec::new3(0.0, 1.0, 0.0)
                );
                let projection = FMat::projection_frustum_invz(-1.0, 1.0, -1.0, 1.0, 1.0);

                FMat::mul(&projection, &view)
            },
            pipeline,
            pipeline_layout,
            allocator,
            swapchain,
            dc,
        })
    }

    /// Create framebuffer image from some external one
    unsafe fn create_framebuffer_image_external(
        &self,
        image: vk::Image,
        format: vk::Format,
        is_depthbuffer: bool,
    ) -> Result<FramebufferImage, vk::Result> {
        let create_info = vk::ImageViewCreateInfo::default()
            .components(vk::ComponentMapping::default())
            .format(format)
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .subresource_range(vk::ImageSubresourceRange::default()
                .aspect_mask(if is_depthbuffer { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR })
                .base_array_layer(0)
                .base_mip_level(0)
                .layer_count(1)
                .level_count(1)
            )
            ;

        let view = unsafe { self.dc.device.create_image_view(&create_info, None) }?;

        Ok(FramebufferImage {
            allocation: None,
            image,
            view
        })
    }

    /// Create completely new framebuffer image
    unsafe fn create_framebuffer_image(
        &self,
        image_create_info: vk::ImageCreateInfo,
        is_depthbuffer: bool
    ) -> Result<FramebufferImage, vk::Result> {
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let image_allocation = DropGuard::new(
            unsafe { self.allocator._create_image(&image_create_info, &allocation_create_info) }?,
            |(image, allocation)| unsafe { self.allocator.destroy_image(*image, allocation) }
        );

        let view_create_info = vk::ImageViewCreateInfo::default()
            // pub flags: ImageViewCreateFlags,
            .image(image_allocation.0)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(image_create_info.format)
            // pub components: ComponentMapping,
            .subresource_range(vk::ImageSubresourceRange::default()
                .aspect_mask(if is_depthbuffer { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR })
                .base_array_layer(0)
                .base_mip_level(0)
                .layer_count(image_create_info.array_layers)
                .level_count(image_create_info.mip_levels)
            )
        ;

        let view = unsafe { self.dc.device.create_image_view(&view_create_info, None) }?;
        let (image, allocation) = image_allocation.into_inner();

        Ok(FramebufferImage {
            allocation: Some(allocation),
            image,
            view
        })
    }

    /// Destroy contents of the framebuffer image
    unsafe fn destroy_framebuffer_image(&self, image: &mut FramebufferImage) {
        if let Some(allocation) = image.allocation.as_mut() {
            unsafe { self.allocator.destroy_image(image.image, allocation) };
        }
        unsafe { self.dc.device.destroy_image_view(image.view, None) };
    }

    /// Create framebuffer
    unsafe fn create_framebuffer(
        &self,
        image: vk::Image,
        image_format: vk::Format,
        swapchain_handle: Arc<SwapchainHandle>,
        swapchain_extent: vk::Extent2D,
    ) -> Result<Framebuffer, vk::Result> {
        let swapchain_image = DropGuard::new(
            unsafe { self.create_framebuffer_image_external(image, image_format, false) }?,
            |img| unsafe { self.destroy_framebuffer_image(img) }
        );

        let depth_image = {
            let create_info = vk::ImageCreateInfo::default()
                // pub flags: ImageCreateFlags,
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT)
                .extent(vk::Extent3D {
                    width: swapchain_extent.width,
                    height: swapchain_extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                // pub queue_family_index_count: u32,
                // pub p_queue_family_indices: *const u32,
                .initial_layout(vk::ImageLayout::UNDEFINED)
                ;

            DropGuard::new(
                unsafe { self.create_framebuffer_image(create_info, true) }?,
                |img| unsafe { self.destroy_framebuffer_image(img); }
            )
        };

        let framebuffer = {
            let attachments = [swapchain_image.view, depth_image.view];

            let create_info = vk::FramebufferCreateInfo::default()
                .flags(vk::FramebufferCreateFlags::empty())
                .render_pass(self.render_pass)
                .attachments(&attachments)
                .width(swapchain_extent.width)
                .height(swapchain_extent.height)
                .layers(1);

            DropGuard::new(
                unsafe { self.dc.device.create_framebuffer(&create_info, None) }?,
                |fb| unsafe { self.dc.device.destroy_framebuffer(*fb, None) }
            )
        };

        Ok(Framebuffer {
            swapchain_image: swapchain_image.into_inner(),
            depth_image: depth_image.into_inner(),
            framebuffer: framebuffer.into_inner(),
            swapchain_handle,
        })
    }

    /// Destroy framebuffer
    unsafe fn destroy_framebuffer(&self, mut fb: Framebuffer) {
        unsafe {
            self.destroy_framebuffer_image(&mut fb.swapchain_image);
            self.destroy_framebuffer_image(&mut fb.depth_image);
            self.dc.device.destroy_framebuffer(fb.framebuffer, None);
        }
    }

    /// Create new frame context
    unsafe fn create_frame_context(&self) -> Result<FrameContext, vk::Result> {
        let command_buffer = {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(self.frame_command_pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            DropGuard::new(
                unsafe { self.dc.device.allocate_command_buffers(&alloc_info) }?[0],
                |cb| unsafe {
                    self.dc.device.free_command_buffers(
                        self.frame_command_pool,
                        std::array::from_ref(cb)
                    )
                }
            )
        };

        let create_semaphore = || -> Result<DropGuard<vk::Semaphore, _>, vk::Result> {
            Ok(DropGuard::new(
                unsafe { self.dc.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?,
                |s| unsafe { self.dc.device.destroy_semaphore(*s, None) }
            ))
        };

        let frame_acuqired_semaphore = create_semaphore()?;
        let render_finished_semaphore = create_semaphore()?;
        let transfer_finished_semaphore = create_semaphore()?;

        let fence = {
            let create_info = vk::FenceCreateInfo::default()
                .flags(vk::FenceCreateFlags::SIGNALED);

            DropGuard::new(
                unsafe { self.dc.device.create_fence(&create_info, None) }?,
                |f| unsafe { self.dc.device.destroy_fence(*f, None) }
            )
        };

        Ok(FrameContext {
            command_buffer: command_buffer.into_inner(),
            fence: fence.into_inner(),
            flush_context: None,
            frame_acquired_semaphore: Cell::new(frame_acuqired_semaphore.into_inner()),
            render_finished_semaphore: render_finished_semaphore.into_inner(),
            transfer_finished_semaphore: transfer_finished_semaphore.into_inner(),
            framebuffer: None,
            render_set: Vec::new()
        })
    }

    unsafe fn destroy_frame_context(&self, mut frame: FrameContext) {
        unsafe {
            if let Some(fb) = frame.framebuffer.take() {
                self.destroy_framebuffer(fb);
            }

            self.dc.device.free_command_buffers(
                self.frame_command_pool,
                std::array::from_ref(&frame.command_buffer)
            );

            self.dc.device.destroy_fence(frame.fence, None);
            self.dc.device.destroy_semaphore(frame.frame_acquired_semaphore.get(), None);
            self.dc.device.destroy_semaphore(frame.transfer_finished_semaphore, None);
            self.dc.device.destroy_semaphore(frame.render_finished_semaphore, None);
        }
    }

    /// Resize frames to match desired frame amount
    unsafe fn resize_frames(&mut self, new_amount: usize) -> Result<(), vk::Result> {
        if self.frames.len() == new_amount {
            return Ok(());
        }

        if new_amount > self.frames.len() {
            // Create new frames
            let mut new_frames = DropGuard::new(
                Vec::<FrameContext>::with_capacity(new_amount - self.frames.len()),
                |frames| frames
                    .drain(..)
                    .for_each(|fc| unsafe { self.destroy_frame_context(fc) })
            );
            for _ in 0..new_amount - self.frames.len() {
                new_frames.push(unsafe { self.create_frame_context() }?);
            }

            self.frames.append(&mut new_frames.into_inner());
        } else {
            let frame_destroy_list = self.frames.drain(new_amount..).collect::<Vec<_>>();
            for frame in frame_destroy_list {
                unsafe { self.destroy_frame_context(frame) };
            }
        }

        Ok(())
    }

    /// Update frame context to match swapchain state
    unsafe fn update_frame_context(
        &mut self,
        index: usize,
        swapchain_handle: Arc<SwapchainHandle>
    ) -> Result<(), vk::Result> {
        let mut fb_opt = self.frames[index].framebuffer.take();

        // Perform filter_map
        if let Some(fb) = fb_opt {
            fb_opt = if fb.swapchain_handle != swapchain_handle {
                unsafe { self.destroy_framebuffer(fb) };
                None
            } else {
                Some(fb)
            };
        }

        // Initialize framebuffer
        self.frames[index].framebuffer = Some(match fb_opt {
            Some(fb) => fb,
            None => unsafe {
                self.create_framebuffer(
                    self.swapchain.images()[index],
                    self.swapchain.image_format(),
                    swapchain_handle,
                    self.swapchain.extent()
                )?
            }
        });

        Ok(())
    }

    /// Create new mesh
    pub fn create_mesh(&self, vertices: &[Vertex], indices: &[u32]) -> Result<Arc<Mesh>, vk::Result> {
        let vt_buf_size = vertices.len() * std::mem::size_of::<Vertex>();
        let ind_buf_size = indices.len() * std::mem::size_of::<u32>();

        let guarded_buffer_and_allocation = {
            let create_info = vk::BufferCreateInfo::default()
                .queue_family_indices(std::array::from_ref(&self.dc.queue_family_index))
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .size((vt_buf_size + ind_buf_size) as u64)
                .usage(vk::BufferUsageFlags::empty()
                    | vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                );
            let alloc_info = vk_mem::AllocationCreateInfo {
                flags: vk_mem::AllocationCreateFlags::empty(),
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            DropGuard::new(
                unsafe { self.allocator.create_buffer(&create_info, &alloc_info) }?,
                |(buffer, allocation)| unsafe { self.allocator.destroy_buffer(*buffer, allocation) }
            )
        };
        let (buffer, _) = &*guarded_buffer_and_allocation;

        // Write vertex and index buffer data
        unsafe {
            self.allocator.write_buffer(
                *buffer,
                0,
                std::slice::from_raw_parts(vertices.as_ptr() as *const u8, vt_buf_size)
            )?;
            self.allocator.write_buffer(
                *buffer,
                vt_buf_size,
                std::slice::from_raw_parts(indices.as_ptr() as *const u8, ind_buf_size)
            )?;
        }

        let (buffer, allocation) = guarded_buffer_and_allocation.into_inner();

        Ok(Arc::new(Mesh {
            _dc: self.dc.clone(),
            allocator: self.allocator.clone(),
            buffer_allocation: allocation,
            buffer: buffer,
            vertex_span: 0..vt_buf_size,
            index_span: vt_buf_size..vt_buf_size + ind_buf_size,
            index_count: indices.len(),
        }))
    }

    /// Create new instance
    pub fn create_instance(&self, mesh: Arc<Mesh>, material: Material) -> Result<Arc<Instance>, vk::Result> {
        Ok(Arc::new(Instance {
            mesh,
            _material: material,
            render_set: Arc::downgrade(&self.render_set),
            transform: Cell::new(math::FMat::identity()),
        }))
    }

    /// Render next frame
    pub fn render_frame(&mut self) -> Result<(), vk::Result> {
        let frame_acquired_semaphore = self.buffered_semaphore.get();

        // Get swapchain image
        let (swapchain_guard, swapchain_image_index) = {
            let (guard, index, resized) = unsafe {
                self.swapchain.next_image(frame_acquired_semaphore)?
            };

            // Resize if required
            if resized {
                // device_wait_idle here is required to preserve synchronization here
                unsafe { self.dc.device.device_wait_idle() }?;
                unsafe { self.resize_frames(self.swapchain.images().len()) }?;
            }

            (guard, index)
        };

        // Wait for frame fences and reset'em
        unsafe {
            let frame = &self.frames[swapchain_image_index as usize];
            self.dc.device.wait_for_fences(std::array::from_ref(&frame.fence), true, u64::MAX)?;
            self.dc.device.reset_fences(std::array::from_ref(&frame.fence))?;
        }

        // Update frame framebuffer
        unsafe { self.update_frame_context(swapchain_image_index as usize, swapchain_guard) }?;

        // Get frame and set semaphore
        let frame = &mut self.frames[swapchain_image_index as usize];
        frame.frame_acquired_semaphore.swap(&self.buffered_semaphore);

        // Take current render set snapshot
        frame.render_set = self.render_set.snapshot();

        // Replace flush context with the new one
        frame.flush_context.replace(
            self.allocator.clone().flush(frame.transfer_finished_semaphore)?
        );

        // Reset rendering command buffer
        unsafe {
            self.dc.device.reset_command_buffer(frame.command_buffer, vk::CommandBufferResetFlags::default())?
        };

        // Begin command buffer
        unsafe {
            self.dc.device.begin_command_buffer(
                frame.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            )?;
        }

        let clear_values = {
            macro_rules! value {
                ((color $field: ident $val: expr)) => {
                    vk::ClearValue {
                        color: vk::ClearColorValue { $field: $val }
                    }
                };
                ((depth [$d: expr, $s: expr])) => {
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: $d,
                            stencil: $s,
                        }
                    }
                };
            }
            macro_rules! clear_values {
                ($($value: tt)*) => { [$( value!($value) ),*] };
            }

            // Clojure-ish style)
            clear_values! [
                (color float32 [0.30, 0.47, 0.80, 0.0])
                (depth [0.0, 0])
            ]
        };

        let begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(frame.framebuffer.as_ref().unwrap().framebuffer)
            .render_area(vk::Rect2D::default()
                .offset(vk::Offset2D::default())
                .extent(self.swapchain.extent())
            )
            .clear_values(&clear_values)
        ;

        unsafe {
            self.dc.device.cmd_begin_render_pass(
                frame.command_buffer,
                &begin_info,
                vk::SubpassContents::INLINE,
            )
        };

        // Bind (the) pipeline
        unsafe {
            self.dc.device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline
            );
        }

        // Set scissor and viewport
        unsafe {
            let extent = self.swapchain.extent();

            let scissor = vk::Rect2D::default().extent(extent);
            self.dc.device.cmd_set_scissor(frame.command_buffer, 0, std::array::from_ref(&scissor));

            let viewport = vk::Viewport {
                width: extent.width as f32,
                height: extent.height as f32,
                x: 0.0,
                y: 0.0,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.dc.device.cmd_set_viewport(frame.command_buffer, 0, std::array::from_ref(&viewport));
        }

        unsafe {
            let time = std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();

            static mut FRAME_COUNT: u32 = 0;
            static mut LAST_MEASURE: f64 = 0.0;

            FRAME_COUNT += 1;
            if time - LAST_MEASURE >= 1.0 {
                let fps = FRAME_COUNT as f64 / (time - LAST_MEASURE);
                LAST_MEASURE = time;
                FRAME_COUNT = 0;

                if fps >= 0.001 {
                    println!("FPS: {}", fps);
                }
            }
        }

        // Render!
        for instance in &frame.render_set {
            // Calculate world-view-projection matrix
            let world_view_projection = FMat::mul(
                &self.matrix_view_projection,
                // &FMat::translate(FVec::new3(0.0, time.sin() as f32, 0.0)),
                &instance.transform.get(),
            );

            // Emit draw commands
            unsafe {
                self.dc.device.cmd_bind_vertex_buffers(
                    frame.command_buffer, 0,
                    &[instance.mesh.buffer],
                    &[instance.mesh.vertex_span.start as u64]
                );

                self.dc.device.cmd_bind_index_buffer(
                    frame.command_buffer,
                    instance.mesh.buffer,
                    instance.mesh.index_span.start as u64,
                    vk::IndexType::UINT32
                );

                self.dc.device.cmd_push_constants(
                    frame.command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    std::slice::from_raw_parts(
                        (&world_view_projection as *const FMat) as *const u8,
                        std::mem::size_of::<FMat>()
                    )
                );

                self.dc.device.cmd_draw_indexed(
                    frame.command_buffer,
                    instance.mesh.index_count as u32,
                    1,
                    0,
                    0,
                    0
                );
            }
        }

        unsafe {
            self.dc.device.cmd_end_render_pass(frame.command_buffer);
        }

        unsafe {
            self.dc.device.end_command_buffer(frame.command_buffer)?;
        }

        let submit_wait_semaphores = [
            frame_acquired_semaphore,
            frame.transfer_finished_semaphore,
        ];
        let submit_stage_flags = [
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        ];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&submit_wait_semaphores)
            .wait_dst_stage_mask(&submit_stage_flags)
            .signal_semaphores(std::array::from_ref(&frame.render_finished_semaphore))
            .command_buffers(std::array::from_ref(&frame.command_buffer));

        unsafe {
            self.dc.device.queue_submit(
                self.dc.queue,
                std::array::from_ref(&submit_info),
                frame.fence
            )?;
        }

        let vulkan_swapchain = unsafe { self.swapchain.handle() };
        let mut present_result = vk::Result::default();

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(std::array::from_ref(&frame.render_finished_semaphore))
            .image_indices(std::array::from_ref(&swapchain_image_index))
            .swapchains(std::array::from_ref(&vulkan_swapchain))
            .results(std::array::from_mut(&mut present_result));

        // Present and check for success
        unsafe { self.dc.device_swapchain.queue_present(self.dc.queue, &present_info) }?;

        Ok(())
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        // Clear render set

        unsafe {
            // Wait all device operations finish
            _ = self.dc.device.device_wait_idle();

            // Destroy frames
            _ = self.resize_frames(0);

            self.dc.device.destroy_pipeline(self.pipeline, None);
            self.dc.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.dc.device.destroy_render_pass(self.render_pass, None);
            self.dc.device.destroy_semaphore(self.buffered_semaphore.get(), None);
            self.dc.device.destroy_command_pool(self.frame_command_pool, None);
        }
    }
}
