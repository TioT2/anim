//! Render component that performs actual low-level rendering (manages meshes, materials, instances, etc.)

use ash::vk;

// STD imports
use std::{
    cell::{Cell, RefCell},
    collections::HashSet,
    ffi::{CStr, CString},
    sync::Arc
};

// ANIM imports
use crate::{
    math::{self, FMat, FVec},
    render::core::{
        device_context::DeviceContext,
        shader_compiler::{ShaderCompiler, ShaderCompilerError},
        memory::{Allocator, FlushContext},
        swapchain::{Swapchain, SwapchainHandle},
        util::DropGuard
    }
};

mod device_context;
mod shader_compiler;
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

/// Protect vulkan object with guard with dc
macro_rules! vulkan_guard {
    ($v: expr, $dc: expr, $d: ident) => { DropGuard::new($v, |v| unsafe { $dc.device.$d(*v, None) }) };
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

impl<T> std::hash::Hash for BlindArc<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(self.0.as_ref() as *const _, state)
    }
}

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

    /// Shader compiler related error
    ShaderCompilerError(ShaderCompilerError),

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

impl From<ShaderCompilerError> for CoreInitError {
    fn from(value: ShaderCompilerError) -> Self {
        Self::ShaderCompilerError(value)
    }
}

/// Framebuffer image
struct FramebufferImage {
    /// Image allocation (may be null)
    allocation: Option<vk_mem::Allocation>,

    /// Image itself
    image: vk::Image,

    /// Image view
    view: vk::ImageView,
}

/// Standalone framebuffer implementation
struct Framebuffer {
    /// Device context
    dc: Arc<DeviceContext>,

    /// Memory allocator
    allocator: Arc<Allocator>,

    /// Handle of the swapchain
    swapchain_handle: Arc<SwapchainHandle>,

    /// Swapchain image
    swapchain_image: FramebufferImage,

    /// Depthbuffer image
    depth_image: FramebufferImage,

    /// Framebuffer itself
    framebuffer: vk::Framebuffer,
}

impl Framebuffer {
    /// Allocate image for fb
    unsafe fn create_framebuffer_image(
        dc: &DeviceContext,
        allocator: &Allocator,
        image_create_info: vk::ImageCreateInfo,
        is_depthbuffer: bool
    ) -> Result<FramebufferImage, vk::Result> {
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let image_allocation = DropGuard::new(
            unsafe { allocator.create_image(&image_create_info, &allocation_create_info) }?,
            |(image, allocation)| unsafe { allocator.destroy_image(*image, allocation) }
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

        let view = unsafe { dc.device.create_image_view(&view_create_info, None) }?;
        let (image, allocation) = image_allocation.into_inner();

        Ok(FramebufferImage {
            allocation: Some(allocation),
            image,
            view
        })
    }

    /// Create fb image from pre-created image
    unsafe fn create_framebuffer_image_external(
        dc: &DeviceContext,
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

        let view = unsafe { dc.device.create_image_view(&create_info, None) }?;

        Ok(FramebufferImage {
            allocation: None,
            image,
            view
        })
    }

    /// Destroy fb image
    unsafe fn destroy_framebuffer_image(
        dc: &DeviceContext,
        allocator: &Allocator,
        image: &mut FramebufferImage
    ) {
        if let Some(allocation) = image.allocation.as_mut() {
            unsafe { allocator.destroy_image(image.image, allocation) };
        }
        unsafe { dc.device.destroy_image_view(image.view, None) };
    }

    /// Create new framebuffer
    pub unsafe fn new(
        dc: Arc<DeviceContext>,
        allocator: Arc<Allocator>,
        render_pass: vk::RenderPass,
        swapchain_image_format: vk::Format,
        swapchain_image_index: usize,
        swapchain_handle: Arc<SwapchainHandle>,
    ) -> Result<Self, vk::Result> {
        let swapchain_extent = swapchain_handle.extent();
        let destroy_image = |img: &mut FramebufferImage| unsafe {
            Self::destroy_framebuffer_image(dc.as_ref(), allocator.as_ref(), img)
        };

        let swapchain_image = DropGuard::new(
            unsafe { Self::create_framebuffer_image_external(
                dc.as_ref(),
                swapchain_handle.images()[swapchain_image_index],
                swapchain_image_format,
                false
            ) }?,
            destroy_image
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
                unsafe { Self::create_framebuffer_image(dc.as_ref(), allocator.as_ref(), create_info, true) }?,
                destroy_image
            )
        };

        let framebuffer = {
            let attachments = [swapchain_image.view, depth_image.view];

            let create_info = vk::FramebufferCreateInfo::default()
                .flags(vk::FramebufferCreateFlags::empty())
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(swapchain_extent.width)
                .height(swapchain_extent.height)
                .layers(1);

            DropGuard::new(
                unsafe { dc.device.create_framebuffer(&create_info, None) }?,
                |fb| unsafe { dc.device.destroy_framebuffer(*fb, None) }
            )
        };

        Ok(Self {
            swapchain_image: swapchain_image.into_inner(),
            depth_image: depth_image.into_inner(),
            framebuffer: framebuffer.into_inner(),
            swapchain_handle,
            dc,
            allocator,
        })
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        // Image destroy function
        let destroy_image = |img| unsafe {
            Self::destroy_framebuffer_image(self.dc.as_ref(), self.allocator.as_ref(), img);
        };

        destroy_image(&mut self.swapchain_image);
        destroy_image(&mut self.depth_image);
        unsafe { self.dc.device.destroy_framebuffer(self.framebuffer, None) };
    }
}

/// Item of the device matrix buffer
#[repr(C)]
#[derive(Copy, Clone)]
struct MatrixDeviceBufferItem {
    /// World matirx
    pub world: FMat,

    /// World-view-projection matrix
    pub world_view_projection: FMat,

    /// Inverse world matrix (3x3)
    pub world_inverse: math::Mat<f32, 3, 4>,
}

/// Uniform buffer that holds camera buffer data
#[repr(C)]
#[derive(Copy, Clone)]
struct CameraBufferData {
    /// View-projection matirx
    pub view_projection: FMat,

    /// View matrix
    pub view: FMat,

    /// Projection matrix
    pub projection: FMat,

    /// Forward camera direction
    pub dir_forward: FVec,

    /// Right camera direction
    pub dir_right: FVec,

    /// Up camera direction
    pub dir_up: FVec,

    /// Camera location
    pub location: FVec,
}

/// Structure that contains matrix buffer data
#[derive(Default)]
struct MatrixBuffer {
    /// Actual count of matrix buffer elements
    length: usize,

    /// Buffer (potential) capacity
    capacity: usize,

    /// Host matrix buffer
    host_buffer: vk::Buffer,

    /// Host matrix allocation
    host_allocation: Option<vk_mem::Allocation>,

    /// Device matrix buffer
    device_buffer: vk::Buffer,

    /// Device matrix allocation
    device_allocation: Option<vk_mem::Allocation>,
}

/// In-flight frame representation
struct FrameContext {
    /// Frame acquision semaphore (for frame output start)
    frame_acquired_semaphore: Cell<vk::Semaphore>,

    /// Semaphore that indicates rendering process end (for presentation)
    render_finished_semaphore: vk::Semaphore,

    /// Transfer operation finish semaphore
    transfer_finished_semaphore: vk::Semaphore,

    /// Fence to wait then frame is reused
    fence: vk::Fence,

    /// Command buffer used for frame commands
    command_buffer: vk::CommandBuffer,

    /// New framebuffer structure
    framebuffer: Option<Framebuffer>,

    /// Context of the flush operations
    flush_context: Option<FlushContext>,

    /// Buffer that contains matrix data
    matrix_buffer: MatrixBuffer,

    /// Camera buffer
    camera_buffer: vk::Buffer,

    /// Corresponding allocation
    camera_buffer_allocation: vk_mem::Allocation,

    /// Pool for the matrix_descriptor_set field
    frame_descriptor_pool: vk::DescriptorPool,

    /// Matirx descriptor set
    matrix_descriptor_set: vk::DescriptorSet,

    /// Descriptor set used in rendering process
    render_descriptor_set: vk::DescriptorSet,

    /// Current frame render set
    render_set: Vec<Arc<Instance>>,
}

/// Core object of the renderer
pub struct Core {
    /// Context of the device
    dc: Arc<DeviceContext>,

    /// Memory allocator handle
    allocator: Arc<Allocator>,

    /// Shader compiler
    _shader_compiler: ShaderCompiler,

    /// Swapchain
    swapchain: Swapchain,

    /// Command pool for per-frame allocation command buffer
    frame_command_pool: vk::CommandPool,

    /// Unsignaled semaphore to use as a new frame semaphore
    buffered_semaphore: Cell<vk::Semaphore>,

    /// Main render pass
    render_pass: vk::RenderPass,

    /// Temp pipeline layout storage
    render_pipeline_layout: vk::PipelineLayout,

    /// Depth-only pipeline
    render_depth_pipeline: vk::Pipeline,

    /// Color pipeline
    render_color_pipeline: vk::Pipeline,

    /// Matrix descriptor set layout
    matrix_ds_layout: vk::DescriptorSetLayout,

    /// DS layout used during rendering process
    render_ds_layout: vk::DescriptorSetLayout,

    /// Matrix pipeline layout
    matrix_pipeline_layout: vk::PipelineLayout,

    /// Matrix pipeline
    matrix_pipeline: vk::Pipeline,

    /// Set of frames
    frames: Vec<Option<Box<FrameContext>>>,

    /// Set of rendered objects
    render_set: Arc<RenderSet>,

    // /// View * Projection matrix
    // matrix_view_projection: FMat,
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

        let depth_depth_att_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_depth_att_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);

        let subpasses = [
            vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .depth_stencil_attachment(&depth_depth_att_ref),
            vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(std::array::from_ref(&target_att_ref))
                .depth_stencil_attachment(&color_depth_att_ref),
        ];

        let dependencies = [
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(1)
                .src_stage_mask(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
                .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
                // pub dependency_flags: DependencyFlags,
        ];

        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .dependencies(&dependencies)
            .subpasses(&subpasses);

        unsafe { dc.device.create_render_pass(&create_info, None) }
    }

    /// Create pipeline used for matrix computation
    unsafe fn create_matrix_compute_pipeline(
        dc: &DeviceContext,
        shader_compiler: &ShaderCompiler
    ) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline), vk::Result> {
        let ds_layout = {
            // Generate DS layout binding
            let binding = |index, ty| vk::DescriptorSetLayoutBinding::default()
                .binding(index)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE);

            let bindings = [
                binding(0, vk::DescriptorType::STORAGE_BUFFER),
                binding(1, vk::DescriptorType::STORAGE_BUFFER),
                binding(2, vk::DescriptorType::UNIFORM_BUFFER),
            ];

            let create_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&bindings);

            DropGuard::new(
                unsafe { dc.device.create_descriptor_set_layout(&create_info, None) }?,
                |dsl| unsafe { dc.device.destroy_descriptor_set_layout(*dsl, None) }
            )
        };

        let pipeline_layout = {
            let create_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::array::from_ref(&*ds_layout))
                ;

            DropGuard::new(
                unsafe { dc.device.create_pipeline_layout(&create_info, None) }?,
                |pl| unsafe { dc.device.destroy_pipeline_layout(*pl, None) }
            )
        };

        let shader_module = {
            let spirv = shader_compiler
                .compile_shader("/anim/matrix_compute.hlsl", "cs_main", "cs_5_1")
                .unwrap();

            let module = unsafe {
                dc.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spirv), None)
            }?;

            DropGuard::new(
                module,
                |module| unsafe { dc.device.destroy_shader_module(*module, None) }
            )
        };

        let stage_create_info = vk::PipelineShaderStageCreateInfo::default()
            // pub flags: PipelineShaderStageCreateFlags,
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(*shader_module)
            .name(c"cs_main")
            // pub p_specialization_info: *const SpecializationInfo<'a>,
            ;

        let create_info = vk::ComputePipelineCreateInfo::default()
            // pub flags: PipelineCreateFlags,
            .stage(stage_create_info)
            // pub stage: PipelineShaderStageCreateInfo<'a>,
            .layout(*pipeline_layout)
            // pub base_pipeline_handle: Pipeline,
            // pub base_pipeline_index: i32,
            ;

        let pipeline = unsafe {
            dc.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                std::array::from_ref(&create_info),
                None
            ).map_err(|(_, err)| err)?[0]
        };

        Ok((
            ds_layout.into_inner(),
            pipeline_layout.into_inner(),
            pipeline
        ))
    }

    /// Create set of pipelines with same usecase
    unsafe fn create_pipeline_family(
        dc: &DeviceContext,
        shader_compiler: &ShaderCompiler,
        render_pass: vk::RenderPass
    ) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline, vk::Pipeline), vk::Result> {
        let ds_layout = {
            let binding = |index, ty| vk::DescriptorSetLayoutBinding::default()
                .binding(index)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

            let bindings = [
                binding(0, vk::DescriptorType::STORAGE_BUFFER),
                binding(1, vk::DescriptorType::UNIFORM_BUFFER),
            ];

            let create_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&bindings);
            let layout = unsafe { dc.device.create_descriptor_set_layout(&create_info, None) }?;
            vulkan_guard!(layout, dc, destroy_descriptor_set_layout)
        };

        let layout = {
            let push_constant_range = vk::PushConstantRange::default()
                .offset(0)
                .size(std::mem::size_of::<u32>() as u32)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                ;

            let create_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::array::from_ref(&*ds_layout))
                .push_constant_ranges(std::array::from_ref(&push_constant_range))
                ;

            let layout = unsafe { dc.device.create_pipeline_layout(&create_info, None) }?;
            vulkan_guard!(layout, dc, destroy_pipeline_layout)
        };

        let shader_modules = {
            let build = |main_fn_name: &str, shader_model: &str| {
                let spirv = shader_compiler
                    .compile_shader("/anim/model.hlsl", main_fn_name, shader_model)
                    .unwrap();

                Ok(DropGuard::new(
                    unsafe {
                        dc.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spirv), None)
                    }?,
                    |sm| unsafe { dc.device.destroy_shader_module(*sm, None) }
                ))
            };

            DropGuard::zip_multiple([
                build("vs_main",       "vs_5_1")?,
                build("fs_main",       "ps_5_1")?,
                build("vs_depth_main", "vs_5_1")?,
            ].into_iter())
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
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            ;

        let dynamic_states = [
            vk::DynamicState::SCISSOR,
            vk::DynamicState::VIEWPORT,
        ];

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states)
            ;

        let base_create_info = vk::GraphicsPipelineCreateInfo::default()
            // .stages(&color_stage_create_infos)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            // pub p_tessellation_state: *const PipelineTessellationStateCreateInfo<'a>,
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            // .depth_stencil_state(&color_depth_stencil_state)
            // .color_blend_state(&color_color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(*layout)
            .render_pass(render_pass)
            // .subpass(0)
            ;

        // Construct shader stage create info
        let make_stage_info = |sm, sn, st| vk::PipelineShaderStageCreateInfo::default()
            .module(sm).stage(st).name(sn);

        let depth_stage_create_infos = [
            make_stage_info(shader_modules[2], c"vs_depth_main", vk::ShaderStageFlags::VERTEX),
        ];

        let depth_depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
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

        let depth_color_blend_state = vk::PipelineColorBlendStateCreateInfo::default();

        let depth_create_info = base_create_info
            .stages(&depth_stage_create_infos)
            .depth_stencil_state(&depth_depth_stencil_state)
            .color_blend_state(&depth_color_blend_state)
            .subpass(0)
            ;

        let color_stage_create_infos = [
            make_stage_info(shader_modules[0], c"vs_main", vk::ShaderStageFlags::VERTEX),
            make_stage_info(shader_modules[1], c"fs_main", vk::ShaderStageFlags::FRAGMENT),
        ];

        let color_depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            // pub flags: PipelineDepthStencilStateCreateFlags,
            .depth_test_enable(true)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            // pub front: StencilOpState,
            // pub back: StencilOpState,
            // pub min_depth_bounds: f32,
            // pub max_depth_bounds: f32,
            ;

        let target_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::array::from_ref(&target_attachment));

        let color_create_info = base_create_info
            .stages(&color_stage_create_infos)
            .depth_stencil_state(&color_depth_stencil_state)
            .color_blend_state(&color_color_blend_state)
            .subpass(1)
            ;

        let create_infos = [depth_create_info, color_create_info];

        let pipelines_opt = unsafe {
            dc.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &create_infos,
                None
            )
        };

        match pipelines_opt {
            Ok(pipelines) => {
                let depth_pipeline = pipelines[0];
                let color_pipeline = pipelines[1];

                Ok((ds_layout.into_inner(), layout.into_inner(), depth_pipeline, color_pipeline))
            }
            Err((pipelines, error)) => {
                for pipeline in pipelines {
                    unsafe { dc.device.destroy_pipeline(pipeline, None) };
                }

                Err(error)
            }
        }
    }

    /// Construct instance
    pub fn new(
        window_context: Arc<dyn WindowContext>,
        application_name: Option<&CStr>
    ) -> Result<Self, CoreInitError> {
        let dc = Arc::new(DeviceContext::new(window_context, application_name)?);
        let swapchain = Swapchain::new(dc.clone(), true)?;
        let allocator = Arc::new(Allocator::new(dc.clone())?);
        let shader_compiler = ShaderCompiler::new()?;

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

        let acquision_semaphore = unsafe { dc.device.create_semaphore(&Default::default(), None)? };
        let acquision_semaphore = vulkan_guard!(acquision_semaphore, dc, destroy_semaphore);

        let (render_ds_layout, render_pipeline_layout, render_depth_pipeline, render_color_pipeline) = unsafe {
            Self::create_pipeline_family(dc.as_ref(), &shader_compiler, *render_pass)
        }?;
        let render_ds_layout = vulkan_guard!(render_ds_layout, dc, destroy_descriptor_set_layout);
        let render_pipeline_layout = vulkan_guard!(render_pipeline_layout, dc, destroy_pipeline_layout);
        let render_depth_pipeline = vulkan_guard!(render_depth_pipeline, dc, destroy_pipeline);
        let render_color_pipeline = vulkan_guard!(render_color_pipeline, dc, destroy_pipeline);

        // Create matrix pipeline
        let (matrix_ds_layout, matrix_pipeline_layout, matrix_pipeline) = unsafe {
            Self::create_matrix_compute_pipeline(dc.as_ref(), &shader_compiler)
        }?;
        let matrix_ds_layout = vulkan_guard!(matrix_ds_layout, dc, destroy_descriptor_set_layout);
        let matrix_pipeline_layout = vulkan_guard!(matrix_pipeline_layout, dc, destroy_pipeline_layout);
        let matrix_pipeline = vulkan_guard!(matrix_pipeline, dc, destroy_pipeline);

        Ok(Self {
            frame_command_pool: frame_command_pool.into_inner(),
            render_pass: render_pass.into_inner(),
            buffered_semaphore: Cell::new(acquision_semaphore.into_inner()),
            frames: Vec::new(),
            render_set: Arc::new(RenderSet::new()),

            render_depth_pipeline: render_depth_pipeline.into_inner(),
            render_color_pipeline: render_color_pipeline.into_inner(),
            render_ds_layout: render_ds_layout.into_inner(),
            render_pipeline_layout: render_pipeline_layout.into_inner(),

            matrix_ds_layout: matrix_ds_layout.into_inner(),
            matrix_pipeline_layout: matrix_pipeline_layout.into_inner(),
            matrix_pipeline: matrix_pipeline.into_inner(),

            _shader_compiler: shader_compiler,
            allocator,
            swapchain,
            dc,
        })
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

        let camera_buffer = {
            let create_info = vk::BufferCreateInfo::default()
                .size(std::mem::size_of::<CameraBufferData>() as u64)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                ;
            let alloc_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                preferred_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                ..Default::default()
            };

            DropGuard::new(
                unsafe { self.allocator.create_buffer(&create_info, &alloc_info) }?,
                |(b, a)| unsafe { self.allocator.destroy_buffer(*b, a) }
            )
        };

        let frame_descriptor_pool = {
            let pool_sizes = [
                vk::DescriptorPoolSize::default()
                    .descriptor_count(3)
                    .ty(vk::DescriptorType::STORAGE_BUFFER),
                vk::DescriptorPoolSize::default()
                    .descriptor_count(2)
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
            ];

            let create_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(2)
                .pool_sizes(&pool_sizes);

            DropGuard::new(
                unsafe { self.dc.device.create_descriptor_pool(&create_info, None) }?,
                |dp| unsafe { self.dc.device.destroy_descriptor_pool(*dp, None) }
            )
        };

        let (render_descriptor_set, matrix_descriptor_set) = {
            let layouts = [
                self.render_ds_layout,
                self.matrix_ds_layout
            ];

            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(*frame_descriptor_pool)
                .set_layouts(&layouts)
                ;

            let sets = unsafe { self.dc.device.allocate_descriptor_sets(&alloc_info) }?;
            (sets[0], sets[1])
        };

        // Write camera buffer to render and matrix descriptor sets
        {
            let write_camera_buffer = |set, index| {
                let camera_buffer_info = vk::DescriptorBufferInfo::default()
                    .buffer(camera_buffer.0)
                    .offset(0)
                    .range(std::mem::size_of::<CameraBufferData>() as vk::DeviceSize);

                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(index)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::array::from_ref(&camera_buffer_info));

                unsafe { self.dc.device.update_descriptor_sets(std::array::from_ref(&write), &[]) };
            };

            write_camera_buffer(render_descriptor_set, 1);
            write_camera_buffer(matrix_descriptor_set, 2);
        }

        // Write camera buffer to matrix descriptor set
        {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(camera_buffer.0)
                .offset(0)
                .range(std::mem::size_of::<CameraBufferData>() as vk::DeviceSize);

            let descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(matrix_descriptor_set)
                .dst_binding(2)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::array::from_ref(&buffer_info))
                ;

            // Write camera uniform buffer descriptor to the matrix descriptor set
            unsafe { self.dc.device.update_descriptor_sets(std::array::from_ref(&descriptor_write), &[]) };

        }

        Ok(FrameContext {
            command_buffer: command_buffer.into_inner(),
            fence: fence.into_inner(),
            flush_context: None,
            frame_acquired_semaphore: Cell::new(frame_acuqired_semaphore.into_inner()),
            render_finished_semaphore: render_finished_semaphore.into_inner(),
            transfer_finished_semaphore: transfer_finished_semaphore.into_inner(),
            framebuffer: None,
            camera_buffer: camera_buffer.0,
            camera_buffer_allocation: camera_buffer.into_inner().1,
            matrix_buffer: MatrixBuffer::default(),
            matrix_descriptor_set: matrix_descriptor_set,
            render_descriptor_set: render_descriptor_set,
            frame_descriptor_pool: frame_descriptor_pool.into_inner(),
            render_set: Vec::new()
        })
    }

    unsafe fn destroy_frame_context(&self, mut frame: FrameContext) {
        unsafe {
            _ = frame.framebuffer.take();

            // Free matrix buffer
            _ = self.realloc_matrix_buffer(&mut frame.matrix_buffer, 0);

            // Free camera buffer
            self.allocator.destroy_buffer(frame.camera_buffer, &mut frame.camera_buffer_allocation);

            self.dc.device.free_command_buffers(
                self.frame_command_pool,
                std::array::from_ref(&frame.command_buffer)
            );

            self.dc.device.destroy_fence(frame.fence, None);
            self.dc.device.destroy_descriptor_pool(frame.frame_descriptor_pool, None);
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

            self.frames.extend(new_frames
                .into_inner()
                .into_iter()
                .map(Box::new)
                .map(Some));
        } else {
            // Get list of destroyed frames
            let frame_destroy_list = self.frames
                .drain(new_amount..)
                .map(|f| *f.unwrap())
                .collect::<Vec<_>>();

            for frame in frame_destroy_list {
                unsafe { self.destroy_frame_context(frame) };
            }
        }

        Ok(())
    }

    /// Reallocate matrix buffer data to match certain capacity
    /// # Note
    /// This function may be used for buffer deallocation
    /// In case of capacity == 0, new buffers are just not allocated
    unsafe fn realloc_matrix_buffer(&self, matrix_buffer: &mut MatrixBuffer, capacity: usize) -> Result<(), vk::Result> {

        // Allocate new buffers for host and device
        let new_host_device_buffers = if capacity > 0 {
            let host = {
                let host_buffer_info = vk::BufferCreateInfo::default()
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .size((std::mem::size_of::<FMat>() * capacity) as u64)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    ;
                let host_alloc_info = vk_mem::AllocationCreateInfo {
                    flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                    usage: vk_mem::MemoryUsage::AutoPreferHost,
                    required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    ..Default::default()
                };

                DropGuard::new(
                    unsafe { self.allocator.create_buffer(&host_buffer_info, &host_alloc_info)? },
                    |(db, da)| unsafe { self.allocator.destroy_buffer(*db, da) }
                )
            };

            let device = {
                let device_buffer_info = vk::BufferCreateInfo::default()
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .size((std::mem::size_of::<MatrixDeviceBufferItem>() * capacity) as u64)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    ;
                let device_alloc_info = vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferDevice,
                    preferred_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    ..Default::default()
                };

                DropGuard::new(
                    unsafe { self.allocator.create_buffer(&device_buffer_info, &device_alloc_info)? },
                    |(db, da)| unsafe { self.allocator.destroy_buffer(*db, da) }
                )
            };

            Some((host, device))
        } else {
            None
        };

        // Destroy old buffers
        if let Some(host_allocation) = matrix_buffer.host_allocation.as_mut() {
            unsafe { self.allocator.destroy_buffer(matrix_buffer.host_buffer, host_allocation) };
        }
        if let Some(device_allocation) = matrix_buffer.device_allocation.as_mut() {
            unsafe { self.allocator.destroy_buffer(matrix_buffer.device_buffer, device_allocation) };
        }

        // Set new buffers
        if let Some((host, device)) = new_host_device_buffers {
            let host = host.into_inner();
            let device = device.into_inner();

            matrix_buffer.host_buffer = host.0;
            matrix_buffer.host_allocation = Some(host.1);

            matrix_buffer.device_buffer = device.0;
            matrix_buffer.device_allocation = Some(device.1);
        } else {
            matrix_buffer.host_buffer = vk::Buffer::null();
            matrix_buffer.host_allocation = None;

            matrix_buffer.device_buffer = vk::Buffer::null();
            matrix_buffer.device_allocation = None;
        }

        matrix_buffer.capacity = capacity;

        Ok(())
    }

    /// Update frame context to match swapchain state
    unsafe fn update_frame_context(
        &mut self,
        frame: &mut FrameContext,
        index: usize,
        swapchain_handle: Arc<SwapchainHandle>
    ) -> Result<(), vk::Result> {
        let construction_required = frame.framebuffer
            .as_ref()
            .filter(|fb| fb.swapchain_handle == swapchain_handle)
            .is_none();

        if construction_required {
            frame.framebuffer = Some(unsafe {
                Framebuffer::new(
                    self.dc.clone(),
                    self.allocator.clone(),
                    self.render_pass,
                    self.swapchain.image_format(),
                    index,
                    swapchain_handle
                )?
            });
        }

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

    /// Record contents for the depth and color subpass
    fn record_subpass_contents<const SUBPASS_INDEX: u32>(&self, frame: &FrameContext) {
        // Bind pipeline
        unsafe {
            self.dc.device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                match SUBPASS_INDEX {
                    0 => self.render_depth_pipeline,
                    1 => self.render_color_pipeline,
                    _ => panic!("Invalid subpass index")
                }
            );

            // Bind render descriptor set
            self.dc.device.cmd_bind_descriptor_sets(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.render_pipeline_layout,
                0,
                &[frame.render_descriptor_set],
                &[]
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

        // Render!
        for (instance_index, instance) in frame.render_set.iter().enumerate() {
            let instance_index = instance_index as u32;

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

                // Push instance index
                self.dc.device.cmd_push_constants(
                    frame.command_buffer,
                    self.render_pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    std::slice::from_raw_parts(
                        (&instance_index as *const u32) as *const u8,
                        std::mem::size_of::<u32>()
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
    }

    /// Fill frame matrix buffer with contents
    fn write_matrix_buffer(&self, frame: &mut FrameContext) -> Result<(), vk::Result> {
        // Resize matrix buffer if it's required
        if frame.matrix_buffer.capacity < frame.render_set.len() {
            unsafe { self.realloc_matrix_buffer(&mut frame.matrix_buffer, frame.render_set.len()) }?;

            let gen_buffer_info = |buf, len| vk::DescriptorBufferInfo::default()
                .buffer(buf)
                .range(len as vk::DeviceSize);

            let gen_descriptor_write = |descriptor_set, index, info| vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(index)
                .dst_array_element(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::array::from_ref(info))
                ;

            let host_buffer_info = gen_buffer_info(
                frame.matrix_buffer.host_buffer,
                std::mem::size_of::<FMat>() * frame.matrix_buffer.capacity
            );
            let device_buffer_info = gen_buffer_info(
                frame.matrix_buffer.device_buffer,
                std::mem::size_of::<MatrixDeviceBufferItem>() * frame.matrix_buffer.capacity
            );

            unsafe {
                self.dc.device.update_descriptor_sets(&[
                    gen_descriptor_write(frame.matrix_descriptor_set, 0, &host_buffer_info),
                    gen_descriptor_write(frame.matrix_descriptor_set, 1, &device_buffer_info),
                    gen_descriptor_write(frame.render_descriptor_set, 0, &device_buffer_info),
                ], &[]);
            }
        }

        frame.matrix_buffer.length = frame.render_set.len();

        let map_fn = |data: &mut [u8]| {
            for (index, instance) in frame.render_set.iter().enumerate() {
                // Memcpy required because of undefined alignment
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        instance.transform.as_ptr(),
                        (data.as_mut_ptr() as *mut FMat).add(index),
                        1
                    );
                }
            }
        };

        unsafe {
            self.allocator.map_host_allocation(
                &mut frame.matrix_buffer.host_allocation.unwrap(),
                map_fn
            )?;
        }

        Ok(())
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

        // Acquire frame
        let mut frame = self.frames[swapchain_image_index as usize].take().unwrap();

        // Wait for frame fences and reset'em
        unsafe {
            self.dc.device.wait_for_fences(std::array::from_ref(&frame.fence), true, u64::MAX)?;
            self.dc.device.reset_fences(std::array::from_ref(&frame.fence))?;
        }

        // Update frame framebuffer
        unsafe { self.update_frame_context(frame.as_mut(), swapchain_image_index as usize, swapchain_guard) }?;

        // Get frame and set semaphore
        frame.frame_acquired_semaphore.swap(&self.buffered_semaphore);

        // Take current render set snapshot
        frame.render_set = self.render_set.snapshot();

        // Write frame camera buffer
        {
            let location = FVec::new3(4.0, 4.0, 4.0);
            let at = FVec::new3(0.0, 0.0, 0.0);
            let view = FMat::view(location, at, FVec::new3(0.0, 1.0, 0.0));
            let projection = FMat::projection_frustum_invz(-1.0, 1.0, -1.0, 1.0, 1.0);

            let camera_buffer_data = CameraBufferData {
                view_projection: FMat::mul(&projection, &view),
                view,
                projection,
                dir_forward: FVec::new3(
                    -view.data[2][0],
                    -view.data[2][1],
                    -view.data[2][2],
                ),
                dir_right: FVec::new3(
                    view.data[0][0],
                    view.data[0][1],
                    view.data[0][2],
                ),
                dir_up: FVec::new3(
                    view.data[1][0],
                    view.data[1][1],
                    view.data[1][2],
                ),
                location,
            };

            unsafe {
                let buffer_data = std::slice::from_raw_parts(
                    (&camera_buffer_data as *const CameraBufferData) as *const u8,
                    std::mem::size_of::<CameraBufferData>()
                );

                self.allocator.write_buffer(frame.camera_buffer, 0, buffer_data)?;
            }
        }

        // Replace flush context with the new one
        frame.flush_context.replace(
            self.allocator.clone().flush(frame.transfer_finished_semaphore)?
        );

        // Fill matrix buffer with data
        self.write_matrix_buffer(frame.as_mut())?;

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

        unsafe {
            self.dc.device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.matrix_pipeline,
            );

            self.dc.device.cmd_bind_descriptor_sets(
                frame.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.matrix_pipeline_layout,
                0,
                std::array::from_ref(&frame.matrix_descriptor_set),
                &[]
            );

            self.dc.device.cmd_dispatch(
                frame.command_buffer,
                frame.matrix_buffer.length as u32,
                1,
                1,
            );

            self.dc.device.cmd_pipeline_barrier(
                frame.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::VERTEX_INPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[]
            );
        }

        // Construct clear values
        let clear_values = {
            // Construct value
            macro_rules! value {
                (color $field: ident $val: expr) => {
                    vk::ClearValue { color: vk::ClearColorValue { $field: $val } }
                };
                (depth [$d: expr, $s: expr]) => {
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: $d,
                            stencil: $s,
                        }
                    }
                };
            }

            [
                value!(color float32 [0.30, 0.47, 0.80, 0.0]),
                value!(depth [0.0, 0]),
            ]
        };

        let begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(frame.framebuffer.as_ref().unwrap().framebuffer)
            .render_area(vk::Rect2D::default()
                .offset(vk::Offset2D::default())
                .extent(self.swapchain.extent())
            )
            .clear_values(&clear_values);

        unsafe {
            self.dc.device.cmd_begin_render_pass(
                frame.command_buffer,
                &begin_info,
                vk::SubpassContents::INLINE,
            )
        };

        self.record_subpass_contents::<0>(frame.as_ref());

        unsafe {
            self.dc.device.cmd_next_subpass(
                frame.command_buffer,
                vk::SubpassContents::INLINE
            );
        }

        self.record_subpass_contents::<1>(frame.as_ref());

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

        self.frames[swapchain_image_index as usize] = Some(frame);

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

            // Destroy matrix computation related objects
            self.dc.device.destroy_pipeline(self.matrix_pipeline, None);
            self.dc.device.destroy_pipeline_layout(self.matrix_pipeline_layout, None);
            self.dc.device.destroy_descriptor_set_layout(self.matrix_ds_layout, None);

            // Destroy render-related objects
            self.dc.device.destroy_pipeline(self.render_depth_pipeline, None);
            self.dc.device.destroy_pipeline(self.render_color_pipeline, None);
            self.dc.device.destroy_descriptor_set_layout(self.render_ds_layout, None);
            self.dc.device.destroy_pipeline_layout(self.render_pipeline_layout, None);

            self.dc.device.destroy_render_pass(self.render_pass, None);
            self.dc.device.destroy_semaphore(self.buffered_semaphore.get(), None);
            self.dc.device.destroy_command_pool(self.frame_command_pool, None);
        }
    }
}
