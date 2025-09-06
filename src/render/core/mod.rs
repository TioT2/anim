//! Render component that performs actual low-level rendering (manages meshes, materials, instances, etc.)

use ash::vk;

// STD imports
use std::{
    cell::Cell,
    ffi::{CStr, CString},
    sync::Arc
};

// ANIM imports
use crate::{
    math::{self, FVec},
    render::core::{
        device_context::DeviceContext,
        memory::{Allocator, FlushContext},
        swapchain::{Swapchain, SwapchainGuard},
        util::DropGuard
    }
};

mod device_context;
mod memory;
mod swapchain;
mod util;

/// Fixed vertex format
#[repr(C)]
pub struct Vertex {
    /// Vertex position
    pub position: math::Mat<f32, 3>,

    /// Vertex texture coordinate
    pub tex_coord: math::Mat<f32, 2>,

    /// Vertex normal (octmap, r16g16_snorm)
    pub normal: u32,

    /// Vertex tangent (octmap, r16g16_snorm)
    pub tangent: u32,

    /// Misc data (tangent sign, ...)
    pub misc: u32,
}

impl Vertex {
    /// Pack vector into octmap
    pub fn pack_direction_octmap(dx: f32, dy: f32, dz: f32) -> u32 {
        let length = (dx.abs() + dy.abs() + dz.abs()).recip();
        let (mut x, mut y) = (dx * length, dy * length);
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

    /// Unpack octmapped vector
    pub fn unpack_direction_octmap(packed: u32) -> FVec {
        let (x, y) = (
            ((packed & 0xFFFF) as u16).cast_signed(),
            ((packed >> 16) as u16).cast_signed()
        );

        let (x, y) = ((x as f32) / 32767.0, (y as f32) / 32767.0);
        let z = 1.0 - x.abs() - y.abs();
        let t = (-z).clamp(0.0, 1.0);
        FVec::new3(x - x.copysign(t), y - y.copysign(t), z).normalized()
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
    /// Device context reference
    _dc: Arc<DeviceContext>,

    /// Allocator reference
    allocator: Arc<Allocator>,

    /// Mesh memory allocation
    buffer_allocation: vk_mem::Allocation,

    /// Mesh bufffer
    buffer: vk::Buffer,

    /// Vertex buffer memory region
    _vertex_span: std::ops::Range<usize>,

    /// Index buffer memory region
    _index_span: std::ops::Range<usize>,

    /// Count of the mesh indices
    _index_count: usize,
}

impl Drop for Mesh {
    fn drop(&mut self) {
        unsafe { self.allocator.destroy_buffer(self.buffer, &mut self.buffer_allocation) };
    }
}

/// Rendered instance representation structure
pub struct Instance {
    /// Underlying mesh
    _mesh: Arc<Mesh>,

    /// Instance transformation matrix
    transform: Cell<math::FMat>,

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

/// Representation of the in-flight frame
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

    /// Framebuffer
    framebuffer: vk::Framebuffer,

    /// Swapchain image
    swapchain_image: vk::Image,

    /// Swapchain image view
    swapchain_image_view: vk::ImageView,

    /// Context of the flush operations
    flush_context: Option<FlushContext>,

    /// Swapchain
    swapchain_guard: Option<SwapchainGuard>,
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

    /// Set of frames
    frames: Vec<FrameContext>,
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

        let target_att_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::array::from_ref(&target_att_ref));

        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(std::array::from_ref(&target_attachment))
            .subpasses(std::array::from_ref(&subpass));

        unsafe { dc.device.create_render_pass(&create_info, None) }
    }

    /// Create (the) pipeline
    unsafe fn _create_pipeline(&self) -> Result<(vk::PipelineLayout, vk::Pipeline), vk::Result> {
        let layout = {
            let push_constant_range = vk::PushConstantRange::default();

            let create_info = vk::PipelineLayoutCreateInfo::default()
                .push_constant_ranges(std::array::from_ref(&push_constant_range))
                ;

            DropGuard::new(
                unsafe { self.dc.device.create_pipeline_layout(&create_info, None) }?,
                |l| unsafe { self.dc.device.destroy_pipeline_layout(*l, None) }
            )
        };

        // Shader loading and compilation is not implemented yet(
        if true { todo!(); }

        let shader_stages = {
            let vertex_stage = vk::PipelineShaderStageCreateInfo::default();
            let fragment_stage = vk::PipelineShaderStageCreateInfo::default();

            [
                vertex_stage,
                fragment_stage
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
            .primitive_restart_enable(true)
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1)
            ;

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
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

        let target_attachment = vk::PipelineColorBlendAttachmentState::default();

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::array::from_ref(&target_attachment))
            ;

        let dynamic_states = [
            vk::DynamicState::SCISSOR,
            vk::DynamicState::VIEWPORT,
        ];

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states)
            ;

        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            // pub p_tessellation_state: *const PipelineTessellationStateCreateInfo<'a>,
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            // pub p_depth_stencil_state: *const PipelineDepthStencilStateCreateInfo<'a>,
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(*layout)
            .render_pass(self.render_pass)
            .subpass(0)
            ;

        let pipeline = unsafe {
            self.dc.device.create_graphics_pipelines(
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

        Ok(Self {
            frame_command_pool: frame_command_pool.into_inner(),
            render_pass: render_pass.into_inner(),
            buffered_semaphore: Cell::new(acquision_semaphore.into_inner()),
            frames: Vec::new(),
            allocator,
            swapchain,
            dc,
        })
    }

    /// Reize frame set to match amount of the swapchain images
    unsafe fn resize_frames(
        &mut self,
        swapchain_images: &[vk::Image],
        swapchain_extent: vk::Extent2D,
        swapchain_image_format: vk::Format,
    ) -> Result<(), vk::Result> {

        // Await all frames completion before updating target buffers contents
        if self.frames.len() != 0 {
            unsafe { self.dc.device.queue_wait_idle(self.dc.queue)? };
        }

        // Create new framebuffer vector
        let mut new_framebuffers = DropGuard::new(
            Vec::<FrameContext>::with_capacity(swapchain_images.len()),
            |resources| resources.drain(..).for_each(|fb| unsafe {
                self.dc.device.destroy_framebuffer(fb.framebuffer, None);
                self.dc.device.destroy_image_view(fb.swapchain_image_view, None);
            })
        );

        for image in swapchain_images {
            let view = {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(swapchain_image_format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                    );

                DropGuard::new(
                    unsafe { self.dc.device.create_image_view(&create_info, None) }?,
                    |view| unsafe { self.dc.device.destroy_image_view(*view, None); }
                )
            };

            let fb = {
                let create_info = vk::FramebufferCreateInfo::default()
                    .flags(vk::FramebufferCreateFlags::empty())
                    .render_pass(self.render_pass)
                    .attachments(std::array::from_ref::<vk::ImageView>(&view))
                    .width(swapchain_extent.width)
                    .height(swapchain_extent.height)
                    .layers(1);

                DropGuard::new(
                    unsafe { self.dc.device.create_framebuffer(&create_info, None) }?,
                    |fb| unsafe { self.dc.device.destroy_framebuffer(*fb, None) }
                )
            };

            new_framebuffers.push(FrameContext {
                swapchain_image: *image,
                swapchain_image_view: view.into_inner(),
                framebuffer: fb.into_inner(),
                ..Default::default()
            })
        }

        if self.frames.len() <= swapchain_images.len() {
            let mut new_frames = DropGuard::new(
                Vec::<FrameContext>::with_capacity(swapchain_images.len() - self.frames.len()),
                |fs| fs
                    .drain(..)
                    .for_each(|item| unsafe {
                        self.dc.device.free_command_buffers(
                            self.frame_command_pool,
                            std::array::from_ref(&item.command_buffer)
                        );
                        self.dc.device.destroy_fence(item.fence, None);
                        self.dc.device.destroy_semaphore(item.frame_acquired_semaphore.get(), None);
                        self.dc.device.destroy_semaphore(item.render_finished_semaphore, None);
                    })
            );

            for _new_frame_index in self.frames.len()..swapchain_images.len() {
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

                let make_semaphore = || -> Result<DropGuard<vk::Semaphore, _>, vk::Result> {
                    Ok(DropGuard::new(
                        unsafe { self.dc.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?,
                        |s| unsafe { self.dc.device.destroy_semaphore(*s, None) }
                    ))
                };

                let frame_acuqired_semaphore = make_semaphore()?;
                let render_finished_semaphore = make_semaphore()?;
                let transfer_finished_semaphore = make_semaphore()?;

                let fence = {
                    let create_info = vk::FenceCreateInfo::default()
                        .flags(vk::FenceCreateFlags::SIGNALED);

                    DropGuard::new(
                        unsafe { self.dc.device.create_fence(&create_info, None) }?,
                        |f| unsafe { self.dc.device.destroy_fence(*f, None) }
                    )
                };

                new_frames.push(FrameContext {
                    command_buffer: command_buffer.into_inner(),
                    fence: fence.into_inner(),
                    frame_acquired_semaphore: Cell::new(frame_acuqired_semaphore.into_inner()),
                    render_finished_semaphore: render_finished_semaphore.into_inner(),
                    transfer_finished_semaphore: transfer_finished_semaphore.into_inner(),
                    ..Default::default()
                })
            }

            // Append new frames
            self.frames.append(&mut new_frames);
        } else {
            // Remove old frames
            for frame in self.frames.drain(swapchain_images.len()..) {
                unsafe {
                    self.dc.device.destroy_framebuffer(frame.framebuffer, None);
                    self.dc.device.destroy_image_view(frame.swapchain_image_view, None);

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
        }

        // Assign new framebuffers
        for (frame, res) in Iterator::zip(self.frames.iter_mut(), new_framebuffers.into_inner().into_iter()) {
            unsafe {
                self.dc.device.destroy_framebuffer(frame.framebuffer, None);
                self.dc.device.destroy_image_view(frame.swapchain_image_view, None);
            }

            frame.framebuffer = res.framebuffer;
            frame.swapchain_image = res.swapchain_image;
            frame.swapchain_image_view = res.swapchain_image_view;
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
                0,
                std::slice::from_raw_parts(indices.as_ptr() as *const u8, ind_buf_size)
            )?;
        }

        let (buffer, allocation) = guarded_buffer_and_allocation.into_inner();

        Ok(Arc::new(Mesh {
            _dc: self.dc.clone(),
            allocator: self.allocator.clone(),
            buffer_allocation: allocation,
            buffer: buffer,
            _vertex_span: 0..vt_buf_size,
            _index_span: vt_buf_size..vt_buf_size + ind_buf_size,
            _index_count: indices.len(),
        }))
    }

    /// Create new instance
    pub fn create_instance(&self, mesh: Arc<Mesh>, material: Material) -> Result<Arc<Instance>, vk::Result> {
        Ok(Arc::new(Instance {
            _mesh: mesh,
            _material: material,
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
                unsafe {
                    self.resize_frames(
                        self.swapchain.images().to_owned().as_slice(),
                        self.swapchain.extent(),
                        self.swapchain.image_format()
                    )?;
                }
            }

            (guard, index)
        };

        let frame = &mut self.frames[swapchain_image_index as usize];
        frame.frame_acquired_semaphore.swap(&self.buffered_semaphore);

        // Wait for fences and reset'em
        unsafe {
            self.dc.device.wait_for_fences(std::array::from_ref(&frame.fence), true, u64::MAX)?;
            self.dc.device.reset_fences(std::array::from_ref(&frame.fence))?;
        }

        // Replace flush context with the new one
        frame.flush_context.replace(
            self.allocator.clone().flush(frame.transfer_finished_semaphore)?
        );
        frame.swapchain_guard.replace(swapchain_guard);

        // Reset command buffer used for rendering
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

        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.30, 0.47, 0.80, 0.0],
            }
        };

        let begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(frame.framebuffer)
            .render_area(vk::Rect2D::default()
                .offset(vk::Offset2D::default())
                .extent(self.swapchain.extent())
            )
            .clear_values(std::array::from_ref(&clear_value))
        ;

        unsafe {
            self.dc.device.cmd_begin_render_pass(
                frame.command_buffer,
                &begin_info,
                vk::SubpassContents::INLINE,
            )
        };

        // Fill render pass with... nothing!

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

        let swapchain_handle = unsafe { self.swapchain.handle() };
        let mut present_result = vk::Result::default();

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(std::array::from_ref(&frame.render_finished_semaphore))
            .image_indices(std::array::from_ref(&swapchain_image_index))
            .swapchains(std::array::from_ref(&swapchain_handle))
            .results(std::array::from_mut(&mut present_result));

        // Present and check for success
        unsafe { self.dc.device_swapchain.queue_present(self.dc.queue, &present_info) }?;

        Ok(())
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        unsafe {
            // Wait all device operations finish
            _ = self.dc.device.device_wait_idle();

            // Destroy all vulkan data
            _ = self.resize_frames(
                &[],
                self.swapchain.extent(),
                self.swapchain.image_format(),
            );
            self.dc.device.destroy_render_pass(self.render_pass, None);
            self.dc.device.destroy_semaphore(self.buffered_semaphore.get(), None);
            self.dc.device.destroy_command_pool(self.frame_command_pool, None);
        }
    }
}
