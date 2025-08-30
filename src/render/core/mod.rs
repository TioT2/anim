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
        // memory::Allocator,
        swapchain::Swapchain,
        util::DropGuard
    }
};

mod device_context;
// mod memory;
mod swapchain;
mod util;

/// Vertex format common for all models
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

    /// Metadata (tangent sign, ...)
    pub meta: u32,
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

// /// Material descriptor
// pub struct Material {
//     /// RGB color
//     pub base_color: [f32; 3],
//
//     /// Model metalness
//     pub metallic: f32,
//
//     /// Model roughness
//     pub roughness: f32,
// }

// /// Mesh - structure that holds some vertex data
// pub struct Mesh {
//     /// Device context reference
//     dc: Arc<DeviceContext>,

//     /// Allocator reference
//     allocator: Arc<Allocator>,

//     /// Mesh memory allocation
//     buffer_allocation: vk_mem::Allocation,

//     /// Mesh bufffer
//     buffer: vk::Buffer,

//     /// Vertex buffer memory region
//     vertex_span: std::ops::Range<usize>,

//     /// Index buffer memory region
//     index_span: std::ops::Range<usize>,

//     /// Count of the mesh indices
//     index_count: usize,
// }

// /// Rendered instance representation structure
// pub struct Instance {
//     /// Underlying mesh
//     mesh: Arc<Mesh>,

//     /// Instance transformation matrix
//     transform: Cell<math::FMat>,

//     /// Material structure
//     material: Material,
// }

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
struct FrameContext {
    /// Frame acquision semaphore (for frame output start)
    frame_acquired_semaphore: Cell<vk::Semaphore>,

    /// Semaphore that indicates rendering process end (for presentation)
    render_finished_semaphore: vk::Semaphore,

    /// Fence to wait then frame is reused
    fence: vk::Fence,

    /// Command buffer used for frame contents
    command_buffer: vk::CommandBuffer,

    /// Unique image for swapchain
    swapchain_image: vk::Image,

    /// View of the swapchain image
    swapchain_image_view: vk::ImageView,

    /// Main framebuffer
    framebuffer: vk::Framebuffer,
}

/// Core object of the renderer
pub struct Core {
    /// Context of the device
    dc: Arc<DeviceContext>,

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

        Ok(Self {
            frame_command_pool: frame_command_pool.into_inner(),
            render_pass: render_pass.into_inner(),
            buffered_semaphore: Cell::new(acquision_semaphore.into_inner()),
            frames: Vec::new(),
            swapchain,
            dc,
        })
    }

    /// Reize frame set to match amount of the swapchain images
    /// # TODO
    /// Refactor this sh*t giant.
    unsafe fn resize_frames_2(
        &mut self,
        swapchain_images: &[vk::Image],
        swapchain_extent: vk::Extent2D,
        swapchain_image_format: vk::Format,
    ) -> Result<(), vk::Result> {

        // Await all frames completion before updating target buffers contents
        if self.frames.len() != 0 {
            let frame_fences = self.frames.iter().map(|f| f.fence).collect::<Vec<_>>();
            unsafe { self.dc.device.wait_for_fences(&frame_fences, true, u64::MAX)? };
        }

        /// Set of surface resources to be created
        struct FrameRes {
            /// Swapchain image
            image: vk::Image,

            /// Swapchain image view
            view: vk::ImageView,

            /// Framebuffer
            fb: vk::Framebuffer,
        }

        // Destroy vector of the some resource
        let mut frame_res = DropGuard::new(
            Vec::<FrameRes>::new(),
            |resources| resources
                .drain(..)
                .for_each(|item| unsafe {
                    self.dc.device.destroy_framebuffer(item.fb, None);
                    self.dc.device.destroy_image_view(item.view, None);
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

            frame_res.push(FrameRes {
                image: *image,
                view: view.into_inner(),
                fb: fb.into_inner(),
            })
        }

        if self.frames.len() <= swapchain_images.len() {
            struct NewFrameRes {
                /// Command buffer
                command_buffer: vk::CommandBuffer,

                /// Fence
                fence: vk::Fence,

                /// Frame acquision semaphore
                frame_acquired_semaphore: vk::Semaphore,

                /// Render finish semaphore
                render_finished_semaphore: vk::Semaphore,
            }

            let mut new_frame_res = DropGuard::new(
                Vec::<NewFrameRes>::new(),
                |res| res
                    .drain(..)
                    .for_each(|item| unsafe {
                        self.dc.device.free_command_buffers(
                            self.frame_command_pool,
                            std::array::from_ref(&item.command_buffer)
                        );
                        self.dc.device.destroy_fence(item.fence, None);
                        self.dc.device.destroy_semaphore(item.frame_acquired_semaphore, None);
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

                let fence = {
                    let create_info = vk::FenceCreateInfo::default()
                        .flags(vk::FenceCreateFlags::SIGNALED);

                    DropGuard::new(
                        unsafe { self.dc.device.create_fence(&create_info, None) }?,
                        |f| unsafe { self.dc.device.destroy_fence(*f, None) }
                    )
                };

                new_frame_res.push(NewFrameRes {
                    command_buffer: command_buffer.into_inner(),
                    fence: fence.into_inner(),
                    frame_acquired_semaphore: frame_acuqired_semaphore.into_inner(),
                    render_finished_semaphore: render_finished_semaphore.into_inner()
                })
            }

            // Append new frames
            for frame in new_frame_res.into_inner().into_iter() {
                self.frames.push(FrameContext {
                    command_buffer: frame.command_buffer,
                    fence: frame.fence,
                    frame_acquired_semaphore: Cell::new(frame.frame_acquired_semaphore),
                    render_finished_semaphore: frame.render_finished_semaphore,
                    framebuffer: vk::Framebuffer::null(),
                    swapchain_image: vk::Image::null(),
                    swapchain_image_view: vk::ImageView::null(),
                });
            }
        } else {
            // Destroy frame resources
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
                    self.dc.device.destroy_semaphore(frame.render_finished_semaphore, None);
                }
            }
        }

        for (frame, res) in Iterator::zip(self.frames.iter_mut(), frame_res.into_inner().into_iter()) {
            // Destroy old resources
            unsafe {
                self.dc.device.destroy_image_view(frame.swapchain_image_view, None);
                self.dc.device.destroy_framebuffer(frame.framebuffer, None);
            }

            frame.swapchain_image = res.image;
            frame.swapchain_image_view = res.view;
            frame.framebuffer = res.fb;
        }

        Ok(())
    }

    /// Render next frame
    pub fn render_frame(&mut self) -> Result<(), vk::Result> {
        let frame_acquired_semaphore = self.buffered_semaphore.get();

        // Get swapchain image
        let swapchain_image_index = {
            let (index, resized) = unsafe {
                self.swapchain.next_image(frame_acquired_semaphore)?
            };

            // Resize if required
            if resized {
                unsafe {
                    self.resize_frames_2(
                        self.swapchain.images().to_owned().as_slice(),
                        self.swapchain.extent(),
                        self.swapchain.image_format()
                    )?;
                }
            }

            index
        };

        let frame = &self.frames[swapchain_image_index as usize];
        frame.frame_acquired_semaphore.swap(&self.buffered_semaphore);

        // Wait for fences and reset'em
        unsafe {
            self.dc.device.wait_for_fences(std::array::from_ref(&frame.fence), true, u64::MAX)?;
            self.dc.device.reset_fences(std::array::from_ref(&frame.fence))?;
        }

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

        let stage_flags = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(std::array::from_ref(&frame_acquired_semaphore))
            .signal_semaphores(std::array::from_ref(&frame.render_finished_semaphore))
            .wait_dst_stage_mask(std::array::from_ref(&stage_flags))
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
            _ = self.resize_frames_2(
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
