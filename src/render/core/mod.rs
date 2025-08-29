//! Render component that performs actual low-level rendering (manages meshes, materials, instances, etc.)

use std::{cell::Cell, collections::VecDeque, ffi::{CStr, CString}, sync::Arc};

use ash::vk::{self, Handle};

use crate::{math::{self, FVec}, render::core::util::DropGuard};

mod device_context;
mod util;
pub use device_context::DeviceContext;

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

/// Frame storage descriptor
pub struct Swapchain {
    /// Device context reference
    dc: Arc<DeviceContext>,

    /// Format of the surface
    surface_format: vk::SurfaceFormatKHR,

    /// Presentation mode
    present_mode: vk::PresentModeKHR,

    /// Swapchain, actually
    swapchain: vk::SwapchainKHR,

    /// Extent of the swapchain images
    extent: vk::Extent2D,

    /// Is image clipping allowed or not
    allow_image_clipping: bool,

    /// Set of the swapchain images
    images: Vec<vk::Image>,

    /// If true, resize operation is requested.
    resize_request: Cell<bool>,
}

impl Swapchain {
    /// Pick surface format from surface format set
    fn pick_surface_format(dc: &DeviceContext) -> Result<vk::SurfaceFormatKHR, CoreInitError> {
        let surface_formats = unsafe { dc.instance_surface.get_physical_device_surface_formats(
            dc.physical_device,
            dc.surface
        ) }?;

        let required_surface_format = vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };

        // Pick surface format
        if surface_formats.contains(&required_surface_format) {
            Ok(required_surface_format)
        } else {
            Err(CoreInitError::SuitableSurfaceFormatMissing)
        }
    }

    /// Pick swapchain presentation mode
    fn pick_present_mode(dc: &DeviceContext) -> Result<vk::PresentModeKHR, CoreInitError> {
        // Get present modes
        let present_modes = unsafe {
            dc.instance_surface.get_physical_device_surface_present_modes(
                dc.physical_device,
                dc.surface
            )?
        };

        // Find one from suitable list
        [vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::FIFO]
            .into_iter()
            .find(|mode| present_modes.contains(&mode))
            .ok_or(CoreInitError::SuitablePresentModeMissing)
    }

    /// Create swapchain
    unsafe fn create_swapchain(&self) -> Result<(vk::SwapchainKHR, vk::Extent2D), vk::Result> {
        let surface_caps = unsafe {
            self.dc.instance_surface.get_physical_device_surface_capabilities(
                self.dc.physical_device,
                self.dc.surface
            )?
        };

        // Get image count
        let image_count = 3.clamp(
            surface_caps.min_image_count,
            if surface_caps.max_image_count == 0 { u32::MAX } else { surface_caps.max_image_count }
        );

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.dc.surface)
            .min_image_count(image_count)
            .image_format(self.surface_format.format)
            .image_color_space(self.surface_format.color_space)
            .image_extent(surface_caps.current_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(std::array::from_ref(&self.dc.queue_family_index))
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::INHERIT)
            .present_mode(self.present_mode)
            .clipped(self.allow_image_clipping)
            .old_swapchain(self.swapchain);

        Ok((
            unsafe { self.dc.device_swapchain.create_swapchain(&swapchain_create_info, None)? },
            surface_caps.current_extent
        ))
    }

    /// Create new swapchain
    pub fn new(dc: Arc<DeviceContext>, allow_image_clipping: bool) -> Result<Self, CoreInitError> {
        let surface_format = Self::pick_surface_format(&dc)?;
        let present_mode = Self::pick_present_mode(&dc)?;

        Ok(Self {
            swapchain: vk::SwapchainKHR::null(),
            extent: vk::Extent2D::default(),
            images: Vec::new(),
            allow_image_clipping,
            surface_format,
            present_mode,
            dc,
            resize_request: Cell::new(true), // Request resize on the first frame
        })
    }

    /// Acquire next image from swapchain
    pub unsafe fn next_image(
        &mut self,
        semaphore: vk::Semaphore
    ) -> Result<(u32, vk::Image, bool), vk::Result> {
        let resized = self.resize_request.get();

        if resized {
            // Recreate swapchain
            let (swapchain, extent) = unsafe { self.create_swapchain()? };

            // Destroy old swapchain
            unsafe { self.dc.device_swapchain.destroy_swapchain(self.swapchain, None) };

            // Update
            self.swapchain = swapchain;
            self.extent = extent;
            self.images = unsafe {
                self.dc.device_swapchain.get_swapchain_images(self.swapchain)?
            };
        }

        let (image_index, resize_request) = unsafe {
            self.dc.device_swapchain.acquire_next_image(
                self.swapchain,
                u64::MAX,
                semaphore,
                vk::Fence::null()
            )?
        };
        // Set resize flag if required
        self.resize_request.set(resize_request);

        Ok((image_index, self.images[image_index as usize], resized))
    }

    /// Format of the swapchain images
    pub const fn image_format(&self) -> vk::Format {
        self.surface_format.format
    }

    /// Count of images
    pub const fn image_count(&self) -> usize {
        self.images.len()
    }

    /// Get vulkan-level swapchain handle
    pub unsafe fn handle(&self) -> vk::SwapchainKHR {
        self.swapchain
    }
}

// Swapchain wrapper destructor
impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            if !self.swapchain.is_null() {
                self.dc.device_swapchain.destroy_swapchain(self.swapchain, None);
            }
        }
    }
}

/// Single in-flight frames
struct FrameContext {
    /// Frame acquision semaphore
    frame_acquired_semaphore: Cell<vk::Semaphore>,

    /// Semaphore that indicates rendering process end
    render_finished_semaphore: vk::Semaphore,

    /// Fence to wait then frame is reused
    fence: vk::Fence,

    /// Image of the swapchain image used as a frame rendering destination
    swapchain_image_index: Cell<u32>,

    /// Command buffer used for frame contents
    command_buffer: vk::CommandBuffer,
}

/// Core object of the renderer
pub struct Core {
    /// Context of the device
    dc: Arc<DeviceContext>,

    /// Swapchain
    swapchain: Swapchain,

    /// Command pool for per-frame operations
    frame_command_pool: vk::CommandPool,

    /// Unsignaled semaphore to use as a new frame semaphore
    buffered_semaphore: Cell<vk::Semaphore>,

    /// Set of frames
    frames: VecDeque<FrameContext>,
}

/// Rendering error occured
#[derive(Debug)]
pub enum CoreRendringError {
    /// Vulkan error
    VulkanError(vk::Result),

    /// Cannot acquire next image for rendering
    ImageAcquisionError,
}

impl From<vk::Result> for CoreRendringError {
    fn from(value: vk::Result) -> Self {
        Self::VulkanError(value)
    }
}

impl Core {
    /// Construct instance
    pub fn new(
        window_context: Arc<dyn WindowContext>,
        application_name: Option<&CStr>
    ) -> Result<Self, CoreInitError> {
        let dc = Arc::new(DeviceContext::new(window_context, application_name)?);
        let swapchain = Swapchain::new(dc.clone(), true)?;

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

        let mut result = Self {
            frame_command_pool: frame_command_pool.into_inner(),
            buffered_semaphore: Cell::new(acquision_semaphore.into_inner()),
            frames: VecDeque::new(),
            swapchain,
            dc,
        };

        // Resize frame set to be non-empty
        unsafe { result.resize_frames(2) }?;

        Ok(result)
    }

    /// Resize frame set (mainly to match amount of the swapchain frames)
    unsafe fn resize_frames(&mut self, new_amount: usize) -> Result<(), vk::Result> {
        if new_amount == self.frames.len() {
            return Ok(());
        }

        if new_amount < self.frames.len() {
            // Drain frames issued for destruction
            let frames = self.frames.drain(new_amount..).collect::<Vec<_>>();

            let fences = frames.iter()
                .map(|f| f.fence)
                .collect::<Vec<_>>();

            let command_buffers = frames.iter()
                .map(|f| f.command_buffer)
                .collect::<Vec<_>>();

            // Wait for frames finish and destroy corresponding command buffers
            unsafe {
                self.dc.device.wait_for_fences(&fences, true, u64::MAX)?;
                self.dc.device.free_command_buffers(self.frame_command_pool, &command_buffers);
            }

            // Destroy semaphores and fences
            for frame in frames {
                unsafe {
                    self.dc.device.destroy_fence(frame.fence, None);
                    self.dc.device.destroy_semaphore(frame.frame_acquired_semaphore.get(), None);
                    self.dc.device.destroy_semaphore(frame.render_finished_semaphore, None);
                }
            }
        } else {
            let buffer_count = new_amount - self.frames.len();

            // Create guarded command buffers
            let command_buffers = {
                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.frame_command_pool)
                    .command_buffer_count(buffer_count as u32)
                    .level(vk::CommandBufferLevel::PRIMARY);

                DropGuard::new(
                    unsafe { self.dc.device.allocate_command_buffers(&alloc_info)? },
                    |set| unsafe { self.dc.device.free_command_buffers(self.frame_command_pool, &set) }
                )
            };

            // Create infos
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();
            let fence_create_info = vk::FenceCreateInfo::default()
                .flags(vk::FenceCreateFlags::SIGNALED);

            let make_fence = || Ok(DropGuard::new(
                unsafe { self.dc.device.create_fence(&fence_create_info, None)? },
                |fence| unsafe { self.dc.device.destroy_fence(*fence, None) }
            ));
            let make_semaphore = || Ok(DropGuard::new(
                unsafe { self.dc.device.create_semaphore(&semaphore_create_info, None)? },
                |semaphore| unsafe { self.dc.device.destroy_semaphore(*semaphore, None); }
            ));

            // Create guarded fences and semaphores
            let data = (0..buffer_count)
                .map(|_| Ok((make_fence()?, make_semaphore()?, make_semaphore()?)))
                .collect::<Result<Vec<_>, vk::Result>>()?;

            // Compose command buffers and data
            self.frames.extend(command_buffers
                .into_inner()
                .into_iter()
                .zip(data.into_iter())
                .map(|(command_buffer, (fence, semaphore1, semaphore2))| FrameContext {
                    frame_acquired_semaphore: Cell::new(semaphore1.into_inner()),
                    render_finished_semaphore: semaphore2.into_inner(),
                    fence: fence.into_inner(),
                    swapchain_image_index: Cell::new(0),
                    command_buffer,
                })
            );
        }

        Ok(())
    }

    /// Render next frame
    pub fn render_frame(&mut self) -> Result<(), CoreRendringError> {
        let frame_acquired_semaphore = self.buffered_semaphore.get();

        // println!("{:016X}", frame_acquired_semaphore.as_raw());

        // Get swapchain image
        let (swapchain_image_index, swapchain_image) = {
            let (index, image, resized) = unsafe {
                self.swapchain.next_image(frame_acquired_semaphore)?
            };

            if resized {
                unsafe { self.resize_frames(self.swapchain.image_count()) }?;
            }

            (index, image)
        };

        let frame = &self.frames[swapchain_image_index as usize];
        frame.swapchain_image_index.set(swapchain_image_index);
        frame.frame_acquired_semaphore.swap(&self.buffered_semaphore);

        // Wait for fences and reset'em
        unsafe {
            self.dc.device.wait_for_fences(std::array::from_ref(&frame.fence), true, u64::MAX)?;
            self.dc.device.reset_fences(std::array::from_ref(&frame.fence))?;
        }

        // Get image view of the swapchain
        let _swapchain_image_view = {
            let create_info = vk::ImageViewCreateInfo::default()
                .components(vk::ComponentMapping::default())
                .format(self.swapchain.image_format())
                .image(swapchain_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .subresource_range(vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .base_mip_level(0)
                    .layer_count(1)
                    .level_count(1)
                );

            // Create view and wrap it with guard
            DropGuard::new(
                unsafe { self.dc.device.create_image_view(&create_info, None)? },
                |view| unsafe { self.dc.device.destroy_image_view(*view, None) }
            )
        };

        // Reset command buffer used for rendering
        unsafe {
            self.dc.device.reset_command_buffer(frame.command_buffer, vk::CommandBufferResetFlags::default())?
        };

        // Fill command buffer with nothing and submit it for fence
        unsafe {
            self.dc.device.begin_command_buffer(
                frame.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            )?;

            let image_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::NONE)
                .dst_access_mask(vk::AccessFlags::NONE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(self.dc.queue_family_index)
                .dst_queue_family_index(self.dc.queue_family_index)
                .image(swapchain_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_array_layer: 0,
                    base_mip_level: 0,
                    layer_count: 1,
                    level_count: 1,
                });

            // Transfer state of the image
            self.dc.device.cmd_pipeline_barrier(
                frame.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::array::from_ref(&image_barrier)
            );

            self.dc.device.end_command_buffer(frame.command_buffer)?;

            let stage_flags = vk::PipelineStageFlags::ALL_COMMANDS;
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(std::array::from_ref(&frame_acquired_semaphore))
                .signal_semaphores(std::array::from_ref(&frame.render_finished_semaphore))
                .wait_dst_stage_mask(std::array::from_ref(&stage_flags))
                .command_buffers(std::array::from_ref(&frame.command_buffer));

            self.dc.device.queue_submit(self.dc.queue, std::array::from_ref(&submit_info), frame.fence)?;
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
            _ = self.resize_frames(0);
            self.dc.device.destroy_semaphore(self.buffered_semaphore.get(), None);
            self.dc.device.destroy_command_pool(self.frame_command_pool, None);
        }
    }
}
