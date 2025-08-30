//! Swapchain implementation

use std::{cell::Cell, sync::Arc};

use ash::vk::{self, Handle};

use crate::render::core::{CoreInitError, DeviceContext};

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

        // Swapchain will be created on the first frame
        Ok(Self {
            swapchain: vk::SwapchainKHR::null(),
            extent: vk::Extent2D::default(),
            images: Vec::new(),
            allow_image_clipping,
            surface_format,
            present_mode,
            dc,
            resize_request: Cell::new(true),
        })
    }

    /// Acquire next image from swapchain
    pub unsafe fn next_image(
        &mut self,
        semaphore: vk::Semaphore
    ) -> Result<(u32, bool), vk::Result> {
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

        Ok((image_index, resized))
    }

    /// Get swapchain image set
    pub unsafe fn images(&self) -> &[vk::Image] {
        &self.images
    }

    /// Format of the swapchain images
    pub const fn image_format(&self) -> vk::Format {
        self.surface_format.format
    }

    /// Get image extent
    pub const fn extent(&self) -> vk::Extent2D {
        self.extent
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
