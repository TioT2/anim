//! Device context implementation file

use std::{collections::HashMap, ffi::{CStr, CString}, sync::Arc};

use ash::vk::{self, Handle};

use crate::render::core::{util::DropGuard, CoreInitError, WindowContext};

/// Constant structure that holds global low-level Vulkan objects, such as instance, device, queue, etc.
pub struct DeviceContext {
    /// Window context reference
    _wc: Arc<dyn WindowContext>,

    /// Vulkan entry (must outlive all vulkan structures)
    _entry: ash::Entry,

    /// Vulkan API version
    pub _api_version: u32,

    /// Vulkan instance
    pub instance: ash::Instance,

    /// Surface extension-level instance handle
    pub instance_surface: ash::khr::surface::Instance,

    /// Surface handle
    pub surface: vk::SurfaceKHR,

    /// Physical device
    pub physical_device: vk::PhysicalDevice,

    /// Main queue family index
    pub queue_family_index: u32,

    /// Device
    pub device: ash::Device,

    /// Swapchain extension specific device
    pub device_swapchain: ash::khr::swapchain::Device,

    /// Main queue
    pub queue: vk::Queue,
}

macro_rules! bitflags_to_string {
    ($flags: expr, $separator: expr, $($flag: expr => $name: expr),* $(,)?) => {
        {
            let flags = $flags;
            let mut result = String::new();

            $(
                if flags.contains($flag) {
                    if !result.is_empty() {
                        result.push_str($separator);
                    }
                    result.push_str($name);
                }
            )*

            result
        }
    };
}

/// Vulkan debug message handler
unsafe extern "system" fn debug_message_handler(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message_severity_string = bitflags_to_string!(message_severity, " | ",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR   => "error",
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "verbose",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO    => "info",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "warning",
    );

    let message_types_string = bitflags_to_string!(message_types, " | ",
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL                => "general",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION             => "validation",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE            => "performance",
        vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING => "device address binding",
    );

    println!("SEVERITY: {}", message_severity_string);
    println!("TYPES: {}", message_types_string);

    // Display callback
    if let Some(callback_data) = unsafe { p_callback_data.as_ref() } {
        if let Some(id) = unsafe { callback_data.message_id_name_as_c_str() } {
            println!("ID: {}", id.to_string_lossy());
        }
        if let Some(msg) = unsafe { callback_data.message_as_c_str() } {
            println!("MESSAGE: {}", msg.to_string_lossy());
        }
    }
    println!();

    // Follow by Vulkan spec
    false as vk::Bool32
}

impl DeviceContext {
    /// Get instance layer set
    fn get_instance_layers(
        entry: &ash::Entry
    ) -> Result<HashMap<CString, vk::LayerProperties>, CoreInitError> {
        unsafe { entry.enumerate_instance_layer_properties() }?
            .into_iter()
            .map(|layer| Ok((
                layer
                    .layer_name_as_c_str()
                    .map_err(CoreInitError::InvalidCStr)?
                    .to_owned(),
                layer
            )))
            .collect()
    }

    /// Get extensions available then layer is active
    fn get_instance_layer_extensions(
        entry: &ash::Entry,
        name: Option<&CStr>
    ) -> Result<HashMap<CString, u32>, CoreInitError> {
        unsafe { entry.enumerate_instance_extension_properties(name) }?
            .into_iter()
            .map(|ext| Ok((
                ext
                    .extension_name_as_c_str()
                    .map_err(CoreInitError::InvalidCStr)?
                    .to_owned(),
                ext.spec_version
            )))
            .collect()
    }

    /// Get device extension set
    fn get_device_extensions(
        instance: &ash::Instance,
        device: vk::PhysicalDevice
    ) -> Result<HashMap<CString, u32>, CoreInitError> {
        unsafe { instance.enumerate_device_extension_properties(device) }?
            .into_iter()
            .map(|ext| Ok((
                ext
                    .extension_name_as_c_str()
                    .map_err(CoreInitError::InvalidCStr)?
                    .to_owned(),
                ext.spec_version
            )))
            .collect()
    }

    /// Validate required extension and layer sets
    fn validate_instance_reqs(
        entry: &ash::Entry,
        layers: &[CString],
        extensions: &[CString]
    ) -> Result<(), CoreInitError> {
        // Get available layeres
        let available_layers = Self::get_instance_layers(entry)?;

        // Check all layers for being present
        for layer in layers {
            if !available_layers.contains_key(layer) {
                return Err(CoreInitError::InstanceLayerNotPresent(layer.clone()));
            }
        }

        // Get available extensions and filter them by layers
        let available_extensions = Self::get_instance_layer_extensions(entry, None)?;

        // Check all extensions for being present
        for extension in extensions {
            if !available_extensions.contains_key(extension) {
                return Err(CoreInitError::InstanceExtensionNotPresent(extension.clone()));
            }
        }

        Ok(())
    }

    /// Pick queue that can be used as main
    fn find_main_queue_family_index(
        instance: &ash::Instance,
        instance_surface: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice
    ) -> Result<Option<u32>, CoreInitError> {
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_flags = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;

        for (index, family) in families.iter().enumerate() {
            let index = index as u32;
            if family.queue_flags & queue_flags != queue_flags {
                continue;
            }

            let surface_support = unsafe {
                instance_surface.get_physical_device_surface_support(
                    physical_device, index, surface)?
            };

            if !surface_support {
                continue;
            }

            return Ok(Some(index));
        }

        return Ok(None);
    }

    /// Pick physical device
    fn pick_physical_device(
        instance: &ash::Instance,
        instance_surface: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
        required_extensions: &[CString],
    ) -> Result<(vk::PhysicalDevice, u32), CoreInitError> {
        let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
        let mut best_physial_device = None;

        for physical_device in physical_devices {
            let surface_formats = unsafe {
                instance_surface.get_physical_device_surface_formats(physical_device, surface)?
            };

            let required_format = vk::SurfaceFormatKHR {
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                format: vk::Format::B8G8R8A8_SRGB,
            };

            if !surface_formats.contains(&required_format) {
                continue;
            }

            let present_modes = unsafe {
                instance_surface.get_physical_device_surface_present_modes(physical_device, surface)?
            };

            if !present_modes.contains(&vk::PresentModeKHR::FIFO) {
                continue;
            }

            let device_extensions = Self::get_device_extensions(instance, physical_device)?;

            // Skip devices without required extensions
            if !required_extensions.iter().all(|ext| device_extensions.contains_key(ext)) {
                continue;
            }

            let main_queue = Self::find_main_queue_family_index(
                instance,
                instance_surface,
                surface,
                physical_device
            )?;
            let Some(main_queue) = main_queue else {
                continue;
            };

            let props = unsafe { instance.get_physical_device_properties(physical_device) };
            let rate = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 4,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                _ => 1,
            };

            // Device descriptor
            let desc = (rate, physical_device, main_queue);

            let Some((prev_rate, _, _)) = best_physial_device else {
                best_physial_device = Some(desc);
                continue;
            };
            if rate > prev_rate {
                best_physial_device = Some(desc);
            }
        }

        match best_physial_device {
            Some((_, device, main_queue)) => Ok((device, main_queue)),
            None => Err(CoreInitError::SuitablePhysicalDeviceMissing)
        }
    }

    fn create_instance(
        entry: &ash::Entry,
        application_name: Option<&CStr>,
        layers: &[CString],
        extensions: &[CString],
        api_version: u32
    ) -> Result<ash::Instance, CoreInitError> {
        let mut debug_messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::empty()
                // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
            )
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::empty()
                // | vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING
                | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
            )
            .user_data(std::ptr::null_mut())
            .pfn_user_callback(Some(debug_message_handler))
            ;

        let mut app_info = vk::ApplicationInfo::default()
            .engine_name(c"anim")
            .api_version(api_version);

        if let Some(application_name) = application_name {
            app_info = app_info.application_name(application_name);
        }

        let layer_names = layers
            .iter()
            .map(|layer| layer.as_ptr())
            .collect::<Vec<_>>();
        let extension_names = extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&extension_names)
            .push_next(&mut debug_messenger_info);

        Ok(unsafe { entry.create_instance(&instance_create_info, None) }?)
    }

    /// Create physical device
    fn create_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
        extensions: &[CString]
    ) -> Result<ash::Device, CoreInitError> {
        let extension_names = extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        let priority = 1.0;
        let queue_create_infos = [
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(std::array::from_ref(&priority))
        ];

        let device_create_info = vk::DeviceCreateInfo::default()
            .enabled_extension_names(&extension_names)
            .queue_create_infos(&queue_create_infos);

        Ok(unsafe { instance.create_device(physical_device, &device_create_info, None) }?)
    }

    /// Construct new device context
    pub fn new(
        window_context: Arc<dyn WindowContext>,
        applciation_name: Option<&CStr>
    ) -> Result<Self, CoreInitError> {
        let entry = unsafe { ash::Entry::load() }.map_err(CoreInitError::EntryLoadingError)?;

        let instance_layers = vec! [
            c"VK_LAYER_KHRONOS_validation".to_owned()
        ];

        let mut instance_extensions = window_context
            .get_instance_extensions()
            .map_err(CoreInitError::WindowContextError)?;
        instance_extensions.push(c"VK_EXT_debug_utils".to_owned());

        // Validate instance layers and extensions
        Self::validate_instance_reqs(
            &entry,
            &instance_layers,
            &instance_extensions
        )?;

        const API_VERSION: u32 = vk::API_VERSION_1_2;

        // Create instance and wrap it in RAII guard
        let instance = DropGuard::new(
            Self::create_instance(
                &entry,
                applciation_name,
                &instance_layers,
                &instance_extensions,
                API_VERSION
            )?,
            |i| unsafe { i.destroy_instance(None) }
        );

        let instance_surface = ash::khr::surface::Instance::new(&entry, &instance);

        // Create surface
        let surface = DropGuard::new(
            vk::SurfaceKHR::from_raw(window_context
                .create_surface(instance.handle().as_raw() as usize)
                .map_err(CoreInitError::WindowContextError)? as _
            ),
            |s: &mut vk::SurfaceKHR| unsafe { instance_surface.destroy_surface(*s, None) }
        );

        let device_extensions = vec! [
            c"VK_KHR_swapchain".to_owned()
        ];

        // Pick physical device
        let (physical_device, queue_family_index) = Self::pick_physical_device(
            &instance,
            &instance_surface,
            *surface,
            &device_extensions
        )?;

        // Create device
        let device = DropGuard::new(
            Self::create_device(&instance, physical_device, queue_family_index, &device_extensions)?,
            |d| unsafe { d.destroy_device(None); }
        );

        let device_swapchain = ash::khr::swapchain::Device::new(&instance, &device);

        // Get queue from device
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // let props = unsafe {
        //     instance.get_physical_device_format_properties(physical_device, vk::Format::R16G16B16A16_SNORM)
        // };
        // println!("properties: ");
        // // vk::FormatFeatureFlags::SAMPLED_IMAGE;
        // println!("    usages: {}", bitflags_to_string!(props.buffer_features, " | ",
        //     vk::FormatFeatureFlags::SAMPLED_IMAGE => "sampled_image",
        //     vk::FormatFeatureFlags::VERTEX_BUFFER => "vertex_buffer",
        //     vk::FormatFeatureFlags::COLOR_ATTACHMENT => "color_attachment",
        // ));

        Ok(Self {
            _wc: window_context,
            _entry: entry,
            _api_version: API_VERSION,
            surface: surface.into_inner(),
            instance: instance.into_inner(),
            instance_surface,
            physical_device,
            queue_family_index,
            device: device.into_inner(),
            device_swapchain,
            queue,
        })
    }
}

// Device context requires automatical destruction of it's resources
impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance_surface.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
