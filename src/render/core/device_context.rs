//! Device context implementation file

use std::{collections::HashMap, ffi::{CStr, CString}, sync::Arc};

use ash::vk::{self, Handle};

use crate::render::core::{CoreInitError, WindowContext};

/// Constant structure that holds global low-level Vulkan objects, such as instance, device, queue, etc.
pub struct DeviceContext {
    /// Window context reference
    pub wc: Arc<dyn WindowContext>,

    /// Instance
    pub entry: ash::Entry,

    /// Vulkan instance
    pub instance: ash::Instance,

    /// Surface extension-level instance handle
    pub instance_surface: ash::khr::surface::Instance,

    /// Surface handle
    pub surface: vk::SurfaceKHR,
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

/// Local RAII wrapper for unsafe API objects
struct DropGuard<I, D: FnOnce(&mut I)> {
    /// Guarded item
    item: I,

    /// Drop function
    drop: Option<D>,
}

impl<I, D: FnOnce(&mut I)> DropGuard<I, D> {
    /// Create new drop guard
    pub fn new(item: I, drop: D) -> Self {
        Self { item, drop: Some(drop) }
    }

    /// Convert drop guard into intenral type, drop 'drop' function
    pub fn into_inner(mut self) -> I {
        unsafe {
            let item = std::ptr::read(&self.item);

            // Destroy drop function
            std::ptr::drop_in_place(&mut self.drop);

            // Do not destroy self, as all of it's fields are already consumed
            std::mem::forget(self);

            item
        }
    }
}

impl<I, D: FnOnce(&mut I)> std::ops::Deref for DropGuard<I, D> {
    type Target = I;

    fn deref(&self) -> &Self::Target {
        &self.item
    }
}

impl<I, D: FnOnce(&mut I)> std::ops::DerefMut for DropGuard<I, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.item
    }
}

impl<I, D: FnOnce(&mut I)> Drop for DropGuard<I, D> {
    fn drop(&mut self) {
        if let Some(drop) = self.drop.take() {
            drop(&mut self.item);
        }
    }
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
            .collect::<Result<HashMap<CString, vk::LayerProperties>, CoreInitError>>()
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
            .collect::<Result<HashMap<CString, u32>, CoreInitError>>()
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

    /// Construct new device context
    pub fn new(
        window_context: Arc<dyn WindowContext>,
        applciation_name: Option<&CStr>
    ) -> Result<Self, CoreInitError> {
        let entry = unsafe { ash::Entry::load() }.map_err(CoreInitError::EntryLoadingError)?;

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
            .api_version(vk::API_VERSION_1_2);

        if let Some(application_name) = applciation_name {
            app_info = app_info.application_name(application_name);
        }

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

        let instance_layer_names = instance_layers
            .iter()
            .map(|layer| layer.as_ptr())
            .collect::<Vec<_>>();
        let instance_extension_names = instance_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&instance_layer_names)
            .enabled_extension_names(&instance_extension_names)
            .push_next(&mut debug_messenger_info);

        // Create instance and wrap it in Vulkan RAII guard
        let instance = DropGuard::new(
            unsafe { entry.create_instance(&instance_create_info, None) }?,
            |i: &mut ash::Instance| unsafe { i.destroy_instance(None) }
        );

        let instance_surface = ash::khr::surface::Instance::new(&entry, &instance);

        // Create surface and wrap it in Vulkan RAII guard
        let surface = DropGuard::new(
            vk::SurfaceKHR::from_raw(window_context
                .create_surface(instance.handle().as_raw() as usize)
                .map_err(CoreInitError::WindowContextError)? as _
            ),
            |s: &mut vk::SurfaceKHR| unsafe { instance_surface.destroy_surface(*s, None) }
        );

        Ok(Self {
            wc: window_context,
            entry,
            surface: surface.into_inner(),
            instance: instance.into_inner(),
            instance_surface,
        })
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe {
            self.instance_surface.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
