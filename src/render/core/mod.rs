//! Renderer core

use std::{ffi::{CStr, CString}, sync::Arc};

use ash::vk;

use crate::math;

mod device_context;
pub use device_context::DeviceContext;

/// Standard vertex structure
pub struct Vertex {
    /// Position
    pub position: math::Mat<f32, 3>,

    /// Texture coordinate
    pub tex_coord: math::Mat<f32, 2>,

    /// Normal
    pub normal: math::Mat<f32, 3>,
}

/// Vulkan-compatible surface
pub trait WindowContext {
    /// Enumerate required instance extensions
    fn get_instance_extensions(&self) -> Result<Vec<CString>, String>;

    /// Create new surface
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


/// Core object of the renderer
pub struct Core {
    /// Initialize device context
    _dc: Arc<DeviceContext>,
}

impl Core {
    /// Construct instance
    pub fn new(
        window_context: Arc<dyn WindowContext>,
        application_name: Option<&CStr>
    ) -> Result<Self, CoreInitError> {
        let dc = DeviceContext::new(window_context, application_name)?;
        let dc = Arc::new(dc);

        Ok(Self {
            _dc: dc,
        })
    }
}
