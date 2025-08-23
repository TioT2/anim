//! Renderer subsystem implementation file

use std::{ffi::CStr, sync::Arc};

pub mod core;

/// Render context structure
pub struct Render {
    /// Renderer main object
    _core: Arc<core::Core>,
}

/// Renderer construction error
#[derive(Debug)]
pub enum RenderInitError {
    /// Render core initialization failed
    CoreInitError(core::CoreInitError),
}

impl Render {
    /// Construct render
    pub fn new(
        window_context: Arc<dyn core::WindowContext>,
        application_name: Option<&CStr>
    ) -> Result<Self, RenderInitError> {
        let core = core::Core::new(window_context, application_name).map_err(RenderInitError::CoreInitError)?;
        let core = Arc::new(core);

        Ok(Self {
            _core: core,
        })
    }
}
