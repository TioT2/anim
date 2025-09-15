//! Renderer animation subsystem implementation file

use std::{cell::RefCell, ffi::CStr, sync::Arc};

pub mod core;
pub mod model_loader;

/// Render context structure
pub struct Render {
    /// Renderer main object
    core: Arc<RefCell<core::Core>>,
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
        let core = Arc::new(RefCell::new(core));

        Ok(Self { core })
    }

    /// Render next frame
    pub fn next_frame(&self) {
        self.core.borrow_mut().render_frame().unwrap();
    }

    /// Create core-level mesh
    pub fn create_core_mesh(&self, vertices: &[core::Vertex], indices: &[u32]) -> Arc<core::Mesh> {
        self.core.borrow().create_mesh(vertices, indices).unwrap()
    }

    /// Create new core instance
    pub fn create_core_instance(
        &self,
        mesh: Arc<core::Mesh>,
        material: core::Material
    ) -> Arc<core::Instance> {
        self.core.borrow().create_instance(mesh, material).unwrap()
    }
}
