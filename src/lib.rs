//! Animation project

pub mod render;
pub mod math;
pub mod util;

#[cfg(debug_assertions)]
pub const DEBUG_ASSERTIONS_ENABLED: bool = true;

#[cfg(not(debug_assertions))]
pub const DEBUG_ASSERTIONS_ENABLED: bool = false;
