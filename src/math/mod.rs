//! Math (linear algebra, etc.) implementation module

mod mat;
pub use mat::*;

mod fast;
pub use fast::*;

// Construct column-vector
#[macro_export]
macro_rules! vector {
    ($($item: expr),+) => { crate::math::Mat::new_column([$($item),*]) }
}
pub use vector;
