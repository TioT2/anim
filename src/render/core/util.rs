//! Utilities for working with unsafe VulkanAPI

/// Local RAII wrapper for unsafe API objects
pub struct DropGuard<I, D: FnOnce(&mut I)> {
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
