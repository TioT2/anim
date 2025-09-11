//! Utilities for working with unsafe VulkanAPI

use std::mem::ManuallyDrop;

/// Local RAII wrapper for unsafe API objects
pub struct DropGuard<I, D: FnOnce(&mut I)> {
    /// Guarded item
    item: I,

    /// Drop function
    drop: ManuallyDrop<D>,
}

impl<I, D: FnOnce(&mut I)> DropGuard<I, D> {
    /// Create new drop guard
    pub fn new(item: I, drop: D) -> Self {
        Self { item, drop: ManuallyDrop::new(drop) }
    }

    /// Zip two drop guards into one
    pub fn zip<I2, D2: FnOnce(&mut I2)>(
        self,
        second: DropGuard<I2, D2>
    ) -> DropGuard<(I, I2), impl FnOnce(&mut (I, I2))> {
        let (i1, d1) = self.decompose();
        let (i2, d2) = second.decompose();

        DropGuard {
            drop: ManuallyDrop::new(move |(i1, i2): &mut (I, I2)| {
                d1(i1);
                d2(i2);
            }),
            item: (i1, i2)
        }
    }

    /// Decompose drop guard into pieces
    fn decompose(self) -> (I, D) {
        unsafe {
            // Consume fields
            let item = std::ptr::read(&self.item);
            let drop = std::ptr::read(&self.drop);

            // Do not destroy self, as all of it's fields are already consumed
            std::mem::forget(self);

            (item, ManuallyDrop::into_inner(drop))
        }
    }

    /// Convert drop guard into intenral type, do not apply drop function
    pub fn into_inner(self) -> I {
        self.decompose().0
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
        let drop = unsafe { ManuallyDrop::take(&mut self.drop) };
        drop(&mut self.item);
    }
}
