//! Generic RxC Mat implementation

use std::mem::MaybeUninit;

/// Column-major generic Mat type with consistent binary layout
#[repr(transparent)]
pub struct Mat<T, const R: usize, const C: usize = 1>(pub [[T; R]; C]);

impl<T: Clone, const R: usize, const C: usize> Clone for Mat<T, R, C> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Copy, const R: usize, const C: usize> Copy for Mat<T, R, C> {}

impl<T: Default, const R: usize, const C: usize> Default for Mat<T, R, C> {
    fn default() -> Self {
        Self(std::array::from_fn(|_| std::array::from_fn(|_| Default::default())))
    }
}

impl<T, const R: usize, const C: usize> Mat<T, R, C> {

    /// Count of matrix rows
    pub const ROW_COUNT: usize = R;

    /// Count of matrix columns
    pub const COLUMN_COUNT: usize = C;

    /// Map Mat type
    pub fn map<D, F: FnMut(T) -> D>(self, mut map_fn: F) -> Mat<D, R, C> {
        Mat::<D, R, C>(self.0.map(|col| col.map(&mut map_fn)))
    }
}

impl<T, const R: usize, const C: usize> Mat<MaybeUninit<T>, R, C> {
    /// New uninit matrix
    pub unsafe fn new_uninit() -> Self {
        Self(unsafe { MaybeUninit::<[[MaybeUninit<T>; R]; C]>::uninit().assume_init_read() })
    }

    /// New zeroed uninit matirx
    pub unsafe fn new_zeroed() -> Self {
        Self(unsafe { MaybeUninit::<[[MaybeUninit<T>; R]; C]>::zeroed().assume_init_read() })
    }

    /// Assume that matrix is initialized
    pub unsafe fn assume_init(self) -> Mat<T, R, C> {
        self.map(|u| unsafe { u.assume_init() })
    }
}

macro_rules! matrix_foreach_index {
    ($x: ident, $y: ident, $action: expr) => {
        for $x in 0..Self::COLUMN_COUNT {
            for $y in 0..Self::ROW_COUNT {
                $action
            }
        }
    };
}

macro_rules! matrix_impl_matrix_operator {
    ($trait_name: ident, $fn_name: ident, $assign_trait_name: ident, $assign_fn_name: ident) => {
        impl<
            T: std::ops::$trait_name<T, Output = T>,
            const R: usize,
            const C: usize
        > std::ops::$trait_name<Mat<T, R, C>> for Mat<T, R, C> {
            type Output = Self;

            fn $fn_name(self, rhs: Mat<T, R, C>) -> Self::Output {
                let lhs = self.map(MaybeUninit::new);
                let rhs = rhs.map(MaybeUninit::new);
                let mut uninit = unsafe { Mat::<MaybeUninit<T>, R, C>::new_uninit() };

                matrix_foreach_index!(x, y, {
                    uninit.0[x][y].write(unsafe {
                        std::ops::$trait_name::$fn_name(lhs.0[x][y].assume_init_read(), rhs.0[x][y].assume_init_read())
                    });
                });

                unsafe { uninit.assume_init() }
            }
        }

        impl<
            T: std::ops::$assign_trait_name<T>,
            const R: usize,
            const C: usize
        > std::ops::$assign_trait_name<Self> for Mat<T, R, C> {
            fn $assign_fn_name(&mut self, rhs: Self) {
                let src = rhs.map(MaybeUninit::new);
                matrix_foreach_index!(x, y, unsafe { self.0[x][y].$assign_fn_name(src.0[x][y].assume_init_read()) });
            }
        }
    }
}

macro_rules! matrix_impl_component_operator {
    ($trait_name: ident, $fn_name: ident, $assign_trait_name: ident, $assign_fn_name: ident) => {
        impl<
            F: Clone,
            T: std::ops::$assign_trait_name<F>,
            const R: usize,
            const C: usize
        > std::ops::$assign_trait_name<F> for Mat<T, R, C> {
            fn $assign_fn_name(&mut self, rhs: F) {
                for col in &mut self.0 {
                    for item in col {
                        item.$assign_fn_name(rhs.clone());
                    }
                }
            }
        }

        impl<
            F: Clone,
            T: std::ops::$trait_name<F, Output = T>,
            const R: usize,
            const C: usize
        > std::ops::$trait_name<F> for Mat<T, R, C> {
            type Output = Self;

            fn $fn_name(self, rhs: F) -> Self::Output {
                let mut uninit = self.map(MaybeUninit::new);
                matrix_foreach_index!(x, y, unsafe {
                    uninit.0[x][y].write(
                        std::ops::$trait_name::$fn_name(uninit.0[x][y].assume_init_read(), rhs.clone())
                    );
                });
                unsafe { uninit.assume_init() }
            }
        }
    };
}


// Implement additive operators
matrix_impl_matrix_operator!(Add, add, AddAssign, add_assign);
matrix_impl_matrix_operator!(Sub, sub, SubAssign, sub_assign);

// Implement multiplicative operators
matrix_impl_component_operator!(Mul, mul, MulAssign, mul_assign);
matrix_impl_component_operator!(Div, div, DivAssign, div_assign);

impl<T, const R: usize> Mat<T, R, 1> {
    /// Construct matrix from single column
    pub fn new_column(col: [T; R]) -> Self {
        Self([col])
    }

    pub fn dot(self, rhs: Self) -> T
    where
        T: std::ops::Mul<T, Output = T> + std::iter::Sum
    {
        let [lhs] = self.0;
        let [rhs] = rhs.0;

        std::iter::zip(lhs.into_iter(), rhs.into_iter())
            .map(|(l, r)| l * r)
            .sum()
    }
}

impl<T: Clone, const N: usize> Mat<T, N, N> {
    /// Construct diagonal matrix
    pub fn diagonal(diagonal: [T; N], zero: T) -> Self {
        let mut uninit = unsafe { Mat::<MaybeUninit<T>, N, N>::new_uninit() };
        matrix_foreach_index!(x, y, {
            uninit.0[x][y].write(if x == y {
                diagonal[x].clone()
            } else {
                zero.clone()
            });
        });
        unsafe { uninit.assume_init() }
    }
}
