//! Fast vector and matrix implementation file

/// 4-component vector (may be used as 3-component)
#[derive(Copy, Clone)]
#[repr(C)]
pub struct FVec {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl std::ops::Index<usize> for FVec {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Invalid FVec index: {}", index)
        }
    }
}

impl FVec {
    /// Construct new fast vector
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Construct new fast vector
    pub fn new3(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, w: 0.0 }
    }

    /// X vector component
    pub fn x(self) -> f32 { self.x }

    /// Y vector component
    pub fn y(self) -> f32 { self.y }

    /// Z vector component
    pub fn z(self) -> f32 { self.z }

    /// W vector component
    pub fn w(self) -> f32 { self.w }

    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    pub fn length(self) -> f32 {
        Self::dot(self, self).sqrt()
    }

    pub fn normalized(self) -> Self {
        self * self.length().recip()
    }

    pub fn normalize(&mut self) {
        *self *= self.length().recip();
    }
}

macro_rules! impl_vec_operator {
    ($trait: ident, $fn: ident, $trait_assign: ident, $fn_assign: ident, $($x: ident),*) => {
        impl std::ops::$trait<Self> for FVec {
            type Output = Self;

            fn $fn(self, rhs: Self) -> Self {
                Self { $( $x: std::ops::$trait::$fn(self.$x, rhs.$x), )* }
            }
        }

        impl std::ops::$trait<f32> for FVec {
            type Output = Self;

            fn $fn(self, rhs: f32) -> Self {
                Self { $( $x: std::ops::$trait::$fn(self.$x, rhs), )* }
            }
        }

        impl std::ops::$trait_assign<Self> for FVec {
            fn $fn_assign(&mut self, rhs: Self) {
                $( self.$x.$fn_assign(rhs.$x); )*
            }
        }

        impl std::ops::$trait_assign<f32> for FVec {
            fn $fn_assign(&mut self, rhs: f32) {
                $( self.$x.$fn_assign(rhs); )*
            }
        }
    };
}

impl_vec_operator!(Add, add, AddAssign, add_assign, x, y, z, w);
impl_vec_operator!(Sub, sub, SubAssign, sub_assign, x, y, z, w);
impl_vec_operator!(Mul, mul, MulAssign, mul_assign, x, y, z, w);
impl_vec_operator!(Div, div, DivAssign, div_assign, x, y, z, w);

/// Column-major 4x4 floating-point matrix
#[derive(Copy, Clone)]
#[repr(C)]
pub struct FMat {
    /// Matrix contents
    pub data: [[f32; 4]; 4],
}

impl FMat {
    /// Construct 4x4 matrix from it's data
    pub fn new(data: [[f32; 4]; 4]) -> Self {
        Self { data }
    }

    /// Construct identity matrix
    pub fn identity() -> Self {
        Self::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }
}
