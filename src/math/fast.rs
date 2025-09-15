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
    /// Construct new vector
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Construct new vector
    pub const fn new3(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, w: 0.0 }
    }

    /// Produce zero vector
    pub const fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }
    }

    /// X vector component
    pub const fn x(self) -> f32 { self.x }

    /// Y vector component
    pub const fn y(self) -> f32 { self.y }

    /// Z vector component
    pub const fn z(self) -> f32 { self.z }

    /// W vector component
    pub const fn w(self) -> f32 { self.w }

    pub const fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    /// Calculate cross product of self and rhs treated as XYZ 3-component vectors.
    pub const fn cross(self, rhs: Self) -> Self {
        Self::new3(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
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

macro_rules! fmat4_foreach {
    ($action: ident) => {
        $action!(0, 0); $action!(0, 1); $action!(0, 2); $action!(0, 3);
        $action!(1, 0); $action!(1, 1); $action!(1, 2); $action!(1, 3);
        $action!(2, 0); $action!(2, 1); $action!(2, 2); $action!(2, 3);
        $action!(3, 0); $action!(3, 1); $action!(3, 2); $action!(3, 3);
    };
}

impl FMat {
    /// Construct 4x4 matrix from it's data
    pub const fn new(data: [[f32; 4]; 4]) -> Self {
        Self { data }
    }

    /// Matrix multiplication function
    pub const fn mul(&self, rhs: &Self) -> Self {
        let mut res = Self { data: [[0.0; 4]; 4] };

        macro_rules! mul {
            ($i: expr, $j: expr) => {
                res.data[$i][$j] = 0.0
                    + self.data[0][$j] * rhs.data[$i][0]
                    + self.data[1][$j] * rhs.data[$i][1]
                    + self.data[2][$j] * rhs.data[$i][2]
                    + self.data[3][$j] * rhs.data[$i][3]
                ;
            }
        }
        fmat4_foreach!(mul);

        res
    }

    /// Construct identity matrix
    pub const fn identity() -> Self {
        Self::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Translate matrix for some distance
    pub const fn translate(d: FVec) -> Self {
        Self::new([
            [  1.0,   0.0,   0.0, 0.0],
            [  0.0,   1.0,   0.0, 0.0],
            [  0.0,   0.0,   1.0, 0.0],
            [d.x(), d.y(), d.z(), 1.0],
        ])
    }

    /// Calculate view matrix from normalized vectors
    pub const fn view_normalized(dir: FVec, right: FVec, up: FVec, loc: FVec) -> Self {
        Self::new([
            [right.x,         up.x,         -dir.x,       0.0],
            [right.y,         up.y,         -dir.y,       0.0],
            [right.z,         up.z,         -dir.z,       0.0],
            [-loc.dot(right), -loc.dot(up), loc.dot(dir), 1.0],
        ])
    }

    /// Produce view matrix
    pub fn view(loc: FVec, at: FVec, up: FVec) -> Self {
        let dir = (at - loc).normalized();
        let right = dir.cross(up).normalized();
        let up = right.cross(dir).normalized();

        Self::view_normalized(dir, right, up, loc)
    }

    /// Build frustum projection matrix for inverse-z trick
    pub const fn projection_frustum_invz(l: f32, r: f32, b: f32, t: f32, n: f32) -> Self {
        Self::new([
            [2.0 * n / (r - l),  0.0,               0.0,  0.0 ],
            [0.0,               -2.0 * n / (t - b), 0.0,  0.0 ],
            [(r + l) / (r - l), -(t + b) / (t - b), 0.0, -1.0 ],
            [0.0,                0.0,                 n,  0.0 ],
        ])
    }

    /// Transform interpreting this matrix as 3x3.
    pub const fn transform_vector(&self, v: FVec) -> FVec {
        FVec::new3(
            v.x * self.data[0][0] + v.y * self.data[1][0] + v.z * self.data[2][0],
            v.x * self.data[0][1] + v.y * self.data[1][1] + v.z * self.data[2][1],
            v.x * self.data[0][2] + v.y * self.data[1][2] + v.z * self.data[2][2],
        )
    }

    /// Transform interpreting this matrix as 4x3 (e.g. without perspective)
    pub const fn transform_point(&self, v: FVec) -> FVec {
        FVec::new3(
            v.x * self.data[0][0] + v.y * self.data[1][0] + v.z * self.data[2][0] + self.data[3][0],
            v.x * self.data[0][1] + v.y * self.data[1][1] + v.z * self.data[2][1] + self.data[3][1],
            v.x * self.data[0][2] + v.y * self.data[1][2] + v.z * self.data[2][2] + self.data[3][2],
        )
    }

    /// Perform 4x4 transformation
    pub const fn transform4x4(&self, v: FVec) -> FVec {
        let w_inv = (v.x * self.data[0][3] + v.y * self.data[1][3] + v.z * self.data[2][3] + self.data[3][3]).recip();

        FVec::new3(
            (v.x * self.data[0][0] + v.y * self.data[1][0] + v.z * self.data[2][0] + self.data[3][0]) * w_inv,
            (v.x * self.data[0][1] + v.y * self.data[1][1] + v.z * self.data[2][1] + self.data[3][1]) * w_inv,
            (v.x * self.data[0][2] + v.y * self.data[1][2] + v.z * self.data[2][2] + self.data[3][2]) * w_inv,
        )
    }

    /// Transform v as 4-component vector
    pub const fn transform(&self, v: FVec) -> FVec {
        FVec::new(
            v.x * self.data[0][0] + v.y * self.data[1][0] + v.z * self.data[2][0] + v.w * self.data[3][0],
            v.x * self.data[0][1] + v.y * self.data[1][1] + v.z * self.data[2][1] + v.w * self.data[3][1],
            v.x * self.data[0][2] + v.y * self.data[1][2] + v.z * self.data[2][2] + v.w * self.data[3][2],
            v.x * self.data[0][3] + v.y * self.data[1][3] + v.z * self.data[2][3] + v.w * self.data[3][3],
        )
    }
}
