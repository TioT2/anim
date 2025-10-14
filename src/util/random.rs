//! Random generator implementation file

/// Splitmix64 based random generator (mainly used as an initializer for another generators)
pub struct Splitmix64(u64);

impl Splitmix64 {
    /// Create new splitmix64
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next splitmix64
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let r0 = self.0;
        let r1 = (r0 ^ (r0 >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        let r2 = (r1 ^ (r1 >> 27)).wrapping_mul(0x94D049BB133111EB);
        r2 ^ (r2 >> 31)
    }
}

/// Xoshiro256 based random generator (preferred if you need good uniform distribution)
pub struct Xoshiro256(u64, u64, u64, u64);

impl Xoshiro256 {
    /// Create new random generator
    pub fn new(seed: u64) -> Self {
        let mut splitmix = Splitmix64::new(seed);

        Self(splitmix.next_u64(), splitmix.next_u64(), splitmix.next_u64(), splitmix.next_u64())
    }

    /// Generate next u64
    pub fn next_u64(&mut self) -> u64 {
        let result = self.3
            .wrapping_add(self.0)
            .rotate_left(23)
            .wrapping_add(self.0);
        let t = self.1 << 17;

        self.2 ^= self.0;
        self.3 ^= self.1;
        self.1 ^= self.2;
        self.0 ^= self.3;

        self.2 ^= t;
        self.3 = self.3.rotate_left(45);

        result
    }

    /// Generate next unit 32-bit float
    pub fn next_unit_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
}
