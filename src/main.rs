use std::{ffi::CString, sync::Arc};

use anim::{math::{FMat, FVec}, render::core::Material};

/// Random number generator
pub struct RandomGenerator(u64, u64, u64, u64);

impl RandomGenerator {
    /// Splitmix64 for initial state generation
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let r0 = *state;
        let r1 = (r0 ^ (r0 >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        let r2 = (r1 ^ (r1 >> 27)).wrapping_mul(0x94D049BB133111EB);
        r2 ^ (r2 >> 31)
    }

    /// Create new random generator
    pub fn new(mut seed: u64) -> Self {
        Self(
            Self::splitmix64(&mut seed),
            Self::splitmix64(&mut seed),
            Self::splitmix64(&mut seed),
            Self::splitmix64(&mut seed)
        )
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

/// Window newtype to implement render function
pub struct WindowContext(Arc<sdl2::video::Window>);

// Implement render window context for the window
impl anim::render::core::WindowContext for WindowContext {
    fn get_instance_extensions(&self) -> Result<Vec<CString>, String> {
        Ok(self.0
            .vulkan_instance_extensions()?
            .into_iter()
            .map(|str| {
                let mut bytes = str.as_bytes().to_owned();
                bytes.push(0);
                CString::from_vec_with_nul(bytes).unwrap()
            })
            .collect::<Vec<CString>>()
        )
    }

    fn create_surface(&self, instance: usize) -> Result<usize, String> {
        self.0
            .vulkan_create_surface(instance)
            .map(|item| item as usize)
    }
}

fn main() {
    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();
    let mut event_pump = sdl.event_pump().unwrap();

    let window = video
        .window("anim", 800, 600)
        .vulkan()
        .build()
        .unwrap();
    let window = Arc::new(window);

    // Initialize renderer
    let render = anim::render::Render::new(
        Arc::new(WindowContext(window.clone())),
        Some(c"anim")
    ).unwrap();

    let mut random_generator = RandomGenerator::new(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );

    // let instance = {
    //     let text = std::fs::read_to_string(".temp/rei.obj").unwrap();
    //     let obj = anim::render::model_loader::parse(&text).unwrap();
    //     let mesh = render.create_core_mesh(&obj.vertices, &obj.indices);

    //     render.create_core_instance(mesh, anim::render::core::Material {
    //         base_color: [0.30, 0.47, 0.80],
    //         metallic: 0.0,
    //         roughness: 0.0
    //     })
    // };

    let mut instances = {
        // Create mesh
        let text = std::fs::read_to_string(".temp/rei.obj").unwrap();
        let obj = anim::render::model_loader::parse(&text).unwrap();
        let mesh = render.create_core_mesh(&obj.vertices, &obj.indices);

        let mut instances = Vec::with_capacity(1000);

        // XYZ triple iterator
        let xyz = std::iter::once(0)
            .flat_map(|i| std::iter::zip(std::iter::repeat(i), 0..5))
            .flat_map(|i| std::iter::zip(std::iter::repeat(i), 0..5))
            .flat_map(|i| std::iter::zip(std::iter::repeat(i), 0..5))
            .map(|(((_, x), y), z)| (x, y, z));

        for (x, y, z) in xyz {
            let instance = render.create_core_instance(mesh.clone(), Material {
                base_color: [0.30, 0.47, 0.80],
                metallic: 0.2,
                roughness: 0.8
            });
            instance.set_transform(FMat::translate(FVec::new3(
                -x as f32 * 1.0,
                -y as f32 * 1.0,
                -z as f32 * 1.0,
            )));
            instance.enable();
            instances.push(instance);
        }

        instances
    };

    let mut fps_frame_count = 0;
    let mut fps_last_measure = std::time::Instant::now();

    'main_loop: loop {
        'event_loop: while let Some(event) = event_pump.poll_event() {
            type Event = sdl2::event::Event;

            match event {
                Event::Window { window_id, win_event, .. } => {
                    if window_id != window.id() {
                        continue 'event_loop;
                    }

                    type WinEvent = sdl2::event::WindowEvent;

                    match win_event {
                        WinEvent::Close => {
                            break 'main_loop
                        }
                        _ => {}
                    }
                }
                Event::KeyDown { scancode, .. } => 'key_event: {
                    let Some(scancode) = scancode else {
                        break 'key_event;
                    };

                    if scancode == sdl2::keyboard::Scancode::R {
                        if instances.len() == 0 {
                            break 'key_event;
                        }

                        let index = random_generator.next_u64() as usize % instances.len();
                        let value = instances.swap_remove(index as usize % instances.len());

                        value.disable();
                    }
                }
                Event::Quit { .. } => break 'main_loop,
                _ => {}
            }
        }

        fps_frame_count += 1;

        let time_now = std::time::Instant::now();
        let fps_duration = time_now.duration_since(fps_last_measure);

        if fps_duration >= std::time::Duration::from_secs(1) {
            let fps = fps_frame_count as f64 / fps_duration.as_secs_f64();
            fps_frame_count = 0;
            fps_last_measure = time_now;
            println!("FPS: {}", fps);
        }

        // Render next frame (literally)
        render.next_frame();
    }

    // Instance **must be** disabled
    // instance.disable();
}
