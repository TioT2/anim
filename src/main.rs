use std::{ffi::CString, sync::Arc};

use anim::{math::{FMat, FVec}, render::core::Material};

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

    let mut random_generator = anim::util::random::Xoshiro256::new(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );

    let mut instances = {
        // Create mesh
        let text = std::fs::read_to_string(".temp/rei.obj").unwrap();
        let obj = anim::render::model_loader::parse(&text).unwrap();
        let mesh = render.create_core_mesh(&obj.vertices, &obj.indices);

        // Dimensions
        const COUNT_X: i32 = 5;
        const COUNT_Y: i32 = 5;
        const COUNT_Z: i32 = 5;

        let mut instances = Vec::with_capacity((COUNT_X * COUNT_Y * COUNT_Z) as usize);

        // XYZ triple iterator
        let xyz = std::iter::once(0)
            .flat_map(|i| std::iter::zip(std::iter::repeat(i), 0..COUNT_X))
            .flat_map(|i| std::iter::zip(std::iter::repeat(i), 0..COUNT_Y))
            .flat_map(|i| std::iter::zip(std::iter::repeat(i), 0..COUNT_Z))
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

                    if win_event == WinEvent::Close {
                        break 'main_loop;
                    }
                }
                Event::KeyDown { scancode, .. } => 'key_event: {
                    let Some(scancode) = scancode else {
                        break 'key_event;
                    };

                    if scancode == sdl2::keyboard::Scancode::R {
                        if instances.is_empty() {
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
}
