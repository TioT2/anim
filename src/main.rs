use std::{ffi::CString, sync::Arc};

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

    let instance = {
        let text = std::fs::read_to_string(".temp/rei.obj").unwrap();
        let obj = anim::render::model_loader::parse(&text).unwrap();
        let mesh = render.create_core_mesh(&obj.vertices, &obj.indices);

        render.create_core_instance(mesh, anim::render::core::Material {
            base_color: [0.30, 0.47, 0.80],
            metallic: 0.0,
            roughness: 0.0
        })
    };

    instance.enable();

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
                    match scancode {
                        sdl2::keyboard::Scancode::D => instance.disable(),
                        sdl2::keyboard::Scancode::E => instance.enable(),
                        _ => {}
                    }
                }
                Event::Quit { .. } => break 'main_loop,
                _ => {}
            }
        }

        // Render next frame (literally)
        render.next_frame();
    }

    // Instance **must be** disabled
    // instance.disable();
}
