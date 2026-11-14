//! nixvm's built-in display: a Wayland compositor that nixvm itself runs
//! for the guest. Guest clients connect to its socket (`/tmp/wayland-0` by
//! default, with `WAYLAND_DISPLAY` pointing there) as they would to any
//! compositor; they draw into shared-memory buffers (memfds passed over the
//! socket) and nixvm composites their windows into one RGBA frame that the
//! embedder shows — the browser demo on a `<canvas>`, the CLI as a PNG — and
//! feeds the user's keyboard and pointer back as Wayland input events.
//!
//! It is a [`Service`]: the schedulers give it a turn between guest slices,
//! so it works the same on one CPU, several, and in the single-threaded
//! browser. The embedder holds a [`Display`] handle (cheap to clone) to read
//! frames and push input from its own thread or event loop.

use std::sync::{Arc, Mutex};

use crate::kernel::{Kernel, Service};

/// Where the display listens unless told otherwise.
pub const DEFAULT_SOCKET: &str = "/tmp/wayland-0";

/// A user input event, in display pixels and Linux input codes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Input {
    /// The pointer moved to `(x, y)` (display pixels, may be fractional).
    PointerMove { x: f64, y: f64 },
    /// A pointer button changed: `button` is a Linux `BTN_*` code
    /// (`BTN_LEFT` = 0x110, `BTN_RIGHT` = 0x111, `BTN_MIDDLE` = 0x112).
    PointerButton { button: u32, pressed: bool },
    /// Scrolling, in pixels: positive `dy` scrolls down.
    Scroll { dx: f64, dy: f64 },
    /// The pointer left the display.
    PointerLeave,
    /// A key changed: `code` is a Linux evdev `KEY_*` code (`KEY_A` = 30).
    Key { code: u32, pressed: bool },
    /// The display (the embedder's viewport) is now `width` × `height`.
    Resize { width: u32, height: u32 },
}

/// A handle on the display: the embedder's side. Clones share one display.
#[derive(Clone)]
pub struct Display {
    inner: Arc<Mutex<State>>,
}

/// The compositor's state, shared between the embedder's handle and the
/// [`Service`] the kernel runs.
struct State {
    width: u32,
    height: u32,
    /// The composited frame, `width * height` RGBA pixels, row-major.
    pixels: Vec<u8>,
    /// Bumped each time `pixels` changes.
    generation: u64,
    /// The generation the embedder last took with [`Display::frame`].
    shown: u64,
    /// Input from the embedder not yet delivered to clients.
    input: Vec<Input>,
    /// The host listener (see [`Kernel::served_listen`]).
    listener: Option<usize>,
}

/// The background where no window covers the display.
const BACKGROUND: [u8; 4] = [0x20, 0x24, 0x2c, 0xff];

impl std::fmt::Debug for Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (w, h) = self.size();
        f.debug_struct("Display")
            .field("width", &w)
            .field("height", &h)
            .finish()
    }
}

impl Display {
    /// A display `width` × `height` pixels, not yet serving any guest.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let mut st = State {
            width,
            height,
            pixels: Vec::new(),
            generation: 1,
            shown: 0,
            input: Vec::new(),
            listener: None,
        };
        st.clear();
        Self {
            inner: Arc::new(Mutex::new(st)),
        }
    }

    /// Serve the guest of `kernel`: listen on `socket` (an absolute guest
    /// path) and install the compositor as one of the kernel's services.
    ///
    /// # Errors
    /// A negative errno if the socket can't be made (see
    /// [`Kernel::served_listen`]).
    pub fn attach(&self, kernel: &Kernel, socket: &str) -> Result<(), i64> {
        let listener = kernel.served_listen(socket)?;
        self.inner.lock().unwrap().listener = Some(listener);
        kernel.add_service(Box::new(self.clone()));
        Ok(())
    }

    /// The display's size in pixels.
    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        let st = self.inner.lock().unwrap();
        (st.width, st.height)
    }

    /// Queue an input event for the guest.
    pub fn input(&self, ev: Input) {
        self.inner.lock().unwrap().input.push(ev);
    }

    /// Hand `show` the current frame (RGBA, `width * height * 4` bytes,
    /// row-major) if it changed since the last call, and tell the guest's
    /// clients a frame was presented (their frame callbacks fire). Returns
    /// whether `show` was called.
    pub fn frame(&self, show: impl FnOnce(&[u8], u32, u32)) -> bool {
        let mut st = self.inner.lock().unwrap();
        if st.shown == st.generation {
            return false;
        }
        st.shown = st.generation;
        show(&st.pixels, st.width, st.height);
        true
    }

    /// The current frame as a PNG file.
    #[must_use]
    pub fn png(&self) -> Vec<u8> {
        let st = self.inner.lock().unwrap();
        png::encode_rgba(&st.pixels, st.width, st.height)
    }
}

impl State {
    fn clear(&mut self) {
        let n = self.width as usize * self.height as usize;
        self.pixels.clear();
        self.pixels.reserve(n * 4);
        for _ in 0..n {
            self.pixels.extend_from_slice(&BACKGROUND);
        }
        self.generation += 1;
    }
}

impl Service for Display {
    fn service(&mut self, kernel: &Kernel) {
        let mut st = self.inner.lock().unwrap();
        let Some(listener) = st.listener else {
            return;
        };
        // TODO(display): the Wayland compositor. For now, connections are
        // accepted and closed, and input is dropped.
        while let Some(conn) = kernel.served_accept(listener) {
            kernel.served_close(conn);
        }
        for ev in std::mem::take(&mut st.input) {
            if let Input::Resize { width, height } = ev {
                st.width = width;
                st.height = height;
                st.clear();
            }
        }
    }
}

mod png;
