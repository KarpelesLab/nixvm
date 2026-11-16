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
//!
//! How frames are made is a `compose::Compositor`'s business: on the CPU,
//! or — when the VM has a GPU — on the GPU, in the host GL context the
//! guest renders in, the frame read back only when the embedder takes it
//! ([`Display::frame`]) or, in the browser, blitted to the page's WebGL2
//! canvas without reading it back at all ([`Display::present`]).
//!
//! What it speaks: `wl_compositor` (v6), `wl_subcompositor`, `wl_shm`
//! (ARGB/XRGB/ABGR/XBGR 8888, RGB565), `wl_seat` (pointer and keyboard, a
//! US XKB keymap), `wl_output` and `zxdg_output_manager_v1`, `xdg_wm_base`
//! (toplevels, popups placed by their positioner), `wl_data_device_manager`
//! (a clipboard between guest clients), `zxdg_decoration_manager_v1`,
//! `wp_viewporter`, and — when the VM has a GPU — `zwp_linux_dmabuf_v1`
//! (v4), so GL clients show what they render on the virtual GPU (see
//! `dmabuf`). The window policy is a kiosk's: toplevels fill the
//! display, the newest on top with the keyboard; a dialog (a toplevel with
//! a parent) is centered at its own size.

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
    /// Scrolling, in pixels: positive `dy` scrolls down. About 50 pixels
    /// make one wheel step.
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

/// What the embedder's handle and the compositor share: the frame, and the
/// input on its way in.
struct State {
    width: u32,
    height: u32,
    /// What makes the frames.
    compose: server::Shared,
    /// The frame as last read from the compositor, `width * height` RGBA
    /// pixels, row-major.
    pixels: Vec<u8>,
    /// Bumped each time the compositor makes a new frame.
    generation: u64,
    /// The generation `pixels` holds.
    read: u64,
    /// The generation the embedder last took with [`Display::frame`].
    shown: u64,
    /// Input from the embedder not yet delivered to clients.
    input: Vec<Input>,
    /// How many times the embedder has taken a frame: each is a
    /// presentation, which fires the clients' frame callbacks.
    presented: u64,
    /// When it last did (monotonic ms), 0 if never.
    last_frame_ms: u64,
    /// How many clients are connected.
    clients: usize,
}

/// The compositor as the kernel runs it: the [`Service`] behind a
/// [`Display`].
struct Server {
    shared: Arc<Mutex<State>>,
    comp: server::Compositor,
    listener: usize,
    /// [`Kernel::io_events`] when the compositor last looked.
    last_io: u64,
    /// Whether to composite on the VM's GPU when it has one, and whether
    /// that was settled (it is once a client has connected).
    gl_wanted: bool,
    gl_settled: bool,
}

/// How often frame callbacks fire when no embedder takes frames (ms).
const IDLE_FRAME_MS: u64 = 16;

/// How long without [`Display::frame`] before the embedder counts as gone
/// and frame callbacks run on [`IDLE_FRAME_MS`] instead (ms).
const EMBEDDER_IDLE_MS: u64 = 100;

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
        let (width, height) = (width.clamp(1, 8192), height.clamp(1, 8192));
        let compose: server::Shared =
            Arc::new(Mutex::new(Box::new(compose::Software::new(width, height))));
        let st = State {
            width,
            height,
            compose,
            pixels: Vec::new(),
            generation: 1,
            read: 0,
            shown: 0,
            input: Vec::new(),
            presented: 0,
            last_frame_ms: 0,
            clients: 0,
        };
        Self {
            inner: Arc::new(Mutex::new(st)),
        }
    }

    /// Serve the guest of `kernel`: listen on `socket` (an absolute guest
    /// path) and install the compositor as one of the kernel's services.
    /// The keyboard's keymap is written next to it (`<socket>.xkb`).
    ///
    /// # Errors
    /// A negative errno if the socket can't be made (see
    /// [`Kernel::served_listen`]) or the keymap written.
    pub fn attach(&self, kernel: &Kernel, socket: &str) -> Result<(), i64> {
        let (w, h) = self.size();
        let shared = Arc::clone(&self.inner.lock().unwrap().compose);
        let mut comp = server::Compositor::new(w, h, shared);
        let km = keymap::keymap();
        comp.keymap_path = format!("{socket}.xkb");
        comp.keymap_len = u32::try_from(km.len()).unwrap_or(0);
        comp.format_table_path = format!("{socket}.formats");
        kernel.served_write_file(&comp.keymap_path, &km, 0o444)?;
        let listener = kernel.served_listen(socket)?;
        kernel.add_service(Box::new(Server {
            shared: Arc::clone(&self.inner),
            comp,
            listener,
            last_io: u64::MAX,
            // `NIXVM_DISPLAY_GL=0` keeps the CPU compositor with a GPU.
            gl_wanted: std::env::var("NIXVM_DISPLAY_GL").map_or(true, |v| v != "0"),
            gl_settled: false,
        }));
        Ok(())
    }

    /// The display's size in pixels.
    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        let st = self.inner.lock().unwrap();
        (st.width, st.height)
    }

    /// How many guest clients are connected.
    #[must_use]
    pub fn clients(&self) -> usize {
        self.inner.lock().unwrap().clients
    }

    /// A number that changes whenever the frame does.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.inner.lock().unwrap().generation
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
        st.presented += 1;
        st.last_frame_ms = server::now_ms().max(1);
        if st.shown == st.generation {
            return false;
        }
        st.shown = st.generation;
        st.read_frame();
        show(&st.pixels, st.width, st.height);
        true
    }

    /// Show the current frame on the embedder's GL surface, if it changed
    /// since the last call or `force` (the surface lost its contents), and
    /// count it presented as [`Self::frame`] does. Only a display
    /// compositing on the GPU, whose GL context draws to that surface (the
    /// browser's display canvas), can: false otherwise, and when nothing
    /// was drawn.
    #[must_use]
    pub fn present(&self, force: bool) -> bool {
        let mut st = self.inner.lock().unwrap();
        st.presented += 1;
        st.last_frame_ms = server::now_ms().max(1);
        if st.shown == st.generation && !force {
            return false;
        }
        let shown = st.compose.lock().unwrap().present();
        if shown {
            st.shown = st.generation;
        }
        shown
    }

    /// Which compositor makes the frames: `software`, or `gl` once it
    /// composites on the VM's GPU.
    #[must_use]
    pub fn compositor(&self) -> &'static str {
        let compose = Arc::clone(&self.inner.lock().unwrap().compose);
        compose.lock().unwrap().name()
    }

    /// The current frame as a PNG file.
    #[must_use]
    pub fn png(&self) -> Vec<u8> {
        let mut st = self.inner.lock().unwrap();
        st.read_frame();
        png::encode_rgba(&st.pixels, st.width, st.height)
    }
}

impl State {
    /// Bring `pixels` up to the compositor's latest frame.
    fn read_frame(&mut self) {
        if self.read == self.generation {
            return;
        }
        let n = self.width as usize * self.height as usize * 4;
        if self.pixels.len() != n {
            self.pixels.clear();
        }
        self.compose.lock().unwrap().read(&mut self.pixels);
        self.read = self.generation;
    }
}

impl Service for Server {
    fn service(&mut self, kernel: &Kernel) {
        let io = kernel.io_events();
        let (input, presented, last_frame_ms) = {
            let mut st = self.shared.lock().unwrap();
            (
                std::mem::take(&mut st.input),
                st.presented,
                st.last_frame_ms,
            )
        };
        if !self.gl_settled {
            self.settle_gl(kernel);
        }
        let now = server::now_ms();
        let comp = &mut self.comp;
        let frames = comp.frames_pending();
        let presented_since = frames && presented != comp.presented;
        let idle_due = frames
            && now >= comp.last_fire_ms + IDLE_FRAME_MS
            && now >= last_frame_ms + EMBEDDER_IDLE_MS;
        if io == self.last_io
            && input.is_empty()
            && !presented_since
            && !idle_due
            && !comp.scene_dirty
            && !comp.cursor_dirty
        {
            return;
        }
        self.last_io = io;
        // The frame was shown before whatever comes in now: its callbacks
        // go first, and commits read from here on wait for the next one.
        if presented_since {
            comp.fire_frames(presented, false);
        } else if idle_due {
            comp.fire_frames(0, true);
        }
        comp.presented = presented;
        comp.pump(kernel, self.listener);
        for ev in input {
            comp.input(ev);
        }
        comp.flush(kernel);
        let mut st = self.shared.lock().unwrap();
        let st = &mut *st;
        st.clients = comp.clients.len();
        if comp.scene_dirty || comp.cursor_dirty {
            st.width = comp.width;
            st.height = comp.height;
            comp.composite(&mut st.generation);
        }
    }
}

impl Server {
    /// Composite on the VM's GPU if it has one (it may be attached after
    /// the display, but before the guest's first client connects).
    fn settle_gl(&mut self, kernel: &Kernel) {
        if !self.gl_wanted {
            self.gl_settled = true;
            return;
        }
        if let Some(gl) = dmabuf::gpu_frames(kernel).and_then(|g| g.compositor()) {
            *self.comp.compose.lock().unwrap() = gl;
            self.comp.scene_dirty = true;
            self.gl_settled = true;
        } else if !self.comp.clients.is_empty() {
            self.gl_settled = true;
        }
    }
}

mod compose;
mod dmabuf;
#[cfg(feature = "gpu")]
mod gl;
mod keymap;
mod png;
mod protocol;
mod render;
mod seat;
mod server;
mod shell;
mod wire;
