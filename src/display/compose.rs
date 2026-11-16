//! Compositing: turning the surfaces' contents into the display's frame.
//!
//! The protocol side (`server`, `protocol`) decides *what* is shown — which
//! surfaces, where, at what size, under which cursor — and hands a
//! [`Compositor`] each surface's new pixels as they are committed and the
//! list of [`Draw`]s that make a frame. How the pixels are kept and blended
//! is the compositor's business:
//!
//! - [`Software`]: surfaces are premultiplied ARGB images in host memory and
//!   frames are blended on the CPU (`render::Canvas`). Always available.
//! - `display::gl::GlCompositor` (the `gpu` feature, when the VM has a GPU):
//!   surfaces are textures on the GPU's host GL context — a GPU buffer is
//!   copied there GPU-to-GPU, never read back — and frames are drawn into
//!   a framebuffer that is read back only when the embedder takes a frame
//!   (only the part that changed since it last did), or, in the browser,
//!   blitted to the page's canvas with no read-back at all.
//!
//! Both produce the same pixels for the same scene: nearest-neighbour
//! scaling by the same integer mapping, `over` with exact rounding.

use super::dmabuf::{DmaBuffer, GpuFrames};
use super::render::{self, Canvas, Image};

/// A rectangle: x, y, width, height.
pub type Rect = (i32, i32, i32, i32);

/// What a compositor keeps contents under: a surface (client, object id),
/// or [`ARROW`].
pub type Key = (u32, u32);

/// The default pointer's key (clients are numbered from 1).
pub const ARROW: Key = (0, 0);

/// One surface drawn into the frame: buffer rectangle `src` scaled to
/// `w` × `h` at `(x, y)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Draw {
    pub key: Key,
    pub src: Rect,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Draw {
    /// The display area it covers.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    #[must_use]
    pub fn area(&self) -> Rect {
        (self.x, self.y, self.w, self.h)
    }
}

/// Fills row `i` of a written rectangle with premultiplied ARGB pixels.
pub type Fill<'a> = &'a mut dyn FnMut(usize, &mut [u32]);

/// How a frame gets made. See the module docs.
pub trait Compositor: Send {
    /// What it is, for logs (`software`, `gl`).
    fn name(&self) -> &'static str;
    /// Surface `key`'s buffer is `size` (width, height) and `opaque` or
    /// not; `rect` of it (all of it if `None`) changed: `fill` writes each
    /// of its rows. Contents outside `rect` are kept if the size didn't
    /// change, else zero.
    fn write(
        &mut self,
        key: Key,
        size: (u32, u32),
        opaque: bool,
        rect: Option<Rect>,
        fill: Fill<'_>,
    );
    /// Surface `key` now shows GPU buffer `buf`: `rect` of it (all of it if
    /// `None`) changed. Returns false if the buffer couldn't be used.
    fn write_gpu(
        &mut self,
        key: Key,
        gpu: &dyn GpuFrames,
        buf: &DmaBuffer,
        rect: Option<Rect>,
    ) -> bool;
    /// Surface `key` has no content any more (its buffer was detached, or
    /// it is gone).
    fn remove(&mut self, key: Key);
    /// Make the frame: `width` × `height`, the background, `scene`'s
    /// surfaces bottom to top, then `cursor`. `scene_changed` is false when
    /// only the cursor moved since the last call.
    fn render(
        &mut self,
        width: u32,
        height: u32,
        scene: &[Draw],
        cursor: Option<Draw>,
        scene_changed: bool,
    );
    /// The frame as RGBA bytes into `out` (`width * height * 4`, row-major,
    /// alpha opaque). `out` holds the frame read last time, so only what
    /// changed since needs writing.
    fn read(&mut self, out: &mut Vec<u8>);
    /// Show the frame on the embedder's GL surface (the browser's display
    /// canvas, its drawing buffer the frame's size) without reading it
    /// back. False if this compositor can't.
    fn present(&mut self) -> bool {
        false
    }
}

/// The box covering both `a` and `b` (either may be empty).
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
#[must_use]
pub fn union(a: Option<Rect>, b: Rect) -> Option<Rect> {
    if b.2 <= 0 || b.3 <= 0 {
        return a;
    }
    let Some(a) = a else {
        return Some(b);
    };
    let x0 = a.0.min(b.0);
    let y0 = a.1.min(b.1);
    let x1 = (a.0 + a.2).max(b.0 + b.2);
    let y1 = (a.1 + a.3).max(b.1 + b.3);
    Some((x0, y0, x1 - x0, y1 - y0))
}

/// `r` clipped to a `width` × `height` frame; `None` if nothing is left.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub fn clip(r: Rect, width: u32, height: u32) -> Option<Rect> {
    let x0 = r.0.max(0);
    let y0 = r.1.max(0);
    let x1 = (r.0 + r.2).min(width as i32);
    let y1 = (r.1 + r.3).min(height as i32);
    (x0 < x1 && y0 < y1).then_some((x0, y0, x1 - x0, y1 - y0))
}

/// What changed between two frames: draws `prev` → `next` (each with the
/// content version it showed). A draw that moved, changed or came or went
/// damages where it was and where it is.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
#[must_use]
pub fn damage(prev: &[(Draw, u64)], next: &[(Draw, u64)]) -> Option<Rect> {
    let mut d = None;
    let n = prev.len().max(next.len());
    for i in 0..n {
        match (prev.get(i), next.get(i)) {
            (Some(a), Some(b)) if a == b => {}
            (a, b) => {
                if let Some(a) = a {
                    d = union(d, a.0.area());
                }
                if let Some(b) = b {
                    d = union(d, b.0.area());
                }
            }
        }
    }
    d
}

/// The CPU compositor.
pub struct Software {
    images: std::collections::HashMap<Key, Image>,
    width: u32,
    height: u32,
    /// The scene without the cursor, premultiplied ARGB.
    scene: Vec<u32>,
    /// The scene with it.
    out: Vec<u32>,
}

impl Software {
    /// A compositor whose frame is `width` × `height` of background.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let mut images = std::collections::HashMap::new();
        images.insert(ARROW, render::arrow());
        let mut s = Self {
            images,
            width: 0,
            height: 0,
            scene: Vec::new(),
            out: Vec::new(),
        };
        s.render(width, height, &[], None, true);
        s
    }

    fn draw(images: &std::collections::HashMap<Key, Image>, canvas: &mut Canvas<'_>, d: &Draw) {
        if let Some(img) = images.get(&d.key) {
            canvas.draw(img, d.src, d.x, d.y, d.w, d.h);
        }
    }
}

impl Compositor for Software {
    fn name(&self) -> &'static str {
        "software"
    }

    fn write(
        &mut self,
        key: Key,
        (w, h): (u32, u32),
        opaque: bool,
        rect: Option<Rect>,
        fill: Fill<'_>,
    ) {
        let img = self.images.entry(key).or_default();
        if img.width != w || img.height != h {
            img.width = w;
            img.height = h;
            img.pixels = vec![0; w as usize * h as usize];
        }
        img.opaque = opaque;
        #[allow(clippy::cast_sign_loss)]
        let (x0, y0, rw, rh) = match rect {
            Some((x, y, rw, rh)) => (x as usize, y as usize, rw as usize, rh as usize),
            None => (0, 0, w as usize, h as usize),
        };
        for r in 0..rh {
            let d0 = (y0 + r) * w as usize + x0;
            fill(r, &mut img.pixels[d0..d0 + rw]);
        }
    }

    #[allow(clippy::cast_sign_loss)]
    fn write_gpu(
        &mut self,
        key: Key,
        gpu: &dyn GpuFrames,
        buf: &DmaBuffer,
        rect: Option<Rect>,
    ) -> bool {
        let (w, h) = (buf.width as usize, buf.height as usize);
        // A bottom-up buffer is read whole and flipped.
        let (x0, y0, rw, rh) = match rect {
            Some((x, y, rw, rh)) if !buf.y_invert => {
                (x as usize, y as usize, rw as usize, rh as usize)
            }
            _ => (0, 0, w, h),
        };
        if rw == 0 || rh == 0 {
            return true;
        }
        let mut bytes = vec![0u8; rw * rh * 4];
        #[allow(clippy::cast_possible_truncation)]
        let r = (x0 as u32, y0 as u32, rw as u32, rh as u32);
        if !gpu.read(buf.res, r, &mut bytes) {
            return false;
        }
        let opaque = render::is_opaque(buf.format);
        let rect = Some((x0 as i32, y0 as i32, rw as i32, rh as i32));
        #[allow(clippy::cast_possible_wrap)]
        self.write(key, (buf.width, buf.height), opaque, rect, &mut |i, row| {
            let src = if buf.y_invert { rh - 1 - i } else { i };
            render::convert_row(buf.format, &bytes[src * rw * 4..(src + 1) * rw * 4], row);
        });
        true
    }

    fn remove(&mut self, key: Key) {
        if key != ARROW {
            self.images.remove(&key);
        }
    }

    fn render(
        &mut self,
        width: u32,
        height: u32,
        scene: &[Draw],
        cursor: Option<Draw>,
        scene_changed: bool,
    ) {
        let n = width as usize * height as usize;
        if scene_changed || width != self.width || height != self.height {
            self.width = width;
            self.height = height;
            self.scene.clear();
            self.scene.resize(n, super::server::BACKGROUND);
            let mut canvas = Canvas {
                width,
                height,
                pixels: &mut self.scene,
            };
            for d in scene {
                Self::draw(&self.images, &mut canvas, d);
            }
        }
        self.out.clear();
        self.out.extend_from_slice(&self.scene);
        if let Some(c) = cursor {
            let mut canvas = Canvas {
                width,
                height,
                pixels: &mut self.out,
            };
            Self::draw(&self.images, &mut canvas, &c);
        }
    }

    fn read(&mut self, out: &mut Vec<u8>) {
        render::to_rgba(&self.out, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn damage_covers_what_moved_or_changed() {
        let d = |x, y| Draw {
            key: (1, 1),
            src: (0, 0, 10, 10),
            x,
            y,
            w: 10,
            h: 10,
        };
        assert_eq!(damage(&[(d(0, 0), 1)], &[(d(0, 0), 1)]), None);
        // New content where it was.
        assert_eq!(
            damage(&[(d(0, 0), 1)], &[(d(0, 0), 2)]),
            Some((0, 0, 10, 10))
        );
        // Moved: both places.
        assert_eq!(
            damage(&[(d(0, 0), 1)], &[(d(20, 5), 1)]),
            Some((0, 0, 30, 15))
        );
        // Gone, or new.
        assert_eq!(damage(&[(d(5, 5), 1)], &[]), Some((5, 5, 10, 10)));
        assert_eq!(damage(&[], &[(d(5, 5), 1)]), Some((5, 5, 10, 10)));
        assert_eq!(clip((-5, -5, 10, 10), 4, 4), Some((0, 0, 4, 4)));
        assert_eq!(clip((5, 5, 10, 10), 4, 4), None);
    }

    #[test]
    fn software_composites_writes_and_cursor() {
        let mut c = Software::new(4, 2);
        let mut out = Vec::new();
        c.read(&mut out);
        assert_eq!(out.len(), 4 * 2 * 4);
        assert_eq!(&out[0..4], &[0x20, 0x24, 0x2c, 0xff], "background");
        // A 2x1 opaque red surface, then its second pixel green.
        c.write((1, 2), (2, 1), true, None, &mut |_, row| {
            row.fill(0xffff_0000);
        });
        c.write((1, 2), (2, 1), true, Some((1, 0, 1, 1)), &mut |_, row| {
            row.fill(0xff00_ff00);
        });
        let d = Draw {
            key: (1, 2),
            src: (0, 0, 2, 1),
            x: 1,
            y: 1,
            w: 2,
            h: 1,
        };
        c.render(4, 2, &[d], None, true);
        c.read(&mut out);
        assert_eq!(&out[20..24], &[0xff, 0, 0, 0xff]);
        assert_eq!(&out[24..28], &[0, 0xff, 0, 0xff]);
        // The arrow's tip (black) at (0, 0) over the scene.
        let a = Draw {
            key: ARROW,
            src: (0, 0, 12, 19),
            x: 0,
            y: 0,
            w: 12,
            h: 19,
        };
        c.render(4, 2, &[d], Some(a), false);
        c.read(&mut out);
        assert_eq!(&out[0..4], &[0, 0, 0, 0xff]);
        c.remove((1, 2));
        c.render(4, 2, &[d], None, true);
        c.read(&mut out);
        assert_eq!(&out[20..24], &[0x20, 0x24, 0x2c, 0xff], "removed");
    }
}
