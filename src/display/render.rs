//! Pixels: converting clients' shared-memory buffers, and compositing
//! surfaces into the frame.
//!
//! Everything is kept as premultiplied ARGB in a `u32` (`0xAARRGGBB`), the
//! layout of `wl_shm`'s ARGB8888 read as a little-endian word; the frame is
//! converted to RGBA bytes only when it is handed out.

/// `wl_shm` formats nixvm reads (the two every compositor must, and a few
/// more that cost a line each).
pub const ARGB8888: u32 = 0;
pub const XRGB8888: u32 = 1;
pub const ABGR8888: u32 = 0x3432_4241;
pub const XBGR8888: u32 = 0x3432_4258;
pub const RGB565: u32 = 0x3631_4752;

/// The formats advertised, in `wl_shm.format` order.
pub const FORMATS: &[u32] = &[ARGB8888, XRGB8888, ABGR8888, XBGR8888, RGB565];

/// Bytes per pixel of a supported format, `None` if unsupported.
pub fn bytes_per_pixel(format: u32) -> Option<usize> {
    match format {
        ARGB8888 | XRGB8888 | ABGR8888 | XBGR8888 => Some(4),
        RGB565 => Some(2),
        _ => None,
    }
}

/// Convert row `src` of `format` pixels into `dst` (premultiplied ARGB).
pub fn convert_row(format: u32, src: &[u8], dst: &mut [u32]) {
    match format {
        ARGB8888 => {
            for (d, s) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
                *d = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
            }
        }
        XRGB8888 => {
            for (d, s) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
                *d = u32::from_le_bytes([s[0], s[1], s[2], 0xff]);
            }
        }
        ABGR8888 => {
            for (d, s) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
                *d = u32::from_le_bytes([s[2], s[1], s[0], s[3]]);
            }
        }
        XBGR8888 => {
            for (d, s) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
                *d = u32::from_le_bytes([s[2], s[1], s[0], 0xff]);
            }
        }
        RGB565 => {
            for (d, s) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                let v = u32::from(u16::from_le_bytes([s[0], s[1]]));
                let r = (v >> 11) & 0x1f;
                let g = (v >> 5) & 0x3f;
                let b = v & 0x1f;
                let r = (r << 3) | (r >> 2);
                let g = (g << 2) | (g >> 4);
                let b = (b << 3) | (b >> 2);
                *d = 0xff00_0000 | (r << 16) | (g << 8) | b;
            }
        }
        _ => {}
    }
}

/// Whether pixels of `format` are always opaque.
pub fn is_opaque(format: u32) -> bool {
    matches!(format, XRGB8888 | XBGR8888 | RGB565)
}

/// A surface's content: the last buffer committed to it, copied.
#[derive(Debug, Clone, Default)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
    pub opaque: bool,
}

/// The frame being composited.
pub struct Canvas<'a> {
    pub width: u32,
    pub height: u32,
    pub pixels: &'a mut [u32],
}

/// `src` over `dst`, both premultiplied.
#[inline]
fn over(src: u32, dst: u32) -> u32 {
    let a = src >> 24;
    if a == 0xff {
        return src;
    }
    if a == 0 {
        return dst;
    }
    let inv = 255 - a;
    // Two channels at a time: 0x00RR00BB and 0x00AA00GG.
    let rb = (((dst & 0x00ff_00ff) * inv + 0x0080_0080) >> 8) & 0x00ff_00ff;
    let ag = ((((dst >> 8) & 0x00ff_00ff) * inv + 0x0080_0080) >> 8) & 0x00ff_00ff;
    let d = rb | (ag << 8);
    // Premultiplied channels never overflow: src + dst*(1-a) <= 255.
    src.wrapping_add(d)
}

impl Canvas<'_> {
    /// Draw `img`'s source rectangle `src` (buffer pixels: x, y, w, h) at
    /// `(x, y)` scaled to `dw` × `dh`, clipped to the canvas.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        clippy::too_many_arguments
    )]
    pub fn draw(
        &mut self,
        img: &Image,
        src: (i32, i32, i32, i32),
        x: i32,
        y: i32,
        dw: i32,
        dh: i32,
    ) {
        let (sx, sy, sw, sh) = src;
        if dw <= 0 || dh <= 0 || sw <= 0 || sh <= 0 || img.width == 0 {
            return;
        }
        let (cw, ch) = (self.width as i32, self.height as i32);
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = (x + dw).min(cw);
        let y1 = (y + dh).min(ch);
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let iw = img.width as i32;
        let ih = img.height as i32;
        let same = sw == dw && sh == dh;
        for py in y0..y1 {
            let v = if same {
                sy + (py - y)
            } else {
                sy + ((i64::from(py - y) * i64::from(sh)) / i64::from(dh)) as i32
            };
            if v < 0 || v >= ih {
                continue;
            }
            let row = &img.pixels[(v * iw) as usize..((v + 1) * iw) as usize];
            let out = &mut self.pixels[(py * cw) as usize..((py + 1) * cw) as usize];
            if same {
                let u0 = sx + (x0 - x);
                // Clip to the image too.
                let ua = u0.max(0);
                let ub = (u0 + (x1 - x0)).min(iw);
                if ua >= ub {
                    continue;
                }
                let o0 = (x0 + (ua - u0)) as usize;
                let n = (ub - ua) as usize;
                let s = &row[ua as usize..ua as usize + n];
                let d = &mut out[o0..o0 + n];
                if img.opaque {
                    d.copy_from_slice(s);
                } else {
                    for (dp, &sp) in d.iter_mut().zip(s) {
                        *dp = over(sp, *dp);
                    }
                }
            } else {
                for px in x0..x1 {
                    let u = sx + ((i64::from(px - x) * i64::from(sw)) / i64::from(dw)) as i32;
                    if u < 0 || u >= iw {
                        continue;
                    }
                    let sp = row[u as usize];
                    let dp = &mut out[px as usize];
                    *dp = if img.opaque { sp } else { over(sp, *dp) };
                }
            }
        }
    }
}

/// Convert premultiplied ARGB to straight RGBA bytes (alpha forced opaque:
/// the frame is the display, there's nothing behind it).
pub fn to_rgba(src: &[u32], dst: &mut Vec<u8>) {
    dst.clear();
    dst.reserve(src.len() * 4);
    for &p in src {
        dst.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, p as u8, 0xff]);
    }
}

/// The default pointer: an arrow, black-edged white, hotspot at its tip.
const ARROW: &[&str] = &[
    "X           ",
    "XX          ",
    "X.X         ",
    "X..X        ",
    "X...X       ",
    "X....X      ",
    "X.....X     ",
    "X......X    ",
    "X.......X   ",
    "X........X  ",
    "X.........X ",
    "X......XXXXX",
    "X...X..X    ",
    "X..XX..X    ",
    "X.X  X..X   ",
    "XX   X..X   ",
    "X     X..X  ",
    "      X..X  ",
    "       XX   ",
];

/// [`ARROW`] as an image.
#[allow(clippy::cast_possible_truncation)]
pub fn arrow() -> Image {
    let w = ARROW[0].len();
    let mut pixels = Vec::with_capacity(w * ARROW.len());
    for row in ARROW {
        for c in row.bytes() {
            pixels.push(match c {
                b'X' => 0xff00_0000,
                b'.' => 0xffff_ffff,
                _ => 0,
            });
        }
    }
    Image {
        width: w as u32,
        height: ARROW.len() as u32,
        pixels,
        opaque: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_convert_to_premultiplied_argb() {
        let mut d = [0u32; 1];
        convert_row(ARGB8888, &[0x10, 0x20, 0x30, 0x80], &mut d);
        assert_eq!(d[0], 0x8030_2010);
        convert_row(XRGB8888, &[0x10, 0x20, 0x30, 0x00], &mut d);
        assert_eq!(d[0], 0xff30_2010);
        convert_row(ABGR8888, &[0x30, 0x20, 0x10, 0x40], &mut d);
        assert_eq!(d[0], 0x4030_2010);
        convert_row(RGB565, &0xf800u16.to_le_bytes(), &mut d);
        assert_eq!(d[0], 0xffff_0000);
    }

    #[test]
    fn blending_is_source_over() {
        assert_eq!(over(0xff12_3456, 0xff00_0000), 0xff12_3456);
        assert_eq!(over(0x0000_0000, 0xffab_cdef), 0xffab_cdef);
        // Half-transparent white (premultiplied 0x80) over black.
        assert_eq!(over(0x8080_8080, 0xff00_0000), 0xff80_8080);
        // ... over white stays white.
        assert_eq!(over(0x8080_8080, 0xffff_ffff), 0xffff_ffff);
    }

    #[test]
    fn draw_clips_and_scales() {
        let img = Image {
            width: 2,
            height: 2,
            pixels: vec![
                1 | 0xff00_0000,
                2 | 0xff00_0000,
                3 | 0xff00_0000,
                4 | 0xff00_0000,
            ],
            opaque: true,
        };
        let mut px = vec![0u32; 16];
        let mut c = Canvas {
            width: 4,
            height: 4,
            pixels: &mut px,
        };
        // Partly off the top-left: only the bottom-right pixel lands.
        c.draw(&img, (0, 0, 2, 2), -1, -1, 2, 2);
        assert_eq!(c.pixels[0] & 0xff, 4);
        assert_eq!(c.pixels[1], 0);
        // Scaled 2x into the bottom-right quarter.
        c.draw(&img, (0, 0, 2, 2), 2, 2, 2, 2);
        c.draw(&img, (0, 0, 2, 2), 0, 2, 4, 2);
        assert_eq!(c.pixels[8] & 0xff, 1);
        assert_eq!(c.pixels[9] & 0xff, 1);
        assert_eq!(c.pixels[10] & 0xff, 2);
        assert_eq!(c.pixels[12] & 0xff, 3);
    }

    #[test]
    fn rgba_output() {
        let mut out = Vec::new();
        to_rgba(&[0xff11_2233], &mut out);
        assert_eq!(out, [0x11, 0x22, 0x33, 0xff]);
        let a = arrow();
        assert_eq!(a.pixels.len() as u32, a.width * a.height);
        assert_eq!(a.pixels[0], 0xff00_0000, "the tip is the hotspot");
    }
}
