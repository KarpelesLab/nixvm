//! virgl formats (Gallium's `pipe_format` numbering) and how each one lives on
//! a GLES 3.0 / WebGL2 host: its texture internal format, the format/type its
//! pixels are uploaded and read back with, and its vertex-attribute shape.
//!
//! Only formats a GLES 3.0 host holds natively are here, plus the BGRA family
//! every window system wants: GLES has no BGRA texture, so those are stored as
//! RGBA and their red and blue are swapped on the CPU as pixels cross between
//! the guest's buffer and the host texture ([`Format::swap_rb`]). Which of
//! them the guest may use for what comes from the masks in the capset
//! ([`super::caps`]), built from this same table.

use super::gl;

/// What a format holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Normalized or float colour: sampled as `vec4`.
    Float,
    /// Unsigned integer colour: sampled as `uvec4`.
    Uint,
    /// Signed integer colour: sampled as `ivec4`.
    Sint,
    /// Depth only.
    Depth,
    /// Depth and stencil.
    DepthStencil,
}

/// One virgl format on the host. See the module docs.
#[derive(Debug, Clone, Copy)]
pub struct Format {
    pub id: u32,
    /// Bytes per pixel (a block of one pixel for every format here).
    pub bpp: u32,
    /// `glTexStorage*`'s internal format.
    pub internal: u32,
    /// The `format`/`type` pixels cross with (`glTexSubImage*`,
    /// `glReadPixels`).
    pub format: u32,
    pub ty: u32,
    pub kind: Kind,
    /// Stored as RGBA with red and blue swapped (a BGRA format).
    pub swap_rb: bool,
    /// Has no alpha channel of its own (`X8`): alpha reads as one.
    pub no_alpha: bool,
    /// May be rendered to.
    pub render: bool,
    /// As a vertex attribute: component count, GL type, normalized.
    pub vertex: Option<(u32, u32, bool)>,
}

const fn f(id: u32, bpp: u32, internal: u32, format: u32, ty: u32, kind: Kind) -> Format {
    Format {
        id,
        bpp,
        internal,
        format,
        ty,
        kind,
        swap_rb: false,
        no_alpha: false,
        render: true,
        vertex: None,
    }
}

const fn v(mut fm: Format, n: u32, ty: u32, norm: bool) -> Format {
    fm.vertex = Some((n, ty, norm));
    fm
}

const fn norender(mut fm: Format) -> Format {
    fm.render = false;
    fm
}

const fn bgr(mut fm: Format, no_alpha: bool) -> Format {
    fm.swap_rb = true;
    fm.no_alpha = no_alpha;
    fm
}

const fn xa(mut fm: Format) -> Format {
    fm.no_alpha = true;
    fm
}

/// A vertex-only format in BGRA order: fetched as RGBA, then the vertex
/// shader swaps red and blue (GLES has no BGRA vertex formats).
const fn vob(id: u32, bpp: u32, n: u32, ty: u32, norm: bool) -> Format {
    let mut fm = vo(id, bpp, n, ty, norm);
    fm.swap_rb = true;
    fm
}

/// A vertex-only format: never a texture.
const fn vo(id: u32, bpp: u32, n: u32, ty: u32, norm: bool) -> Format {
    Format {
        id,
        bpp,
        internal: 0,
        format: 0,
        ty: 0,
        kind: Kind::Float,
        swap_rb: false,
        no_alpha: false,
        render: false,
        vertex: Some((n, ty, norm)),
    }
}

use Kind::{Depth, DepthStencil, Float, Sint, Uint};
use gl::{BYTE, FLOAT, HALF_FLOAT, INT, SHORT, UNSIGNED_BYTE, UNSIGNED_INT, UNSIGNED_SHORT};

/// Every format the host holds, by virgl id.
pub const FORMATS: &[Format] = &[
    // The BGRA family, stored as RGBA.
    v(
        bgr(f(1, 4, gl::RGBA8, gl::RGBA, UNSIGNED_BYTE, Float), false),
        4,
        UNSIGNED_BYTE,
        true,
    ), // B8G8R8A8_UNORM
    bgr(f(2, 4, gl::RGBA8, gl::RGBA, UNSIGNED_BYTE, Float), true), // B8G8R8X8_UNORM
    bgr(
        f(100, 4, gl::SRGB8_ALPHA8, gl::RGBA, UNSIGNED_BYTE, Float),
        false,
    ), // B8G8R8A8_SRGB
    bgr(
        f(101, 4, gl::SRGB8_ALPHA8, gl::RGBA, UNSIGNED_BYTE, Float),
        true,
    ), // B8G8R8X8_SRGB
    f(7, 2, gl::RGB565, gl::RGB, gl::UNSIGNED_SHORT_5_6_5, Float), // B5G6R5_UNORM
    v(
        f(
            8,
            4,
            gl::RGB10_A2,
            gl::RGBA,
            gl::UNSIGNED_INT_2_10_10_10_REV,
            Float,
        ),
        4,
        gl::UNSIGNED_INT_2_10_10_10_REV,
        true,
    ), // R10G10B10A2_UNORM
    // Depth / stencil.
    f(
        16,
        2,
        gl::DEPTH_COMPONENT16,
        gl::DEPTH_COMPONENT,
        UNSIGNED_SHORT,
        Depth,
    ), // Z16_UNORM
    f(
        18,
        4,
        gl::DEPTH_COMPONENT32F,
        gl::DEPTH_COMPONENT,
        FLOAT,
        Depth,
    ), // Z32_FLOAT
    f(
        19,
        4,
        gl::DEPTH24_STENCIL8,
        gl::DEPTH_STENCIL,
        gl::UNSIGNED_INT_24_8,
        DepthStencil,
    ), // Z24_UNORM_S8_UINT
    f(
        21,
        4,
        gl::DEPTH_COMPONENT24,
        gl::DEPTH_COMPONENT,
        UNSIGNED_INT,
        Depth,
    ), // Z24X8_UNORM
    f(
        126,
        8,
        gl::DEPTH32F_STENCIL8,
        gl::DEPTH_STENCIL,
        gl::FLOAT_32_UNSIGNED_INT_24_8_REV,
        DepthStencil,
    ), // Z32_FLOAT_S8X24_UINT
    // 32-bit float.
    v(f(28, 4, gl::R32F, gl::RED, FLOAT, Float), 1, FLOAT, false),
    v(f(29, 8, gl::RG32F, gl::RG, FLOAT, Float), 2, FLOAT, false),
    v(
        norender(f(30, 12, gl::RGB32F, gl::RGB, FLOAT, Float)),
        3,
        FLOAT,
        false,
    ),
    v(
        f(31, 16, gl::RGBA32F, gl::RGBA, FLOAT, Float),
        4,
        FLOAT,
        false,
    ),
    // Vertex-only scaled/normalized 32-bit and 16/8-bit scaled forms.
    vo(32, 4, 1, UNSIGNED_INT, true),
    vo(33, 8, 2, UNSIGNED_INT, true),
    vo(34, 12, 3, UNSIGNED_INT, true),
    vo(35, 16, 4, UNSIGNED_INT, true),
    vo(36, 4, 1, UNSIGNED_INT, false),
    vo(37, 8, 2, UNSIGNED_INT, false),
    vo(38, 12, 3, UNSIGNED_INT, false),
    vo(39, 16, 4, UNSIGNED_INT, false),
    vo(40, 4, 1, INT, true),
    vo(41, 8, 2, INT, true),
    vo(42, 12, 3, INT, true),
    vo(43, 16, 4, INT, true),
    vo(44, 4, 1, INT, false),
    vo(45, 8, 2, INT, false),
    vo(46, 12, 3, INT, false),
    vo(47, 16, 4, INT, false),
    // 16-bit unorm/snorm/scaled (vertex only on GLES 3.0).
    vo(48, 2, 1, UNSIGNED_SHORT, true),
    vo(49, 4, 2, UNSIGNED_SHORT, true),
    vo(50, 6, 3, UNSIGNED_SHORT, true),
    vo(51, 8, 4, UNSIGNED_SHORT, true),
    vo(52, 2, 1, UNSIGNED_SHORT, false),
    vo(53, 4, 2, UNSIGNED_SHORT, false),
    vo(54, 6, 3, UNSIGNED_SHORT, false),
    vo(55, 8, 4, UNSIGNED_SHORT, false),
    vo(56, 2, 1, SHORT, true),
    vo(57, 4, 2, SHORT, true),
    vo(58, 6, 3, SHORT, true),
    vo(59, 8, 4, SHORT, true),
    vo(60, 2, 1, SHORT, false),
    vo(61, 4, 2, SHORT, false),
    vo(62, 6, 3, SHORT, false),
    vo(63, 8, 4, SHORT, false),
    // 8-bit unorm.
    v(
        f(64, 1, gl::R8, gl::RED, UNSIGNED_BYTE, Float),
        1,
        UNSIGNED_BYTE,
        true,
    ),
    v(
        f(65, 2, gl::RG8, gl::RG, UNSIGNED_BYTE, Float),
        2,
        UNSIGNED_BYTE,
        true,
    ),
    v(
        f(66, 3, gl::RGB8, gl::RGB, UNSIGNED_BYTE, Float),
        3,
        UNSIGNED_BYTE,
        true,
    ),
    v(
        f(67, 4, gl::RGBA8, gl::RGBA, UNSIGNED_BYTE, Float),
        4,
        UNSIGNED_BYTE,
        true,
    ),
    vo(69, 1, 1, UNSIGNED_BYTE, false),
    vo(70, 2, 2, UNSIGNED_BYTE, false),
    vo(71, 3, 3, UNSIGNED_BYTE, false),
    vo(72, 4, 4, UNSIGNED_BYTE, false),
    // 8-bit snorm.
    v(
        norender(f(74, 1, gl::R8_SNORM, gl::RED, BYTE, Float)),
        1,
        BYTE,
        true,
    ),
    v(
        norender(f(75, 2, gl::RG8_SNORM, gl::RG, BYTE, Float)),
        2,
        BYTE,
        true,
    ),
    v(
        norender(f(76, 3, gl::RGB8_SNORM, gl::RGB, BYTE, Float)),
        3,
        BYTE,
        true,
    ),
    v(
        norender(f(77, 4, gl::RGBA8_SNORM, gl::RGBA, BYTE, Float)),
        4,
        BYTE,
        true,
    ),
    vo(82, 1, 1, BYTE, false),
    vo(83, 2, 2, BYTE, false),
    vo(84, 3, 3, BYTE, false),
    vo(85, 4, 4, BYTE, false),
    // Half float.
    v(
        f(91, 2, gl::R16F, gl::RED, HALF_FLOAT, Float),
        1,
        HALF_FLOAT,
        false,
    ),
    v(
        f(92, 4, gl::RG16F, gl::RG, HALF_FLOAT, Float),
        2,
        HALF_FLOAT,
        false,
    ),
    v(
        norender(f(93, 6, gl::RGB16F, gl::RGB, HALF_FLOAT, Float)),
        3,
        HALF_FLOAT,
        false,
    ),
    v(
        f(94, 8, gl::RGBA16F, gl::RGBA, HALF_FLOAT, Float),
        4,
        HALF_FLOAT,
        false,
    ),
    // sRGB.
    norender(f(97, 3, gl::SRGB8, gl::RGB, UNSIGNED_BYTE, Float)), // R8G8B8_SRGB
    f(104, 4, gl::SRGB8_ALPHA8, gl::RGBA, UNSIGNED_BYTE, Float),  // R8G8B8A8_SRGB
    // Packed floats.
    f(
        124,
        4,
        gl::R11F_G11F_B10F,
        gl::RGB,
        gl::UNSIGNED_INT_10F_11F_11F_REV,
        Float,
    ),
    norender(f(
        125,
        4,
        gl::RGB9_E5,
        gl::RGB,
        gl::UNSIGNED_INT_5_9_9_9_REV,
        Float,
    )),
    xa(f(134, 4, gl::RGBA8, gl::RGBA, UNSIGNED_BYTE, Float)), // R8G8B8X8_UNORM
    vo(173, 4, 4, gl::INT_2_10_10_10_REV, true),              // R10G10B10A2_SNORM
    // The rest of the 10/10/10/2 and BGRA vertex formats.
    vo(123, 4, 4, gl::UNSIGNED_INT_2_10_10_10_REV, false), // R10G10B10A2_USCALED
    vo(172, 4, 4, gl::INT_2_10_10_10_REV, false),          // R10G10B10A2_SSCALED
    vob(131, 4, 4, gl::UNSIGNED_INT_2_10_10_10_REV, true), // B10G10R10A2_UNORM
    vob(174, 4, 4, gl::UNSIGNED_INT_2_10_10_10_REV, false), // B10G10R10A2_USCALED
    vob(175, 4, 4, gl::INT_2_10_10_10_REV, false),         // B10G10R10A2_SSCALED
    vob(176, 4, 4, gl::INT_2_10_10_10_REV, true),          // B10G10R10A2_SNORM
    vob(380, 4, 4, BYTE, true),                            // B8G8R8A8_SNORM
    vob(381, 4, 4, BYTE, false),                           // B8G8R8A8_SSCALED
    vob(383, 4, 4, UNSIGNED_BYTE, false),                  // B8G8R8A8_USCALED
    // Integer.
    v(
        f(177, 1, gl::R8UI, gl::RED_INTEGER, UNSIGNED_BYTE, Uint),
        1,
        UNSIGNED_BYTE,
        false,
    ),
    v(
        f(178, 2, gl::RG8UI, gl::RG_INTEGER, UNSIGNED_BYTE, Uint),
        2,
        UNSIGNED_BYTE,
        false,
    ),
    v(
        norender(f(179, 3, gl::RGB8UI, gl::RGB_INTEGER, UNSIGNED_BYTE, Uint)),
        3,
        UNSIGNED_BYTE,
        false,
    ),
    v(
        f(180, 4, gl::RGBA8UI, gl::RGBA_INTEGER, UNSIGNED_BYTE, Uint),
        4,
        UNSIGNED_BYTE,
        false,
    ),
    v(
        f(181, 1, gl::R8I, gl::RED_INTEGER, BYTE, Sint),
        1,
        BYTE,
        false,
    ),
    v(
        f(182, 2, gl::RG8I, gl::RG_INTEGER, BYTE, Sint),
        2,
        BYTE,
        false,
    ),
    v(
        norender(f(183, 3, gl::RGB8I, gl::RGB_INTEGER, BYTE, Sint)),
        3,
        BYTE,
        false,
    ),
    v(
        f(184, 4, gl::RGBA8I, gl::RGBA_INTEGER, BYTE, Sint),
        4,
        BYTE,
        false,
    ),
    v(
        f(185, 2, gl::R16UI, gl::RED_INTEGER, UNSIGNED_SHORT, Uint),
        1,
        UNSIGNED_SHORT,
        false,
    ),
    v(
        f(186, 4, gl::RG16UI, gl::RG_INTEGER, UNSIGNED_SHORT, Uint),
        2,
        UNSIGNED_SHORT,
        false,
    ),
    v(
        norender(f(
            187,
            6,
            gl::RGB16UI,
            gl::RGB_INTEGER,
            UNSIGNED_SHORT,
            Uint,
        )),
        3,
        UNSIGNED_SHORT,
        false,
    ),
    v(
        f(188, 8, gl::RGBA16UI, gl::RGBA_INTEGER, UNSIGNED_SHORT, Uint),
        4,
        UNSIGNED_SHORT,
        false,
    ),
    v(
        f(189, 2, gl::R16I, gl::RED_INTEGER, SHORT, Sint),
        1,
        SHORT,
        false,
    ),
    v(
        f(190, 4, gl::RG16I, gl::RG_INTEGER, SHORT, Sint),
        2,
        SHORT,
        false,
    ),
    v(
        norender(f(191, 6, gl::RGB16I, gl::RGB_INTEGER, SHORT, Sint)),
        3,
        SHORT,
        false,
    ),
    v(
        f(192, 8, gl::RGBA16I, gl::RGBA_INTEGER, SHORT, Sint),
        4,
        SHORT,
        false,
    ),
    v(
        f(193, 4, gl::R32UI, gl::RED_INTEGER, UNSIGNED_INT, Uint),
        1,
        UNSIGNED_INT,
        false,
    ),
    v(
        f(194, 8, gl::RG32UI, gl::RG_INTEGER, UNSIGNED_INT, Uint),
        2,
        UNSIGNED_INT,
        false,
    ),
    v(
        norender(f(195, 12, gl::RGB32UI, gl::RGB_INTEGER, UNSIGNED_INT, Uint)),
        3,
        UNSIGNED_INT,
        false,
    ),
    v(
        f(196, 16, gl::RGBA32UI, gl::RGBA_INTEGER, UNSIGNED_INT, Uint),
        4,
        UNSIGNED_INT,
        false,
    ),
    v(
        f(197, 4, gl::R32I, gl::RED_INTEGER, INT, Sint),
        1,
        INT,
        false,
    ),
    v(
        f(198, 8, gl::RG32I, gl::RG_INTEGER, INT, Sint),
        2,
        INT,
        false,
    ),
    v(
        norender(f(199, 12, gl::RGB32I, gl::RGB_INTEGER, INT, Sint)),
        3,
        INT,
        false,
    ),
    v(
        f(200, 16, gl::RGBA32I, gl::RGBA_INTEGER, INT, Sint),
        4,
        INT,
        false,
    ),
    f(
        253,
        4,
        gl::RGB10_A2UI,
        gl::RGBA_INTEGER,
        gl::UNSIGNED_INT_2_10_10_10_REV,
        Uint,
    ), // R10G10B10A2_UINT
];

/// The host form of virgl format `id`, if it has one.
#[must_use]
pub fn lookup(id: u32) -> Option<&'static Format> {
    FORMATS.iter().find(|f| f.id == id)
}

impl Format {
    /// The bits of each of red, green, blue and alpha in a renderable
    /// unsigned normalized (linear) format, or `None` for every other kind.
    #[must_use]
    pub fn unorm_bits(&self) -> Option<[u32; 4]> {
        Some(match self.internal {
            gl::R8 => [8, 0, 0, 0],
            gl::RG8 => [8, 8, 0, 0],
            gl::RGB8 => [8, 8, 8, 0],
            gl::RGBA8 => [8, 8, 8, 8],
            gl::RGB565 => [5, 6, 5, 0],
            gl::RGB10_A2 => [10, 10, 10, 2],
            _ => return None,
        })
    }

    /// `value`, a float colour to clear a buffer of this format to, as the
    /// exact value of the stored integer it converts to, ties rounding up
    /// (as llvmpipe and WebGL's SwiftShader do; GLES leaves the tie to the
    /// implementation, and NVIDIA's rounds down, so a 0.5 clear read back
    /// 0x7f there and 0x80 elsewhere). Unchanged for other formats.
    #[must_use]
    pub fn quantize_clear(&self, value: [f32; 4]) -> [f32; 4] {
        let Some(bits) = self.unorm_bits() else {
            return value;
        };
        let mut out = value;
        for (v, b) in out.iter_mut().zip(bits) {
            if b > 0 && !v.is_nan() {
                let max = f64::from((1u32 << b) - 1);
                let q = (f64::from(v.clamp(0.0, 1.0)) * max + 0.5).floor();
                *v = (q / max) as f32;
            }
        }
        out
    }

    /// Whether the format can be a texture at all (not vertex-only).
    #[must_use]
    pub fn is_texture(&self) -> bool {
        self.internal != 0
    }

    /// Whether a vertex attribute of this format is fed as integers
    /// (`glVertexAttribIPointer`).
    #[must_use]
    pub fn vertex_is_integer(&self) -> bool {
        matches!(self.kind, Kind::Uint | Kind::Sint) && self.internal != 0
    }

    /// Rewrite `px` (pixels of this format, as the guest lays them out) into
    /// the host's layout in place, or back: the same swap both ways.
    pub fn convert_in_place(&self, px: &mut [u8]) {
        if self.swap_rb && self.bpp == 4 {
            for p in px.as_chunks_mut::<4>().0 {
                p.swap(0, 2);
            }
        }
    }

    /// The `format`, `type` and bytes per pixel `glReadPixels` reads this
    /// format with. GLES 3.0 — and WebGL2, strictly — only promise
    /// `RGBA`/`UNSIGNED_BYTE` for normalized colour (and
    /// `UNSIGNED_INT_2_10_10_10_REV` for `RGB10_A2`), `RGBA`/`FLOAT` for
    /// float and `RGBA_INTEGER` with `INT`/`UNSIGNED_INT` for integers;
    /// anything else is read that way and packed with [`Self::pack_read`].
    #[must_use]
    pub fn read_as(&self) -> (u32, u32, u32) {
        use gl::{RGBA, RGBA_INTEGER, UNSIGNED_INT_2_10_10_10_REV as U2101010};
        match (self.kind, self.format, self.ty) {
            // Readable as is (depth isn't read at all on GLES).
            (Kind::Float, RGBA, UNSIGNED_BYTE | U2101010)
            | (Kind::Depth | Kind::DepthStencil, ..) => (self.format, self.ty, self.bpp),
            (
                Kind::Float,
                _,
                FLOAT
                | HALF_FLOAT
                | gl::UNSIGNED_INT_10F_11F_11F_REV
                | gl::UNSIGNED_INT_5_9_9_9_REV,
            ) => (RGBA, FLOAT, 16),
            (Kind::Float, ..) => (RGBA, UNSIGNED_BYTE, 4),
            (Kind::Uint, ..) => (RGBA_INTEGER, UNSIGNED_INT, 16),
            (Kind::Sint, ..) => (RGBA_INTEGER, INT, 16),
        }
    }

    /// Pack `src`, pixels read as [`Self::read_as`] says, into `out` in
    /// this format's own layout (`out.len() / bpp` pixels).
    pub fn pack_read(&self, src: &[u8], out: &mut [u8]) {
        let (_, rty, rbpp) = self.read_as();
        if (rty, rbpp) == (self.ty, self.bpp) {
            let n = out.len().min(src.len());
            out[..n].copy_from_slice(&src[..n]);
            return;
        }
        let channels = match self.format {
            gl::RED | gl::RED_INTEGER => 1,
            gl::RG | gl::RG_INTEGER => 2,
            gl::RGB | gl::RGB_INTEGER => 3,
            _ => 4,
        };
        let word = |p: &[u8], c: usize| {
            u32::from_le_bytes([p[c * 4], p[c * 4 + 1], p[c * 4 + 2], p[c * 4 + 3]])
        };
        let float = |p: &[u8], c: usize| f32::from_bits(word(p, c));
        let pixels = src
            .chunks_exact(rbpp as usize)
            .zip(out.chunks_exact_mut(self.bpp as usize));
        for (p, o) in pixels {
            match (rty, self.ty) {
                // From RGBA8: fewer channels, or 5/6/5 bits.
                (UNSIGNED_BYTE, gl::UNSIGNED_SHORT_5_6_5) => {
                    let q = |v: u8, max: u32| (u32::from(v) * max * 2 + 255) / 510;
                    let v = (q(p[0], 31) << 11) | (q(p[1], 63) << 5) | q(p[2], 31);
                    o.copy_from_slice(&(v as u16).to_le_bytes());
                }
                // From RGBA8 or RGBA float: fewer channels.
                (UNSIGNED_BYTE, _) | (FLOAT, FLOAT) => o.copy_from_slice(&p[..o.len()]),
                // From RGBA float: halves, or packed floats.
                (FLOAT, HALF_FLOAT) => {
                    for (c, h) in o.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                        let v = small_float(float(p, c), 10, true) as u16;
                        h.copy_from_slice(&v.to_le_bytes());
                    }
                }
                (FLOAT, gl::UNSIGNED_INT_10F_11F_11F_REV) => {
                    let v = small_float(float(p, 0), 6, false)
                        | (small_float(float(p, 1), 6, false) << 11)
                        | (small_float(float(p, 2), 5, false) << 22);
                    o.copy_from_slice(&v.to_le_bytes());
                }
                (FLOAT, _) => o.fill(0), // RGB9_E5: never rendered to
                // From RGBA 32-bit integers: narrower, fewer, or 10/10/10/2.
                (_, gl::UNSIGNED_INT_2_10_10_10_REV) => {
                    let v = (word(p, 0) & 0x3ff)
                        | ((word(p, 1) & 0x3ff) << 10)
                        | ((word(p, 2) & 0x3ff) << 20)
                        | (word(p, 3) << 30);
                    o.copy_from_slice(&v.to_le_bytes());
                }
                (_, ty) => {
                    let size = match ty {
                        UNSIGNED_BYTE | BYTE => 1,
                        UNSIGNED_SHORT | SHORT => 2,
                        _ => 4,
                    };
                    for c in 0..channels {
                        o[c * size..(c + 1) * size]
                            .copy_from_slice(&word(p, c).to_le_bytes()[..size]);
                    }
                }
            }
        }
    }

    /// Pixels on their way to the host: also give `X8` formats an opaque
    /// alpha, since whatever the guest left in that byte must not show.
    pub fn to_host(&self, px: &mut [u8]) {
        self.convert_in_place(px);
        if self.no_alpha && self.bpp == 4 && self.format == gl::RGBA {
            for p in px.as_chunks_mut::<4>().0 {
                p[3] = 0xff;
            }
        }
    }
}

/// `v` as a float with a 5-bit exponent and `mantissa` bits (and a sign bit
/// when `signed`), rounded to nearest even: a half float (10, signed), or
/// the 11- and 10-bit floats of `R11F_G11F_B10F` (6 and 5, unsigned, where
/// negatives become zero).
fn small_float(v: f32, mantissa: u32, signed: bool) -> u32 {
    let bits = v.to_bits();
    let inf = 0x1f << mantissa;
    let abs = bits & 0x7fff_ffff;
    if abs > 0x7f80_0000 {
        return inf | (1 << (mantissa - 1)); // NaN
    }
    if bits >> 31 == 1 && !signed {
        return 0;
    }
    let sign = if signed {
        (bits >> 31) << (5 + mantissa)
    } else {
        0
    };
    // Shift right by `s`, rounding to nearest even.
    let round = |x: u32, s: u32| {
        let (q, rem, half) = (x >> s, x & ((1 << s) - 1), 1 << (s - 1));
        if rem > half || (rem == half && q & 1 == 1) {
            q + 1
        } else {
            q
        }
    };
    let exp = (abs >> 23) as i32 - 127 + 15;
    let shift = 23 - mantissa;
    let v = if abs >= 0x7f80_0000 || exp >= 31 {
        inf
    } else if exp > 0 {
        // A carry out of the mantissa bumps the exponent, up to infinity.
        round(((exp as u32) << 23) | (abs & 0x7f_ffff), shift).min(inf)
    } else if exp < -(mantissa as i32) {
        0 // under half the smallest subnormal
    } else {
        round((abs & 0x7f_ffff) | 0x80_0000, shift + (1 - exp) as u32)
    };
    sign | v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clear_to_a_half_rounds_up_on_every_host() {
        let q = |internal, v| {
            FORMATS
                .iter()
                .find(|f| f.internal == internal)
                .unwrap()
                .quantize_clear(v)
        };
        let half = [0.25, 0.5, 1.0, 0.75];
        // 0.5 of 255 is 127.5: stored as 128, whatever the host's tie rule.
        assert_eq!((q(gl::RGBA8, half)[1] * 255.0).round(), 128.0);
        assert!((q(gl::RGBA8, half)[1] * 255.0 - 128.0).abs() < 1e-3);
        // 0.5 of 63 (RGB565's green) is 31.5: 32.
        assert!((q(gl::RGB565, half)[1] * 63.0 - 32.0).abs() < 1e-3);
        // Values that aren't ties keep their nearest integer.
        assert!((q(gl::RG8, half)[0] * 255.0 - 64.0).abs() < 1e-3);
        assert!((q(gl::RGBA8, half)[3] * 255.0 - 191.0).abs() < 1e-3);
        // Out of range is clamped; float and sRGB formats are left alone.
        assert_eq!(q(gl::RGBA8, [-1.0, 2.0, 0.0, 1.0]), [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(q(gl::RGBA16F, half), half);
        assert_eq!(q(gl::SRGB8_ALPHA8, half), half);
    }

    #[test]
    fn small_floats() {
        let h = |v| small_float(v, 10, true);
        assert_eq!(h(1.0), 0x3c00);
        assert_eq!(h(-2.0), 0xc000);
        assert_eq!(h(0.5), 0x3800);
        assert_eq!(h(65504.0), 0x7bff);
        assert_eq!(h(65520.0), 0x7c00); // rounds up to infinity
        assert_eq!(h(1e9), 0x7c00);
        assert_eq!(h(f32::INFINITY), 0x7c00);
        assert_eq!(h(2f32.powi(-24)), 0x0001); // smallest subnormal
        assert_eq!(h(2f32.powi(-26)), 0);
        assert_eq!(h(1.0 + 2f32.powi(-11)), 0x3c00); // tie: to even
        assert_eq!(h(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        assert_eq!(h(0.0), 0);
        assert!(h(f32::NAN) & 0x3ff != 0);
        assert_eq!(small_float(1.0, 6, false), 15 << 6);
        assert_eq!(small_float(1.0, 5, false), 15 << 5);
        assert_eq!(small_float(-1.0, 6, false), 0);
        assert_eq!(small_float(0.25, 5, false), 13 << 5);
    }

    #[test]
    fn readback_packs_what_gles_can_read() {
        // R8: read as RGBA8, keep red.
        let f = lookup(64).unwrap();
        assert_eq!(f.read_as(), (gl::RGBA, UNSIGNED_BYTE, 4));
        let mut out = [0u8; 2];
        f.pack_read(&[10, 20, 30, 40, 50, 60, 70, 80], &mut out);
        assert_eq!(out, [10, 50]);
        // RGBA8 is read as is.
        let f = lookup(67).unwrap();
        let mut out = [0u8; 4];
        f.pack_read(&[1, 2, 3, 4], &mut out);
        assert_eq!(out, [1, 2, 3, 4]);
        // B5G6R5: requantized from 8 bits.
        let f = lookup(7).unwrap();
        let mut out = [0u8; 2];
        f.pack_read(&[255, 0, 255, 255], &mut out);
        assert_eq!(u16::from_le_bytes(out), 0xf81f);
        f.pack_read(&[0, 255, 0, 255], &mut out);
        assert_eq!(u16::from_le_bytes(out), 0x07e0);
        f.pack_read(&[132, 130, 8, 255], &mut out); // 16/31, 32/63, 1/31
        assert_eq!(u16::from_le_bytes(out), (16 << 11) | (32 << 5) | 1);
        // RG16F: read as RGBA float, halved.
        let f = lookup(92).unwrap();
        assert_eq!(f.read_as(), (gl::RGBA, FLOAT, 16));
        let src: Vec<u8> = [1.0f32, -2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut out = [0u8; 4];
        f.pack_read(&src, &mut out);
        assert_eq!(out, [0x00, 0x3c, 0x00, 0xc0]);
        // R32F: one float of four.
        let f = lookup(28).unwrap();
        f.pack_read(&src, &mut out);
        assert_eq!(f32::from_le_bytes(out), 1.0);
        // R11G11B10F.
        let f = lookup(124).unwrap();
        let src: Vec<u8> = [1.0f32, 1.0, 1.0, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        f.pack_read(&src, &mut out);
        assert_eq!(
            u32::from_le_bytes(out),
            (15 << 6) | (15 << 6 << 11) | (15 << 5 << 22)
        );
        // RG8UI: read as RGBA_INTEGER words, narrowed.
        let f = lookup(178).unwrap();
        assert_eq!(f.read_as(), (gl::RGBA_INTEGER, UNSIGNED_INT, 16));
        let src: Vec<u8> = [7u32, 200, 9, 1]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut out = [0u8; 2];
        f.pack_read(&src, &mut out);
        assert_eq!(out, [7, 200]);
        // RGB10_A2UI: packed from words.
        let f = lookup(253).unwrap();
        let src: Vec<u8> = [1u32, 2, 3, 1]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut out = [0u8; 4];
        f.pack_read(&src, &mut out);
        assert_eq!(
            u32::from_le_bytes(out),
            1 | (2 << 10) | (3 << 20) | (1 << 30)
        );
        // R32I stays a word.
        let f = lookup(197).unwrap();
        assert_eq!(f.read_as(), (gl::RGBA_INTEGER, INT, 16));
        let src: Vec<u8> = [-5i32, 0, 0, 1]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        f.pack_read(&src, &mut out);
        assert_eq!(i32::from_le_bytes(out), -5);
    }

    #[test]
    fn ids_are_unique() {
        for (i, a) in FORMATS.iter().enumerate() {
            assert!(
                FORMATS[i + 1..].iter().all(|b| b.id != a.id),
                "format {} twice",
                a.id
            );
        }
    }

    #[test]
    fn bgra_swaps_red_and_blue_and_x8_gets_alpha() {
        let f = lookup(2).unwrap(); // B8G8R8X8_UNORM
        let mut px = [1, 2, 3, 4];
        f.to_host(&mut px);
        assert_eq!(px, [3, 2, 1, 0xff]);
        let f = lookup(1).unwrap(); // B8G8R8A8_UNORM
        let mut px = [1, 2, 3, 4];
        f.to_host(&mut px);
        assert_eq!(px, [3, 2, 1, 4]);
        f.convert_in_place(&mut px);
        assert_eq!(px, [1, 2, 3, 4]);
    }

    #[test]
    fn vertex_formats() {
        let f = lookup(30).unwrap(); // R32G32B32_FLOAT
        assert_eq!(f.vertex, Some((3, FLOAT, false)));
        assert!(!f.vertex_is_integer());
        let f = lookup(196).unwrap(); // R32G32B32A32_UINT
        assert!(f.vertex_is_integer());
        assert!(lookup(36).unwrap().vertex.is_some()); // R32_USCALED
        assert!(!lookup(36).unwrap().is_texture());
    }
}
