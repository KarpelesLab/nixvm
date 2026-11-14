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
    bgr(f(1, 4, gl::RGBA8, gl::RGBA, UNSIGNED_BYTE, Float), false), // B8G8R8A8_UNORM
    bgr(f(2, 4, gl::RGBA8, gl::RGBA, UNSIGNED_BYTE, Float), true),  // B8G8R8X8_UNORM
    bgr(
        f(100, 4, gl::SRGB8_ALPHA8, gl::RGBA, UNSIGNED_BYTE, Float),
        false,
    ), // B8G8R8A8_SRGB
    bgr(
        f(101, 4, gl::SRGB8_ALPHA8, gl::RGBA, UNSIGNED_BYTE, Float),
        true,
    ), // B8G8R8X8_SRGB
    f(7, 2, gl::RGB565, gl::RGB, gl::UNSIGNED_SHORT_5_6_5, Float),  // B5G6R5_UNORM
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

#[cfg(test)]
mod tests {
    use super::*;

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
