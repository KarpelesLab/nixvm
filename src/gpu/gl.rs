//! The GLES 3.0 enum values nixvm's renderer speaks. WebGL2 uses the same
//! numbers, so one set serves every backend.
#![allow(missing_docs)]

// Pixel formats.
pub const DEPTH_COMPONENT: u32 = 0x1902;
pub const RED: u32 = 0x1903;
pub const RGB: u32 = 0x1907;
pub const RGBA: u32 = 0x1908;
pub const RG: u32 = 0x8227;
pub const RG_INTEGER: u32 = 0x8228;
pub const RED_INTEGER: u32 = 0x8D94;
pub const RGB_INTEGER: u32 = 0x8D98;
pub const RGBA_INTEGER: u32 = 0x8D99;
pub const DEPTH_STENCIL: u32 = 0x84F9;

// Types.
pub const BYTE: u32 = 0x1400;
pub const UNSIGNED_BYTE: u32 = 0x1401;
pub const SHORT: u32 = 0x1402;
pub const UNSIGNED_SHORT: u32 = 0x1403;
pub const INT: u32 = 0x1404;
pub const UNSIGNED_INT: u32 = 0x1405;
pub const FLOAT: u32 = 0x1406;
pub const HALF_FLOAT: u32 = 0x140B;
pub const UNSIGNED_SHORT_5_6_5: u32 = 0x8363;
pub const UNSIGNED_INT_2_10_10_10_REV: u32 = 0x8368;
pub const INT_2_10_10_10_REV: u32 = 0x8D9F;
pub const UNSIGNED_INT_24_8: u32 = 0x84FA;
pub const UNSIGNED_INT_10F_11F_11F_REV: u32 = 0x8C3B;
pub const UNSIGNED_INT_5_9_9_9_REV: u32 = 0x8C3E;
pub const FLOAT_32_UNSIGNED_INT_24_8_REV: u32 = 0x8DAD;

// Sized internal formats.
pub const RGB8: u32 = 0x8051;
pub const RGBA8: u32 = 0x8058;
pub const RGB10_A2: u32 = 0x8059;
pub const RGB10_A2UI: u32 = 0x906F;
pub const R8: u32 = 0x8229;
pub const RG8: u32 = 0x822B;
pub const R16F: u32 = 0x822D;
pub const R32F: u32 = 0x822E;
pub const RG16F: u32 = 0x822F;
pub const RG32F: u32 = 0x8230;
pub const R8I: u32 = 0x8231;
pub const R8UI: u32 = 0x8232;
pub const R16I: u32 = 0x8233;
pub const R16UI: u32 = 0x8234;
pub const R32I: u32 = 0x8235;
pub const R32UI: u32 = 0x8236;
pub const RG8I: u32 = 0x8237;
pub const RG8UI: u32 = 0x8238;
pub const RG16I: u32 = 0x8239;
pub const RG16UI: u32 = 0x823A;
pub const RG32I: u32 = 0x823B;
pub const RG32UI: u32 = 0x823C;
pub const RGBA32F: u32 = 0x8814;
pub const RGB32F: u32 = 0x8815;
pub const RGBA16F: u32 = 0x881A;
pub const RGB16F: u32 = 0x881B;
pub const R11F_G11F_B10F: u32 = 0x8C3A;
pub const RGB9_E5: u32 = 0x8C3D;
pub const SRGB8: u32 = 0x8C41;
pub const SRGB8_ALPHA8: u32 = 0x8C43;
pub const RGB565: u32 = 0x8D62;
pub const RGBA32UI: u32 = 0x8D70;
pub const RGB32UI: u32 = 0x8D71;
pub const RGBA16UI: u32 = 0x8D76;
pub const RGB16UI: u32 = 0x8D77;
pub const RGBA8UI: u32 = 0x8D7C;
pub const RGB8UI: u32 = 0x8D7D;
pub const RGBA32I: u32 = 0x8D82;
pub const RGB32I: u32 = 0x8D83;
pub const RGBA16I: u32 = 0x8D88;
pub const RGB16I: u32 = 0x8D89;
pub const RGBA8I: u32 = 0x8D8E;
pub const RGB8I: u32 = 0x8D8F;
pub const R8_SNORM: u32 = 0x8F94;
pub const RG8_SNORM: u32 = 0x8F95;
pub const RGB8_SNORM: u32 = 0x8F96;
pub const RGBA8_SNORM: u32 = 0x8F97;
pub const DEPTH_COMPONENT16: u32 = 0x81A5;
pub const DEPTH_COMPONENT24: u32 = 0x81A6;
pub const DEPTH_COMPONENT32F: u32 = 0x8CAC;
pub const DEPTH24_STENCIL8: u32 = 0x88F0;
pub const DEPTH32F_STENCIL8: u32 = 0x8CAD;

// Buffer targets and usage.
pub const ARRAY_BUFFER: u32 = 0x8892;
pub const ELEMENT_ARRAY_BUFFER: u32 = 0x8893;
pub const UNIFORM_BUFFER: u32 = 0x8A11;
pub const COPY_READ_BUFFER: u32 = 0x8F36;
pub const COPY_WRITE_BUFFER: u32 = 0x8F37;
pub const STREAM_DRAW: u32 = 0x88E0;
pub const DYNAMIC_DRAW: u32 = 0x88E8;

// Textures.
pub const TEXTURE_2D: u32 = 0x0DE1;
pub const TEXTURE_3D: u32 = 0x806F;
pub const TEXTURE_2D_ARRAY: u32 = 0x8C1A;
pub const TEXTURE_CUBE_MAP: u32 = 0x8513;
pub const TEXTURE_CUBE_MAP_POSITIVE_X: u32 = 0x8515;
pub const TEXTURE0: u32 = 0x84C0;
pub const TEXTURE_MAG_FILTER: u32 = 0x2800;
pub const TEXTURE_MIN_FILTER: u32 = 0x2801;
pub const TEXTURE_WRAP_S: u32 = 0x2802;
pub const TEXTURE_WRAP_T: u32 = 0x2803;
pub const TEXTURE_WRAP_R: u32 = 0x8072;
pub const TEXTURE_MIN_LOD: u32 = 0x813A;
pub const TEXTURE_MAX_LOD: u32 = 0x813B;
pub const TEXTURE_BASE_LEVEL: u32 = 0x813C;
pub const TEXTURE_MAX_LEVEL: u32 = 0x813D;
pub const TEXTURE_COMPARE_MODE: u32 = 0x884C;
pub const TEXTURE_COMPARE_FUNC: u32 = 0x884D;
pub const COMPARE_REF_TO_TEXTURE: u32 = 0x884E;
pub const NEAREST: u32 = 0x2600;
pub const LINEAR: u32 = 0x2601;
pub const NEAREST_MIPMAP_NEAREST: u32 = 0x2700;
pub const LINEAR_MIPMAP_NEAREST: u32 = 0x2701;
pub const NEAREST_MIPMAP_LINEAR: u32 = 0x2702;
pub const LINEAR_MIPMAP_LINEAR: u32 = 0x2703;
pub const REPEAT: u32 = 0x2901;
pub const CLAMP_TO_EDGE: u32 = 0x812F;
pub const MIRRORED_REPEAT: u32 = 0x8370;

// Framebuffers.
pub const FRAMEBUFFER: u32 = 0x8D40;
pub const READ_FRAMEBUFFER: u32 = 0x8CA8;
pub const DRAW_FRAMEBUFFER: u32 = 0x8CA9;
pub const COLOR_ATTACHMENT0: u32 = 0x8CE0;
pub const DEPTH_ATTACHMENT: u32 = 0x8D00;
pub const STENCIL_ATTACHMENT: u32 = 0x8D20;
pub const DEPTH_STENCIL_ATTACHMENT: u32 = 0x821A;
pub const NONE: u32 = 0;
pub const COLOR: u32 = 0x1800;
pub const DEPTH: u32 = 0x1801;
pub const STENCIL: u32 = 0x1802;
pub const COLOR_BUFFER_BIT: u32 = 0x4000;
pub const DEPTH_BUFFER_BIT: u32 = 0x0100;
pub const STENCIL_BUFFER_BIT: u32 = 0x0400;
pub const FRAMEBUFFER_COMPLETE: u32 = 0x8CD5;

// Capabilities.
pub const CULL_FACE: u32 = 0x0B44;
pub const DEPTH_TEST: u32 = 0x0B71;
pub const STENCIL_TEST: u32 = 0x0B90;
pub const BLEND: u32 = 0x0BE2;
pub const SCISSOR_TEST: u32 = 0x0C11;
pub const POLYGON_OFFSET_FILL: u32 = 0x8037;
pub const SAMPLE_ALPHA_TO_COVERAGE: u32 = 0x809E;
pub const RASTERIZER_DISCARD: u32 = 0x8C89;
pub const PRIMITIVE_RESTART_FIXED_INDEX: u32 = 0x8D69;
pub const DITHER: u32 = 0x0BD0;

// Faces and winding.
pub const FRONT: u32 = 0x0404;
pub const BACK: u32 = 0x0405;
pub const FRONT_AND_BACK: u32 = 0x0408;
pub const CW: u32 = 0x0900;
pub const CCW: u32 = 0x0901;

// Comparison functions (`NEVER + n` for Gallium's `PIPE_FUNC_*` n).
pub const NEVER: u32 = 0x0200;
pub const ALWAYS: u32 = 0x0207;

// Stencil ops.
pub const KEEP: u32 = 0x1E00;
pub const ZERO: u32 = 0;
pub const REPLACE: u32 = 0x1E01;
pub const INCR: u32 = 0x1E02;
pub const DECR: u32 = 0x1E03;
pub const INVERT: u32 = 0x150A;
pub const INCR_WRAP: u32 = 0x8507;
pub const DECR_WRAP: u32 = 0x8508;

// Blending.
pub const ONE: u32 = 1;
pub const SRC_COLOR: u32 = 0x0300;
pub const ONE_MINUS_SRC_COLOR: u32 = 0x0301;
pub const SRC_ALPHA: u32 = 0x0302;
pub const ONE_MINUS_SRC_ALPHA: u32 = 0x0303;
pub const DST_ALPHA: u32 = 0x0304;
pub const ONE_MINUS_DST_ALPHA: u32 = 0x0305;
pub const DST_COLOR: u32 = 0x0306;
pub const ONE_MINUS_DST_COLOR: u32 = 0x0307;
pub const SRC_ALPHA_SATURATE: u32 = 0x0308;
pub const CONSTANT_COLOR: u32 = 0x8001;
pub const ONE_MINUS_CONSTANT_COLOR: u32 = 0x8002;
pub const CONSTANT_ALPHA: u32 = 0x8003;
pub const ONE_MINUS_CONSTANT_ALPHA: u32 = 0x8004;
pub const FUNC_ADD: u32 = 0x8006;
pub const MIN: u32 = 0x8007;
pub const MAX: u32 = 0x8008;
pub const FUNC_SUBTRACT: u32 = 0x800A;
pub const FUNC_REVERSE_SUBTRACT: u32 = 0x800B;

// Primitives.
pub const POINTS: u32 = 0;
pub const TRIANGLES: u32 = 4;

// Errors.
pub const NO_ERROR: u32 = 0;
