//! The virgl capability set nixvm hands Mesa (`DRM_IOCTL_VIRTGPU_GET_CAPS`):
//! what the guest's `virgl` driver builds its GL feature level from.
//!
//! It describes a GLES 3.0 host — the WebGL2 feature level — the way
//! virglrenderer describes one when it runs on GLES (`vrend_renderer_fill_caps`
//! with `gles_ver == 30`): GLSL level 130, `VIRGL_CAP_HOST_IS_GLES`, the
//! formats of [`super::format::FORMATS`], and GLES 3.0's minimum limits, so
//! whatever the guest is promised every WebGL2 implementation can keep. Mesa
//! then offers applications OpenGL ES 3.0 and desktop GL 3.0.
//!
//! The layout is `union virgl_caps` from virglrenderer's `virgl_hw.h`
//! (`struct virgl_caps_v2`, 1408 bytes); the offsets below are that struct's.

use super::format::{FORMATS, Kind};

/// `VIRTGPU_DRM_CAPSET_VIRGL`.
pub const CAPSET_VIRGL: u32 = 1;
/// `VIRTGPU_DRM_CAPSET_VIRGL2`.
pub const CAPSET_VIRGL2: u32 = 2;

/// `sizeof(struct virgl_caps_v1)`.
pub const V1_SIZE: usize = 308;
/// `sizeof(struct virgl_caps_v2)`.
pub const V2_SIZE: usize = 1408;

/// The renderer string Mesa shows in `GL_RENDERER` ("virgl (...)").
pub const RENDERER: &str = "nixvm GLES 3.0";

// Field offsets in `struct virgl_caps_v2`.
const MAX_VERSION: usize = 0;
const MASK_SAMPLER: usize = 4;
const MASK_RENDER: usize = 68;
const MASK_DEPTHSTENCIL: usize = 132;
const MASK_VERTEX: usize = 196;
const BSET: usize = 260;
const GLSL_LEVEL: usize = 264;
const MAX_TEXTURE_ARRAY_LAYERS: usize = 268;
const MAX_STREAMOUT_BUFFERS: usize = 272;
const MAX_RENDER_TARGETS: usize = 280;
const MAX_SAMPLES: usize = 284;
const PRIM_MASK: usize = 288;
const MAX_UNIFORM_BLOCKS: usize = 296;
const MAX_VIEWPORTS: usize = 300;
const MIN_ALIASED_POINT_SIZE: usize = 308;
const MAX_ALIASED_POINT_SIZE: usize = 312;
const MIN_SMOOTH_POINT_SIZE: usize = 316;
const MAX_SMOOTH_POINT_SIZE: usize = 320;
const MIN_ALIASED_LINE_WIDTH: usize = 324;
const MAX_ALIASED_LINE_WIDTH: usize = 328;
const MIN_SMOOTH_LINE_WIDTH: usize = 332;
const MAX_SMOOTH_LINE_WIDTH: usize = 336;
const MAX_TEXTURE_LOD_BIAS: usize = 340;
const MAX_VERTEX_OUTPUTS: usize = 352;
const MAX_VERTEX_ATTRIBS: usize = 356;
const MIN_TEXEL_OFFSET: usize = 364;
const MAX_TEXEL_OFFSET: usize = 368;
const UNIFORM_BUFFER_OFFSET_ALIGNMENT: usize = 384;
const CAPABILITY_BITS: usize = 392;
const MAX_VERTEX_ATTRIB_STRIDE: usize = 428;
const MAX_TEXTURE_2D_SIZE: usize = 484;
const MAX_TEXTURE_3D_SIZE: usize = 488;
const MAX_TEXTURE_CUBE_SIZE: usize = 492;
const HOST_FEATURE_CHECK_VERSION: usize = 556;
const MASK_READBACK: usize = 560;
const MASK_SCANOUT: usize = 624;
const RENDERER_STR: usize = 696;
const MAX_ANISOTROPY: usize = 760;
const MAX_TEXTURE_SAMPLERS: usize = 764;
const MAX_CONST_BUFFER_SIZE: usize = 832;
const MAX_UNIFORM_BLOCK_SIZE: usize = 1372;

// `struct virgl_caps_bool_set1` bits.
const BSET_PRIMITIVE_RESTART: u32 = 1 << 6;
const BSET_BLEND_EQ_SEP: u32 = 1 << 7;
const BSET_INSTANCEID: u32 = 1 << 8;
const BSET_VERTEX_ELEMENT_INSTANCE_DIVISOR: u32 = 1 << 9;
const BSET_SEAMLESS_CUBE_MAP: u32 = 1 << 10;
const BSET_OCCLUSION_QUERY: u32 = 1 << 11;
const BSET_UBO: u32 = 1 << 18;

// `capability_bits`.
const VIRGL_CAP_SRGB_WRITE_CONTROL: u32 = 1 << 15;
const VIRGL_CAP_FBO_MIXED_COLOR_FORMATS: u32 = 1 << 18;
const VIRGL_CAP_HOST_IS_GLES: u32 = 1 << 19;

/// The `host_feature_check_version` nixvm's decoder keeps up with: it
/// understands every command a guest checking for this level may send.
pub const FEATURE_VERSION: u32 = 15;

struct Caps(Vec<u8>);

impl Caps {
    fn u32(&mut self, off: usize, v: u32) {
        self.0[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, off: usize, v: f32) {
        self.u32(off, v.to_bits());
    }
    fn bit(&mut self, mask: usize, fmt: u32) {
        let off = mask + (fmt / 32) as usize * 4;
        let w = u32::from_le_bytes(self.0[off..off + 4].try_into().unwrap());
        self.u32(off, w | 1 << (fmt % 32));
    }
}

/// Capset `set` at `version` as Mesa reads it, or `None` for one nixvm
/// doesn't offer. Capset 1 is the leading `struct virgl_caps_v1` of 2.
#[must_use]
pub fn capset(set: u32, version: u32) -> Option<Vec<u8>> {
    let (max_version, size) = match set {
        CAPSET_VIRGL if version <= 1 => (1, V1_SIZE),
        CAPSET_VIRGL2 if version <= 2 => (2, V2_SIZE),
        _ => return None,
    };
    let mut c = Caps(vec![0; V2_SIZE]);
    c.u32(MAX_VERSION, max_version);
    for f in FORMATS {
        if f.vertex.is_some() {
            c.bit(MASK_VERTEX, f.id);
        }
        if !f.is_texture() {
            continue;
        }
        c.bit(MASK_SAMPLER, f.id);
        match f.kind {
            Kind::Depth | Kind::DepthStencil => c.bit(MASK_DEPTHSTENCIL, f.id),
            _ if f.render => {
                c.bit(MASK_RENDER, f.id);
                c.bit(MASK_READBACK, f.id);
            }
            _ => {}
        }
    }
    // What a window system scans out: B8G8R8A8/X8 and R8G8B8A8/X8.
    for id in [1, 2, 67, 134] {
        c.bit(MASK_SCANOUT, id);
    }
    c.u32(
        BSET,
        BSET_PRIMITIVE_RESTART
            | BSET_BLEND_EQ_SEP
            | BSET_INSTANCEID
            | BSET_VERTEX_ELEMENT_INSTANCE_DIVISOR
            | BSET_SEAMLESS_CUBE_MAP
            | BSET_OCCLUSION_QUERY
            | BSET_UBO,
    );
    c.u32(GLSL_LEVEL, 130);
    c.u32(MAX_TEXTURE_ARRAY_LAYERS, 256);
    c.u32(MAX_STREAMOUT_BUFFERS, 4);
    c.u32(MAX_RENDER_TARGETS, 4);
    c.u32(MAX_SAMPLES, 0);
    // POINTS, LINES, LINE_LOOP, LINE_STRIP, TRIANGLES, TRIANGLE_STRIP,
    // TRIANGLE_FAN: GLES's primitive set.
    c.u32(PRIM_MASK, 0x7f);
    // Constant buffer 0 (plain uniforms) and GLES 3.0's 12 uniform blocks.
    c.u32(MAX_UNIFORM_BLOCKS, 13);
    c.u32(MAX_VIEWPORTS, 1);
    if set == CAPSET_VIRGL2 {
        c.f32(MIN_ALIASED_POINT_SIZE, 1.0);
        c.f32(MAX_ALIASED_POINT_SIZE, 255.0);
        c.f32(MIN_SMOOTH_POINT_SIZE, 1.0);
        c.f32(MAX_SMOOTH_POINT_SIZE, 255.0);
        c.f32(MIN_ALIASED_LINE_WIDTH, 1.0);
        c.f32(MAX_ALIASED_LINE_WIDTH, 1.0);
        c.f32(MIN_SMOOTH_LINE_WIDTH, 1.0);
        c.f32(MAX_SMOOTH_LINE_WIDTH, 1.0);
        c.f32(MAX_TEXTURE_LOD_BIAS, 2.0);
        c.u32(MAX_VERTEX_OUTPUTS, 15);
        c.u32(MAX_VERTEX_ATTRIBS, 16);
        c.u32(MIN_TEXEL_OFFSET, (-8i32).cast_unsigned());
        c.u32(MAX_TEXEL_OFFSET, 7);
        c.u32(UNIFORM_BUFFER_OFFSET_ALIGNMENT, 256);
        c.u32(
            CAPABILITY_BITS,
            VIRGL_CAP_HOST_IS_GLES
                | VIRGL_CAP_FBO_MIXED_COLOR_FORMATS
                | VIRGL_CAP_SRGB_WRITE_CONTROL,
        );
        c.u32(MAX_VERTEX_ATTRIB_STRIDE, 255);
        c.u32(MAX_TEXTURE_2D_SIZE, 4096);
        c.u32(MAX_TEXTURE_3D_SIZE, 256);
        c.u32(MAX_TEXTURE_CUBE_SIZE, 4096);
        c.u32(HOST_FEATURE_CHECK_VERSION, FEATURE_VERSION);
        let name = RENDERER.as_bytes();
        c.0[RENDERER_STR..RENDERER_STR + name.len()].copy_from_slice(name);
        c.f32(MAX_ANISOTROPY, 1.0);
        c.u32(MAX_TEXTURE_SAMPLERS, 16);
        // Per stage (VS, FS, GS, TCS, TES, CS): GLES 3.0's 224 vec4s of
        // default-block uniforms in a vertex shader, 224 in a fragment one.
        for stage in 0..6 {
            c.u32(MAX_CONST_BUFFER_SIZE + stage * 4, 224 * 16);
        }
        c.u32(MAX_UNIFORM_BLOCK_SIZE, 16384);
    }
    c.0.truncate(size);
    Some(c.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(c: &[u8], off: usize) -> u32 {
        u32::from_le_bytes(c[off..off + 4].try_into().unwrap())
    }

    #[test]
    fn capset2_describes_a_gles3_host() {
        let c = capset(CAPSET_VIRGL2, 0).unwrap();
        assert_eq!(c.len(), V2_SIZE);
        assert_eq!(word(&c, MAX_VERSION), 2);
        assert_eq!(word(&c, GLSL_LEVEL), 130);
        assert_ne!(word(&c, CAPABILITY_BITS) & VIRGL_CAP_HOST_IS_GLES, 0);
        // B8G8R8A8_UNORM (1) is renderable; Z24_UNORM_S8_UINT (19) is depth.
        assert_ne!(word(&c, MASK_RENDER) & 1 << 1, 0);
        assert_ne!(word(&c, MASK_DEPTHSTENCIL) & 1 << 19, 0);
        assert_eq!(word(&c, MASK_RENDER) & 1 << 19, 0);
        // R32G32B32_FLOAT (30) is a vertex format.
        assert_ne!(word(&c, MASK_VERTEX) & 1 << 30, 0);
        assert!(c[RENDERER_STR..].starts_with(RENDERER.as_bytes()));
    }

    #[test]
    fn capset1_is_the_v1_prefix_and_unknown_sets_are_refused() {
        let c = capset(CAPSET_VIRGL, 1).unwrap();
        assert_eq!(c.len(), V1_SIZE);
        assert_eq!(word(&c, MAX_VERSION), 1);
        assert!(capset(CAPSET_VIRGL2, 3).is_none());
        assert!(capset(4, 0).is_none());
    }
}
