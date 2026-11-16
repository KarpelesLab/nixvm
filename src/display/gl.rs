//! The GL compositor: the display's frames made on the VM's GPU, in the
//! host GL context the guest's rendering already runs in (native EGL, or
//! the browser's WebGL2).
//!
//! Every surface's content is a texture (RGBA8, premultiplied, rows top
//! first). A `wl_shm` buffer's damaged rows are uploaded into it; a GPU
//! buffer (`zwp_linux_dmabuf_v1`) is copied into it with a framebuffer blit
//! — GPU to GPU, nothing read back — so the client gets its buffer back at
//! once, as with shared memory. A frame is drawn into a framebuffer texture
//! (row 0 the display's top row): the background, then a quad per surface
//! whose fragment shader fetches texels by the same integer mapping the
//! software compositor uses, blended `ONE, ONE_MINUS_SRC_ALPHA` — so the
//! pixels match [`super::compose::Software`]'s.
//!
//! The frame stays on the GPU until the embedder wants it: natively
//! [`Compositor::read`] reads back only what changed since its last read,
//! once per frame the embedder takes rather than once per client commit; in
//! the browser [`Compositor::present`] blits it to the display canvas (the
//! context's default framebuffer) and nothing is read back at all.

use std::collections::HashMap;

use super::compose::{self, Compositor, Draw, Fill, Key, Rect};
use super::dmabuf::{DmaBuffer, GpuFrames};
use crate::gpu::{GpuBackend, GpuDevice, gl};

const TRIANGLE_STRIP: u32 = 5;

const VS: &str = "#version 300 es
// u[0]: destination x, y, w, h; u[1]: source x, y, w, h (buffer texels);
// u[2]: frame width, height, opaque.
uniform highp uvec4 u[3];
void main() {
    highp ivec4 d = ivec4(u[0]);
    highp vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    highp vec2 p = vec2(d.xy) + c * vec2(d.zw);
    gl_Position = vec4(p / vec2(u[2].xy) * 2.0 - 1.0, 0.0, 1.0);
}
";

const FS: &str = "#version 300 es
precision highp float;
precision highp int;
uniform highp uvec4 u[3];
uniform highp sampler2D t;
out vec4 o;
void main() {
    ivec4 d = ivec4(u[0]);
    ivec4 s = ivec4(u[1]);
    ivec2 p = ivec2(gl_FragCoord.xy) - d.xy;
    ivec2 q = s.xy + (p * s.zw) / d.zw;
    ivec2 size = textureSize(t, 0);
    if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, size))) discard;
    vec4 c = texelFetch(t, q, 0);
    if (u[2].z != 0u) c.a = 1.0;
    o = c;
}
";

/// A surface's texture.
struct Tex {
    gl: u32,
    width: u32,
    height: u32,
    opaque: bool,
    /// Bumped at each write: a draw of a newer version is damage.
    version: u64,
}

/// The GL objects the compositor owns.
struct Objects {
    program: u32,
    loc_u: i32,
    vao: u32,
    /// The frame: `frame_tex` attached to `fbo`.
    fbo: u32,
    frame_tex: u32,
    /// Source and destination of the blits that copy GPU buffers in.
    copy_read: u32,
    copy_draw: u32,
}

/// See the module docs.
pub struct GlCompositor {
    dev: GpuDevice,
    o: Objects,
    textures: HashMap<Key, Tex>,
    next_version: u64,
    width: u32,
    height: u32,
    /// The last frame's draws, with the versions they showed.
    last: Vec<(Draw, u64)>,
    /// What changed since [`Compositor::read`] last ran.
    unread: Option<Rect>,
    /// Upload staging.
    rgba: Vec<u8>,
    row: Vec<u32>,
}

impl GlCompositor {
    /// A compositor on `dev`'s GL context; `None` (with a note) if its
    /// shaders don't build there.
    pub fn new(dev: GpuDevice) -> Option<Self> {
        let o = dev.with_gl(|b, _| -> Result<Objects, String> {
            let program = b.create_program(VS, FS)?;
            let loc_u = b.uniform_location(program, "u");
            let loc_t = b.uniform_location(program, "t");
            b.use_program(program);
            b.uniform_1i(loc_t, 0);
            b.use_program(0);
            Ok(Objects {
                program,
                loc_u,
                vao: b.create_vertex_array(),
                fbo: b.create_framebuffer(),
                frame_tex: 0,
                copy_read: b.create_framebuffer(),
                copy_draw: b.create_framebuffer(),
            })
        });
        match o {
            Ok(o) => Some(Self {
                dev,
                o,
                textures: HashMap::new(),
                next_version: 1,
                width: 0,
                height: 0,
                last: Vec::new(),
                unread: None,
                rgba: Vec::new(),
                row: Vec::new(),
            }),
            Err(e) => {
                eprintln!("nixvm display: no GL compositing, the shaders failed: {e}");
                None
            }
        }
    }

    /// A version number for a fresh write.
    fn bump(&mut self) -> u64 {
        self.next_version += 1;
        self.next_version
    }
}

/// A texture for surface contents: RGBA8, one level, nearest.
fn new_texture(b: &mut dyn GpuBackend, width: u32, height: u32) -> u32 {
    let t = b.create_texture();
    b.active_texture(gl::TEXTURE0);
    b.bind_texture(gl::TEXTURE_2D, t);
    b.tex_storage(gl::TEXTURE_2D, 1, gl::RGBA8, width.max(1), height.max(1), 0);
    for (p, v) in [
        (gl::TEXTURE_MIN_FILTER, gl::NEAREST),
        (gl::TEXTURE_MAG_FILTER, gl::NEAREST),
        (gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE),
        (gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE),
    ] {
        #[allow(clippy::cast_possible_wrap)]
        b.tex_parameter(gl::TEXTURE_2D, p, v as i32);
    }
    t
}

/// The fixed-function state compositing wants (whatever the guest's
/// rendering left behind).
fn plain_state(b: &mut dyn GpuBackend) {
    for cap in [
        gl::SCISSOR_TEST,
        gl::DEPTH_TEST,
        gl::STENCIL_TEST,
        gl::CULL_FACE,
        gl::POLYGON_OFFSET_FILL,
        gl::SAMPLE_ALPHA_TO_COVERAGE,
        gl::RASTERIZER_DISCARD,
        gl::DITHER,
        gl::BLEND,
    ] {
        b.enable(cap, false);
    }
    b.color_mask(true, true, true, true);
}

impl GlCompositor {
    /// Surface `key`'s texture, made (or remade) `width` × `height`.
    /// Whether it is new (its contents undefined).
    fn ensure(&mut self, key: Key, width: u32, height: u32, opaque: bool) -> bool {
        let fresh = match self.textures.get(&key) {
            Some(t) => t.width != width || t.height != height,
            None => true,
        };
        let version = self.bump();
        if fresh {
            let old = self.textures.remove(&key).map(|t| t.gl);
            let gl = self.dev.with_gl(|b, _| {
                if let Some(old) = old {
                    b.delete_texture(old);
                }
                new_texture(b, width, height)
            });
            self.textures.insert(
                key,
                Tex {
                    gl,
                    width,
                    height,
                    opaque,
                    version,
                },
            );
        } else if let Some(t) = self.textures.get_mut(&key) {
            t.opaque = opaque;
            t.version = version;
        }
        fresh
    }

    fn upload_arrow(&mut self) {
        let a = super::render::arrow();
        self.write(
            compose::ARROW,
            (a.width, a.height),
            false,
            None,
            &mut |i, row| {
                let w = a.width as usize;
                row.copy_from_slice(&a.pixels[i * w..(i + 1) * w]);
            },
        );
    }
}

impl Compositor for GlCompositor {
    fn name(&self) -> &'static str {
        "gl"
    }

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn write(
        &mut self,
        key: Key,
        (w, h): (u32, u32),
        opaque: bool,
        rect: Option<Rect>,
        fill: Fill<'_>,
    ) {
        let fresh = self.ensure(key, w, h, opaque);
        // New storage is undefined: a partial write fills the rest with 0.
        let (x0, y0, rw, rh) = match rect {
            Some((x, y, rw, rh)) if !fresh => (x as u32, y as u32, rw as u32, rh as u32),
            Some(r) => {
                let (x, y, rw, rh) = (r.0 as usize, r.1 as usize, r.2 as usize, r.3 as usize);
                let mut full = vec![0u32; w as usize * h as usize];
                for i in 0..rh {
                    let d0 = (y + i) * w as usize + x;
                    fill(i, &mut full[d0..d0 + rw]);
                }
                let fw = w as usize;
                return self.write(key, (w, h), opaque, None, &mut |i, row| {
                    row.copy_from_slice(&full[i * fw..(i + 1) * fw]);
                });
            }
            None => (0, 0, w, h),
        };
        if rw == 0 || rh == 0 {
            return;
        }
        let (rwu, rhu) = (rw as usize, rh as usize);
        self.row.resize(rwu, 0);
        self.rgba.clear();
        self.rgba.reserve(rwu * rhu * 4);
        for i in 0..rhu {
            fill(i, &mut self.row);
            for &p in &self.row {
                self.rgba.extend_from_slice(&[
                    (p >> 16) as u8,
                    (p >> 8) as u8,
                    p as u8,
                    (p >> 24) as u8,
                ]);
            }
        }
        let tex = self.textures[&key].gl;
        let data = &self.rgba;
        self.dev.with_gl(|b, _| {
            b.active_texture(gl::TEXTURE0);
            b.bind_texture(gl::TEXTURE_2D, tex);
            b.tex_sub_image(
                gl::TEXTURE_2D,
                0,
                x0,
                y0,
                0,
                rw,
                rh,
                0,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                data,
            );
        });
    }

    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn write_gpu(
        &mut self,
        key: Key,
        _gpu: &dyn GpuFrames,
        buf: &DmaBuffer,
        rect: Option<Rect>,
    ) -> bool {
        let (w, h) = (buf.width, buf.height);
        let opaque = super::render::is_opaque(buf.format);
        let fresh = self.ensure(key, w, h, opaque);
        // A bottom-up buffer, or fresh storage, is copied whole.
        let (x0, y0, rw, rh) = match rect {
            Some(r) if !buf.y_invert && !fresh => r,
            _ => (0, 0, w as i32, h as i32),
        };
        if rw <= 0 || rh <= 0 {
            return true;
        }
        let tex = self.textures[&key].gl;
        let (copy_read, copy_draw) = (self.o.copy_read, self.o.copy_draw);
        let (wi, hi) = (w as i32, h as i32);
        let y_invert = buf.y_invert;
        let res = buf.res;
        self.dev.with_gl(|b, lookup| {
            let Some(src) = lookup(res) else {
                return false;
            };
            if src.width < w || src.height < h {
                return false;
            }
            plain_state(b);
            b.bind_framebuffer(gl::READ_FRAMEBUFFER, copy_read);
            b.framebuffer_texture(
                gl::READ_FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                src.gl,
                0,
                None,
            );
            b.read_buffer(gl::COLOR_ATTACHMENT0);
            b.bind_framebuffer(gl::DRAW_FRAMEBUFFER, copy_draw);
            b.framebuffer_texture(
                gl::DRAW_FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                tex,
                0,
                None,
            );
            b.draw_buffers(&[gl::COLOR_ATTACHMENT0]);
            let s = [x0, y0, x0 + rw, y0 + rh];
            let d = if y_invert { [0, hi, wi, 0] } else { s };
            b.blit_framebuffer(s, d, gl::COLOR_BUFFER_BIT, gl::NEAREST);
            // Don't keep the guest's texture attached (it may be deleted).
            b.framebuffer_texture(
                gl::READ_FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                0,
                0,
                None,
            );
            true
        })
    }

    fn remove(&mut self, key: Key) {
        if let Some(t) = self.textures.remove(&key) {
            self.dev.with_gl(|b, _| b.delete_texture(t.gl));
        }
    }

    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn render(
        &mut self,
        width: u32,
        height: u32,
        scene: &[Draw],
        cursor: Option<Draw>,
        _scene_changed: bool,
    ) {
        if cursor.is_some_and(|c| c.key == compose::ARROW)
            && !self.textures.contains_key(&compose::ARROW)
        {
            self.upload_arrow();
        }
        let resized = width != self.width || height != self.height || self.o.frame_tex == 0;
        let next: Vec<(Draw, u64)> = scene
            .iter()
            .chain(cursor.as_ref())
            .map(|d| (*d, self.textures.get(&d.key).map_or(0, |t| t.version)))
            .collect();
        let damage = if resized {
            Some((0, 0, width as i32, height as i32))
        } else {
            compose::damage(&self.last, &next).and_then(|r| compose::clip(r, width, height))
        };
        self.last = next;
        let Some(damage) = damage else {
            return;
        };
        self.unread = compose::union(self.unread, damage);
        self.width = width;
        self.height = height;
        let o = &mut self.o;
        let textures = &self.textures;
        let last = &self.last;
        self.dev.with_gl(|b, _| {
            if resized {
                if o.frame_tex != 0 {
                    b.delete_texture(o.frame_tex);
                }
                o.frame_tex = new_texture(b, width, height);
                b.bind_framebuffer(gl::DRAW_FRAMEBUFFER, o.fbo);
                b.framebuffer_texture(
                    gl::DRAW_FRAMEBUFFER,
                    gl::COLOR_ATTACHMENT0,
                    gl::TEXTURE_2D,
                    o.frame_tex,
                    0,
                    None,
                );
            }
            plain_state(b);
            b.bind_framebuffer(gl::DRAW_FRAMEBUFFER, o.fbo);
            b.draw_buffers(&[gl::COLOR_ATTACHMENT0]);
            b.viewport(0, 0, width as i32, height as i32);
            let bg = super::server::BACKGROUND;
            let ch = |s: u32| f32::from(((bg >> s) & 0xff) as u8) / 255.0;
            b.clear_buffer_fv(gl::COLOR, 0, [ch(16), ch(8), ch(0), 1.0]);
            b.use_program(o.program);
            b.bind_vertex_array(o.vao);
            b.active_texture(gl::TEXTURE0);
            b.bind_sampler(0, 0);
            b.blend_func_separate(
                gl::ONE,
                gl::ONE_MINUS_SRC_ALPHA,
                gl::ONE,
                gl::ONE_MINUS_SRC_ALPHA,
            );
            b.blend_equation_separate(gl::FUNC_ADD, gl::FUNC_ADD);
            for (d, _) in last {
                let Some(t) = textures.get(&d.key) else {
                    continue;
                };
                if d.w <= 0 || d.h <= 0 || d.src.2 <= 0 || d.src.3 <= 0 {
                    continue;
                }
                b.bind_texture(gl::TEXTURE_2D, t.gl);
                b.enable(gl::BLEND, !t.opaque);
                let u = [
                    d.x as u32,
                    d.y as u32,
                    d.w as u32,
                    d.h as u32,
                    d.src.0 as u32,
                    d.src.1 as u32,
                    d.src.2 as u32,
                    d.src.3 as u32,
                    width,
                    height,
                    u32::from(t.opaque),
                    0,
                ];
                b.uniform_4uiv(o.loc_u, &u);
                b.draw_arrays(TRIANGLE_STRIP, 0, 4, 1);
            }
            b.enable(gl::BLEND, false);
            b.bind_vertex_array(0);
            b.use_program(0);
        });
    }

    #[allow(clippy::cast_sign_loss)]
    fn read(&mut self, out: &mut Vec<u8>) {
        let (w, h) = (self.width as usize, self.height as usize);
        if self.o.frame_tex == 0 {
            // Nothing drawn yet: the background.
            out.clear();
            let bg = super::server::BACKGROUND;
            #[allow(clippy::cast_possible_truncation)]
            let px = [(bg >> 16) as u8, (bg >> 8) as u8, bg as u8, 0xff];
            out.extend(std::iter::repeat_n(px, w * h).flatten());
            return;
        }
        let rect = if out.len() == w * h * 4 {
            match self
                .unread
                .and_then(|r| compose::clip(r, self.width, self.height))
            {
                Some(r) => r,
                None => return,
            }
        } else {
            out.clear();
            out.resize(w * h * 4, 0);
            #[allow(clippy::cast_possible_wrap)]
            (0, 0, w as i32, h as i32)
        };
        self.unread = None;
        let (x, y, rw, rh) = (
            rect.0 as usize,
            rect.1 as usize,
            rect.2 as usize,
            rect.3 as usize,
        );
        let mut px = vec![0u8; rw * rh * 4];
        let fbo = self.o.fbo;
        self.dev.with_gl(|b, _| {
            b.bind_framebuffer(gl::READ_FRAMEBUFFER, fbo);
            b.read_buffer(gl::COLOR_ATTACHMENT0);
            #[allow(clippy::cast_possible_truncation)]
            b.read_pixels(
                rect.0,
                rect.1,
                rw as u32,
                rh as u32,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                &mut px,
            );
        });
        // Framebuffer row y is display row y (see the vertex shader).
        for (i, src) in px.chunks_exact(rw * 4).enumerate() {
            let d0 = ((y + i) * w + x) * 4;
            let dst = &mut out[d0..d0 + rw * 4];
            dst.copy_from_slice(src);
            for p in dst.as_chunks_mut::<4>().0 {
                p[3] = 0xff;
            }
        }
    }

    fn present(&mut self) -> bool {
        if !cfg!(target_arch = "wasm32") || self.o.frame_tex == 0 {
            // Natively the context is surfaceless: nothing to present on.
            return false;
        }
        #[allow(clippy::cast_possible_wrap)]
        let (w, h) = (self.width as i32, self.height as i32);
        let fbo = self.o.fbo;
        self.dev.with_gl(|b, _| {
            plain_state(b);
            b.bind_framebuffer(gl::READ_FRAMEBUFFER, fbo);
            b.read_buffer(gl::COLOR_ATTACHMENT0);
            b.bind_framebuffer(gl::DRAW_FRAMEBUFFER, 0);
            // The canvas's row 0 is its bottom.
            b.blit_framebuffer(
                [0, 0, w, h],
                [0, h, w, 0],
                gl::COLOR_BUFFER_BIT,
                gl::NEAREST,
            );
        });
        true
    }
}

impl Drop for GlCompositor {
    fn drop(&mut self) {
        let o = &self.o;
        let textures: Vec<u32> = self.textures.values().map(|t| t.gl).collect();
        self.dev.with_gl(|b, _| {
            for t in textures {
                b.delete_texture(t);
            }
            if o.frame_tex != 0 {
                b.delete_texture(o.frame_tex);
            }
            b.delete_framebuffer(o.fbo);
            b.delete_framebuffer(o.copy_read);
            b.delete_framebuffer(o.copy_draw);
            b.delete_program(o.program);
        });
    }
}

#[cfg(all(test, target_os = "linux", not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::display::compose::{ARROW, Software};

    /// Premultiplied pixels from a little PRNG: alpha anywhere, colour
    /// channels at most alpha.
    fn pixels(seed: u32, n: usize, opaque: bool) -> Vec<u32> {
        let mut x = seed.wrapping_mul(2_654_435_761).max(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                let a = if opaque { 255 } else { x >> 24 };
                let c = |s: u32| ((x >> s) & 0xff) * a / 255;
                (a << 24) | (c(16) << 16) | (c(8) << 8) | c(0)
            })
            .collect()
    }

    fn write_all(c: &mut dyn Compositor, key: Key, w: u32, h: u32, opaque: bool, px: &[u32]) {
        c.write(key, (w, h), opaque, None, &mut |i, row| {
            row.copy_from_slice(&px[i * w as usize..(i + 1) * w as usize]);
        });
    }

    fn read_both(gl: &mut GlCompositor, sw: &mut Software, a: &mut Vec<u8>, what: &str) {
        let mut b = Vec::new();
        gl.read(a);
        sw.read(&mut b);
        assert_eq!(a.len(), b.len(), "{what}: sizes");
        let bad: Vec<usize> = (0..a.len() / 4)
            .filter(|&i| a[i * 4..i * 4 + 4] != b[i * 4..i * 4 + 4])
            .collect();
        assert!(
            bad.is_empty(),
            "{what}: {} pixels differ, the first (#{:?}): gl {:?} software {:?}",
            bad.len(),
            bad.first(),
            bad.first().map(|&i| &a[i * 4..i * 4 + 4]),
            bad.first().map(|&i| &b[i * 4..i * 4 + 4]),
        );
    }

    #[test]
    fn the_gl_compositor_matches_the_software_one() {
        let Some(dev) = GpuDevice::from_spec("egl", false) else {
            eprintln!("no host EGL/GLES 3, skipping");
            return;
        };
        let mut gl = GlCompositor::new(dev).expect("the compositor's shaders build");
        let (fw, fh) = (97, 61);
        let mut sw = Software::new(fw, fh);
        let surfaces = [
            ((1, 1), 40, 30, true),
            ((1, 2), 23, 17, false),
            ((2, 1), 64, 9, false),
        ];
        for (i, &(key, w, h, opaque)) in surfaces.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let px = pixels(i as u32 + 1, (w * h) as usize, opaque);
            write_all(&mut gl, key, w, h, opaque, &px);
            write_all(&mut sw, key, w, h, opaque, &px);
        }
        let draw = |key, src, x, y, w, h| Draw {
            key,
            src,
            x,
            y,
            w,
            h,
        };
        let scene = [
            // Opaque, 1:1, partly off the left.
            draw((1, 1), (0, 0, 40, 30), -7, 3, 40, 30),
            // Translucent, scaled up and off the bottom.
            draw((1, 2), (0, 0, 23, 17), 20, 20, 59, 47),
            // A sub-rectangle scaled down, over both.
            draw((2, 1), (5, 1, 50, 7), 10, 15, 31, 5),
        ];
        let cursor = draw(ARROW, (0, 0, 12, 19), 90, 50, 12, 19);
        let mut frame = Vec::new();
        gl.render(fw, fh, &scene, Some(cursor), true);
        sw.render(fw, fh, &scene, Some(cursor), true);
        read_both(&mut gl, &mut sw, &mut frame, "the first frame");
        // A partial write, then a frame read into the previous one: only
        // the damage is read, and it still matches.
        let patch = pixels(9, 6 * 4, false);
        for c in [&mut gl as &mut dyn Compositor, &mut sw] {
            c.write(
                (1, 2),
                (23, 17),
                false,
                Some((3, 2, 6, 4)),
                &mut |i, row| {
                    row.copy_from_slice(&patch[i * 6..(i + 1) * 6]);
                },
            );
            c.render(fw, fh, &scene, None, true);
        }
        read_both(&mut gl, &mut sw, &mut frame, "after a partial write");
        // Content removed, and a resize.
        gl.remove((1, 1));
        sw.remove((1, 1));
        gl.render(50, 40, &scene, Some(cursor), true);
        sw.render(50, 40, &scene, Some(cursor), true);
        read_both(&mut gl, &mut sw, &mut frame, "after a resize");
        assert_eq!(frame.len(), 50 * 40 * 4);
    }
}
