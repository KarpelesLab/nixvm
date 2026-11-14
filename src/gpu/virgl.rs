//! The virgl command-stream decoder: what virglrenderer's `vrend_decode.c`
//! does, for the commands a GLES 3.0-level guest sends.
//!
//! A command buffer is a sequence of commands, each a header dword
//! (`cmd | object_type << 8 | length << 16`) and `length` payload dwords
//! (layouts in virglrenderer's `virgl_protocol.h`). Each is parsed here and
//! handed to the [`Renderer`]. A malformed or unknown command is logged and
//! skipped; like virtio, the guest has no way to be told which.

use super::Box3;
use super::renderer::{
    Blend, DrawInfo, Dsa, IndexBuffer, Object, Query, Rasterizer, Renderer, RtBlend, SamplerState,
    SamplerView, ShaderObj, Surface, Ubo, VertexBuffer, VertexElement, Viewport,
};

// enum virgl_context_cmd.
pub const NOP: u32 = 0;
pub const CREATE_OBJECT: u32 = 1;
pub const BIND_OBJECT: u32 = 2;
pub const DESTROY_OBJECT: u32 = 3;
pub const SET_VIEWPORT_STATE: u32 = 4;
pub const SET_FRAMEBUFFER_STATE: u32 = 5;
pub const SET_VERTEX_BUFFERS: u32 = 6;
pub const CLEAR: u32 = 7;
pub const DRAW_VBO: u32 = 8;
pub const RESOURCE_INLINE_WRITE: u32 = 9;
pub const SET_SAMPLER_VIEWS: u32 = 10;
pub const SET_INDEX_BUFFER: u32 = 11;
pub const SET_CONSTANT_BUFFER: u32 = 12;
pub const SET_STENCIL_REF: u32 = 13;
pub const SET_BLEND_COLOR: u32 = 14;
pub const SET_SCISSOR_STATE: u32 = 15;
pub const BLIT: u32 = 16;
pub const RESOURCE_COPY_REGION: u32 = 17;
pub const BIND_SAMPLER_STATES: u32 = 18;
pub const BEGIN_QUERY: u32 = 19;
pub const END_QUERY: u32 = 20;
pub const GET_QUERY_RESULT: u32 = 21;
pub const SET_POLYGON_STIPPLE: u32 = 22;
pub const SET_CLIP_STATE: u32 = 23;
pub const SET_SAMPLE_MASK: u32 = 24;
pub const SET_STREAMOUT_TARGETS: u32 = 25;
pub const SET_RENDER_CONDITION: u32 = 26;
pub const SET_UNIFORM_BUFFER: u32 = 27;
pub const SET_SUB_CTX: u32 = 28;
pub const CREATE_SUB_CTX: u32 = 29;
pub const DESTROY_SUB_CTX: u32 = 30;
pub const BIND_SHADER: u32 = 31;
pub const SET_TESS_STATE: u32 = 32;
pub const SET_MIN_SAMPLES: u32 = 33;
pub const SET_FRAMEBUFFER_STATE_NO_ATTACH: u32 = 38;
pub const TEXTURE_BARRIER: u32 = 39;
pub const SET_DEBUG_FLAGS: u32 = 41;
pub const SET_TWEAKS: u32 = 46;
pub const GET_MEMORY_INFO: u32 = 50;
pub const SEND_STRING_MARKER: u32 = 51;
pub const LINK_SHADER: u32 = 52;

// enum virgl_object_type.
pub const OBJ_BLEND: u32 = 1;
pub const OBJ_RASTERIZER: u32 = 2;
pub const OBJ_DSA: u32 = 3;
pub const OBJ_SHADER: u32 = 4;
pub const OBJ_VERTEX_ELEMENTS: u32 = 5;
pub const OBJ_SAMPLER_VIEW: u32 = 6;
pub const OBJ_SAMPLER_STATE: u32 = 7;
pub const OBJ_SURFACE: u32 = 8;
pub const OBJ_QUERY: u32 = 9;
pub const OBJ_STREAMOUT_TARGET: u32 = 10;
pub const OBJ_MSAA_SURFACE: u32 = 11;

/// `VIRGL_CMD0`: a command header.
#[must_use]
pub const fn cmd0(cmd: u32, obj: u32, len: u32) -> u32 {
    cmd | obj << 8 | len << 16
}

fn f(w: u32) -> f32 {
    f32::from_bits(w)
}

/// Decode and execute the command buffer `cmds` of context `ctx`.
pub fn decode(r: &mut Renderer, ctx: u32, cmds: &[u32]) {
    let mut i = 0;
    while i < cmds.len() {
        let hdr = cmds[i];
        let (cmd, obj, len) = (hdr & 0xff, (hdr >> 8) & 0xff, (hdr >> 16) as usize);
        let Some(p) = cmds.get(i + 1..i + 1 + len) else {
            r.warn(format!("command {cmd} overruns the buffer"));
            return;
        };
        i += 1 + len;
        if r.trace() {
            eprintln!("nixvm gpu: ctx {ctx} cmd {cmd} obj {obj} len {len}");
        }
        if let Err(e) = command(r, ctx, cmd, obj, p) {
            r.warn(format!("command {cmd} (object {obj}): {e}"));
        }
    }
}

/// Payload word `n` (`p[n - 1]`: protocol offsets count the header as 0).
fn w(p: &[u32], n: usize) -> Result<u32, String> {
    p.get(n - 1)
        .copied()
        .ok_or_else(|| format!("payload too short for word {n}"))
}

#[allow(clippy::too_many_lines)]
fn command(r: &mut Renderer, ctx: u32, cmd: u32, obj: u32, p: &[u32]) -> Result<(), String> {
    match cmd {
        NOP
        | SET_POLYGON_STIPPLE
        | SET_CLIP_STATE
        | SET_SAMPLE_MASK
        | SET_MIN_SAMPLES
        | SET_FRAMEBUFFER_STATE_NO_ATTACH
        | TEXTURE_BARRIER
        | SET_DEBUG_FLAGS
        | SET_TWEAKS
        | SEND_STRING_MARKER
        | LINK_SHADER
        | SET_TESS_STATE
        | GET_QUERY_RESULT
        | BEGIN_QUERY
        | SET_RENDER_CONDITION
        | SET_STREAMOUT_TARGETS => {}
        CREATE_OBJECT => create_object(r, ctx, obj, p)?,
        BIND_OBJECT => r.bind_object(ctx, obj, w(p, 1)?),
        DESTROY_OBJECT => r.destroy_object(ctx, w(p, 1)?),
        SET_VIEWPORT_STATE => {
            // Only viewport 0 matters on GLES.
            if w(p, 1)? == 0 && p.len() >= 7 {
                r.set_viewport(
                    ctx,
                    Viewport {
                        scale: [f(p[1]), f(p[2]), f(p[3])],
                        translate: [f(p[4]), f(p[5]), f(p[6])],
                    },
                );
            }
        }
        SET_FRAMEBUFFER_STATE => {
            let n = w(p, 1)? as usize;
            let zsurf = w(p, 2)?;
            let cbufs = p.get(2..2 + n).ok_or("short framebuffer state")?.to_vec();
            r.set_framebuffer(ctx, cbufs, zsurf);
        }
        SET_VERTEX_BUFFERS => {
            let vbs = p
                .as_chunks::<3>()
                .0
                .iter()
                .map(|c| VertexBuffer {
                    stride: c[0],
                    offset: c[1],
                    res: c[2],
                })
                .collect();
            r.set_vertex_buffers(ctx, vbs);
        }
        SET_INDEX_BUFFER => {
            let ib = if p.len() >= 3 {
                IndexBuffer {
                    res: p[0],
                    size: p[1],
                    offset: p[2],
                }
            } else {
                IndexBuffer::default()
            };
            r.set_index_buffer(ctx, ib);
        }
        CLEAR => {
            let buffers = w(p, 1)?;
            let color = [w(p, 2)?, w(p, 3)?, w(p, 4)?, w(p, 5)?];
            let depth = f64::from_bits(u64::from(w(p, 6)?) | u64::from(w(p, 7)?) << 32);
            let stencil = w(p, 8)?;
            r.clear(ctx, buffers, color, depth, stencil);
        }
        DRAW_VBO => {
            if p.len() >= 20 {
                return Err("indirect draws are not supported".into());
            }
            let info = DrawInfo {
                start: w(p, 1)?,
                count: w(p, 2)?,
                mode: w(p, 3)?,
                indexed: w(p, 4)? != 0,
                instance_count: w(p, 5)?,
                index_bias: w(p, 6)?.cast_signed(),
                start_instance: w(p, 7)?,
                primitive_restart: w(p, 8)? != 0,
                restart_index: w(p, 9)?,
                min_index: w(p, 10)?,
                max_index: w(p, 11)?,
            };
            r.draw(ctx, &info);
        }
        RESOURCE_INLINE_WRITE => {
            let res = w(p, 1)?;
            let level = w(p, 2)?;
            let stride = w(p, 4)?;
            let layer_stride = w(p, 5)?;
            let bx = Box3 {
                x: w(p, 6)?,
                y: w(p, 7)?,
                z: w(p, 8)?,
                w: w(p, 9)?,
                h: w(p, 10)?,
                d: w(p, 11)?,
            };
            let data: Vec<u8> = p[11..].iter().flat_map(|d| d.to_le_bytes()).collect();
            r.inline_write(res, level, bx, stride, layer_stride, &data);
        }
        SET_SAMPLER_VIEWS => {
            let stage = w(p, 1)? as usize;
            let start = w(p, 2)? as usize;
            r.set_sampler_views(ctx, stage, start, &p[2..]);
        }
        BIND_SAMPLER_STATES => {
            let stage = w(p, 1)? as usize;
            let start = w(p, 2)? as usize;
            r.bind_sampler_states(ctx, stage, start, &p[2..]);
        }
        SET_CONSTANT_BUFFER => {
            let stage = w(p, 1)? as usize;
            let index = w(p, 2)?;
            if index == 0 {
                r.set_constant_buffer(ctx, stage, p[2..].to_vec());
            }
        }
        SET_UNIFORM_BUFFER => {
            let stage = w(p, 1)? as usize;
            let index = w(p, 2)? as usize;
            let ubo = Ubo {
                offset: w(p, 3)?,
                len: w(p, 4)?,
                res: w(p, 5)?,
            };
            r.set_uniform_buffer(ctx, stage, index, ubo);
        }
        SET_STENCIL_REF => {
            let v = w(p, 1)?;
            r.set_stencil_ref(ctx, v & 0xff, (v >> 8) & 0xff);
        }
        SET_BLEND_COLOR => {
            r.set_blend_color(ctx, [f(w(p, 1)?), f(w(p, 2)?), f(w(p, 3)?), f(w(p, 4)?)]);
        }
        SET_SCISSOR_STATE => {
            if w(p, 1)? == 0 {
                let (a, b) = (w(p, 2)?, w(p, 3)?);
                r.set_scissor(ctx, [a & 0xffff, a >> 16, b & 0xffff, b >> 16]);
            }
        }
        RESOURCE_COPY_REGION => {
            let sbox = Box3 {
                x: w(p, 8)?,
                y: w(p, 9)?,
                z: w(p, 10)?,
                w: w(p, 11)?,
                h: w(p, 12)?,
                d: w(p, 13)?,
            };
            r.copy_region(
                w(p, 1)?,
                w(p, 2)?,
                w(p, 3)?,
                w(p, 4)?,
                w(p, 5)?,
                w(p, 6)?,
                w(p, 7)?,
                sbox,
            );
        }
        BLIT => {
            let s0 = w(p, 1)?;
            let mask = s0 & 0xff;
            let filter = if (s0 >> 8) & 3 == 0 {
                super::gl::NEAREST
            } else {
                super::gl::LINEAR
            };
            // PIPE_MASK_RGBA = 0xf, Z = 0x10, S = 0x20.
            let mut glmask = 0;
            if mask & 0xf != 0 {
                glmask |= super::gl::COLOR_BUFFER_BIT;
            }
            if mask & 0x10 != 0 {
                glmask |= super::gl::DEPTH_BUFFER_BIT;
            }
            if mask & 0x20 != 0 {
                glmask |= super::gl::STENCIL_BUFFER_BIT;
            }
            let (dst, dl) = (w(p, 4)?, w(p, 5)?);
            let (dx, dy, dz, dw, dh, dd) = (
                w(p, 7)?,
                w(p, 8)?,
                w(p, 9)?,
                w(p, 10)?,
                w(p, 11)?,
                w(p, 12)?,
            );
            let (src, sl) = (w(p, 13)?, w(p, 14)?);
            let (sx, sy, sz, sw, sh) = (w(p, 16)?, w(p, 17)?, w(p, 18)?, w(p, 19)?, w(p, 20)?);
            let rect = |x: u32, y: u32, w: u32, h: u32| {
                let (x, y) = (x.cast_signed(), y.cast_signed());
                [x, y, x + w.cast_signed(), y + h.cast_signed()]
            };
            for z in 0..dd.max(1) {
                r.blit_layer(
                    src,
                    sl,
                    sz + z,
                    rect(sx, sy, sw, sh),
                    dst,
                    dl,
                    dz + z,
                    rect(dx, dy, dw, dh),
                    glmask,
                    filter,
                );
            }
        }
        END_QUERY => r.end_query(ctx, w(p, 1)?),
        SET_SUB_CTX => r.set_sub_ctx(ctx, w(p, 1)?),
        CREATE_SUB_CTX => r.create_sub_ctx(ctx, w(p, 1)?),
        DESTROY_SUB_CTX => r.destroy_sub_ctx(ctx, w(p, 1)?),
        BIND_SHADER => r.bind_shader(ctx, w(p, 2)? as usize, w(p, 1)?),
        _ => return Err("unsupported command".into()),
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn create_object(r: &mut Renderer, ctx: u32, obj: u32, p: &[u32]) -> Result<(), String> {
    let handle = w(p, 1)?;
    let o = match obj {
        OBJ_BLEND => {
            let s0 = w(p, 2)?;
            let mut b = Blend {
                independent: s0 & 1 != 0,
                alpha_to_coverage: s0 & 8 != 0,
                ..Blend::default()
            };
            for (i, rt) in b.rt.iter_mut().enumerate() {
                let s = p.get(3 + i).copied().unwrap_or(0);
                *rt = RtBlend {
                    enable: s & 1 != 0,
                    rgb_func: (s >> 1) & 7,
                    rgb_src: (s >> 4) & 0x1f,
                    rgb_dst: (s >> 9) & 0x1f,
                    alpha_func: (s >> 14) & 7,
                    alpha_src: (s >> 17) & 0x1f,
                    alpha_dst: (s >> 22) & 0x1f,
                    colormask: (s >> 27) & 0xf,
                };
            }
            Object::Blend(Box::new(b))
        }
        OBJ_RASTERIZER => Object::Rasterizer(Rasterizer {
            s0: w(p, 2)?,
            point_size: f(w(p, 3)?),
            line_width: f(w(p, 6)?),
            offset_units: f(w(p, 7)?),
            offset_scale: f(w(p, 8)?),
        }),
        OBJ_DSA => Object::Dsa(Dsa {
            s0: w(p, 2)?,
            stencil: [w(p, 3)?, w(p, 4)?],
            alpha_ref: f(w(p, 5)?),
        }),
        OBJ_SHADER => {
            let stage = w(p, 2)?;
            let offlen = w(p, 3)?;
            let nso = w(p, 5)? as usize;
            // Compute shaders carry their local memory size where others
            // carry the stream-output count.
            let skip = if stage == 5 || nso == 0 {
                5
            } else {
                5 + 4 + 2 * nso
            };
            let text: Vec<u8> = p
                .get(skip..)
                .unwrap_or(&[])
                .iter()
                .flat_map(|d| d.to_le_bytes())
                .collect();
            let cont = offlen & 0x8000_0000 != 0;
            let total = if cont {
                0
            } else {
                (offlen & 0x7fff_ffff) as usize
            };
            let mut text = text;
            if !cont {
                text.truncate(total.max(1));
            }
            Object::Shader(ShaderObj {
                stage,
                text,
                total,
                parsed: None,
                error: None,
            })
        }
        OBJ_VERTEX_ELEMENTS => Object::VertexElements(
            p[1..]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| VertexElement {
                    offset: c[0],
                    divisor: c[1],
                    vb: c[2],
                    format: c[3],
                })
                .collect(),
        ),
        OBJ_SAMPLER_VIEW => {
            let lv = w(p, 5)?;
            Object::SamplerView(SamplerView {
                res: w(p, 2)?,
                format: w(p, 3)? & 0xff_ffff,
                first_level: lv & 0xff,
                last_level: (lv >> 8) & 0xff,
                swizzle: w(p, 6)?,
            })
        }
        OBJ_SAMPLER_STATE => Object::SamplerState(SamplerState {
            s0: w(p, 2)?,
            lod_bias: f(w(p, 3)?),
            min_lod: f(w(p, 4)?),
            max_lod: f(w(p, 5)?),
        }),
        OBJ_SURFACE | OBJ_MSAA_SURFACE => Object::Surface(Surface {
            res: w(p, 2)?,
            format: w(p, 3)?,
            level: w(p, 4)?,
            first_layer: w(p, 5)? & 0xffff,
        }),
        OBJ_QUERY => Object::Query(Query {
            ty: w(p, 2)? & 0xffff,
            offset: w(p, 3)?,
            res: w(p, 4)?,
        }),
        OBJ_STREAMOUT_TARGET => Object::Other(obj),
        _ => return Err("unknown object type".into()),
    };
    r.create_object(ctx, handle, o);
    if r.trace()
        && obj == OBJ_SHADER
        && let Some(t) = r.shader_text(ctx, handle)
    {
        eprintln!("nixvm gpu: shader {handle}:\n{}", t.trim_end_matches('\0'));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::gpu::backend::Recorder;
    use crate::gpu::{Backing, GpuDevice, MemBacking, ResourceDesc, Transfer, target};

    fn words(text: &str) -> Vec<u32> {
        let mut b = text.as_bytes().to_vec();
        b.push(0);
        while !b.len().is_multiple_of(4) {
            b.push(0);
        }
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect()
    }

    fn shader(handle: u32, stage: u32, text: &str) -> Vec<u32> {
        let t = words(text);
        let mut v = vec![
            cmd0(CREATE_OBJECT, OBJ_SHADER, 5 + t.len() as u32),
            handle,
            stage,
            text.len() as u32 + 1,
            0,
            0,
        ];
        v.extend(t);
        v
    }

    const VS: &str = "VERT
DCL IN[0]
DCL IN[1]
DCL OUT[0], POSITION
DCL OUT[1], GENERIC[0]
  0: MOV OUT[0], IN[0]
  1: MOV OUT[1], IN[1]
  2: END
";
    const FS: &str = "FRAG
DCL IN[0], GENERIC[0], PERSPECTIVE
DCL OUT[0], COLOR
DCL CONST[0][0]
  0: MUL OUT[0], IN[0], CONST[0][0]
  1: END
";

    type Log<T> = Arc<Mutex<Vec<T>>>;

    /// A device on a recorder whose calls the test can read.
    fn device() -> (GpuDevice, Log<String>, Log<(String, String)>) {
        let rec = Recorder::new(false);
        let (calls, progs) = (Arc::clone(&rec.calls), Arc::clone(&rec.programs));
        (GpuDevice::new(Box::new(rec), false), calls, progs)
    }

    #[test]
    fn a_triangle_reaches_the_backend() {
        let (dev, calls, progs) = device();
        let ctx = dev.create_context();
        let rt = dev
            .create_resource(
                ResourceDesc {
                    target: target::TEXTURE_2D,
                    format: 1, // B8G8R8A8_UNORM
                    bind: 2,
                    width: 64,
                    height: 64,
                    depth: 1,
                    array_size: 1,
                    ..ResourceDesc::default()
                },
                MemBacking::new(64 * 64 * 4),
            )
            .unwrap();
        let verts: Vec<f32> = vec![
            -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, //
            1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, //
            0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
        ];
        let vbacking = MemBacking::new(96);
        for (i, v) in verts.iter().enumerate() {
            vbacking.write(i as u64 * 4, &v.to_le_bytes());
        }
        let vb = dev
            .create_resource(
                ResourceDesc {
                    target: target::BUFFER,
                    format: 64,
                    bind: 16,
                    width: 96,
                    height: 1,
                    depth: 1,
                    array_size: 1,
                    ..ResourceDesc::default()
                },
                vbacking,
            )
            .unwrap();
        dev.transfer(&Transfer {
            res: vb,
            level: 0,
            bx: Box3 {
                x: 0,
                y: 0,
                z: 0,
                w: 96,
                h: 1,
                d: 1,
            },
            offset: 0,
            stride: 0,
            layer_stride: 0,
            to_host: true,
        })
        .unwrap();
        let mut cmds = vec![];
        cmds.extend(shader(10, 0, VS));
        cmds.extend(shader(11, 1, FS));
        cmds.extend([cmd0(CREATE_OBJECT, OBJ_SURFACE, 5), 20, rt, 1, 0, 0]);
        cmds.extend([cmd0(SET_FRAMEBUFFER_STATE, 0, 3), 1, 0, 20]);
        cmds.extend([
            cmd0(CREATE_OBJECT, OBJ_VERTEX_ELEMENTS, 9),
            30,
            0,
            0,
            0,
            31,
            16,
            0,
            0,
            31,
        ]);
        cmds.extend([cmd0(BIND_OBJECT, OBJ_VERTEX_ELEMENTS, 1), 30]);
        cmds.extend([cmd0(SET_VERTEX_BUFFERS, 0, 3), 32, 0, vb]);
        cmds.extend([
            cmd0(BIND_SHADER, 0, 2),
            10,
            0,
            cmd0(BIND_SHADER, 0, 2),
            11,
            1,
        ]);
        cmds.extend([cmd0(SET_CONSTANT_BUFFER, 0, 6), 1, 0]);
        cmds.extend([1.0f32, 1.0, 1.0, 1.0].map(f32::to_bits));
        // Viewport 64x64 flipped (a window-system buffer).
        cmds.extend([cmd0(SET_VIEWPORT_STATE, 0, 7), 0]);
        cmds.extend([32.0f32, -32.0, 0.5, 32.0, 32.0, 0.5].map(f32::to_bits));
        cmds.extend([cmd0(CLEAR, 0, 8), 4, 0, 0, 0, 0x3f80_0000, 0, 0, 0]);
        cmds.extend([cmd0(DRAW_VBO, 0, 12), 0, 3, 4, 0, 1, 0, 0, 0, 0, 0, !0, 0]);
        dev.submit(ctx, &cmds).unwrap();
        let calls = calls.lock().unwrap();
        let has = |s: &str| calls.iter().any(|c| c.starts_with(s));
        assert!(has("tex_storage(3553, 1, 32856, 64, 64, 0)"), "{calls:#?}");
        assert!(
            has("clear_buffer_fv(6144, 0, [0.0, 0.0, 0.0, 1.0])"),
            "{calls:#?}"
        );
        assert!(has("viewport(0, 0, 64, 64)"), "{calls:#?}");
        assert!(
            has("vertex_attrib_pointer(1, 4, 5126, false, false, 32, 16)"),
            "{calls:#?}"
        );
        assert!(has("uniform_1f") && has("uniform_4uiv"), "{calls:#?}");
        // Gallium's default front face is clockwise; flipped, it's CCW.
        assert!(
            has("front_face(2305)"),
            "flipped viewport inverts the winding: {calls:#?}"
        );
        assert!(
            calls.last().unwrap().starts_with("draw_arrays(4, 0, 3, 1)"),
            "{calls:#?}"
        );
        let progs = progs.lock().unwrap();
        assert_eq!(progs.len(), 1);
        assert!(progs[0].1.contains("nx_out0 = nx_o0;"));
    }

    #[test]
    fn transfers_round_trip_and_swap_bgra() {
        let (dev, calls, _) = device();
        let backing = MemBacking::new(16);
        backing.write(0, &[1, 2, 3, 4, 5, 6, 7, 8]);
        let tex = dev
            .create_resource(
                ResourceDesc {
                    target: target::TEXTURE_2D,
                    format: 1,
                    width: 2,
                    height: 2,
                    depth: 1,
                    array_size: 1,
                    ..ResourceDesc::default()
                },
                backing,
            )
            .unwrap();
        let t = Transfer {
            res: tex,
            level: 0,
            bx: Box3 {
                x: 0,
                y: 1,
                z: 0,
                w: 2,
                h: 1,
                d: 1,
            },
            offset: 8,
            stride: 8,
            layer_stride: 0,
            to_host: true,
        };
        dev.transfer(&t).unwrap();
        dev.transfer(&Transfer {
            to_host: false,
            ..t
        })
        .unwrap();
        assert!(
            dev.transfer(&Transfer {
                bx: Box3 { w: 3, ..t.bx },
                ..t
            })
            .is_err()
        );
        let calls = calls.lock().unwrap();
        assert!(
            calls
                .iter()
                .any(|c| c.starts_with("tex_sub_image(3553, 0, 0, 1, 0, 2, 1, 0, 6408, 5121, 8)"))
        );
        assert!(
            calls
                .iter()
                .any(|c| c.starts_with("read_pixels(0, 1, 2, 1, 6408, 5121, 8)"))
        );
    }

    #[test]
    fn shaders_split_across_commands_are_joined() {
        let (dev, _, _) = device();
        let ctx = dev.create_context();
        let t = words(VS);
        let (a, b) = t.split_at(t.len() / 2);
        let mut cmds = vec![
            cmd0(CREATE_OBJECT, OBJ_SHADER, 5 + a.len() as u32),
            10,
            0,
            VS.len() as u32 + 1,
            0,
            0,
        ];
        cmds.extend(a);
        cmds.extend([
            cmd0(CREATE_OBJECT, OBJ_SHADER, 5 + b.len() as u32),
            10,
            0,
            (a.len() as u32 * 4) | 0x8000_0000,
            0,
            0,
        ]);
        cmds.extend(b);
        dev.submit(ctx, &cmds).unwrap();
        let mut r = dev.inner.lock().unwrap();
        let text = r.shader_text(ctx, 10).unwrap();
        assert!(text.starts_with("VERT") && text.contains("END"));
    }

    #[test]
    fn garbage_is_skipped_not_fatal() {
        let (dev, _, _) = device();
        let ctx = dev.create_context();
        dev.submit(ctx, &[cmd0(200, 0, 1), 5, cmd0(DRAW_VBO, 0, 2), 0])
            .unwrap();
        dev.submit(ctx, &[cmd0(CLEAR, 0, 50)]).unwrap();
        assert!(dev.submit(ctx + 1, &[]).is_err());
    }
}
