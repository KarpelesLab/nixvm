//! The renderer: virgl's Gallium-level state, kept per context and
//! sub-context the way virglrenderer keeps it, and turned into GLES 3.0
//! calls on a [`GpuBackend`] at each draw, clear, copy and transfer.
//!
//! GL state is applied in full before every draw (the guest's state objects
//! are small and draws are few next to the cost of the ioctl that carried
//! them), so contexts sharing the one host GL context never see each
//! other's leftovers. Gallium conventions that GLES lacks are emulated here:
//!
//! - **Viewport y flip.** A Gallium viewport may scale y negatively (Mesa
//!   does for window-system buffers, whose row 0 is the top). GL's can't;
//!   the vertex shader negates `gl_Position.y` instead (`nx_flip_y`), which
//!   inverts the winding too. Gallium's `front_ccw` is measured with window
//!   y pointing down, GL's with it up, so unflipped the face names swap. Rendering always goes to textures
//!   attached to an FBO, whose row 0 is window y 0 in both APIs, so the
//!   guest's memory layout comes out right.
//! - **Base vertex.** GLES 3.0 has no `glDrawElementsBaseVertex`: the vertex
//!   attribute offsets are moved by `index_bias × stride` instead.
//! - **Index buffers** are drawn from a host-only element buffer filled from
//!   the guest buffer's shadow copy: WebGL forbids a buffer that held indices
//!   from serving as anything else, and Mesa streams vertices and indices
//!   through one buffer.
//! - **BGRA** formats are stored as RGBA, swapped on transfer
//!   ([`super::format`]).
//!
//! Buffers keep a shadow copy of their contents on the host side so reads
//! (`TRANSFER_FROM_HOST`, copies between buffers, index uploads) never need
//! GL to read a buffer back, which GLES 3.0 / WebGL2 make awkward.

use std::collections::HashMap;
use std::sync::Arc;

use super::format::{self, Format, Kind};
use super::glsl;
use super::tgsi;
use super::{Backing, Box3, EINVAL, Errno, GpuBackend, ResourceDesc, Transfer, gl, target};

/// The largest buffer resource, and all of them together: host memory the
/// guest's limits don't account for.
const MAX_BUFFER: u32 = 256 << 20;
const MAX_BUFFER_BYTES: u64 = 1 << 30;

/// The host object behind a resource.
enum Host {
    Buffer {
        gl: u32,
        shadow: Vec<u8>,
    },
    Texture {
        gl: u32,
        gl_target: u32,
    },
    /// A format or target the host can't hold: transfers land in the
    /// backing only.
    None,
}

pub(crate) struct Resource {
    desc: ResourceDesc,
    fmt: Option<&'static Format>,
    backing: Arc<dyn Backing>,
    host: Host,
}

impl Resource {
    fn level_size(&self, level: u32) -> (u32, u32, u32) {
        let m = |v: u32| v.checked_shr(level).unwrap_or(0).max(1);
        let d = if self.desc.target == target::TEXTURE_3D {
            m(self.desc.depth)
        } else {
            self.desc.array_size.max(1)
        };
        (m(self.desc.width), m(self.desc.height), d)
    }
}

/// A virgl state object, by handle (one namespace per sub-context, as in
/// virglrenderer).
#[derive(Debug, Clone)]
pub enum Object {
    Blend(Box<Blend>),
    Rasterizer(Rasterizer),
    Dsa(Dsa),
    Shader(ShaderObj),
    VertexElements(Vec<VertexElement>),
    SamplerView(SamplerView),
    SamplerState(SamplerState),
    Surface(Surface),
    Query(Query),
    /// A stream-output target or anything else nixvm only tracks.
    Other(u32),
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RtBlend {
    pub enable: bool,
    pub rgb_func: u32,
    pub rgb_src: u32,
    pub rgb_dst: u32,
    pub alpha_func: u32,
    pub alpha_src: u32,
    pub alpha_dst: u32,
    pub colormask: u32,
}

#[derive(Debug, Clone, Default)]
pub struct Blend {
    pub independent: bool,
    pub alpha_to_coverage: bool,
    pub rt: [RtBlend; 8],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Rasterizer {
    pub s0: u32,
    pub point_size: f32,
    pub line_width: f32,
    pub offset_units: f32,
    pub offset_scale: f32,
}

impl Rasterizer {
    fn bit(&self, n: u32) -> bool {
        self.s0 & (1 << n) != 0
    }
    #[must_use]
    pub fn flatshade(&self) -> bool {
        self.bit(0)
    }
    fn clip_halfz(&self) -> bool {
        self.bit(2)
    }
    fn discard(&self) -> bool {
        self.bit(3)
    }
    fn cull(&self) -> u32 {
        (self.s0 >> 8) & 3
    }
    fn scissor(&self) -> bool {
        self.bit(14)
    }
    fn front_ccw(&self) -> bool {
        self.bit(15)
    }
    fn offset_tri(&self) -> bool {
        self.bit(20)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Dsa {
    pub s0: u32,
    pub stencil: [u32; 2],
    pub alpha_ref: f32,
}

#[derive(Debug, Clone)]
pub struct ShaderObj {
    pub stage: u32,
    /// The TGSI text, possibly still arriving in pieces.
    pub text: Vec<u8>,
    pub total: usize,
    pub parsed: Option<Arc<tgsi::Shader>>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct VertexElement {
    pub offset: u32,
    pub divisor: u32,
    pub vb: u32,
    pub format: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct SamplerView {
    pub res: u32,
    pub format: u32,
    pub first_level: u32,
    pub last_level: u32,
    pub swizzle: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SamplerState {
    pub s0: u32,
    pub lod_bias: f32,
    pub min_lod: f32,
    pub max_lod: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Surface {
    pub res: u32,
    pub format: u32,
    pub level: u32,
    pub first_layer: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct Query {
    pub ty: u32,
    pub res: u32,
    pub offset: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct VertexBuffer {
    pub stride: u32,
    pub offset: u32,
    pub res: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct IndexBuffer {
    pub res: u32,
    pub size: u32,
    pub offset: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Viewport {
    pub scale: [f32; 3],
    pub translate: [f32; 3],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Ubo {
    pub res: u32,
    pub offset: u32,
    pub len: u32,
}

/// What `DRAW_VBO` carries.
#[derive(Debug, Clone, Copy, Default)]
pub struct DrawInfo {
    pub start: u32,
    pub count: u32,
    pub mode: u32,
    pub indexed: bool,
    pub instance_count: u32,
    pub index_bias: i32,
    pub start_instance: u32,
    pub primitive_restart: bool,
    pub restart_index: u32,
    pub min_index: u32,
    pub max_index: u32,
}

struct Program {
    gl: u32,
    consts: [(i32, u32); 2],
    ubos: [Vec<u32>; 2],
    samplers: [Vec<(u32, i32)>; 2],
    flip_y: i32,
    fb_height: i32,
}

#[derive(Default)]
pub(crate) struct SubCtx {
    objects: HashMap<u32, Object>,
    blend: u32,
    rasterizer: u32,
    dsa: u32,
    vertex_elements: u32,
    shaders: [u32; 6],
    viewport: Viewport,
    scissor: [u32; 4],
    cbufs: Vec<u32>,
    zsurf: u32,
    vbufs: Vec<VertexBuffer>,
    ib: IndexBuffer,
    consts: [Vec<u32>; 6],
    ubos: [[Ubo; 16]; 6],
    views: [[u32; 32]; 6],
    samplers: [[u32; 32]; 6],
    stencil_ref: [u32; 2],
    blend_color: [f32; 4],
    programs: HashMap<(u32, u32, glsl::Options), Option<Program>>,
}

#[derive(Default)]
pub(crate) struct Context {
    subs: HashMap<u32, SubCtx>,
    cur: u32,
}

/// See the module docs.
pub struct Renderer {
    backend: Box<dyn GpuBackend>,
    trace: bool,
    resources: HashMap<u32, Resource>,
    /// Bytes of buffer shadows held for the guest.
    buffer_bytes: u64,
    next_res: u32,
    contexts: HashMap<u32, Context>,
    next_ctx: u32,
    /// Host-only GL objects, made on first use.
    vao: u32,
    draw_fbo: u32,
    read_fbo: u32,
    ebo: u32,
    /// The guest's failures, logged once each.
    warned: std::collections::HashSet<String>,
}

impl std::fmt::Debug for Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer")
            .field("resources", &self.resources.len())
            .field("contexts", &self.contexts.len())
            .finish_non_exhaustive()
    }
}

/// Gallium `PIPE_BLENDFACTOR_*` → GL.
fn blend_factor(f: u32) -> u32 {
    match f {
        0x2 => gl::SRC_COLOR,
        0x3 => gl::SRC_ALPHA,
        0x4 => gl::DST_ALPHA,
        0x5 => gl::DST_COLOR,
        0x6 => gl::SRC_ALPHA_SATURATE,
        0x7 => gl::CONSTANT_COLOR,
        0x8 => gl::CONSTANT_ALPHA,
        0x11 => gl::ZERO,
        0x12 => gl::ONE_MINUS_SRC_COLOR,
        0x13 => gl::ONE_MINUS_SRC_ALPHA,
        0x14 => gl::ONE_MINUS_DST_ALPHA,
        0x15 => gl::ONE_MINUS_DST_COLOR,
        0x17 => gl::ONE_MINUS_CONSTANT_COLOR,
        0x18 => gl::ONE_MINUS_CONSTANT_ALPHA,
        // ONE, and SRC1_* (dual source), which GLES 3.0 lacks.
        _ => gl::ONE,
    }
}

fn blend_func(f: u32) -> u32 {
    match f {
        1 => gl::FUNC_SUBTRACT,
        2 => gl::FUNC_REVERSE_SUBTRACT,
        3 => gl::MIN,
        4 => gl::MAX,
        _ => gl::FUNC_ADD,
    }
}

fn stencil_op(o: u32) -> u32 {
    match o {
        1 => gl::ZERO,
        2 => gl::REPLACE,
        3 => gl::INCR,
        4 => gl::DECR,
        5 => gl::INCR_WRAP,
        6 => gl::DECR_WRAP,
        7 => gl::INVERT,
        _ => gl::KEEP,
    }
}

fn wrap(w: u32) -> i32 {
    (match w {
        0 => gl::REPEAT,
        4 => gl::MIRRORED_REPEAT,
        _ => gl::CLAMP_TO_EDGE,
    }) as i32
}

fn gl_texture_target(desc: &ResourceDesc) -> Option<u32> {
    Some(match desc.target {
        target::TEXTURE_1D | target::TEXTURE_2D | target::TEXTURE_RECT => gl::TEXTURE_2D,
        target::TEXTURE_3D => gl::TEXTURE_3D,
        target::TEXTURE_CUBE => gl::TEXTURE_CUBE_MAP,
        target::TEXTURE_1D_ARRAY | target::TEXTURE_2D_ARRAY => gl::TEXTURE_2D_ARRAY,
        _ => return None,
    })
}

impl Renderer {
    /// A renderer driving `backend`.
    #[must_use]
    pub fn new(backend: Box<dyn GpuBackend>, trace: bool) -> Self {
        Self {
            backend,
            trace,
            resources: HashMap::new(),
            buffer_bytes: 0,
            next_res: 1,
            contexts: HashMap::new(),
            next_ctx: 1,
            vao: 0,
            draw_fbo: 0,
            read_fbo: 0,
            ebo: 0,
            warned: std::collections::HashSet::new(),
        }
    }

    pub(crate) fn begin(&mut self) {
        self.backend.begin();
    }

    pub(crate) fn end(&mut self) {
        self.backend.end();
    }

    pub(crate) fn backend_name(&self) -> String {
        self.backend.name()
    }

    /// Log `msg` (once per distinct message) to stderr.
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn warn(&mut self, msg: String) {
        // Every message when tracing; else each distinct one once, up to 64.
        let fresh = self.warned.len() < 64 && self.warned.insert(msg.clone());
        if self.trace || fresh {
            eprintln!("nixvm gpu: {msg}");
        }
    }

    pub(crate) fn trace(&self) -> bool {
        self.trace
    }

    pub(crate) fn create_context(&mut self) -> u32 {
        let id = self.next_ctx;
        self.next_ctx += 1;
        let mut c = Context::default();
        c.subs.insert(0, SubCtx::default());
        self.contexts.insert(id, c);
        id
    }

    pub(crate) fn destroy_context(&mut self, id: u32) {
        if let Some(c) = self.contexts.remove(&id) {
            for (_, sub) in c.subs {
                self.drop_sub(sub);
            }
        }
    }

    fn drop_sub(&mut self, sub: SubCtx) {
        for p in sub.programs.into_values().flatten() {
            self.backend.delete_program(p.gl);
        }
    }

    pub(crate) fn create_resource(
        &mut self,
        desc: ResourceDesc,
        backing: Arc<dyn Backing>,
    ) -> Result<u32, Errno> {
        // What the capset promised, and no more: these become host
        // allocations the guest's memory limits don't see.
        let (max_2d, max_3d) = (4096, 256);
        let ok = if desc.target == target::BUFFER {
            desc.width > 0
                && desc.width <= MAX_BUFFER
                && self.buffer_bytes + u64::from(desc.width) <= MAX_BUFFER_BYTES
        } else {
            let dims = if desc.target == target::TEXTURE_3D {
                desc.width <= max_3d && desc.height <= max_3d && desc.depth <= max_3d
            } else {
                desc.width <= max_2d && desc.height <= max_2d && desc.depth <= 1
            };
            desc.width > 0 && dims && desc.array_size <= 256 && desc.last_level <= 12
        };
        if !ok {
            return Err(EINVAL);
        }
        let fmt = format::lookup(desc.format);
        let host = if desc.target == target::BUFFER {
            let b = self.backend.create_buffer();
            self.backend.bind_buffer(gl::COPY_WRITE_BUFFER, b);
            self.backend.buffer_data(
                gl::COPY_WRITE_BUFFER,
                desc.width as usize,
                None,
                gl::DYNAMIC_DRAW,
            );
            self.buffer_bytes += u64::from(desc.width);
            Host::Buffer {
                gl: b,
                shadow: vec![0; desc.width as usize],
            }
        } else if let (Some(f), Some(t)) =
            (fmt.filter(|f| f.is_texture()), gl_texture_target(&desc))
        {
            let tex = self.backend.create_texture();
            self.backend.bind_texture(t, tex);
            let depth = match t {
                gl::TEXTURE_3D => desc.depth.max(1),
                gl::TEXTURE_2D_ARRAY => desc.array_size.max(1),
                _ => 0,
            };
            let height =
                if desc.target == target::TEXTURE_1D_ARRAY || desc.target == target::TEXTURE_1D {
                    1
                } else {
                    desc.height.max(1)
                };
            self.backend.tex_storage(
                t,
                desc.last_level + 1,
                f.internal,
                desc.width,
                height,
                depth,
            );
            // Mipmap-less unless the guest says otherwise per view.
            self.backend
                .tex_parameter(t, gl::TEXTURE_MAX_LEVEL, desc.last_level as i32);
            Host::Texture {
                gl: tex,
                gl_target: t,
            }
        } else {
            self.warn(format!(
                "resource target {} format {} has no host form",
                desc.target, desc.format
            ));
            Host::None
        };
        let id = self.next_res;
        self.next_res += 1;
        self.resources.insert(
            id,
            Resource {
                desc,
                fmt,
                backing,
                host,
            },
        );
        Ok(id)
    }

    pub(crate) fn destroy_resource(&mut self, id: u32) {
        if let Some(r) = self.resources.remove(&id) {
            match r.host {
                Host::Buffer { gl, shadow } => {
                    self.buffer_bytes -= shadow.len() as u64;
                    self.backend.delete_buffer(gl);
                }
                Host::Texture { gl, .. } => self.backend.delete_texture(gl),
                Host::None => {}
            }
        }
    }

    fn read_fbo(&mut self) -> u32 {
        if self.read_fbo == 0 {
            self.read_fbo = self.backend.create_framebuffer();
        }
        self.read_fbo
    }

    /// Attach level `level`, layer `layer` of texture resource `res` to the
    /// framebuffer bound at `fb_target`, as `attachment`.
    fn attach(
        &mut self,
        fb_target: u32,
        attachment: u32,
        res: u32,
        level: u32,
        layer: u32,
    ) -> Option<Kind> {
        let r = self.resources.get(&res)?;
        let Host::Texture { gl: tex, gl_target } = r.host else {
            return None;
        };
        let kind = r.fmt?.kind;
        let attachment = match kind {
            Kind::Depth if attachment != gl::NONE => gl::DEPTH_ATTACHMENT,
            Kind::DepthStencil if attachment != gl::NONE => gl::DEPTH_STENCIL_ATTACHMENT,
            _ => attachment,
        };
        match gl_target {
            gl::TEXTURE_2D => self.backend.framebuffer_texture(
                fb_target,
                attachment,
                gl::TEXTURE_2D,
                tex,
                level,
                None,
            ),
            gl::TEXTURE_CUBE_MAP => self.backend.framebuffer_texture(
                fb_target,
                attachment,
                gl::TEXTURE_CUBE_MAP_POSITIVE_X + layer,
                tex,
                level,
                None,
            ),
            _ => self.backend.framebuffer_texture(
                fb_target,
                attachment,
                gl_target,
                tex,
                level,
                Some(layer),
            ),
        }
        Some(kind)
    }

    pub(crate) fn transfer(&mut self, t: &Transfer) -> Result<(), Errno> {
        let Some(r) = self.resources.get_mut(&t.res) else {
            return Err(EINVAL);
        };
        let bx = t.bx;
        match &mut r.host {
            Host::Buffer { gl: b, shadow } => {
                let (x, w) = (bx.x as usize, bx.w as usize);
                if x + w > shadow.len() {
                    return Err(EINVAL);
                }
                if t.to_host {
                    r.backing.read(t.offset, &mut shadow[x..x + w]);
                    let b = *b;
                    let data = shadow[x..x + w].to_vec();
                    self.backend.bind_buffer(gl::COPY_WRITE_BUFFER, b);
                    self.backend
                        .buffer_sub_data(gl::COPY_WRITE_BUFFER, x, &data);
                } else {
                    r.backing.write(t.offset, &shadow[x..x + w]);
                }
                Ok(())
            }
            Host::Texture { gl: tex, gl_target } => {
                let (tex, gl_target) = (*tex, *gl_target);
                let fmt = r.fmt.ok_or(EINVAL)?;
                if t.level > r.desc.last_level {
                    return Err(EINVAL);
                }
                let (lw, lh, ld) = r.level_size(t.level);
                if bx.x + bx.w > lw || bx.y + bx.h > lh || bx.z + bx.d.max(1) > ld.max(1) {
                    return Err(EINVAL);
                }
                let bpp = fmt.bpp as usize;
                let row = bx.w as usize * bpp;
                let stride = if t.stride == 0 {
                    lw as usize * bpp
                } else {
                    t.stride as usize
                };
                let lstride = if t.layer_stride == 0 {
                    stride * lh as usize
                } else {
                    t.layer_stride as usize
                };
                let backing = Arc::clone(&r.backing);
                let mut px = vec![0u8; row * bx.h as usize];
                for z in 0..bx.d.max(1) {
                    let base = t.offset + u64::from(z) * lstride as u64;
                    if t.to_host {
                        for y in 0..bx.h as usize {
                            backing
                                .read(base + (y * stride) as u64, &mut px[y * row..(y + 1) * row]);
                        }
                        fmt.to_host(&mut px);
                        self.backend.bind_texture(gl_target, tex);
                        let (target, zz, depth) = match gl_target {
                            gl::TEXTURE_CUBE_MAP => {
                                (gl::TEXTURE_CUBE_MAP_POSITIVE_X + bx.z + z, 0, 0)
                            }
                            gl::TEXTURE_2D => (gl::TEXTURE_2D, 0, 0),
                            t => (t, bx.z + z, 1),
                        };
                        self.backend.tex_sub_image(
                            target, t.level, bx.x, bx.y, zz, bx.w, bx.h, depth, fmt.format, fmt.ty,
                            &px,
                        );
                    } else {
                        if matches!(fmt.kind, Kind::Depth | Kind::DepthStencil) {
                            // GLES 3.0 can't read depth back.
                            let res = t.res;
                            self.warn(format!("depth readback of resource {res} unsupported"));
                            return Ok(());
                        }
                        let fbo = self.read_fbo();
                        self.backend.bind_framebuffer(gl::READ_FRAMEBUFFER, fbo);
                        self.attach(
                            gl::READ_FRAMEBUFFER,
                            gl::COLOR_ATTACHMENT0,
                            t.res,
                            t.level,
                            bx.z + z,
                        );
                        self.backend.read_buffer(gl::COLOR_ATTACHMENT0);
                        self.backend.read_pixels(
                            bx.x as i32,
                            bx.y as i32,
                            bx.w,
                            bx.h,
                            fmt.format,
                            fmt.ty,
                            &mut px,
                        );
                        fmt.convert_in_place(&mut px);
                        for y in 0..bx.h as usize {
                            backing.write(base + (y * stride) as u64, &px[y * row..(y + 1) * row]);
                        }
                    }
                }
                Ok(())
            }
            Host::None => Ok(()),
        }
    }

    pub(crate) fn submit(&mut self, ctx: u32, cmds: &[u32]) -> Result<(), Errno> {
        if !self.contexts.contains_key(&ctx) {
            return Err(EINVAL);
        }
        super::virgl::decode(self, ctx, cmds);
        Ok(())
    }

    // ---- what the decoder calls ----

    pub(crate) fn sub(&mut self, ctx: u32) -> &mut SubCtx {
        let c = self
            .contexts
            .get_mut(&ctx)
            .expect("context checked by submit");
        let cur = c.cur;
        c.subs.entry(cur).or_default()
    }

    pub(crate) fn create_sub_ctx(&mut self, ctx: u32, id: u32) {
        if let Some(c) = self.contexts.get_mut(&ctx) {
            c.subs.entry(id).or_default();
        }
    }

    pub(crate) fn destroy_sub_ctx(&mut self, ctx: u32, id: u32) {
        let sub = self.contexts.get_mut(&ctx).and_then(|c| {
            if c.cur == id {
                c.cur = 0;
            }
            if id == 0 { None } else { c.subs.remove(&id) }
        });
        if let Some(sub) = sub {
            self.drop_sub(sub);
        }
    }

    pub(crate) fn set_sub_ctx(&mut self, ctx: u32, id: u32) {
        if let Some(c) = self.contexts.get_mut(&ctx) {
            c.subs.entry(id).or_default();
            c.cur = id;
        }
    }

    pub(crate) fn create_object(&mut self, ctx: u32, handle: u32, obj: Object) {
        let sub = self.sub(ctx);
        // A shader arriving in pieces continues the object it started.
        if let (Object::Shader(more), Some(Object::Shader(cur))) =
            (&obj, sub.objects.get_mut(&handle))
            && more.total == 0
            && cur.text.len() < cur.total
        {
            cur.text.extend_from_slice(&more.text);
            Self::finish_shader(cur);
            return;
        }
        let mut obj = obj;
        if let Object::Shader(s) = &mut obj {
            Self::finish_shader(s);
        }
        sub.objects.insert(handle, obj);
    }

    fn finish_shader(s: &mut ShaderObj) {
        if s.text.len() < s.total || s.parsed.is_some() {
            return;
        }
        let end = s.text.iter().position(|&b| b == 0).unwrap_or(s.text.len());
        let text = String::from_utf8_lossy(&s.text[..end]);
        match tgsi::parse(&text) {
            Ok(sh) => s.parsed = Some(Arc::new(sh)),
            Err(e) => s.error = Some(format!("{e}\n{text}")),
        }
    }

    pub(crate) fn destroy_object(&mut self, ctx: u32, handle: u32) {
        let sub = self.sub(ctx);
        if let Some(Object::Shader(_)) = sub.objects.remove(&handle) {
            let dead: Vec<_> = sub
                .programs
                .keys()
                .filter(|(v, f, _)| *v == handle || *f == handle)
                .copied()
                .collect();
            let progs: Vec<_> = dead
                .iter()
                .filter_map(|k| sub.programs.remove(k).flatten())
                .collect();
            for p in progs {
                self.backend.delete_program(p.gl);
            }
        }
    }

    pub(crate) fn bind_object(&mut self, ctx: u32, kind: u32, handle: u32) {
        let sub = self.sub(ctx);
        match kind {
            1 => sub.blend = handle,
            2 => sub.rasterizer = handle,
            3 => sub.dsa = handle,
            5 => sub.vertex_elements = handle,
            _ => {}
        }
    }

    pub(crate) fn shader_text(&mut self, ctx: u32, handle: u32) -> Option<String> {
        match self.sub(ctx).objects.get(&handle) {
            Some(Object::Shader(s)) => Some(String::from_utf8_lossy(&s.text).into_owned()),
            _ => None,
        }
    }

    pub(crate) fn set_viewport(&mut self, ctx: u32, vp: Viewport) {
        self.sub(ctx).viewport = vp;
    }

    pub(crate) fn set_scissor(&mut self, ctx: u32, s: [u32; 4]) {
        self.sub(ctx).scissor = s;
    }

    pub(crate) fn set_framebuffer(&mut self, ctx: u32, cbufs: Vec<u32>, zsurf: u32) {
        let sub = self.sub(ctx);
        sub.cbufs = cbufs;
        sub.zsurf = zsurf;
    }

    pub(crate) fn set_vertex_buffers(&mut self, ctx: u32, vbs: Vec<VertexBuffer>) {
        self.sub(ctx).vbufs = vbs;
    }

    pub(crate) fn set_index_buffer(&mut self, ctx: u32, ib: IndexBuffer) {
        self.sub(ctx).ib = ib;
    }

    pub(crate) fn set_constant_buffer(&mut self, ctx: u32, stage: usize, data: Vec<u32>) {
        if stage < 6 {
            self.sub(ctx).consts[stage] = data;
        }
    }

    pub(crate) fn set_uniform_buffer(&mut self, ctx: u32, stage: usize, index: usize, ubo: Ubo) {
        if stage < 6 && index < 16 {
            self.sub(ctx).ubos[stage][index] = ubo;
        }
    }

    pub(crate) fn set_sampler_views(
        &mut self,
        ctx: u32,
        stage: usize,
        start: usize,
        handles: &[u32],
    ) {
        if stage < 6 {
            let sub = self.sub(ctx);
            for (i, h) in handles.iter().enumerate() {
                if let Some(s) = sub.views[stage].get_mut(start + i) {
                    *s = *h;
                }
            }
        }
    }

    pub(crate) fn bind_sampler_states(
        &mut self,
        ctx: u32,
        stage: usize,
        start: usize,
        handles: &[u32],
    ) {
        if stage < 6 {
            let sub = self.sub(ctx);
            for (i, h) in handles.iter().enumerate() {
                if let Some(s) = sub.samplers[stage].get_mut(start + i) {
                    *s = *h;
                }
            }
        }
    }

    pub(crate) fn bind_shader(&mut self, ctx: u32, stage: usize, handle: u32) {
        if stage < 6 {
            self.sub(ctx).shaders[stage] = handle;
        }
    }

    pub(crate) fn set_stencil_ref(&mut self, ctx: u32, front: u32, back: u32) {
        self.sub(ctx).stencil_ref = [front, back];
    }

    pub(crate) fn set_blend_color(&mut self, ctx: u32, c: [f32; 4]) {
        self.sub(ctx).blend_color = c;
    }

    /// Write a finished query's result where the guest polls for it.
    pub(crate) fn end_query(&mut self, ctx: u32, handle: u32) {
        let Some(Object::Query(q)) = self.sub(ctx).objects.get(&handle).cloned() else {
            return;
        };
        let Some(r) = self.resources.get(&q.res) else {
            return;
        };
        // struct virgl_host_query_state { u32 state; u32 size; u64 result }:
        // done, with a result that lets occlusion tests pass (GLES has no
        // way to count samples, as virglrenderer's GLES path also fakes).
        let result: u64 = match q.ty {
            // PIPE_QUERY_OCCLUSION_COUNTER / PREDICATE / PREDICATE_CONSERVATIVE.
            0..=2 => 1,
            _ => 0,
        };
        let mut st = [0u8; 16];
        st[0..4].copy_from_slice(&1u32.to_le_bytes());
        st[4..8].copy_from_slice(&8u32.to_le_bytes());
        st[8..16].copy_from_slice(&result.to_le_bytes());
        r.backing.write(u64::from(q.offset), &st);
    }

    // ---- drawing ----

    /// Bind the draw framebuffer for the current surfaces; their size and
    /// the colour buffer kinds.
    fn bind_framebuffer(&mut self, ctx: u32) -> Option<(u32, u32)> {
        if self.draw_fbo == 0 {
            self.draw_fbo = self.backend.create_framebuffer();
        }
        let fbo = self.draw_fbo;
        self.backend.bind_framebuffer(gl::DRAW_FRAMEBUFFER, fbo);
        let sub = self.sub(ctx);
        let cbufs = sub.cbufs.clone();
        let surfs: Vec<Option<Surface>> = cbufs
            .iter()
            .map(|h| match sub.objects.get(h) {
                Some(Object::Surface(s)) => Some(*s),
                _ => None,
            })
            .collect();
        let zs = match sub.objects.get(&sub.zsurf) {
            Some(Object::Surface(s)) => Some(*s),
            _ => None,
        };
        let mut size = None;
        let mut bufs = Vec::new();
        for i in 0..8u32 {
            let att = gl::COLOR_ATTACHMENT0 + i;
            match surfs.get(i as usize).copied().flatten() {
                Some(s)
                    if self
                        .attach(gl::DRAW_FRAMEBUFFER, att, s.res, s.level, s.first_layer)
                        .is_some() =>
                {
                    bufs.push(att);
                    if size.is_none() {
                        size = self.resources.get(&s.res).map(|r| {
                            let (w, h, _) = r.level_size(s.level);
                            (w, h)
                        });
                    }
                }
                _ => {
                    if i < 4 {
                        self.backend.framebuffer_texture(
                            gl::DRAW_FRAMEBUFFER,
                            att,
                            gl::TEXTURE_2D,
                            0,
                            0,
                            None,
                        );
                    }
                    bufs.push(gl::NONE);
                }
            }
        }
        while bufs.last() == Some(&gl::NONE) {
            bufs.pop();
        }
        self.backend.draw_buffers(&bufs);
        self.backend.framebuffer_texture(
            gl::DRAW_FRAMEBUFFER,
            gl::DEPTH_STENCIL_ATTACHMENT,
            gl::TEXTURE_2D,
            0,
            0,
            None,
        );
        if let Some(s) = zs
            && self
                .attach(
                    gl::DRAW_FRAMEBUFFER,
                    gl::DEPTH_ATTACHMENT,
                    s.res,
                    s.level,
                    s.first_layer,
                )
                .is_some()
            && size.is_none()
        {
            size = self.resources.get(&s.res).map(|r| {
                let (w, h, _) = r.level_size(s.level);
                (w, h)
            });
        }
        size
    }

    /// The kinds of the bound colour buffers.
    fn cbuf_kinds(&mut self, ctx: u32) -> Vec<Kind> {
        let sub = self.sub(ctx);
        let fmts: Vec<u32> = sub
            .cbufs
            .iter()
            .map(|h| match sub.objects.get(h) {
                Some(Object::Surface(s)) => s.res,
                _ => 0,
            })
            .collect();
        fmts.iter()
            .map(|res| {
                self.resources
                    .get(res)
                    .and_then(|r| r.fmt)
                    .map_or(Kind::Float, |f| f.kind)
            })
            .collect()
    }

    pub(crate) fn clear(
        &mut self,
        ctx: u32,
        buffers: u32,
        color: [u32; 4],
        depth: f64,
        stencil: u32,
    ) {
        if self.bind_framebuffer(ctx).is_none() {
            return;
        }
        let kinds = self.cbuf_kinds(ctx);
        let b = &mut self.backend;
        b.enable(gl::SCISSOR_TEST, false);
        b.enable(gl::RASTERIZER_DISCARD, false);
        b.color_mask(true, true, true, true);
        // PIPE_CLEAR_COLOR0 << n, PIPE_CLEAR_DEPTH = 1, PIPE_CLEAR_STENCIL = 2.
        for (i, k) in kinds.iter().enumerate() {
            if buffers & (4 << i) == 0 {
                continue;
            }
            match k {
                Kind::Uint => b.clear_buffer_uiv(gl::COLOR, i as u32, color),
                Kind::Sint => b.clear_buffer_iv(gl::COLOR, i as u32, color.map(u32::cast_signed)),
                _ => b.clear_buffer_fv(gl::COLOR, i as u32, color.map(f32::from_bits)),
            }
        }
        let (d, s) = (buffers & 1 != 0, buffers & 2 != 0);
        if d {
            b.depth_mask(true);
        }
        if s {
            b.stencil_mask_separate(gl::FRONT_AND_BACK, 0xff);
        }
        match (d, s) {
            (true, true) => b.clear_buffer_fi(depth as f32, stencil as i32),
            (true, false) => b.clear_buffer_fv(gl::DEPTH, 0, [depth as f32; 4]),
            (false, true) => b.clear_buffer_iv(gl::STENCIL, 0, [stencil as i32; 4]),
            _ => {}
        }
    }

    fn shader(&mut self, ctx: u32, stage: usize) -> Option<Result<Arc<tgsi::Shader>, String>> {
        let sub = self.sub(ctx);
        let h = sub.shaders[stage];
        match sub.objects.get(&h) {
            Some(Object::Shader(s)) => match (&s.parsed, &s.error) {
                (Some(p), _) => Some(Ok(Arc::clone(p))),
                (None, Some(e)) => Some(Err(e.clone())),
                _ => Some(Err("shader text incomplete".into())),
            },
            _ => None,
        }
    }

    /// The linked program for the bound shaders under `opts`, made on first
    /// use; `None` (logged) if it can't be.
    fn program(&mut self, ctx: u32, opts: glsl::Options) -> Option<usize> {
        let key = {
            let sub = self.sub(ctx);
            (sub.shaders[0], sub.shaders[1], opts)
        };
        if !self.sub(ctx).programs.contains_key(&key) {
            let prog = self.link(ctx, &opts);
            self.sub(ctx).programs.insert(key, prog);
        }
        self.sub(ctx).programs.get(&key)?.as_ref().map(|_| 0)
    }

    fn link(&mut self, ctx: u32, opts: &glsl::Options) -> Option<Program> {
        let vs = match self.shader(ctx, 0) {
            Some(Ok(s)) => s,
            Some(Err(e)) => {
                self.warn(format!("vertex shader: {e}"));
                return None;
            }
            None => return None,
        };
        let fs = match self.shader(ctx, 1) {
            Some(Ok(s)) => s,
            Some(Err(e)) => {
                self.warn(format!("fragment shader: {e}"));
                return None;
            }
            None => return None,
        };
        let p = match glsl::link(&vs, &fs, opts) {
            Ok(p) => p,
            Err(e) => {
                self.warn(format!("can't translate shaders: {e}\n{vs:#?}"));
                return None;
            }
        };
        if self.trace {
            eprintln!(
                "nixvm gpu: linking\n--- vs ---\n{}--- fs ---\n{}",
                p.vs, p.fs
            );
        }
        let prog = match self.backend.create_program(&p.vs, &p.fs) {
            Ok(g) => g,
            Err(log) => {
                self.warn(format!(
                    "host GLSL compile failed: {log}\n--- vs ---\n{}--- fs ---\n{}",
                    p.vs, p.fs
                ));
                return None;
            }
        };
        let b = &mut self.backend;
        b.use_program(prog);
        let mut out = Program {
            gl: prog,
            consts: [(-1, 0); 2],
            ubos: [Vec::new(), Vec::new()],
            samplers: [Vec::new(), Vec::new()],
            flip_y: b.uniform_location(prog, "nx_flip_y"),
            fb_height: b.uniform_location(prog, "nx_fb_height"),
        };
        for (six, st) in ["vs", "fs"].iter().enumerate() {
            if p.layout.consts[six] > 0 {
                out.consts[six] = (
                    b.uniform_location(prog, &format!("nx_{st}c")),
                    p.layout.consts[six],
                );
            }
            for &k in &p.layout.ubos[six] {
                let idx = b.uniform_block_index(prog, &format!("nx_{st}ub{k}"));
                if idx != u32::MAX {
                    b.uniform_block_binding(prog, idx, six as u32 * 12 + k - 1);
                }
                out.ubos[six].push(k);
            }
            for &s in &p.layout.samplers[six] {
                let loc = b.uniform_location(prog, &format!("nx_{st}s{s}"));
                // Fragment samplers use units 0.., vertex ones 16...
                let unit = if six == 0 { 16 + s } else { s };
                b.uniform_1i(loc, unit as i32);
                out.samplers[six].push((s, loc));
            }
        }
        Some(out)
    }

    #[allow(clippy::too_many_lines)]
    pub(crate) fn draw(&mut self, ctx: u32, info: &DrawInfo) {
        if info.count == 0 {
            return;
        }
        let Some((fb_w, fb_h)) = self.bind_framebuffer(ctx) else {
            return;
        };
        let kinds = self.cbuf_kinds(ctx);
        // State objects.
        let (blend, rs, dsa, ves, vp, scissor, stencil_ref, blend_color) = {
            let sub = self.sub(ctx);
            let get = |h: u32| sub.objects.get(&h).cloned();
            let blend = if let Some(Object::Blend(b)) = get(sub.blend) {
                *b
            } else {
                // None bound: blending off, every channel written.
                let mut b = Blend::default();
                b.rt[0].colormask = 0xf;
                b
            };
            let rs = match get(sub.rasterizer) {
                Some(Object::Rasterizer(r)) => r,
                _ => Rasterizer::default(),
            };
            let dsa = match get(sub.dsa) {
                Some(Object::Dsa(d)) => d,
                _ => Dsa::default(),
            };
            let ves = match get(sub.vertex_elements) {
                Some(Object::VertexElements(v)) => v,
                _ => Vec::new(),
            };
            (
                blend,
                rs,
                dsa,
                ves,
                sub.viewport,
                sub.scissor,
                sub.stencil_ref,
                sub.blend_color,
            )
        };
        let mut opts = glsl::Options {
            flatshade: rs.flatshade(),
            nr_cbufs: kinds.len() as u32,
            ..glsl::Options::default()
        };
        for (i, k) in kinds.iter().enumerate() {
            match k {
                Kind::Uint => opts.uint_cbufs |= 1 << i,
                Kind::Sint => opts.sint_cbufs |= 1 << i,
                _ => {}
            }
        }
        for (i, ve) in ves.iter().enumerate() {
            if let Some(f) = format::lookup(ve.format).filter(|f| f.vertex_is_integer()) {
                if f.kind == Kind::Uint {
                    opts.uint_inputs |= 1 << i;
                } else {
                    opts.sint_inputs |= 1 << i;
                }
            }
        }
        if self.program(ctx, opts).is_none() {
            return;
        }
        let key = {
            let sub = self.sub(ctx);
            (sub.shaders[0], sub.shaders[1], opts)
        };
        let flip = vp.scale[1] < 0.0;
        // Program and its inputs.
        let (prog_gl, consts, ubos, samplers, flip_loc, fbh_loc) = {
            let p = self.sub(ctx).programs[&key]
                .as_ref()
                .expect("checked above");
            (
                p.gl,
                p.consts,
                p.ubos.clone(),
                p.samplers.clone(),
                p.flip_y,
                p.fb_height,
            )
        };
        self.backend.use_program(prog_gl);
        self.backend
            .uniform_1f(flip_loc, if flip { -1.0 } else { 1.0 });
        self.backend
            .uniform_1f(fbh_loc, f32::from(u16::try_from(fb_h).unwrap_or(u16::MAX)));
        for six in 0..2 {
            let (loc, n) = consts[six];
            if loc >= 0 {
                let mut data = self.sub(ctx).consts[six].clone();
                data.resize(n as usize * 4, 0);
                self.backend.uniform_4uiv(loc, &data);
            }
            for &k in &ubos[six] {
                let Some(u) = self.sub(ctx).ubos[six].get(k as usize).copied() else {
                    continue;
                };
                if let Some(Host::Buffer { gl: b, .. }) =
                    self.resources.get(&u.res).map(|r| &r.host)
                {
                    self.backend.bind_buffer_range(
                        gl::UNIFORM_BUFFER,
                        six as u32 * 12 + k - 1,
                        *b,
                        u.offset as usize,
                        u.len as usize,
                    );
                }
            }
            for &(slot, _) in &samplers[six] {
                let unit = if six == 0 { 16 + slot } else { slot };
                self.bind_sampler(ctx, six, slot, unit);
            }
        }
        // Viewport: Gallium's scale/translate as a GL viewport + depth range.
        let (sx, sy) = (vp.scale[0], vp.scale[1].abs());
        self.backend.viewport(
            (vp.translate[0] - sx).round() as i32,
            (vp.translate[1] - sy).round() as i32,
            (2.0 * sx).round() as i32,
            (2.0 * sy).round() as i32,
        );
        let (near, far) = if rs.clip_halfz() {
            (vp.translate[2], vp.translate[2] + vp.scale[2])
        } else {
            (vp.translate[2] - vp.scale[2], vp.translate[2] + vp.scale[2])
        };
        self.backend
            .depth_range(near.clamp(0.0, 1.0), far.clamp(0.0, 1.0));
        // Rasterizer.
        let b = &mut self.backend;
        b.enable(gl::SCISSOR_TEST, rs.scissor());
        if rs.scissor() {
            let [minx, miny, maxx, maxy] = scissor;
            b.scissor(
                minx as i32,
                miny as i32,
                maxx.saturating_sub(minx) as i32,
                maxy.saturating_sub(miny) as i32,
            );
        }
        b.enable(gl::RASTERIZER_DISCARD, rs.discard());
        match rs.cull() {
            0 => b.enable(gl::CULL_FACE, false),
            c => {
                b.enable(gl::CULL_FACE, true);
                b.cull_face(match c {
                    1 => gl::FRONT,
                    2 => gl::BACK,
                    _ => gl::FRONT_AND_BACK,
                });
            }
        }
        // Gallium measures winding with window y pointing down, GL with it
        // pointing up: the same triangle is "CCW" to one and "CW" to the
        // other — unless the vertex shader flipped it vertically too.
        b.front_face(if rs.front_ccw() ^ flip {
            gl::CW
        } else {
            gl::CCW
        });
        b.enable(gl::POLYGON_OFFSET_FILL, rs.offset_tri());
        if rs.offset_tri() {
            b.polygon_offset(rs.offset_scale, rs.offset_units);
        }
        b.line_width(1.0);
        b.enable(gl::PRIMITIVE_RESTART_FIXED_INDEX, info.primitive_restart);
        // Depth / stencil.
        let depth_on = dsa.s0 & 1 != 0;
        b.enable(gl::DEPTH_TEST, depth_on);
        b.depth_mask(dsa.s0 & 2 != 0);
        b.depth_func(gl::NEVER + ((dsa.s0 >> 2) & 7));
        let stencil_on = dsa.stencil[0] & 1 != 0;
        b.enable(gl::STENCIL_TEST, stencil_on);
        if stencil_on {
            let back = usize::from(dsa.stencil[1] & 1 != 0);
            for (face, i) in [(gl::FRONT, 0usize), (gl::BACK, back)] {
                let s = dsa.stencil[i];
                b.stencil_func_separate(
                    face,
                    gl::NEVER + ((s >> 1) & 7),
                    stencil_ref[i] as i32,
                    (s >> 13) & 0xff,
                );
                b.stencil_op_separate(
                    face,
                    stencil_op((s >> 4) & 7),
                    stencil_op((s >> 10) & 7),
                    stencil_op((s >> 7) & 7),
                );
                b.stencil_mask_separate(face, (s >> 21) & 0xff);
            }
        }
        // Blend (GLES 3.0 has one blend state for every buffer: render
        // target 0's).
        let rt = blend.rt[0];
        b.enable(gl::BLEND, rt.enable);
        if rt.enable {
            b.blend_func_separate(
                blend_factor(rt.rgb_src),
                blend_factor(rt.rgb_dst),
                blend_factor(rt.alpha_src),
                blend_factor(rt.alpha_dst),
            );
            b.blend_equation_separate(blend_func(rt.rgb_func), blend_func(rt.alpha_func));
            b.blend_color(blend_color);
        }
        let m = rt.colormask;
        b.color_mask(m & 1 != 0, m & 2 != 0, m & 4 != 0, m & 8 != 0);
        b.enable(gl::SAMPLE_ALPHA_TO_COVERAGE, blend.alpha_to_coverage);
        b.enable(gl::DITHER, false);
        let _ = fb_w;
        // Vertex input.
        if self.vao == 0 {
            self.vao = self.backend.create_vertex_array();
        }
        self.backend.bind_vertex_array(self.vao);
        let vbufs = self.sub(ctx).vbufs.clone();
        if self.trace {
            let c = &self.sub(ctx).consts;
            eprintln!(
                "nixvm gpu: draw {info:?}\n  elements {ves:?}\n  buffers {vbufs:?}\n  vs consts {:?}\n  fs consts {:?}",
                c[0].iter().map(|w| f32::from_bits(*w)).collect::<Vec<_>>(),
                c[1].iter().map(|w| f32::from_bits(*w)).collect::<Vec<_>>()
            );
            for vb in &vbufs {
                if let Some(Host::Buffer { shadow, .. }) =
                    self.resources.get(&vb.res).map(|r| &r.host)
                {
                    let start = vb.offset as usize;
                    let words: Vec<f32> = shadow
                        [start.min(shadow.len())..(start + 48).min(shadow.len())]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c))
                        .collect();
                    eprintln!("  buffer {} @{start}: {words:?}", vb.res);
                }
            }
        }
        let bias = if info.indexed {
            i64::from(info.index_bias)
        } else {
            0
        };
        for i in 0..16u32 {
            let Some(ve) = ves.get(i as usize) else {
                self.backend.enable_vertex_attrib_array(i, false);
                continue;
            };
            let vb = vbufs.get(ve.vb as usize).copied().unwrap_or_default();
            let fmt = format::lookup(ve.format).and_then(|f| f.vertex.map(|v| (f, v)));
            let buf = match self.resources.get(&vb.res).map(|r| &r.host) {
                Some(Host::Buffer { gl: b, .. }) => Some(*b),
                _ => None,
            };
            if let (Some((f, (n, ty, norm))), Some(buf)) = (fmt, buf) {
                {
                    let off =
                        i64::from(vb.offset) + i64::from(ve.offset) + bias * i64::from(vb.stride);
                    self.backend.bind_buffer(gl::ARRAY_BUFFER, buf);
                    self.backend.enable_vertex_attrib_array(i, true);
                    self.backend.vertex_attrib_pointer(
                        i,
                        n,
                        ty,
                        norm,
                        f.vertex_is_integer(),
                        vb.stride,
                        off.max(0) as usize,
                    );
                    self.backend.vertex_attrib_divisor(i, ve.divisor);
                }
            } else {
                {
                    self.backend.enable_vertex_attrib_array(i, false);
                    let what = ve.format;
                    self.warn(format!(
                        "vertex element format {what} or its buffer unsupported"
                    ));
                }
            }
        }
        let mode = info.mode;
        if mode > 6 {
            self.warn(format!("primitive mode {mode} unsupported on GLES"));
            return;
        }
        let instances = info.instance_count.max(1) as i32;
        if info.indexed {
            let ib = self.sub(ctx).ib;
            let (ty, size) = match ib.size {
                1 => (gl::UNSIGNED_BYTE, 1usize),
                2 => (gl::UNSIGNED_SHORT, 2),
                _ => (gl::UNSIGNED_INT, 4),
            };
            let start = ib.offset as usize + info.start as usize * size;
            let len = info.count as usize * size;
            let Some(Host::Buffer { shadow, .. }) = self.resources.get(&ib.res).map(|r| &r.host)
            else {
                self.warn("indexed draw without an index buffer".into());
                return;
            };
            let Some(data) = shadow.get(start..start + len).map(<[u8]>::to_vec) else {
                self.warn("index range outside the index buffer".into());
                return;
            };
            if self.ebo == 0 {
                self.ebo = self.backend.create_buffer();
            }
            self.backend.bind_buffer(gl::ELEMENT_ARRAY_BUFFER, self.ebo);
            self.backend.buffer_data(
                gl::ELEMENT_ARRAY_BUFFER,
                data.len(),
                Some(&data),
                gl::STREAM_DRAW,
            );
            self.backend
                .draw_elements(mode, info.count as i32, ty, 0, instances);
        } else {
            self.backend
                .draw_arrays(mode, info.start as i32, info.count as i32, instances);
        }
        if self.trace {
            let e = self.backend.get_error();
            if e != gl::NO_ERROR {
                eprintln!("nixvm gpu: GL error 0x{e:x} after draw");
            }
        }
    }

    fn bind_sampler(&mut self, ctx: u32, six: usize, slot: u32, unit: u32) {
        let (view, state) = {
            let sub = self.sub(ctx);
            let (Some(&vh), Some(&sh)) = (
                sub.views[six].get(slot as usize),
                sub.samplers[six].get(slot as usize),
            ) else {
                return;
            };
            let v = match sub.objects.get(&vh) {
                Some(Object::SamplerView(v)) => Some(*v),
                _ => None,
            };
            let s = match sub.objects.get(&sh) {
                Some(Object::SamplerState(s)) => Some(*s),
                _ => None,
            };
            (v, s)
        };
        self.backend.active_texture(gl::TEXTURE0 + unit);
        let tex = view
            .and_then(|v| self.resources.get(&v.res))
            .and_then(|r| match r.host {
                Host::Texture { gl, gl_target } => Some((gl, gl_target, r.desc.last_level)),
                _ => None,
            });
        let Some((tex, t, last)) = tex else {
            return;
        };
        self.backend.bind_texture(t, tex);
        let v = view.expect("tex implies view");
        self.backend
            .tex_parameter(t, gl::TEXTURE_BASE_LEVEL, v.first_level as i32);
        self.backend
            .tex_parameter(t, gl::TEXTURE_MAX_LEVEL, v.last_level.min(last) as i32);
        let s = state.unwrap_or_default();
        let s0 = s.s0;
        let min_img = (s0 >> 9) & 1;
        let min_mip = (s0 >> 11) & 3;
        let mag = (s0 >> 13) & 1;
        // PIPE_TEX_MIPFILTER_NEAREST = 0, _LINEAR = 1, _NONE = 2.
        let min = match (min_mip, min_img) {
            (0, 0) => gl::NEAREST_MIPMAP_NEAREST,
            (0, _) => gl::LINEAR_MIPMAP_NEAREST,
            (1, 0) => gl::NEAREST_MIPMAP_LINEAR,
            (1, _) => gl::LINEAR_MIPMAP_LINEAR,
            (_, 0) => gl::NEAREST,
            _ => gl::LINEAR,
        };
        // Samplers are set on the texture: one less object kind for a
        // backend to manage, and every texture is bound fresh each draw.
        let b = &mut self.backend;
        b.bind_sampler(unit, 0);
        b.tex_parameter(t, gl::TEXTURE_MIN_FILTER, min as i32);
        b.tex_parameter(
            t,
            gl::TEXTURE_MAG_FILTER,
            if mag == 0 { gl::NEAREST } else { gl::LINEAR } as i32,
        );
        b.tex_parameter(t, gl::TEXTURE_WRAP_S, wrap(s0 & 7));
        b.tex_parameter(t, gl::TEXTURE_WRAP_T, wrap((s0 >> 3) & 7));
        b.tex_parameter(t, gl::TEXTURE_WRAP_R, wrap((s0 >> 6) & 7));
        if (s0 >> 15) & 1 != 0 {
            b.tex_parameter(
                t,
                gl::TEXTURE_COMPARE_MODE,
                gl::COMPARE_REF_TO_TEXTURE as i32,
            );
            b.tex_parameter(
                t,
                gl::TEXTURE_COMPARE_FUNC,
                (gl::NEVER + ((s0 >> 16) & 7)) as i32,
            );
        } else {
            b.tex_parameter(t, gl::TEXTURE_COMPARE_MODE, gl::NONE as i32);
        }
    }

    /// `RESOURCE_COPY_REGION`: buffers through their shadows, textures
    /// through a framebuffer blit.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn copy_region(
        &mut self,
        dst: u32,
        dst_level: u32,
        dx: u32,
        dy: u32,
        dz: u32,
        src: u32,
        src_level: u32,
        sbox: Box3,
    ) {
        let src_buf = match self.resources.get(&src).map(|r| &r.host) {
            Some(Host::Buffer { shadow, .. }) => shadow
                .get(sbox.x as usize..(sbox.x + sbox.w) as usize)
                .map(<[u8]>::to_vec),
            _ => None,
        };
        if let Some(data) = src_buf {
            if let Some(Resource {
                host: Host::Buffer { gl: b, shadow },
                ..
            }) = self.resources.get_mut(&dst)
            {
                let end = dx as usize + data.len();
                if end <= shadow.len() {
                    shadow[dx as usize..end].copy_from_slice(&data);
                    let b = *b;
                    self.backend.bind_buffer(gl::COPY_WRITE_BUFFER, b);
                    self.backend
                        .buffer_sub_data(gl::COPY_WRITE_BUFFER, dx as usize, &data);
                }
            }
            return;
        }
        let Some(kind) = self.resources.get(&src).and_then(|r| r.fmt).map(|f| f.kind) else {
            return;
        };
        let mask = match kind {
            Kind::Depth => gl::DEPTH_BUFFER_BIT,
            Kind::DepthStencil => gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT,
            _ => gl::COLOR_BUFFER_BIT,
        };
        for z in 0..sbox.d.max(1) {
            self.blit_layer(
                src,
                src_level,
                sbox.z + z,
                [
                    sbox.x as i32,
                    sbox.y as i32,
                    (sbox.x + sbox.w) as i32,
                    (sbox.y + sbox.h) as i32,
                ],
                dst,
                dst_level,
                dz + z,
                [
                    dx as i32,
                    dy as i32,
                    (dx + sbox.w) as i32,
                    (dy + sbox.h) as i32,
                ],
                mask,
                gl::NEAREST,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn blit_layer(
        &mut self,
        src: u32,
        src_level: u32,
        src_layer: u32,
        src_rect: [i32; 4],
        dst: u32,
        dst_level: u32,
        dst_layer: u32,
        dst_rect: [i32; 4],
        mask: u32,
        filter: u32,
    ) {
        let rfbo = self.read_fbo();
        if self.draw_fbo == 0 {
            self.draw_fbo = self.backend.create_framebuffer();
        }
        let dfbo = self.draw_fbo;
        let att = if mask == gl::COLOR_BUFFER_BIT {
            gl::COLOR_ATTACHMENT0
        } else {
            gl::DEPTH_ATTACHMENT
        };
        self.backend.bind_framebuffer(gl::READ_FRAMEBUFFER, rfbo);
        if self
            .attach(gl::READ_FRAMEBUFFER, att, src, src_level, src_layer)
            .is_none()
        {
            return;
        }
        self.backend.bind_framebuffer(gl::DRAW_FRAMEBUFFER, dfbo);
        for i in 0..4 {
            self.backend.framebuffer_texture(
                gl::DRAW_FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0 + i,
                gl::TEXTURE_2D,
                0,
                0,
                None,
            );
        }
        self.backend.framebuffer_texture(
            gl::DRAW_FRAMEBUFFER,
            gl::DEPTH_STENCIL_ATTACHMENT,
            gl::TEXTURE_2D,
            0,
            0,
            None,
        );
        if self
            .attach(gl::DRAW_FRAMEBUFFER, att, dst, dst_level, dst_layer)
            .is_none()
        {
            return;
        }
        if att == gl::COLOR_ATTACHMENT0 {
            self.backend.read_buffer(gl::COLOR_ATTACHMENT0);
            self.backend.draw_buffers(&[gl::COLOR_ATTACHMENT0]);
        }
        self.backend.enable(gl::SCISSOR_TEST, false);
        self.backend.enable(gl::RASTERIZER_DISCARD, false);
        self.backend.color_mask(true, true, true, true);
        self.backend
            .blit_framebuffer(src_rect, dst_rect, mask, filter);
    }

    /// `RESOURCE_INLINE_WRITE`: data carried in the command stream.
    pub(crate) fn inline_write(
        &mut self,
        res: u32,
        level: u32,
        bx: Box3,
        stride: u32,
        layer_stride: u32,
        data: &[u8],
    ) {
        let Some(r) = self.resources.get_mut(&res) else {
            return;
        };
        match &mut r.host {
            Host::Buffer { gl: b, shadow } => {
                let (x, w) = (bx.x as usize, (bx.w as usize).min(data.len()));
                if x + w <= shadow.len() {
                    shadow[x..x + w].copy_from_slice(&data[..w]);
                    let b = *b;
                    self.backend.bind_buffer(gl::COPY_WRITE_BUFFER, b);
                    self.backend
                        .buffer_sub_data(gl::COPY_WRITE_BUFFER, x, &data[..w]);
                }
            }
            Host::Texture { .. } => {
                // Stage it through the backing's layout: a transfer from a
                // scratch backing holding the data.
                let scratch = super::MemBacking::new(data.len());
                scratch.write(0, data);
                let real = std::mem::replace(&mut r.backing, scratch);
                let _ = self.transfer(&Transfer {
                    res,
                    level,
                    bx,
                    offset: 0,
                    stride,
                    layer_stride,
                    to_host: true,
                });
                if let Some(r) = self.resources.get_mut(&res) {
                    r.backing = real;
                }
            }
            Host::None => {}
        }
    }
}
