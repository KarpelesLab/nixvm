//! The renderer: virgl's Gallium-level state, kept per context and
//! sub-context the way virglrenderer keeps it, and turned into GLES 3.0
//! calls on a [`GpuBackend`] at each draw, clear, copy and transfer.
//!
//! GL state is applied in full before every draw, so contexts sharing the
//! one host GL context never see each other's leftovers; a state cache in
//! front of the backend ([`super::cache`]) drops the calls that would set
//! what is already set (peglgears: 53 host calls a draw become 5). Gallium
//! conventions that GLES lacks are emulated here:
//!
//! - **Viewport y flip.** A Gallium viewport may scale y negatively (Mesa
//!   does for window-system buffers, whose row 0 is the top). GL's can't;
//!   the vertex shader negates `gl_Position.y` instead (`nx_flip_y`), which
//!   puts every vertex at the window position Gallium's viewport would.
//!   Gallium's `front_ccw` is measured in those window coordinates with y
//!   pointing down, GL's with it up, so the face names always swap
//!   (flipped or not: a window-system draw with culling showed it). Rendering always goes to textures
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

use super::cache::FxMap;
use std::sync::Arc;

use super::backend::HostCaps;
use super::cache::StateCache;
use super::format::{self, Format, Kind};
use super::glsl::{self, StreamOutput};
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
        /// Bumped on every write: texel-buffer textures made from it are
        /// stale once it moves on.
        version: u64,
    },
    Texture {
        gl: u32,
        gl_target: u32,
        /// A multisampled resource: the renderbuffer it's drawn in (the
        /// texture holds its resolve, for sampling and reading), and
        /// whether that resolve is out of date.
        msaa: Option<Msaa>,
    },
    /// A format or target the host can't hold: transfers land in the
    /// backing only.
    None,
}

#[derive(Clone, Copy)]
struct Msaa {
    rb: u32,
    dirty: bool,
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
    StreamTarget(StreamTarget),
}

/// A stream-output target: where in buffer `res` captured vertices go,
/// and how many bytes of it the captures so far have filled.
#[derive(Debug, Clone, Copy)]
pub struct StreamTarget {
    pub res: u32,
    pub offset: u32,
    pub size: u32,
    pub filled: u32,
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
    /// User clip planes enabled (bit `n`: plane `n`).
    pub clip_planes: u8,
}

impl Rasterizer {
    /// What a draw with no rasterizer bound gets: depth clipping on,
    /// everything else off.
    pub const DEFAULT: Self = Self {
        s0: 1 << 1,
        point_size: 1.0,
        line_width: 1.0,
        offset_units: 0.0,
        offset_scale: 0.0,
        clip_planes: 0,
    };
    fn depth_clip(&self) -> bool {
        self.bit(1)
    }
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
    /// Stream output (transform feedback) of a vertex shader, and each
    /// buffer's vertex stride in dwords.
    pub so: Vec<StreamOutput>,
    pub so_stride: [u32; 4],
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
    /// For a buffer's view: the elements it covers.
    pub first_element: u32,
    pub last_element: u32,
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
    /// Texel-buffer size uniforms: (stage, slot, location).
    buffer_sizes: Vec<(usize, u32, i32)>,
    flip_y: i32,
    point_size: i32,
    clip_planes: i32,
}

#[derive(Default)]
pub(crate) struct SubCtx {
    objects: FxMap<u32, Object>,
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
    /// `SET_CLIP_STATE`'s user clip planes.
    clip_planes: [[f32; 4]; 8],
    /// Bound stream-output targets (object handles), by buffer index.
    so_targets: Vec<u32>,
    /// Queries begun and not yet ended.
    queries: FxMap<u32, QueryRun>,
    programs: FxMap<(u32, u32, glsl::Options), Option<Program>>,
}

/// A query between `BEGIN_QUERY` and `END_QUERY`: what it counts so far.
#[derive(Debug, Clone, Copy, Default)]
struct QueryRun {
    /// When it began (host monotonic nanoseconds).
    begin_ns: u64,
    /// Primitives drawn / captured by stream output since.
    generated: u64,
    emitted: u64,
}

#[derive(Default)]
pub(crate) struct Context {
    subs: FxMap<u32, SubCtx>,
    cur: u32,
}

/// See the module docs.
pub struct Renderer {
    /// The backend, behind a cache that drops redundant state calls.
    backend: StateCache,
    trace: bool,
    resources: FxMap<u32, Resource>,
    /// Bytes of buffer shadows held for the guest.
    buffer_bytes: u64,
    next_res: u32,
    contexts: FxMap<u32, Context>,
    next_ctx: u32,
    /// What the backend's GL offers beyond GLES 3.0.
    host: HostCaps,
    /// Host-only GL objects, made on first use.
    vao: u32,
    draw_fbo: u32,
    read_fbo: u32,
    ebo: u32,
    /// Framebuffers multisample resolves use, so they never disturb the
    /// draw's.
    resolve_fbos: (u32, u32),
    /// Where transform feedback captures before the vertices are spread
    /// into their target buffers.
    xfb_buf: u32,
    /// Texel buffers as 2D textures, by (buffer, format, first and last
    /// element).
    tbos: FxMap<(u32, u32, u32, u32), Tbo>,
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

/// The host's monotonic clock in nanoseconds (timer queries).
fn now_ns() -> u64 {
    u64::try_from(crate::clock::now_monotonic().as_nanos()).unwrap_or(u64::MAX)
}

/// A texel buffer's 2D texture: `TBO_WIDTH` texels a row.
struct Tbo {
    gl: u32,
    /// The buffer generation it was filled from.
    version: u64,
    elements: u32,
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
        // Dual-source (EXT_blend_func_extended; only offered with it).
        0x9 => gl::SRC1_COLOR,
        0xa => gl::SRC1_ALPHA,
        0x19 => gl::ONE_MINUS_SRC1_COLOR,
        0x1a => gl::ONE_MINUS_SRC1_ALPHA,
        _ => gl::ONE,
    }
}

/// Whether Gallium blend factor `f` reads the second source colour.
fn is_src1(f: u32) -> bool {
    matches!(f, 0x9 | 0xa | 0x19 | 0x1a)
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

/// Indices for a draw with primitive restart `restart` (at the type's
/// maximum value, the only restart index GLES has): without restart, an
/// index at that maximum is a real vertex, but WebGL2 always restarts there
/// (`PRIMITIVE_RESTART_FIXED_INDEX` can't be turned off), so byte and short
/// indices holding it are widened to the next type. Elsewhere the copy is
/// harmless.
fn widen_restart_indices(data: Vec<u8>, ty: u32, restart: bool) -> (Vec<u8>, u32) {
    if restart {
        return (data, ty);
    }
    match ty {
        gl::UNSIGNED_BYTE if data.contains(&0xff) => (
            data.iter()
                .flat_map(|&i| u16::from(i).to_ne_bytes())
                .collect(),
            gl::UNSIGNED_SHORT,
        ),
        gl::UNSIGNED_SHORT if data.as_chunks::<2>().0.contains(&[0xff, 0xff]) => (
            data.as_chunks::<2>()
                .0
                .iter()
                .flat_map(|&c| u32::from(u16::from_ne_bytes(c)).to_ne_bytes())
                .collect(),
            gl::UNSIGNED_INT,
        ),
        _ => (data, ty),
    }
}

/// How many primitives `count` vertices make in Gallium primitive mode
/// `mode` (no restarts).
fn prim_count(mode: u32, count: u32) -> u32 {
    match mode {
        0 | 2 => count,
        1 => count / 2,
        3 => count.saturating_sub(1),
        4 => count / 3,
        5 | 6 => count.saturating_sub(2),
        _ => 0,
    }
}

/// Append the vertices transform feedback writes for one run of a
/// Gallium primitive `mode` draw (no restarts within `v`): lists stay as
/// they are, strips, loops and fans become independent primitives (odd
/// strip triangles with their first two vertices swapped, keeping the
/// winding, as GL's capture does).
fn assemble(mode: u32, v: &[u32], out: &mut Vec<u32>) {
    let n = v.len();
    match mode {
        0 => out.extend_from_slice(v),
        1 => out.extend_from_slice(&v[..n / 2 * 2]),
        2 | 3 => {
            for i in 1..n {
                out.extend_from_slice(&[v[i - 1], v[i]]);
            }
            if mode == 2 && n > 1 {
                out.extend_from_slice(&[v[n - 1], v[0]]);
            }
        }
        4 => out.extend_from_slice(&v[..n / 3 * 3]),
        5 => {
            for i in 2..n {
                if i % 2 == 0 {
                    out.extend_from_slice(&[v[i - 2], v[i - 1], v[i]]);
                } else {
                    out.extend_from_slice(&[v[i - 1], v[i - 2], v[i]]);
                }
            }
        }
        6 => {
            for i in 2..n {
                out.extend_from_slice(&[v[0], v[i - 1], v[i]]);
            }
        }
        _ => {}
    }
}

/// A sampler view's swizzle as [`glsl::Options::swizzle`] keeps it: `0`
/// for the identity, else `0x8000 |` the four selectors. A view of a
/// format without alpha (`X8`) reads alpha as one, whatever the
/// resource's texture holds.
fn view_swizzle(v: &SamplerView) -> u16 {
    let mut sw = v.swizzle & 0xfff;
    if format::lookup(v.format).is_some_and(|f| f.no_alpha) {
        for c in 0..4 {
            if (sw >> (3 * c)) & 7 == 3 {
                sw = (sw & !(7 << (3 * c))) | (5 << (3 * c));
            }
        }
    }
    // Identity: X, Y, Z, W.
    if sw == 0o3210 { 0 } else { 0x8000 | sw as u16 }
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
    pub fn new(mut backend: Box<dyn GpuBackend>, trace: bool) -> Self {
        backend.begin();
        let host = backend.host_caps();
        backend.end();
        Self {
            backend: StateCache::new(backend),
            trace,
            resources: FxMap::default(),
            buffer_bytes: 0,
            next_res: 1,
            contexts: FxMap::default(),
            next_ctx: 1,
            vao: 0,
            draw_fbo: 0,
            read_fbo: 0,
            ebo: 0,
            resolve_fbos: (0, 0),
            xfb_buf: 0,
            tbos: FxMap::default(),
            host,
            warned: std::collections::HashSet::new(),
        }
    }

    /// What the backend's GL offers beyond GLES 3.0.
    pub(crate) fn host_caps(&self) -> HostCaps {
        self.host
    }

    /// Use only what `host` says of the backend's GL (less than it has:
    /// to exercise the emulations).
    pub(crate) fn set_host_caps(&mut self, host: HostCaps) {
        self.host = host;
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
                version: 0,
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
            // Multisampled: drawn in a multisampled renderbuffer (GLES 3.0
            // has no multisampled textures), resolved into the texture.
            let msaa = (desc.nr_samples > 1 && t == gl::TEXTURE_2D).then(|| {
                let rb = self.backend.create_renderbuffer();
                let samples = desc.nr_samples.min(super::caps::max_samples(&self.host));
                self.backend
                    .renderbuffer_storage(rb, samples, f.internal, desc.width, height);
                Msaa { rb, dirty: false }
            });
            Host::Texture {
                gl: tex,
                gl_target: t,
                msaa,
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
        if self.trace {
            eprintln!("nixvm gpu: resource {id} = {desc:?}");
        }
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
                Host::Buffer { gl, shadow, .. } => {
                    self.buffer_bytes -= shadow.len() as u64;
                    self.backend.delete_buffer(gl);
                    let views: Vec<_> = self.tbos.keys().filter(|k| k.0 == id).copied().collect();
                    for k in views {
                        if let Some(t) = self.tbos.remove(&k) {
                            self.backend.delete_texture(t.gl);
                        }
                    }
                }
                Host::Texture { gl, msaa, .. } => {
                    self.backend.delete_texture(gl);
                    if let Some(m) = msaa {
                        self.backend.delete_renderbuffer(m.rb);
                    }
                }
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
    /// framebuffer bound at `fb_target`, as `attachment`: its texture (a
    /// multisampled resource's resolve, which callers bring up to date
    /// first with [`Self::resolve`]).
    fn attach(
        &mut self,
        fb_target: u32,
        attachment: u32,
        res: u32,
        level: u32,
        layer: u32,
    ) -> Option<Kind> {
        self.attach_as(fb_target, attachment, res, level, layer, false)
    }

    /// [`Self::attach`], or with `draw` a multisampled resource's
    /// renderbuffer (whose resolve is then stale).
    fn attach_as(
        &mut self,
        fb_target: u32,
        attachment: u32,
        res: u32,
        level: u32,
        layer: u32,
        draw: bool,
    ) -> Option<Kind> {
        let r = self.resources.get_mut(&res)?;
        let Host::Texture {
            gl: tex,
            gl_target,
            ref mut msaa,
        } = r.host
        else {
            return None;
        };
        let kind = r.fmt?.kind;
        let attachment = match kind {
            Kind::Depth if attachment != gl::NONE => gl::DEPTH_ATTACHMENT,
            Kind::DepthStencil if attachment != gl::NONE => gl::DEPTH_STENCIL_ATTACHMENT,
            _ => attachment,
        };
        if draw && let Some(m) = msaa {
            m.dirty = true;
            let rb = m.rb;
            self.backend
                .framebuffer_renderbuffer(fb_target, attachment, rb);
            return Some(kind);
        }
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

    /// Bring multisampled resource `res`'s resolve texture up to date with
    /// what was drawn in its renderbuffer (nothing for other resources).
    fn resolve(&mut self, res: u32) {
        let Some(r) = self.resources.get_mut(&res) else {
            return;
        };
        let Host::Texture {
            gl: tex,
            msaa: Some(ref mut m),
            ..
        } = r.host
        else {
            return;
        };
        if !m.dirty {
            return;
        }
        m.dirty = false;
        let rb = m.rb;
        let (w, h, _) = r.level_size(0);
        let (att, mask) = match r.fmt.map(|f| f.kind) {
            Some(Kind::Depth) => (gl::DEPTH_ATTACHMENT, gl::DEPTH_BUFFER_BIT),
            Some(Kind::DepthStencil) => (
                gl::DEPTH_STENCIL_ATTACHMENT,
                gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT,
            ),
            _ => (gl::COLOR_ATTACHMENT0, gl::COLOR_BUFFER_BIT),
        };
        if self.resolve_fbos.0 == 0 {
            self.resolve_fbos = (
                self.backend.create_framebuffer(),
                self.backend.create_framebuffer(),
            );
        }
        let (rf, df) = self.resolve_fbos;
        let b = &mut self.backend;
        b.bind_framebuffer(gl::READ_FRAMEBUFFER, rf);
        b.framebuffer_renderbuffer(gl::READ_FRAMEBUFFER, att, rb);
        b.bind_framebuffer(gl::DRAW_FRAMEBUFFER, df);
        b.framebuffer_texture(gl::DRAW_FRAMEBUFFER, att, gl::TEXTURE_2D, tex, 0, None);
        if att == gl::COLOR_ATTACHMENT0 {
            b.read_buffer(gl::COLOR_ATTACHMENT0);
            b.draw_buffers(&[gl::COLOR_ATTACHMENT0]);
        }
        b.enable(gl::SCISSOR_TEST, false);
        b.enable(gl::RASTERIZER_DISCARD, false);
        let rect = [0, 0, w as i32, h as i32];
        b.blit_framebuffer(rect, rect, mask, gl::NEAREST);
        // Let go of both, so a destroyed resource isn't kept attached.
        b.framebuffer_renderbuffer(gl::READ_FRAMEBUFFER, att, 0);
        b.framebuffer_texture(gl::DRAW_FRAMEBUFFER, att, gl::TEXTURE_2D, 0, 0, None);
    }

    pub(crate) fn transfer(&mut self, t: &Transfer) -> Result<(), Errno> {
        if !t.to_host {
            self.resolve(t.res);
        }
        let Some(r) = self.resources.get_mut(&t.res) else {
            return Err(EINVAL);
        };
        let bx = t.bx;
        match &mut r.host {
            Host::Buffer {
                gl: b,
                shadow,
                version,
            } => {
                let (x, w) = (bx.x as usize, bx.w as usize);
                if x + w > shadow.len() {
                    return Err(EINVAL);
                }
                if t.to_host {
                    *version += 1;
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
            Host::Texture {
                gl: tex, gl_target, ..
            } => {
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
                        // In a format/type GLES (and WebGL2) can always
                        // read, then packed to the resource's own layout.
                        let (rformat, rty, rbpp) = fmt.read_as();
                        let mut read = vec![0u8; (bx.w * bx.h * rbpp) as usize];
                        self.backend.read_pixels(
                            bx.x as i32,
                            bx.y as i32,
                            bx.w,
                            bx.h,
                            rformat,
                            rty,
                            &mut read,
                        );
                        fmt.pack_read(&read, &mut px);
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

    /// What resource `id` was made as.
    pub(crate) fn resource_desc(&self, id: u32) -> Option<ResourceDesc> {
        self.resources.get(&id).map(|r| r.desc)
    }

    /// Run `f` on the backend for a caller outside the guest's command
    /// stream (the display's GL compositor), with a lookup from resource
    /// ids to their host textures. `f` may change any GL binding or state:
    /// it gets the state cache, which so stays right, and the renderer
    /// applies its full state at every draw.
    pub(crate) fn external<T>(
        &mut self,
        f: impl FnOnce(&mut dyn GpuBackend, &dyn Fn(u32) -> Option<super::HostTexture>) -> T,
    ) -> T {
        let resources = &self.resources;
        let lookup = |id: u32| {
            let r = resources.get(&id)?;
            let Host::Texture {
                gl,
                gl_target: gl::TEXTURE_2D,
                ..
            } = r.host
            else {
                return None;
            };
            let fmt = r.fmt?;
            if fmt.bpp != 4 || matches!(fmt.kind, Kind::Depth | Kind::DepthStencil) {
                return None;
            }
            let (width, height, _) = r.level_size(0);
            Some(super::HostTexture { gl, width, height })
        };
        f(&mut self.backend, &lookup)
    }

    /// Read the box `(x, y, w, h)` of level 0, layer 0 of 4-byte colour
    /// texture `res` into `out` (rows packed, top row first — the guest's
    /// row order, as transfers keep it), in the resource's own byte layout:
    /// what a transfer to the guest would have written, without touching
    /// the guest's pages. How the display shows what the guest rendered.
    pub(crate) fn read_texture(
        &mut self,
        res: u32,
        rect: (u32, u32, u32, u32),
        out: &mut [u8],
    ) -> Result<(), Errno> {
        let (x, y, w, h) = rect;
        self.resolve(res);
        let r = self.resources.get(&res).ok_or(EINVAL)?;
        let Host::Texture { .. } = r.host else {
            return Err(EINVAL);
        };
        let fmt = r.fmt.ok_or(EINVAL)?;
        if fmt.bpp != 4 || matches!(fmt.kind, Kind::Depth | Kind::DepthStencil) {
            return Err(EINVAL);
        }
        let (lw, lh, _) = r.level_size(0);
        let n = w as usize * h as usize * 4;
        if x.checked_add(w).is_none_or(|e| e > lw)
            || y.checked_add(h).is_none_or(|e| e > lh)
            || out.len() < n
        {
            return Err(EINVAL);
        }
        if n == 0 {
            return Ok(());
        }
        let fbo = self.read_fbo();
        self.backend.bind_framebuffer(gl::READ_FRAMEBUFFER, fbo);
        self.attach(gl::READ_FRAMEBUFFER, gl::COLOR_ATTACHMENT0, res, 0, 0);
        self.backend.read_buffer(gl::COLOR_ATTACHMENT0);
        let out = &mut out[..n];
        // In a format/type GLES and WebGL2 can always read, then packed to
        // the resource's layout (as transfers to the guest do).
        let (rformat, rty, rbpp) = fmt.read_as();
        let mut read = vec![0u8; w as usize * h as usize * rbpp as usize];
        self.backend
            .read_pixels(x as i32, y as i32, w, h, rformat, rty, &mut read);
        fmt.pack_read(&read, out);
        fmt.convert_in_place(out);
        Ok(())
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
        let trace = self.trace;
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
        if trace && !matches!(obj, Object::Shader(_)) {
            eprintln!("nixvm gpu: object {handle} = {obj:?}");
        }
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

    pub(crate) fn set_clip_planes(&mut self, ctx: u32, planes: [[f32; 4]; 8]) {
        self.sub(ctx).clip_planes = planes;
    }

    /// `SET_STREAMOUT_TARGETS`: the targets captures go to, by buffer;
    /// a target whose `append` bit is clear starts over at its beginning.
    pub(crate) fn set_streamout_targets(&mut self, ctx: u32, append: u32, handles: &[u32]) {
        let sub = self.sub(ctx);
        for (i, h) in handles.iter().enumerate() {
            if append & (1 << i) == 0
                && let Some(Object::StreamTarget(t)) = sub.objects.get_mut(h)
            {
                t.filled = 0;
            }
        }
        sub.so_targets = handles.to_vec();
    }

    /// `BEGIN_QUERY`: start counting (or timing).
    pub(crate) fn begin_query(&mut self, ctx: u32, handle: u32) {
        let begin_ns = now_ns();
        self.sub(ctx).queries.insert(
            handle,
            QueryRun {
                begin_ns,
                ..QueryRun::default()
            },
        );
    }

    /// Write a finished query's result where the guest polls for it.
    pub(crate) fn end_query(&mut self, ctx: u32, handle: u32) {
        let sub = self.sub(ctx);
        let run = sub.queries.remove(&handle).unwrap_or_default();
        let Some(Object::Query(q)) = sub.objects.get(&handle).cloned() else {
            return;
        };
        let Some(r) = self.resources.get(&q.res) else {
            return;
        };
        // struct virgl_host_query_state { u32 state; u32 size; u64 result },
        // for the VIRGL_QUERY_* type.
        let result: u64 = match q.ty {
            // Occlusion counter / predicates: GLES can't count samples, so
            // they pass (as virglrenderer's GLES path also reports), which
            // is what conditional rendering then assumes.
            0 | 1 | 11 => 1,
            // Timestamp, time elapsed: from the host's clock.
            2 => now_ns(),
            4 => now_ns().saturating_sub(run.begin_ns),
            // Primitives generated / emitted by stream output (counted
            // at each draw).
            5 => run.generated,
            6 | 7 => run.emitted,
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
                        .attach_as(
                            gl::DRAW_FRAMEBUFFER,
                            att,
                            s.res,
                            s.level,
                            s.first_layer,
                            true,
                        )
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
                .attach_as(
                    gl::DRAW_FRAMEBUFFER,
                    gl::DEPTH_ATTACHMENT,
                    s.res,
                    s.level,
                    s.first_layer,
                    true,
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

    /// The format of each bound colour buffer, where known.
    fn cbuf_formats(&mut self, ctx: u32) -> Vec<Option<&'static Format>> {
        let sub = self.sub(ctx);
        let surfaces: Vec<(u32, u32)> = sub
            .cbufs
            .iter()
            .map(|h| match sub.objects.get(h) {
                Some(Object::Surface(s)) => (s.res, s.format),
                _ => (0, 0),
            })
            .collect();
        // The surface's own (view) format is what's rendered.
        surfaces
            .iter()
            .map(|&(r, f)| format::lookup(f).or_else(|| self.resources.get(&r).and_then(|r| r.fmt)))
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
        let fmts = self.cbuf_formats(ctx);
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
                _ => {
                    let value = color.map(f32::from_bits);
                    let value = fmts
                        .get(i)
                        .copied()
                        .flatten()
                        .map_or(value, |f| f.quantize_clear(value));
                    b.clear_buffer_fv(gl::COLOR, i as u32, value);
                }
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
        let so = self.stream_output(ctx).0;
        let p = match glsl::link(&vs, &fs, opts, &so) {
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
        let xfb = (!p.xfb_varyings.is_empty()).then_some((p.xfb_varyings.as_slice(), true));
        let prog = match self.backend.create_program_xfb(&p.vs, &p.fs, xfb) {
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
            buffer_sizes: Vec::new(),
            flip_y: b.uniform_location(prog, "nx_flip_y"),
            point_size: b.uniform_location(prog, "nx_psize"),
            clip_planes: b.uniform_location(prog, "nx_ucp"),
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
            for &s in &p.layout.buffer_sizes[six] {
                let loc = b.uniform_location(prog, &format!("nx_{st}bsz{s}"));
                out.buffer_sizes.push((six, s, loc));
            }
        }
        Some(out)
    }

    /// The bound vertex shader's stream output and buffer strides (in
    /// dwords), if it has any.
    fn stream_output(&mut self, ctx: u32) -> (Vec<StreamOutput>, [u32; 4]) {
        let sub = self.sub(ctx);
        match sub.objects.get(&sub.shaders[0]) {
            Some(Object::Shader(s)) => (s.so.clone(), s.so_stride),
            _ => (Vec::new(), [0; 4]),
        }
    }

    /// The sampler slots the bound shader of stage `six` (0 vertex, 1
    /// fragment) declares.
    fn used_samplers(&mut self, ctx: u32, six: usize) -> Vec<u32> {
        match self.shader(ctx, six) {
            Some(Ok(sh)) => sh
                .decls
                .iter()
                .filter(|d| d.file == tgsi::File::Sampler || d.file == tgsi::File::SamplerView)
                .flat_map(|d| d.first..=d.last)
                .filter(|&s| s < 16)
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Bring the resolves of every multisampled resource the bound views
    /// sample up to date: before the draw's framebuffer is bound, since a
    /// resolve uses framebuffers of its own.
    fn resolve_sampled(&mut self, ctx: u32) {
        let res: Vec<u32> = {
            let sub = self.sub(ctx);
            sub.views[..2]
                .iter()
                .flatten()
                .filter_map(|h| match sub.objects.get(h) {
                    Some(Object::SamplerView(v)) => Some(v.res),
                    _ => None,
                })
                .collect()
        };
        for r in res {
            self.resolve(r);
        }
    }

    /// The program options the draw's state implies (see
    /// [`glsl::Options`]).
    fn draw_options(
        &mut self,
        ctx: u32,
        kinds: &[Kind],
        blend: &Blend,
        rs: &Rasterizer,
        ves: &[VertexElement],
    ) -> glsl::Options {
        let mut opts = glsl::Options {
            flatshade: rs.flatshade(),
            nr_cbufs: kinds.len() as u32,
            clip_planes: rs.clip_planes,
            depth_clamp: !rs.depth_clip() && !self.host.depth_clamp,
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
            let Some(f) = format::lookup(ve.format) else {
                continue;
            };
            if f.vertex_is_integer() {
                if f.kind == Kind::Uint {
                    opts.uint_inputs |= 1 << i;
                } else {
                    opts.sint_inputs |= 1 << i;
                }
            } else if f.swap_rb {
                opts.bgra_inputs |= 1 << i;
            }
        }
        let rt = blend.rt[0];
        opts.dual_src = self.host.blend_func_extended
            && rt.enable
            && [rt.rgb_src, rt.rgb_dst, rt.alpha_src, rt.alpha_dst]
                .into_iter()
                .any(is_src1);
        for six in 0..2 {
            for slot in self.used_samplers(ctx, six) {
                let sub = self.sub(ctx);
                let Some(Object::SamplerView(v)) = sub.objects.get(&sub.views[six][slot as usize])
                else {
                    continue;
                };
                let v = *v;
                if self
                    .resources
                    .get(&v.res)
                    .is_some_and(|r| r.desc.target == target::BUFFER)
                {
                    opts.tbo[six] |= 1 << slot;
                }
                opts.swizzle[six][slot as usize] = view_swizzle(&v);
            }
        }
        opts
    }

    #[allow(clippy::too_many_lines)]
    pub(crate) fn draw(&mut self, ctx: u32, info: &DrawInfo) {
        if info.count == 0 {
            return;
        }
        if self.sub(ctx).shaders[2..5].iter().any(|&h| h != 0) {
            self.warn("geometry and tessellation shaders are not supported".into());
            return;
        }
        self.resolve_sampled(ctx);
        let Some(_) = self.bind_framebuffer(ctx) else {
            return;
        };
        let kinds = self.cbuf_kinds(ctx);
        // State objects.
        let (blend, rs, dsa, ves, vp, scissor, stencil_ref, blend_color, ucp) = {
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
                _ => Rasterizer::DEFAULT,
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
                sub.clip_planes,
            )
        };
        let opts = self.draw_options(ctx, &kinds, &blend, &rs, &ves);
        if self.program(ctx, opts).is_none() {
            return;
        }
        let flip = vp.scale[1] < 0.0;
        self.use_program(ctx, opts, flip, &rs, &ucp);
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
        // other. A flipped viewport changes nothing here: the shader's
        // flip lands every vertex where Gallium's negative scale would, so
        // GL sees the very window coordinates Gallium's front_ccw speaks of
        // (Mesa sets front_ccw for them, inverting it for its FBOs).
        b.front_face(if rs.front_ccw() { gl::CW } else { gl::CCW });
        b.enable(gl::POLYGON_OFFSET_FILL, rs.offset_tri());
        if rs.offset_tri() {
            b.polygon_offset(rs.offset_scale, rs.offset_units);
        }
        b.line_width(1.0);
        b.enable(gl::PRIMITIVE_RESTART_FIXED_INDEX, info.primitive_restart);
        if self.host.depth_clamp {
            b.enable(gl::DEPTH_CLAMP, !rs.depth_clip());
        }
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
        self.apply_blend(&blend, blend_color, kinds.len());
        // Vertex input.
        let vbufs = self.sub(ctx).vbufs.clone();
        if self.trace {
            self.trace_draw(ctx, info, &ves, &vbufs);
        }
        self.bind_vertex_input(info, &ves, &vbufs);
        let mode = info.mode;
        if mode > 6 {
            self.warn(format!("primitive mode {mode} unsupported on GLES"));
            return;
        }
        let instances = info.instance_count.max(1) as i32;
        let indices = if info.indexed {
            let Some(ix) = self.index_data(ctx, info) else {
                return;
            };
            Some(ix)
        } else {
            None
        };
        if let Some((data, ty)) = &indices {
            let (data, ty) = widen_restart_indices(data.clone(), *ty, info.primitive_restart);
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
        // Primitive counts for queries, and stream output.
        let prims = u64::from(prim_count(mode, info.count)) * u64::from(info.instance_count.max(1));
        let emitted = self.stream_out(ctx, info, opts, flip, &rs, &ucp, indices.as_ref());
        for q in self.sub(ctx).queries.values_mut() {
            q.generated += prims;
            q.emitted += emitted;
        }
    }

    /// Make the program for `opts` current and feed it the draw's
    /// uniforms, buffers and textures.
    fn use_program(
        &mut self,
        ctx: u32,
        opts: glsl::Options,
        flip: bool,
        rs: &Rasterizer,
        ucp: &[[f32; 4]; 8],
    ) {
        let key = {
            let sub = self.sub(ctx);
            (sub.shaders[0], sub.shaders[1], opts)
        };
        let Some(Some(p)) = self.sub(ctx).programs.get(&key) else {
            return;
        };
        let (prog_gl, consts, ubos, samplers, sizes) = (
            p.gl,
            p.consts,
            p.ubos.clone(),
            p.samplers.clone(),
            p.buffer_sizes.clone(),
        );
        let (flip_loc, psize_loc, ucp_loc) = (p.flip_y, p.point_size, p.clip_planes);
        self.backend.use_program(prog_gl);
        self.backend
            .uniform_1f(flip_loc, if flip { -1.0 } else { 1.0 });
        if psize_loc >= 0 {
            let size = if rs.point_size > 0.0 {
                rs.point_size
            } else {
                1.0
            };
            self.backend.uniform_1f(psize_loc, size);
        }
        if ucp_loc >= 0 {
            let planes: Vec<f32> = ucp.iter().flatten().copied().collect();
            self.backend.uniform_4fv(ucp_loc, &planes);
        }
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
        for (six, slot, loc) in sizes {
            let sub = self.sub(ctx);
            let n = match sub.objects.get(&sub.views[six][slot as usize]) {
                Some(Object::SamplerView(v)) => v.last_element.saturating_sub(v.first_element) + 1,
                _ => 0,
            };
            self.backend.uniform_1i(loc, n as i32);
        }
    }

    /// Blend state: per colour buffer when the blend asks for it and the
    /// host can (`draw_buffers_indexed`), else buffer 0's for all.
    fn apply_blend(&mut self, blend: &Blend, color: [f32; 4], nr_cbufs: usize) {
        let b = &mut self.backend;
        let set = |b: &mut StateCache, i: Option<u32>, rt: &RtBlend| {
            let (src_rgb, dst_rgb) = (blend_factor(rt.rgb_src), blend_factor(rt.rgb_dst));
            let (src_a, dst_a) = (blend_factor(rt.alpha_src), blend_factor(rt.alpha_dst));
            let (eq_rgb, eq_a) = (blend_func(rt.rgb_func), blend_func(rt.alpha_func));
            let m = rt.colormask;
            let mask = (m & 1 != 0, m & 2 != 0, m & 4 != 0, m & 8 != 0);
            if let Some(i) = i {
                b.enable_i(gl::BLEND, i, rt.enable);
                if rt.enable {
                    b.blend_func_separate_i(i, src_rgb, dst_rgb, src_a, dst_a);
                    b.blend_equation_separate_i(i, eq_rgb, eq_a);
                }
                b.color_mask_i(i, mask.0, mask.1, mask.2, mask.3);
            } else {
                b.enable(gl::BLEND, rt.enable);
                if rt.enable {
                    b.blend_func_separate(src_rgb, dst_rgb, src_a, dst_a);
                    b.blend_equation_separate(eq_rgb, eq_a);
                }
                b.color_mask(mask.0, mask.1, mask.2, mask.3);
            }
        };
        if blend.independent && self.host.draw_buffers_indexed {
            for (i, rt) in blend.rt.iter().enumerate().take(nr_cbufs.clamp(1, 8)) {
                set(b, Some(i as u32), rt);
            }
        } else {
            set(b, None, &blend.rt[0]);
        }
        b.blend_color(color);
        b.enable(gl::SAMPLE_ALPHA_TO_COVERAGE, blend.alpha_to_coverage);
        b.enable(gl::DITHER, false);
    }

    fn trace_draw(
        &mut self,
        ctx: u32,
        info: &DrawInfo,
        ves: &[VertexElement],
        vbufs: &[VertexBuffer],
    ) {
        let c = &self.sub(ctx).consts;
        eprintln!(
            "nixvm gpu: draw {info:?}\n  elements {ves:?}\n  buffers {vbufs:?}\n  vs consts {:?}\n  fs consts {:?}",
            c[0].iter().map(|w| f32::from_bits(*w)).collect::<Vec<_>>(),
            c[1].iter().map(|w| f32::from_bits(*w)).collect::<Vec<_>>()
        );
        for vb in vbufs {
            if let Some(Host::Buffer { shadow, .. }) = self.resources.get(&vb.res).map(|r| &r.host)
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

    /// Point the vertex attributes at the vertex buffers (offsets moved by
    /// the index bias: GLES 3.0 has no base vertex).
    fn bind_vertex_input(
        &mut self,
        info: &DrawInfo,
        ves: &[VertexElement],
        vbufs: &[VertexBuffer],
    ) {
        if self.vao == 0 {
            self.vao = self.backend.create_vertex_array();
        }
        self.backend.bind_vertex_array(self.vao);
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
                let off = i64::from(vb.offset) + i64::from(ve.offset) + bias * i64::from(vb.stride);
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
            } else {
                self.backend.enable_vertex_attrib_array(i, false);
                let what = ve.format;
                self.warn(format!(
                    "vertex element format {what} or its buffer unsupported"
                ));
            }
        }
    }

    /// An indexed draw's indices, from the index buffer's shadow: the bytes
    /// and their GL type.
    fn index_data(&mut self, ctx: u32, info: &DrawInfo) -> Option<(Vec<u8>, u32)> {
        let ib = self.sub(ctx).ib;
        let (ty, size) = match ib.size {
            1 => (gl::UNSIGNED_BYTE, 1usize),
            2 => (gl::UNSIGNED_SHORT, 2),
            _ => (gl::UNSIGNED_INT, 4),
        };
        // The index buffer's offset already points at the draw's first
        // index (Mesa folds `start` into it, as virglrenderer reads it).
        let start = ib.offset as usize;
        let len = info.count as usize * size;
        let Some(Host::Buffer { shadow, .. }) = self.resources.get(&ib.res).map(|r| &r.host) else {
            self.warn("indexed draw without an index buffer".into());
            return None;
        };
        let Some(data) = shadow.get(start..start + len).map(<[u8]>::to_vec) else {
            self.warn("index range outside the index buffer".into());
            return None;
        };
        Some((data, ty))
    }

    /// Transform feedback for the draw just made, if stream output is
    /// bound: the vertex shader runs again over the draw's vertices as
    /// points with the rasterizer off, capturing each vertex's stream
    /// outputs into a scratch buffer; the draw's primitives are then
    /// assembled on the CPU and their vertices written where each output
    /// goes. (GLES 3.0's own capture can't skip components, write several
    /// buffers from one record at offsets, or follow indexed and strip
    /// draws; this does all of Gallium's.) How many primitives were
    /// written.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn stream_out(
        &mut self,
        ctx: u32,
        info: &DrawInfo,
        opts: glsl::Options,
        flip: bool,
        rs: &Rasterizer,
        ucp: &[[f32; 4]; 8],
        indices: Option<&(Vec<u8>, u32)>,
    ) -> u64 {
        let targets = self.sub(ctx).so_targets.clone();
        if targets.iter().all(|&h| h == 0) {
            return 0;
        }
        let (so, strides) = self.stream_output(ctx);
        if so.is_empty() {
            return 0;
        }
        // The vertices the draw reads, in the order the primitives use them.
        let index_list: Vec<u32> = match indices {
            Some((data, ty)) => match *ty {
                gl::UNSIGNED_BYTE => data.iter().map(|&i| u32::from(i)).collect(),
                gl::UNSIGNED_SHORT => data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u32::from(u16::from_ne_bytes(*c)))
                    .collect(),
                _ => data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| u32::from_ne_bytes(*c))
                    .collect(),
            },
            None => (info.start..info.start + info.count).collect(),
        };
        let restart = (info.indexed && info.primitive_restart).then_some(info.restart_index);
        let real = |i: &&u32| Some(**i) != restart;
        let (Some(&lo), Some(&hi)) = (
            index_list.iter().filter(real).min(),
            index_list.iter().filter(real).max(),
        ) else {
            return 0;
        };
        let span = hi - lo + 1;
        let instances = info.instance_count.max(1);
        let record: u32 = so.iter().map(|o| o.count).sum();
        let Some(bytes) = (u64::from(span) * u64::from(instances) * u64::from(record) * 4)
            .try_into()
            .ok()
            .filter(|&b: &usize| b > 0 && b <= MAX_BUFFER as usize)
        else {
            self.warn("stream output capture too large".into());
            return 0;
        };
        // Capture pass.
        let xopts = glsl::Options { xfb: true, ..opts };
        if self.program(ctx, xopts).is_none() {
            return 0;
        }
        self.use_program(ctx, xopts, flip, rs, ucp);
        if self.xfb_buf == 0 {
            self.xfb_buf = self.backend.create_buffer();
        }
        let b = &mut self.backend;
        b.bind_buffer(gl::TRANSFORM_FEEDBACK_BUFFER, self.xfb_buf);
        b.buffer_data(gl::TRANSFORM_FEEDBACK_BUFFER, bytes, None, gl::STREAM_DRAW);
        b.bind_buffer_base(gl::TRANSFORM_FEEDBACK_BUFFER, 0, self.xfb_buf);
        b.enable(gl::RASTERIZER_DISCARD, true);
        b.begin_transform_feedback(gl::POINTS);
        b.draw_arrays(gl::POINTS, lo as i32, span as i32, instances as i32);
        b.end_transform_feedback();
        b.enable(gl::RASTERIZER_DISCARD, rs.discard());
        b.bind_buffer_base(gl::TRANSFORM_FEEDBACK_BUFFER, 0, 0);
        let mut captured = vec![0u8; bytes];
        b.bind_buffer(gl::TRANSFORM_FEEDBACK_BUFFER, self.xfb_buf);
        b.get_buffer_sub_data(gl::TRANSFORM_FEEDBACK_BUFFER, 0, &mut captured);
        b.bind_buffer(gl::TRANSFORM_FEEDBACK_BUFFER, 0);
        // Primitive assembly: the vertex sequence capture writes.
        let mut seq: Vec<u32> = Vec::new();
        for strip in index_list.split(|&i| Some(i) == restart) {
            assemble(info.mode, strip, &mut seq);
        }
        let per_prim = match info.mode {
            0 => 1,
            1..=3 => 2,
            _ => 3,
        };
        // Where each buffer's captures go, and how many whole primitives
        // fit in all of them.
        let mut dest: Vec<(usize, u32, StreamTarget)> = Vec::new();
        for (bi, &h) in targets.iter().enumerate().take(4) {
            if let Some(Object::StreamTarget(t)) = self.sub(ctx).objects.get(&h)
                && so.iter().any(|o| o.buffer as usize == bi)
            {
                dest.push((bi, strides[bi] * 4, *t));
            }
        }
        let prims = seq.len() / per_prim * instances as usize;
        let fit = dest
            .iter()
            .map(|(_, stride, t)| {
                let room = t.size.saturating_sub(t.filled);
                if *stride == 0 {
                    usize::MAX
                } else {
                    (room / (stride * per_prim as u32)) as usize
                }
            })
            .min()
            .unwrap_or(0)
            .min(prims);
        let verts = fit * per_prim;
        let vert_bytes = |v: usize| -> (u32, u32) {
            // The v-th vertex written: which instance and which index.
            let per_inst = seq.len();
            ((v / per_inst) as u32, seq[v % per_inst])
        };
        for (bi, stride, t) in &dest {
            let Some(Resource {
                host:
                    Host::Buffer {
                        gl: buf,
                        shadow,
                        version,
                    },
                ..
            }) = self.resources.get_mut(&t.res)
            else {
                continue;
            };
            let base = (t.offset + t.filled) as usize;
            let end = (base + verts * *stride as usize).min(shadow.len());
            for v in 0..verts {
                let (inst, idx) = vert_bytes(v);
                let rec = (inst * span + (idx - lo)) as usize * record as usize * 4;
                let mut src_off = 0usize;
                for o in &so {
                    let n = o.count as usize * 4;
                    if o.buffer as usize == *bi {
                        let at = base + v * *stride as usize + o.dst_offset as usize * 4;
                        if let (Some(d), Some(s)) = (
                            shadow.get_mut(at..at + n),
                            captured.get(rec + src_off..rec + src_off + n),
                        ) {
                            d.copy_from_slice(s);
                        }
                    }
                    src_off += n;
                }
            }
            *version += 1;
            if base < end {
                let (buf, data) = (*buf, shadow[base..end].to_vec());
                self.backend.bind_buffer(gl::COPY_WRITE_BUFFER, buf);
                self.backend
                    .buffer_sub_data(gl::COPY_WRITE_BUFFER, base, &data);
            }
        }
        let sub = self.sub(ctx);
        for (bi, stride, _) in &dest {
            if let Some(Object::StreamTarget(t)) = sub.objects.get_mut(&targets[*bi]) {
                t.filled += (verts as u32) * stride;
            }
        }
        fit as u64
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
        if let Some(v) = view
            && self
                .resources
                .get(&v.res)
                .is_some_and(|r| r.desc.target == target::BUFFER)
        {
            self.bind_texel_buffer(&v, unit);
            return;
        }
        let tex = view
            .and_then(|v| self.resources.get(&v.res))
            .and_then(|r| match r.host {
                Host::Texture { gl, gl_target, .. } => Some((gl, gl_target, r.desc.last_level)),
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

    /// Bind texel buffer view `v` (on the active unit): its elements in a
    /// 2D texture [`super::caps::TBO_WIDTH`] wide, filled from the
    /// buffer's shadow whenever the buffer has changed since.
    fn bind_texel_buffer(&mut self, v: &SamplerView, unit: u32) {
        let Some(fmt) = format::lookup(v.format).filter(|f| f.is_texture()) else {
            return;
        };
        let w = super::caps::TBO_WIDTH;
        let elements =
            (v.last_element.saturating_sub(v.first_element) + 1).min(super::caps::MAX_TEXEL_BUFFER);
        let key = (v.res, v.format, v.first_element, elements);
        let Some(Host::Buffer {
            shadow, version, ..
        }) = self.resources.get(&v.res).map(|r| &r.host)
        else {
            return;
        };
        let version = *version;
        let fresh = self
            .tbos
            .get(&key)
            .is_some_and(|t| t.version == version && t.elements == elements);
        let rows = elements.div_ceil(w);
        if !fresh {
            let bpp = fmt.bpp as usize;
            let start = v.first_element as usize * bpp;
            let mut data = vec![0u8; (w * rows) as usize * bpp];
            if let Some(src) = shadow.get(start..) {
                let n = src.len().min(elements as usize * bpp);
                data[..n].copy_from_slice(&src[..n]);
            }
            fmt.to_host(&mut data);
            let tex = if let Some(t) = self.tbos.get(&key) {
                t.gl
            } else {
                let t = self.backend.create_texture();
                self.backend.bind_texture(gl::TEXTURE_2D, t);
                self.backend
                    .tex_storage(gl::TEXTURE_2D, 1, fmt.internal, w, rows, 0);
                t
            };
            self.backend.bind_texture(gl::TEXTURE_2D, tex);
            self.backend.tex_sub_image(
                gl::TEXTURE_2D,
                0,
                0,
                0,
                0,
                w,
                rows,
                0,
                fmt.format,
                fmt.ty,
                &data,
            );
            self.tbos.insert(
                key,
                Tbo {
                    gl: tex,
                    version,
                    elements,
                },
            );
        }
        let tex = self.tbos[&key].gl;
        let b = &mut self.backend;
        b.bind_texture(gl::TEXTURE_2D, tex);
        b.bind_sampler(unit, 0);
        b.tex_parameter(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        b.tex_parameter(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
        b.tex_parameter(gl::TEXTURE_2D, gl::TEXTURE_COMPARE_MODE, gl::NONE as i32);
        b.tex_parameter(gl::TEXTURE_2D, gl::TEXTURE_BASE_LEVEL, 0);
        b.tex_parameter(gl::TEXTURE_2D, gl::TEXTURE_MAX_LEVEL, 0);
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
                host:
                    Host::Buffer {
                        gl: b,
                        shadow,
                        version,
                    },
                ..
            }) = self.resources.get_mut(&dst)
            {
                let end = dx as usize + data.len();
                if end <= shadow.len() {
                    *version += 1;
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
        self.resolve(src);
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
            Host::Buffer {
                gl: b,
                shadow,
                version,
            } => {
                let (x, w) = (bx.x as usize, (bx.w as usize).min(data.len()));
                if x + w <= shadow.len() {
                    *version += 1;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restart_indices_widen_only_without_restart() {
        // A byte index of 255 is a vertex unless restart is on.
        let (d, ty) = widen_restart_indices(vec![0, 255, 7], gl::UNSIGNED_BYTE, false);
        assert_eq!(ty, gl::UNSIGNED_SHORT);
        let v: Vec<u16> = d
            .as_chunks()
            .0
            .iter()
            .map(|&c| u16::from_ne_bytes(c))
            .collect();
        assert_eq!(v, [0, 255, 7]);
        let (d, ty) = widen_restart_indices(vec![0, 255, 7], gl::UNSIGNED_BYTE, true);
        assert_eq!((d, ty), (vec![0, 255, 7], gl::UNSIGNED_BYTE));

        let shorts: Vec<u8> = [1u16, 0xffff]
            .iter()
            .flat_map(|i| i.to_ne_bytes())
            .collect();
        let (d, ty) = widen_restart_indices(shorts, gl::UNSIGNED_SHORT, false);
        assert_eq!(ty, gl::UNSIGNED_INT);
        let v: Vec<u32> = d
            .as_chunks()
            .0
            .iter()
            .map(|&c| u32::from_ne_bytes(c))
            .collect();
        assert_eq!(v, [1, 0xffff]);

        // Nothing at the maximum: left alone.
        let (d, ty) = widen_restart_indices(vec![1, 2, 3, 0], gl::UNSIGNED_SHORT, false);
        assert_eq!((d, ty), (vec![1, 2, 3, 0], gl::UNSIGNED_SHORT));
    }

    #[test]
    fn stream_output_assembles_independent_primitives() {
        let mut out = Vec::new();
        // A strip: odd triangles swap their first two vertices.
        assemble(5, &[0, 1, 2, 3, 4], &mut out);
        assert_eq!(out, [0, 1, 2, 2, 1, 3, 2, 3, 4]);
        out.clear();
        assemble(6, &[7, 8, 9, 10], &mut out);
        assert_eq!(out, [7, 8, 9, 7, 9, 10]);
        out.clear();
        // A loop closes; a list drops a partial primitive.
        assemble(2, &[1, 2, 3], &mut out);
        assert_eq!(out, [1, 2, 2, 3, 3, 1]);
        out.clear();
        assemble(4, &[1, 2, 3, 4], &mut out);
        assert_eq!(out, [1, 2, 3]);
        assert_eq!(prim_count(5, 5), 3);
        assert_eq!(prim_count(1, 5), 2);
        assert_eq!(prim_count(3, 1), 0);
    }

    #[test]
    fn view_swizzles_fold_in_missing_alpha() {
        let view = |format, swizzle| SamplerView {
            res: 1,
            format,
            first_level: 0,
            last_level: 0,
            first_element: 0,
            last_element: 0,
            swizzle,
        };
        // RGBA8 with the identity swizzle: nothing to do.
        assert_eq!(view_swizzle(&view(67, 0o3210)), 0);
        // R8G8B8X8: alpha reads one.
        assert_eq!(view_swizzle(&view(134, 0o3210)), 0x8000 | 0o5210);
        // RRR1 stays as it is.
        assert_eq!(view_swizzle(&view(64, 0o5000)), 0x8000 | 0o5000);
    }
}
