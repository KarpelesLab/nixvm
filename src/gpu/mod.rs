//! Optional GPU acceleration: a virtio-gpu device speaking the virgl protocol,
//! rendered with GLES 3.0 — WebGL2 in the browser, host EGL natively.
//!
//! The guest runs stock Mesa. Its `virgl` Gallium driver opens
//! `/dev/dri/renderD128` (the kernel side lives in `kernel::drm`) and drives
//! it with the virtio-gpu ioctls: resources (textures and buffers) whose
//! backing pages are guest memory, transfers between those pages and the
//! host's copies, and command buffers in virgl's wire format — the protocol
//! virglrenderer implements. This module is the host end:
//!
//! - [`GpuDevice`]: the device — resources, contexts, and the [`caps`] Mesa
//!   chooses its GL feature level from.
//! - [`virgl`]: the command-stream decoder.
//! - [`renderer`]: Gallium state (blend/rasterizer/DSA objects, surfaces,
//!   vertex elements, shaders …) turned into GLES 3.0 calls on a
//!   [`GpuBackend`].
//! - [`tgsi`] and [`glsl`]: shaders arrive as TGSI text and are translated to
//!   GLSL ES 3.00 when a vertex and a fragment shader are first drawn with
//!   together.
//! - Backends: [`backend::Recorder`] (tests, `NIXVM_GPU=log`), [`egl`]
//!   (native: host EGL + GLES through `dlopen`, surfaceless), `webgl` (the
//!   browser).
//!
//! Everything a guest does reaches the host synchronously: an ioctl returns
//! once its commands have been executed (or at least issued to GL), so
//! fences and waits are trivially complete.

pub mod backend;
pub mod caps;
#[cfg(all(not(target_arch = "wasm32"), target_os = "linux"))]
pub mod egl;
pub mod format;
pub mod gl;
pub mod glsl;
pub mod nodes;
pub mod renderer;
pub mod tgsi;
pub mod virgl;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub mod webgl;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub use backend::GpuBackend;

/// Where a resource's guest-visible bytes live: the pages of the GEM buffer
/// object the guest maps. Transfers copy between these and the host's copy.
pub trait Backing: Send + Sync {
    /// Read `buf.len()` bytes at `off` (past the end reads zero).
    fn read(&self, off: u64, buf: &mut [u8]);
    /// Write `data` at `off`.
    fn write(&self, off: u64, data: &[u8]);
}

/// Plain memory as a [`Backing`]: for tests and host-side resources.
#[derive(Debug, Default)]
pub struct MemBacking(pub Mutex<Vec<u8>>);

impl MemBacking {
    #[must_use]
    pub fn new(size: usize) -> Arc<Self> {
        Arc::new(Self(Mutex::new(vec![0; size])))
    }
}

impl Backing for MemBacking {
    fn read(&self, off: u64, buf: &mut [u8]) {
        let v = self.0.lock().unwrap();
        for (i, b) in buf.iter_mut().enumerate() {
            *b = v.get(off as usize + i).copied().unwrap_or(0);
        }
    }
    fn write(&self, off: u64, data: &[u8]) {
        let mut v = self.0.lock().unwrap();
        let end = off as usize + data.len();
        if v.len() < end {
            v.resize(end, 0);
        }
        v[off as usize..end].copy_from_slice(data);
    }
}

/// Gallium's `pipe_texture_target` values.
pub mod target {
    pub const BUFFER: u32 = 0;
    pub const TEXTURE_1D: u32 = 1;
    pub const TEXTURE_2D: u32 = 2;
    pub const TEXTURE_3D: u32 = 3;
    pub const TEXTURE_CUBE: u32 = 4;
    pub const TEXTURE_RECT: u32 = 5;
    pub const TEXTURE_1D_ARRAY: u32 = 6;
    pub const TEXTURE_2D_ARRAY: u32 = 7;
    pub const TEXTURE_CUBE_ARRAY: u32 = 8;
}

/// What `DRM_IOCTL_VIRTGPU_RESOURCE_CREATE` describes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResourceDesc {
    pub target: u32,
    pub format: u32,
    pub bind: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub array_size: u32,
    pub last_level: u32,
    pub nr_samples: u32,
    pub flags: u32,
}

/// A 3D box in a resource (`struct drm_virtgpu_3d_box`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Box3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub w: u32,
    pub h: u32,
    pub d: u32,
}

/// One transfer between a resource's backing and the host's copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transfer {
    pub res: u32,
    pub level: u32,
    pub bx: Box3,
    /// Where the box starts in the backing.
    pub offset: u64,
    /// Bytes per row and per layer in the backing (0: tightly packed at the
    /// level's size).
    pub stride: u32,
    pub layer_stride: u32,
    /// Guest → host (`TRANSFER_TO_HOST`) or host → guest.
    pub to_host: bool,
}

/// Why a device call failed: the errno the ioctl returns.
pub type Errno = i32;
const EINVAL: Errno = 22;

/// A virtio-gpu device: the state every guest file descriptor on
/// `/dev/dri/renderD128` shares. Cheap to clone.
#[derive(Clone)]
pub struct GpuDevice {
    inner: Arc<Mutex<renderer::Renderer>>,
    /// The resources' host-side reference counts (one per GEM handle any
    /// file holds): separate from the renderer so the kernel can count
    /// without taking the GL context.
    refs: Arc<Mutex<HashMap<u32, u32>>>,
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice").finish_non_exhaustive()
    }
}

impl GpuDevice {
    /// A device rendering with `backend`. `trace` logs each command the
    /// guest submits to stderr.
    #[must_use]
    pub fn new(backend: Box<dyn GpuBackend>, trace: bool) -> Self {
        Self {
            inner: Arc::new(Mutex::new(renderer::Renderer::new(backend, trace))),
            refs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// The device for `spec` (what `NIXVM_GPU` says): `egl` (native host
    /// GLES, surfaceless), `log` (record and print every GL call), `null`
    /// (record silently). `None` for anything else, or a backend this host
    /// can't provide.
    #[must_use]
    pub fn from_spec(spec: &str, trace: bool) -> Option<Self> {
        let backend: Box<dyn GpuBackend> = match spec {
            "log" => Box::new(backend::Recorder::new(true)),
            "null" => Box::new(backend::Recorder::new(false)),
            #[cfg(all(not(target_arch = "wasm32"), target_os = "linux"))]
            "egl" | "1" | "on" => match egl::Egl::new() {
                Ok(b) => Box::new(b),
                Err(e) => {
                    eprintln!("nixvm: NIXVM_GPU={spec}: {e}");
                    return None;
                }
            },
            _ => return None,
        };
        Some(Self::new(backend, trace))
    }

    /// The device configured by the environment (`NIXVM_GPU`,
    /// `NIXVM_GPU_TRACE`), if any.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let spec = std::env::var("NIXVM_GPU").ok()?;
        if spec.is_empty() || spec == "0" || spec == "off" {
            return None;
        }
        let trace = std::env::var_os("NIXVM_GPU_TRACE").is_some();
        Self::from_spec(&spec, trace)
    }

    /// Run `f` on the renderer with the backend's context current.
    fn with<T>(&self, f: impl FnOnce(&mut renderer::Renderer) -> T) -> T {
        let mut r = self.inner.lock().unwrap();
        r.begin();
        let out = f(&mut r);
        r.end();
        out
    }

    /// The capability set `set` at `version` (`DRM_IOCTL_VIRTGPU_GET_CAPS`).
    #[must_use]
    pub fn capset(&self, set: u32, version: u32) -> Option<Vec<u8>> {
        caps::capset(set, version)
    }

    /// A new rendering context (one per guest DRM file that renders).
    pub fn create_context(&self) -> u32 {
        self.with(renderer::Renderer::create_context)
    }

    /// Drop context `ctx` and everything it created.
    pub fn destroy_context(&self, ctx: u32) {
        self.with(|r| r.destroy_context(ctx));
    }

    /// A new resource backed by `backing`, with one reference (the GEM
    /// handle that names it).
    ///
    /// # Errors
    /// `EINVAL` for a description the host can't hold.
    pub fn create_resource(
        &self,
        desc: ResourceDesc,
        backing: Arc<dyn Backing>,
    ) -> Result<u32, Errno> {
        let id = self.with(|r| r.create_resource(desc, backing))?;
        self.refs.lock().unwrap().insert(id, 1);
        Ok(id)
    }

    /// Another reference to resource `id` (a second handle, a PRIME import).
    pub fn ref_resource(&self, id: u32) {
        *self.refs.lock().unwrap().entry(id).or_insert(0) += 1;
    }

    /// Drop a reference to resource `id`; the last one destroys it.
    pub fn unref_resource(&self, id: u32) {
        let last = {
            let mut refs = self.refs.lock().unwrap();
            match refs.get_mut(&id) {
                Some(n) if *n > 1 => {
                    *n -= 1;
                    false
                }
                Some(_) => {
                    refs.remove(&id);
                    true
                }
                None => false,
            }
        };
        if last {
            self.with(|r| r.destroy_resource(id));
        }
    }

    /// Copy between a resource's backing and the host's copy.
    ///
    /// # Errors
    /// `EINVAL` for an unknown resource or a box outside it.
    pub fn transfer(&self, t: &Transfer) -> Result<(), Errno> {
        self.with(|r| r.transfer(t))
    }

    /// Execute a command buffer of context `ctx`.
    ///
    /// # Errors
    /// `EINVAL` for an unknown context. A malformed command is logged and
    /// skipped, as virglrenderer does — the guest can't be told which.
    pub fn submit(&self, ctx: u32, cmds: &[u32]) -> Result<(), Errno> {
        self.with(|r| r.submit(ctx, cmds))
    }

    /// The renderer name the backend reports.
    #[must_use]
    pub fn backend_name(&self) -> String {
        self.inner.lock().unwrap().backend_name()
    }
}
