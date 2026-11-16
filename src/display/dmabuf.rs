//! `zwp_linux_dmabuf_v1` (v4, with feedback): buffers a client rendered on
//! the virtual GPU, shared as dma-buf descriptors instead of shared memory.
//! This is what Mesa's Wayland EGL platform (and GTK's and Chromium's GL
//! paths on top of it) needs to show anything rendered with the GPU.
//!
//! The feedback names the render node (`/dev/dri/renderD128`) as the main
//! device and offers the 8888 formats, linear or with an implicit
//! (`INVALID`) modifier, from a format table nixvm writes into the guest
//! next to its socket. A buffer made from a dma-buf is a GPU resource: at
//! commit nixvm reads it back from the host GPU (`glReadPixels` on the
//! resource's texture, through [`GpuFrames`]) into the surface's image, as
//! it copies a `wl_shm` buffer, and releases it at once.
//!
//! Everything GPU-specific is behind [`GpuFrames`], so the compositor builds
//! without the `gpu` feature; it then has no GPU and offers no dmabuf
//! global.

use std::sync::Arc;

use super::render;
use super::server::{Compositor, Data, Iface};
use super::wire::{self, Args, Ev, ProtoError};
use crate::kernel::{Kernel, ServedFd};

/// DRM fourccs offered, with the `wl_shm` format whose bytes they share.
pub const FORMATS: &[(u32, u32)] = &[
    (fourcc(*b"AR24"), render::ARGB8888),
    (fourcc(*b"XR24"), render::XRGB8888),
    (fourcc(*b"AB24"), render::ABGR8888),
    (fourcc(*b"XB24"), render::XBGR8888),
];

/// `DRM_FORMAT_MOD_LINEAR` and `DRM_FORMAT_MOD_INVALID` (an implicit
/// modifier: whatever the driver allocated, which a virgl resource is).
pub const MOD_LINEAR: u64 = 0;
pub const MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;
const MODIFIERS: [u64; 2] = [MOD_LINEAR, MOD_INVALID];

const fn fourcc(c: [u8; 4]) -> u32 {
    u32::from_le_bytes(c)
}

/// The `wl_shm` format whose memory layout DRM fourcc `drm` has.
#[must_use]
pub fn shm_format(drm: u32) -> Option<u32> {
    FORMATS.iter().find(|f| f.0 == drm).map(|f| f.1)
}

/// The format table feedback hands out: `(format, padding, modifier)`
/// entries, every format with every modifier.
#[must_use]
pub fn format_table() -> Vec<u8> {
    let mut t = Vec::new();
    for &(f, _) in FORMATS {
        for m in MODIFIERS {
            t.extend_from_slice(&f.to_le_bytes());
            t.extend_from_slice(&0u32.to_le_bytes());
            t.extend_from_slice(&m.to_le_bytes());
        }
    }
    t
}

/// What the compositor needs of the virtual GPU, and nothing more.
pub trait GpuFrames: Send + Sync {
    /// The render node's `dev_t`: the feedback's main device.
    fn render_dev(&self) -> u64;
    /// The GPU resource behind `fd` and the buffer's size in bytes, if it
    /// is a dma-buf of this VM's GPU.
    fn resource(&self, kernel: &Kernel, fd: &ServedFd) -> Option<(u32, u64)>;
    /// The size of colour texture `res` (level 0), if it can be read back.
    fn texture_size(&self, res: u32) -> Option<(u32, u32)>;
    /// Read `rect` (x, y, w, h) of texture `res` into `out`: rows packed,
    /// top first, in the resource's byte layout.
    fn read(&self, res: u32, rect: (u32, u32, u32, u32), out: &mut [u8]) -> bool;
    /// A compositor drawing on this GPU (see [`super::compose`]), if it
    /// can have one.
    fn compositor(&self) -> Option<Box<dyn super::compose::Compositor>> {
        None
    }
}

/// The VM's GPU as the compositor sees it, if one is attached.
#[cfg(feature = "gpu")]
pub fn gpu_frames(kernel: &Kernel) -> Option<Arc<dyn GpuFrames>> {
    Some(Arc::new(VirtGpu(kernel.gpu()?.clone())))
}

/// Without the `gpu` feature there is never a GPU.
#[cfg(not(feature = "gpu"))]
pub fn gpu_frames(_kernel: &Kernel) -> Option<Arc<dyn GpuFrames>> {
    None
}

#[cfg(feature = "gpu")]
struct VirtGpu(crate::gpu::GpuDevice);

#[cfg(feature = "gpu")]
impl GpuFrames for VirtGpu {
    fn render_dev(&self) -> u64 {
        crate::gpu::nodes::render_rdev()
    }
    fn resource(&self, kernel: &Kernel, fd: &ServedFd) -> Option<(u32, u64)> {
        kernel.served_gpu_resource(fd)
    }
    fn texture_size(&self, res: u32) -> Option<(u32, u32)> {
        let d = self.0.resource_desc(res)?;
        let tex2d = d.target == crate::gpu::target::TEXTURE_2D
            || d.target == crate::gpu::target::TEXTURE_RECT;
        let fmt = crate::gpu::format::lookup(d.format)?;
        (tex2d && fmt.bpp == 4).then_some((d.width, d.height))
    }
    fn read(&self, res: u32, rect: (u32, u32, u32, u32), out: &mut [u8]) -> bool {
        self.0.read_texture(res, rect, out).is_ok()
    }
    fn compositor(&self) -> Option<Box<dyn super::compose::Compositor>> {
        super::gl::GlCompositor::new(self.0.clone())
            .map(|c| Box::new(c) as Box<dyn super::compose::Compositor>)
    }
}

/// A `wl_buffer` made from a dma-buf.
#[derive(Debug)]
pub struct DmaBuffer {
    /// Keeps the resource alive while the buffer exists (held, not read).
    #[allow(dead_code)]
    pub fd: ServedFd,
    pub res: u32,
    pub width: u32,
    pub height: u32,
    /// The `wl_shm` format with the same bytes.
    pub format: u32,
    /// Rows bottom first (`Y_INVERT`).
    pub y_invert: bool,
}

/// One plane `add`ed to a params object.
#[derive(Debug)]
pub struct Plane<F> {
    pub fd: F,
    pub offset: u32,
    pub stride: u32,
    pub modifier: u64,
}

/// A `zwp_linux_buffer_params_v1` being filled in.
#[derive(Debug)]
pub struct Params<F> {
    pub planes: [Option<Plane<F>>; 4],
    /// `create`/`create_immed` was called: it can't be used again.
    pub used: bool,
}

impl<F> Default for Params<F> {
    fn default() -> Self {
        Self {
            planes: [None, None, None, None],
            used: false,
        }
    }
}

/// `zwp_linux_buffer_params_v1` errors.
pub const ERR_ALREADY_USED: u32 = 0;
pub const ERR_PLANE_IDX: u32 = 1;
pub const ERR_PLANE_SET: u32 = 2;
pub const ERR_INCOMPLETE: u32 = 3;
pub const ERR_INVALID_FORMAT: u32 = 4;
pub const ERR_INVALID_DIMENSIONS: u32 = 5;
pub const ERR_OUT_OF_BOUNDS: u32 = 6;

/// `zwp_linux_buffer_params_v1.flags`.
const FLAG_Y_INVERT: u32 = 1;

/// Why a buffer can't be made: a protocol error (the client broke the
/// rules), or a plain failure (`failed`: nixvm can't use what it was given).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    Protocol(u32, String),
    Failed(String),
}

/// What a valid `create` describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Valid {
    pub width: u32,
    pub height: u32,
    /// The `wl_shm` format with the same bytes.
    pub format: u32,
    pub y_invert: bool,
}

impl<F> Params<F> {
    /// `add`: plane `idx` is `fd` at `offset`, rows `stride` apart.
    ///
    /// # Errors
    /// A protocol error for a used params object, a plane index past 3 or
    /// one already set; `fd` comes back with it, to be released.
    pub fn add(&mut self, idx: u32, plane: Plane<F>) -> Result<(), (u32, &'static str, Plane<F>)> {
        if self.used {
            return Err((ERR_ALREADY_USED, "params already used", plane));
        }
        let Some(slot) = self.planes.get_mut(idx as usize) else {
            return Err((ERR_PLANE_IDX, "plane index out of bounds", plane));
        };
        if slot.is_some() {
            return Err((ERR_PLANE_SET, "plane already set", plane));
        }
        *slot = Some(plane);
        Ok(())
    }

    /// Check a `create` of a `width` × `height` buffer of DRM `format`
    /// against the planes (one, as every offered format has), where
    /// `size` is the byte size of plane 0's buffer (if known).
    ///
    /// # Errors
    /// See [`Refusal`]: the protocol's errors for incomplete planes, an
    /// unknown format, bad dimensions or a plane outside its buffer; a
    /// failure for flags or a modifier nixvm doesn't take.
    pub fn check(
        &self,
        width: i32,
        height: i32,
        format: u32,
        flags: u32,
        size: Option<u64>,
    ) -> Result<Valid, Refusal> {
        let proto = |c, m: &str| Refusal::Protocol(c, m.to_string());
        if self.used {
            return Err(proto(ERR_ALREADY_USED, "params already used"));
        }
        let Some(shm) = shm_format(format) else {
            return Err(proto(
                ERR_INVALID_FORMAT,
                &format!("format {format:#x} not supported"),
            ));
        };
        let Some(p) = &self.planes[0] else {
            return Err(proto(ERR_INCOMPLETE, "no plane 0"));
        };
        if self.planes[1..].iter().any(Option::is_some) {
            return Err(proto(ERR_INCOMPLETE, "too many planes for the format"));
        }
        if width <= 0 || height <= 0 {
            return Err(proto(ERR_INVALID_DIMENSIONS, "invalid width or height"));
        }
        let (w, h) = (width.unsigned_abs(), height.unsigned_abs());
        let end = u64::from(p.stride)
            .checked_mul(u64::from(h - 1))
            .and_then(|v| v.checked_add(u64::from(p.offset)))
            .and_then(|v| v.checked_add(u64::from(w) * 4));
        let fits = u64::from(p.stride) >= u64::from(w) * 4
            && match (end, size) {
                (Some(e), Some(s)) => e <= s,
                (Some(_), None) => true,
                (None, _) => false,
            };
        if !fits {
            return Err(proto(ERR_OUT_OF_BOUNDS, "plane outside its buffer"));
        }
        if !MODIFIERS.contains(&p.modifier) {
            return Err(Refusal::Failed(format!(
                "modifier {:#x} not supported",
                p.modifier
            )));
        }
        if flags & !FLAG_Y_INVERT != 0 {
            return Err(Refusal::Failed(format!("flags {flags:#x} not supported")));
        }
        if p.offset != 0 {
            return Err(Refusal::Failed("plane offset must be 0".into()));
        }
        Ok(Valid {
            width: w,
            height: h,
            format: shm,
            y_invert: flags & FLAG_Y_INVERT != 0,
        })
    }

    /// Every plane's descriptor, taking them out.
    pub fn take_fds(&mut self) -> Vec<F> {
        self.planes
            .iter_mut()
            .filter_map(|p| p.take().map(|p| p.fd))
            .collect()
    }
}

fn bad_op(obj: u32) -> ProtoError {
    ProtoError::new(obj, wire::ERR_INVALID_METHOD, "bad opcode")
}

impl Compositor {
    /// The VM's GPU, looked up the first time a client asks (it may be
    /// attached after the display).
    pub(super) fn gpu(&mut self, kernel: &Kernel) -> Option<Arc<dyn GpuFrames>> {
        if self.gpu.is_none() {
            self.gpu = gpu_frames(kernel);
        }
        self.gpu.clone()
    }

    /// What a new binding of the dmabuf global is told: for v3 and older
    /// the formats (and modifiers) as events; v4 asks for feedback instead.
    pub(super) fn bind_dmabuf(&mut self, cid: u32, id: u32, version: u32) {
        if version >= 4 {
            return;
        }
        for &(f, _) in FORMATS {
            if version >= 3 {
                for m in MODIFIERS {
                    #[allow(clippy::cast_possible_truncation)]
                    self.send(
                        cid,
                        Ev::new(id, 1).uint(f).uint((m >> 32) as u32).uint(m as u32),
                    );
                }
            } else {
                self.send(cid, Ev::new(id, 0).uint(f));
            }
        }
    }

    pub(super) fn req_dmabuf(
        &mut self,
        kernel: &Kernel,
        cid: u32,
        obj: u32,
        version: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        match op {
            0 => {
                self.destroy(cid, obj);
                Ok(())
            }
            1 => {
                let id = a.uint()?;
                self.create(
                    cid,
                    id,
                    Iface::DmabufParams,
                    version,
                    Data::Params(Box::default()),
                )
            }
            2 | 3 if version >= 4 => {
                let id = a.uint()?;
                if op == 3 {
                    self.expect(cid, a.uint()?, Iface::Surface)?;
                }
                self.create(cid, id, Iface::DmabufFeedback, version, Data::None)?;
                self.send_feedback(kernel, cid, id);
                Ok(())
            }
            _ => Err(bad_op(obj)),
        }
    }

    /// The feedback: one tranche, on the render node, of every format
    /// with every modifier in the table.
    fn send_feedback(&mut self, kernel: &Kernel, cid: u32, id: u32) {
        let Some(gpu) = self.gpu(kernel) else {
            return;
        };
        let table = format_table();
        if !self.format_table_written {
            if kernel
                .served_write_file(&self.format_table_path, &table, 0o444)
                .is_err()
            {
                return;
            }
            self.format_table_written = true;
        }
        let Some(fd) = kernel.served_open_file(&self.format_table_path) else {
            return;
        };
        let dev = gpu.render_dev().to_le_bytes();
        #[allow(clippy::cast_possible_truncation)]
        let n = (table.len() / 16) as u16;
        let indices: Vec<u8> = (0..n).flat_map(u16::to_le_bytes).collect();
        let Some(c) = self.clients.get_mut(&cid) else {
            kernel.served_drop_fd(fd);
            return;
        };
        #[allow(clippy::cast_possible_truncation)]
        c.send_fd(Ev::new(id, 1).uint(table.len() as u32), fd);
        c.send(Ev::new(id, 2).array(&dev));
        c.send(Ev::new(id, 4).array(&dev));
        c.send(Ev::new(id, 6).uint(0));
        c.send(Ev::new(id, 5).array(&indices));
        c.send(Ev::new(id, 3));
        c.send(Ev::new(id, 0));
    }

    pub(super) fn req_dmabuf_feedback(
        &mut self,
        cid: u32,
        obj: u32,
        op: u16,
    ) -> Result<(), ProtoError> {
        if op != 0 {
            return Err(bad_op(obj));
        }
        self.destroy(cid, obj);
        Ok(())
    }

    pub(super) fn req_params(
        &mut self,
        kernel: &Kernel,
        cid: u32,
        obj: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        match op {
            0 => {
                if let Some(o) = self.destroy(cid, obj)
                    && let Data::Params(mut p) = o.data
                {
                    for fd in p.take_fds() {
                        kernel.served_drop_fd(fd);
                    }
                }
                Ok(())
            }
            1 => {
                let fd = a.fd()?;
                let (idx, offset, stride) = (a.uint()?, a.uint()?, a.uint()?);
                let modifier = (u64::from(a.uint()?) << 32) | u64::from(a.uint()?);
                let plane = Plane {
                    fd,
                    offset,
                    stride,
                    modifier,
                };
                let r = match self.object_mut(cid, obj).map(|o| &mut o.data) {
                    Some(Data::Params(p)) => p.add(idx, plane),
                    _ => Err((ERR_ALREADY_USED, "not a params object", plane)),
                };
                if let Err((code, msg, plane)) = r {
                    kernel.served_drop_fd(plane.fd);
                    return Err(ProtoError::new(obj, code, msg));
                }
                Ok(())
            }
            2 | 3 => {
                let immed = op == 3;
                let new_id = if immed { Some(a.uint()?) } else { None };
                let (w, h, format, flags) = (a.int()?, a.int()?, a.uint()?, a.uint()?);
                self.create_dma_buffer(kernel, cid, obj, new_id, (w, h, format, flags))
            }
            _ => Err(bad_op(obj)),
        }
    }

    /// `create` (`new_id` none: answer `created` or `failed`) or
    /// `create_immed` (a failure is fatal for the client).
    fn create_dma_buffer(
        &mut self,
        kernel: &Kernel,
        cid: u32,
        obj: u32,
        new_id: Option<u32>,
        (w, h, format, flags): (i32, i32, u32, u32),
    ) -> Result<(), ProtoError> {
        let gpu = self.gpu(kernel);
        let Some(Data::Params(p)) = self.object_mut(cid, obj).map(|o| &mut o.data) else {
            return Err(bad_op(obj));
        };
        if p.used {
            return Err(ProtoError::new(
                obj,
                ERR_ALREADY_USED,
                "params already used",
            ));
        }
        let found = match (&gpu, &p.planes[0]) {
            (Some(g), Some(pl)) => g.resource(kernel, &pl.fd),
            _ => None,
        };
        let checked = p.check(w, h, format, flags, found.map(|f| f.1));
        p.used = true;
        let mut fds = p.take_fds();
        let made = checked.and_then(|v| {
            let (res, _) = found.ok_or_else(|| Refusal::Failed("not a GPU buffer".into()))?;
            let fits = gpu
                .as_ref()
                .and_then(|g| g.texture_size(res))
                .is_some_and(|(tw, th)| v.width <= tw && v.height <= th);
            if fits {
                Ok((v, res))
            } else {
                Err(Refusal::Failed("not a colour texture that size".into()))
            }
        });
        // Plane 0's descriptor stays with the buffer; any other goes.
        let fd0 = if made.is_ok() && !fds.is_empty() {
            Some(fds.remove(0))
        } else {
            None
        };
        for fd in fds {
            kernel.served_drop_fd(fd);
        }
        let (v, res) = match made {
            Ok(m) => m,
            Err(Refusal::Protocol(code, msg)) => return Err(ProtoError::new(obj, code, msg)),
            Err(Refusal::Failed(msg)) => {
                if new_id.is_some() {
                    // zwp_linux_buffer_params_v1.error.invalid_wl_buffer
                    return Err(ProtoError::new(obj, 7, msg));
                }
                self.send(cid, Ev::new(obj, 1)); // failed
                return Ok(());
            }
        };
        let Some(fd) = fd0 else {
            return Ok(());
        };
        let buf = Arc::new(DmaBuffer {
            fd,
            res,
            width: v.width,
            height: v.height,
            format: v.format,
            y_invert: v.y_invert,
        });
        if let Some(id) = new_id {
            return self.create(cid, id, Iface::Buffer, 1, Data::DmaBuffer(buf));
        }
        let Some(c) = self.clients.get_mut(&cid) else {
            return Ok(());
        };
        let id = c.alloc_id();
        c.objects.insert(
            id,
            super::server::Object {
                iface: Iface::Buffer,
                version: 1,
                data: Data::DmaBuffer(buf),
            },
        );
        c.send(Ev::new(obj, 0).uint(id)); // created
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(stride: u32, modifier: u64) -> Plane<()> {
        Plane {
            fd: (),
            offset: 0,
            stride,
            modifier,
        }
    }

    const AR24: u32 = fourcc(*b"AR24");

    #[test]
    fn the_table_lists_every_format_with_every_modifier() {
        let t = format_table();
        assert_eq!(t.len(), FORMATS.len() * MODIFIERS.len() * 16);
        assert_eq!(&t[0..4], b"AR24");
        assert_eq!(&t[8..16], &MOD_LINEAR.to_le_bytes());
        assert_eq!(&t[24..32], &MOD_INVALID.to_le_bytes());
        assert_eq!(shm_format(AR24), Some(render::ARGB8888));
        assert_eq!(shm_format(fourcc(*b"XB24")), Some(render::XBGR8888));
        assert_eq!(shm_format(fourcc(*b"NV12")), None);
    }

    #[test]
    fn planes_are_added_once_and_in_range() {
        let mut p = Params::default();
        assert!(p.add(0, plane(256, MOD_INVALID)).is_ok());
        assert_eq!(p.add(0, plane(256, 0)).unwrap_err().0, ERR_PLANE_SET);
        assert_eq!(p.add(4, plane(256, 0)).unwrap_err().0, ERR_PLANE_IDX);
        p.used = true;
        assert_eq!(p.add(1, plane(256, 0)).unwrap_err().0, ERR_ALREADY_USED);
        assert_eq!(p.take_fds().len(), 1);
    }

    #[test]
    fn create_is_validated() {
        let proto = |r: Result<Valid, Refusal>| match r {
            Err(Refusal::Protocol(c, _)) => Some(c),
            _ => None,
        };
        let mut p = Params::default();
        assert_eq!(
            proto(p.check(64, 64, AR24, 0, Some(1 << 20))),
            Some(ERR_INCOMPLETE)
        );
        p.add(0, plane(256, MOD_INVALID)).unwrap();
        // A good one: 64x64 ARGB, 256-byte rows, in 16 KiB.
        assert_eq!(
            p.check(64, 64, AR24, 0, Some(16384)),
            Ok(Valid {
                width: 64,
                height: 64,
                format: render::ARGB8888,
                y_invert: false
            })
        );
        assert!(p.check(64, 64, AR24, 1, Some(16384)).unwrap().y_invert);
        assert_eq!(
            proto(p.check(64, 64, fourcc(*b"NV12"), 0, None)),
            Some(ERR_INVALID_FORMAT)
        );
        assert_eq!(
            proto(p.check(0, 64, AR24, 0, None)),
            Some(ERR_INVALID_DIMENSIONS)
        );
        assert_eq!(
            proto(p.check(64, -1, AR24, 0, None)),
            Some(ERR_INVALID_DIMENSIONS)
        );
        // One row too many for the buffer; rows narrower than the width.
        assert_eq!(
            proto(p.check(64, 65, AR24, 0, Some(16384))),
            Some(ERR_OUT_OF_BOUNDS)
        );
        assert_eq!(
            proto(p.check(65, 64, AR24, 0, Some(1 << 20))),
            Some(ERR_OUT_OF_BOUNDS)
        );
        // Unknown flags or modifiers fail without a protocol error.
        assert!(matches!(
            p.check(64, 64, AR24, 2, Some(16384)),
            Err(Refusal::Failed(_))
        ));
        let mut q = Params::default();
        q.add(0, plane(256, 0x0100_0000_0000_0001)).unwrap();
        assert!(matches!(
            q.check(64, 64, AR24, 0, Some(16384)),
            Err(Refusal::Failed(_))
        ));
        // Two planes for a one-plane format.
        q.add(1, plane(256, 0)).unwrap();
        assert_eq!(
            proto(q.check(64, 64, AR24, 0, Some(16384))),
            Some(ERR_INCOMPLETE)
        );
        q.used = true;
        assert_eq!(
            proto(q.check(64, 64, AR24, 0, Some(16384))),
            Some(ERR_ALREADY_USED)
        );
    }
}
