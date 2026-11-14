//! `/dev/dri/renderD128`: the kernel half of nixvm's virtio-gpu (the host
//! half is [`crate::gpu`]).
//!
//! What Linux's `virtio_gpu` DRM driver does for a render node, nixvm does
//! here: the generic DRM ioctls libdrm and Mesa's loader use (`VERSION`,
//! `GET_CAP`, `GEM_CLOSE`, …) and the virtgpu ones Mesa's `virgl` winsys
//! uses (`GETPARAM`, `GET_CAPS`, `RESOURCE_CREATE`, `MAP`, `EXECBUFFER`,
//! `TRANSFER_TO/FROM_HOST`, `WAIT`, `RESOURCE_INFO`, `CONTEXT_INIT`).
//!
//! A GEM buffer object is a shared-memory file of guest frames
//! ([`FileObj`]) — the resource's backing — and mapping the render node at
//! the offset `MAP` hands out maps it exactly as `mmap` of a memfd maps
//! that file's pages. Handles are per open file description, as on Linux,
//! and so is the rendering context, made on first use. The last descriptor
//! going drops its handles and context.
//!
//! The device answers synchronously, so `WAIT` never waits and execbuffers
//! carry no fences (the driver reports version 0.0, which tells Mesa not to
//! ask for fence fds). No blobs, no PRIME sharing yet (phase 2).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::{Errno, Fd, Kernel, ServiceCtx, err};
use crate::gpu::{Backing, Box3, GpuDevice, ResourceDesc, Transfer, caps, nodes};
use crate::vcpu::GuestMemory;
use crate::vcpu::filemap::FileObj;
use crate::vcpu::mem::PAGE_SIZE;

/// An open render node: shared by `dup`s and `fork` copies.
#[derive(Debug, Clone)]
pub struct DrmFile(Arc<DrmInner>);

#[derive(Debug)]
struct DrmInner {
    dev: GpuDevice,
    st: Mutex<DrmState>,
}

#[derive(Debug, Default)]
struct DrmState {
    /// The rendering context, made by `CONTEXT_INIT` or the first
    /// execbuffer.
    ctx: Option<u32>,
    handles: HashMap<u32, Bo>,
    next_handle: u32,
}

#[derive(Debug, Clone)]
struct Bo {
    res: u32,
    obj: Arc<FileObj>,
    size: u64,
}

impl Drop for DrmInner {
    fn drop(&mut self) {
        let st = self.st.get_mut().unwrap();
        for bo in st.handles.values() {
            self.dev.unref_resource(bo.res);
        }
        if let Some(ctx) = st.ctx {
            self.dev.destroy_context(ctx);
        }
    }
}

impl DrmFile {
    /// A stable identity for the open file.
    #[must_use]
    pub fn id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    /// The pages behind map offset `offset`, and the page of them it
    /// starts at.
    pub(super) fn mapping(&self, offset: u64) -> Option<(Arc<FileObj>, u64)> {
        let handle = (offset >> 32) as u32;
        let st = self.0.st.lock().unwrap();
        let bo = st.handles.get(&handle)?;
        Some((Arc::clone(&bo.obj), (offset & 0xffff_ffff) / PAGE_SIZE))
    }
}

/// A buffer object's pages as a resource's [`Backing`].
struct BoBacking(Arc<FileObj>);

impl Backing for BoBacking {
    fn read(&self, off: u64, buf: &mut [u8]) {
        let n = self.0.read(off, buf);
        buf[n..].fill(0);
    }
    fn write(&self, off: u64, data: &[u8]) {
        // Within the object (it never grows past its size); a frame that
        // can't be had leaves the guest's view as it was.
        let size = self.0.size();
        let n = size.saturating_sub(off).min(data.len() as u64) as usize;
        let _ = self.0.write(off, &data[..n], 0);
    }
}

// The ioctl numbers: `DRM_IOWR(nr, struct)`, type 'd'.
const DRM_IOCTL_VERSION: u32 = 0x00;
const DRM_IOCTL_GET_UNIQUE: u32 = 0x01;
const DRM_IOCTL_GEM_CLOSE: u32 = 0x09;
const DRM_IOCTL_GEM_FLINK: u32 = 0x0a;
const DRM_IOCTL_GEM_OPEN: u32 = 0x0b;
const DRM_IOCTL_GET_CAP: u32 = 0x0c;
const DRM_IOCTL_PRIME_HANDLE_TO_FD: u32 = 0x2d;
const DRM_IOCTL_PRIME_FD_TO_HANDLE: u32 = 0x2e;
const VIRTGPU_MAP: u32 = 0x41;
const VIRTGPU_EXECBUFFER: u32 = 0x42;
const VIRTGPU_GETPARAM: u32 = 0x43;
const VIRTGPU_RESOURCE_CREATE: u32 = 0x44;
const VIRTGPU_RESOURCE_INFO: u32 = 0x45;
const VIRTGPU_TRANSFER_FROM_HOST: u32 = 0x46;
const VIRTGPU_TRANSFER_TO_HOST: u32 = 0x47;
const VIRTGPU_WAIT: u32 = 0x48;
const VIRTGPU_GET_CAPS: u32 = 0x49;
const VIRTGPU_CONTEXT_INIT: u32 = 0x4b;

// VIRTGPU_PARAM_*.
const PARAM_3D_FEATURES: u64 = 1;
const PARAM_CAPSET_QUERY_FIX: u64 = 2;
const PARAM_SUPPORTED_CAPSET_IDS: u64 = 7;

/// The largest buffer object a guest may make (and a command buffer).
const MAX_BO: u64 = 1 << 30;
const MAX_CMD: usize = 16 << 20;

fn le32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn le64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

fn put32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn put64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

fn errno(e: i32) -> i64 {
    -i64::from(e)
}

impl Kernel {
    /// Give the guest the virtio-gpu `dev`: from now on `/dev/dri/
    /// renderD128` opens to it, and sysfs shows the device Mesa looks for.
    ///
    /// # Errors
    /// `EBUSY` if a GPU is already attached.
    pub fn attach_gpu(&self, dev: GpuDevice) -> Result<(), i64> {
        self.gpu.set(dev).map_err(|_| err(Errno::EBUSY))?;
        let mut vfs = self.vfs.lock().unwrap();
        vfs.enter(0, "/");
        for (point, fs) in nodes::mounts() {
            vfs.mount(point, fs);
        }
        Ok(())
    }

    /// Open the render node: a new DRM file on the attached GPU.
    pub(super) fn open_render_node(&self, cx: &mut ServiceCtx, flags: u64) -> Option<i64> {
        const O_CLOEXEC: u64 = 0o2_000_000;
        let dev = self.gpu.get()?.clone();
        let file = DrmFile(Arc::new(DrmInner {
            dev,
            st: Mutex::new(DrmState {
                next_handle: 1,
                ..DrmState::default()
            }),
        }));
        let fd = cx.cur.fds.alloc(Fd::Drm(file));
        cx.cur.fds.set_cloexec(fd, flags & O_CLOEXEC != 0);
        Some(i64::from(fd))
    }

    /// An ioctl on a render node.
    #[allow(clippy::too_many_lines, clippy::unused_self)]
    pub(super) fn drm_ioctl(
        &self,
        file: &DrmFile,
        req: u32,
        arg: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const EFAULT: i32 = 14;
        const EINVAL: i32 = 22;
        if (req >> 8) & 0xff != u32::from(b'd') {
            return err(Errno::ENOTTY);
        }
        let nr = req & 0xff;
        let size = ((req >> 16) & 0x3fff) as usize;
        // The argument struct as the guest passed it, zero-extended to what
        // this kernel reads (as Linux's drm_ioctl copies it).
        let mut a = vec![0u8; size.max(64)];
        if size > 0 && mem.read(arg, &mut a[..size]).is_err() {
            return errno(EFAULT);
        }
        let inner = &file.0;
        let dev = &inner.dev;
        let r: Result<bool, i32> = (|| {
            match nr {
                DRM_IOCTL_VERSION => {
                    // struct drm_version: three ints, then (len, ptr) for
                    // name, date and desc.
                    put32(&mut a, 0, 0);
                    put32(&mut a, 4, 0);
                    put32(&mut a, 8, 0);
                    for (i, s) in ["virtio_gpu", "0", "virtio GPU"].iter().enumerate() {
                        let (len_off, ptr_off) = (16 + i * 16, 24 + i * 16);
                        let (len, ptr) = (le64(&a, len_off) as usize, le64(&a, ptr_off));
                        if ptr != 0 && len > 0 {
                            let n = len.min(s.len());
                            mem.write(ptr, &s.as_bytes()[..n]).map_err(|_| EFAULT)?;
                        }
                        put64(&mut a, len_off, s.len() as u64);
                    }
                    Ok(true)
                }
                DRM_IOCTL_GET_UNIQUE => {
                    let unique = format!("pci:{}", nodes::PCI_SLOT);
                    let (len, ptr) = (le64(&a, 0) as usize, le64(&a, 8));
                    if ptr != 0 && len > 0 {
                        let n = len.min(unique.len());
                        mem.write(ptr, &unique.as_bytes()[..n])
                            .map_err(|_| EFAULT)?;
                    }
                    put64(&mut a, 0, unique.len() as u64);
                    Ok(true)
                }
                DRM_IOCTL_GEM_CLOSE => {
                    let h = le32(&a, 0);
                    let bo = inner.st.lock().unwrap().handles.remove(&h).ok_or(EINVAL)?;
                    dev.unref_resource(bo.res);
                    Ok(false)
                }
                // Global names are for the primary node; render nodes refuse.
                DRM_IOCTL_GEM_FLINK | DRM_IOCTL_GEM_OPEN => Err(13),
                DRM_IOCTL_GET_CAP => {
                    // DRM_CAP_PRIME (5), DRM_CAP_SYNCOBJ (0x13) …: none.
                    let cap = le64(&a, 0);
                    if cap == 0 || cap > 0x20 {
                        return Err(EINVAL);
                    }
                    put64(&mut a, 8, 0);
                    Ok(true)
                }
                DRM_IOCTL_PRIME_HANDLE_TO_FD | DRM_IOCTL_PRIME_FD_TO_HANDLE => Err(95),
                VIRTGPU_GETPARAM => {
                    let v: u32 = match le64(&a, 0) {
                        PARAM_3D_FEATURES | PARAM_CAPSET_QUERY_FIX => 1,
                        PARAM_SUPPORTED_CAPSET_IDS => {
                            (1 << caps::CAPSET_VIRGL) | (1 << caps::CAPSET_VIRGL2)
                        }
                        3..=6 | 8 => 0,
                        _ => return Err(EINVAL),
                    };
                    // Linux writes an int.
                    mem.write(le64(&a, 8), &v.to_le_bytes())
                        .map_err(|_| EFAULT)?;
                    Ok(false)
                }
                VIRTGPU_GET_CAPS => {
                    let (id, ver, addr, len) =
                        (le32(&a, 0), le32(&a, 4), le64(&a, 8), le32(&a, 16) as usize);
                    let c = dev.capset(id, ver).ok_or(EINVAL)?;
                    let n = len.min(c.len());
                    mem.write(addr, &c[..n]).map_err(|_| EFAULT)?;
                    Ok(false)
                }
                VIRTGPU_CONTEXT_INIT => {
                    let mut st = inner.st.lock().unwrap();
                    if st.ctx.is_some() {
                        return Err(17); // EEXIST
                    }
                    st.ctx = Some(dev.create_context());
                    Ok(false)
                }
                VIRTGPU_RESOURCE_CREATE => {
                    let desc = ResourceDesc {
                        target: le32(&a, 0),
                        format: le32(&a, 4),
                        bind: le32(&a, 8),
                        width: le32(&a, 12),
                        height: le32(&a, 16),
                        depth: le32(&a, 20),
                        array_size: le32(&a, 24),
                        last_level: le32(&a, 28),
                        nr_samples: le32(&a, 32),
                        flags: le32(&a, 36),
                    };
                    let size =
                        u64::from(le32(&a, 48)).max(PAGE_SIZE).div_ceil(PAGE_SIZE) * PAGE_SIZE;
                    if size > MAX_BO {
                        return Err(12); // ENOMEM
                    }
                    let obj = FileObj::new(mem.phys_arc(), mem.allocator(), size, true);
                    obj.set_origin(nodes::RENDER_NODE);
                    let res = dev.create_resource(desc, Arc::new(BoBacking(Arc::clone(&obj))))?;
                    let mut st = inner.st.lock().unwrap();
                    let h = st.next_handle;
                    st.next_handle += 1;
                    st.handles.insert(h, Bo { res, obj, size });
                    put32(&mut a, 40, h);
                    put32(&mut a, 44, res);
                    Ok(true)
                }
                VIRTGPU_RESOURCE_INFO => {
                    let st = inner.st.lock().unwrap();
                    let bo = st.handles.get(&le32(&a, 0)).ok_or(EINVAL)?;
                    put32(&mut a, 4, bo.res);
                    put32(&mut a, 8, bo.size as u32);
                    put32(&mut a, 12, 0);
                    Ok(true)
                }
                VIRTGPU_MAP => {
                    let h = le32(&a, 8);
                    if !inner.st.lock().unwrap().handles.contains_key(&h) {
                        return Err(EINVAL);
                    }
                    put64(&mut a, 0, u64::from(h) << 32);
                    Ok(true)
                }
                VIRTGPU_TRANSFER_TO_HOST | VIRTGPU_TRANSFER_FROM_HOST => {
                    let res = {
                        let st = inner.st.lock().unwrap();
                        st.handles.get(&le32(&a, 0)).ok_or(EINVAL)?.res
                    };
                    let t = Transfer {
                        res,
                        bx: Box3 {
                            x: le32(&a, 4),
                            y: le32(&a, 8),
                            z: le32(&a, 12),
                            w: le32(&a, 16),
                            h: le32(&a, 20),
                            d: le32(&a, 24),
                        },
                        level: le32(&a, 28),
                        offset: u64::from(le32(&a, 32)),
                        stride: le32(&a, 36),
                        layer_stride: le32(&a, 40),
                        to_host: nr == VIRTGPU_TRANSFER_TO_HOST,
                    };
                    dev.transfer(&t)?;
                    Ok(false)
                }
                VIRTGPU_WAIT => {
                    if !inner.st.lock().unwrap().handles.contains_key(&le32(&a, 0)) {
                        return Err(EINVAL);
                    }
                    Ok(false)
                }
                VIRTGPU_EXECBUFFER => {
                    let (len, cmd) = (le32(&a, 4) as usize, le64(&a, 8));
                    if len > MAX_CMD || !len.is_multiple_of(4) {
                        return Err(EINVAL);
                    }
                    let bytes = mem.read_vec(cmd, len).map_err(|_| EFAULT)?;
                    let words: Vec<u32> = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| u32::from_le_bytes(*c))
                        .collect();
                    let ctx = {
                        let mut st = inner.st.lock().unwrap();
                        *st.ctx.get_or_insert_with(|| dev.create_context())
                    };
                    dev.submit(ctx, &words)?;
                    // No fence: fence_fd stays -1.
                    put32(&mut a, 28, u32::MAX);
                    Ok(size >= 32)
                }
                // SET_CLIENT_CAP (none apply to a render node),
                // RESOURCE_CREATE_BLOB (blobs aren't offered), the rest.
                _ => Err(EINVAL),
            }
        })();
        match r {
            Ok(copy_back) => {
                if copy_back && size > 0 && mem.write(arg, &a[..size]).is_err() {
                    return errno(EFAULT);
                }
                0
            }
            Err(e) => errno(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::backend::Recorder;

    fn file() -> DrmFile {
        DrmFile(Arc::new(DrmInner {
            dev: GpuDevice::new(Box::new(Recorder::new(false)), false),
            st: Mutex::new(DrmState {
                next_handle: 1,
                ..DrmState::default()
            }),
        }))
    }

    /// `DRM_IOWR('d', nr, size)`.
    fn iowr(nr: u32, size: u32) -> u32 {
        (3 << 30) | (size << 16) | (u32::from(b'd') << 8) | nr
    }

    #[test]
    fn version_getparam_caps_create_map_transfer() {
        let (k, mut mem, _, _) = crate::kernel::testutil::setup();
        let f = file();
        let base = crate::kernel::testutil::BASE;
        // DRM_IOCTL_VERSION with room for the name.
        let mut v = [0u8; 64];
        v[16..24].copy_from_slice(&32u64.to_le_bytes());
        v[24..32].copy_from_slice(&(base + 0x100).to_le_bytes());
        mem.write(base, &v).unwrap();
        assert_eq!(k.drm_ioctl(&f, iowr(0, 64), base, &mut mem), 0);
        let mut out = [0u8; 64];
        mem.read(base, &mut out).unwrap();
        assert_eq!(le64(&out, 16), 10);
        let mut name = [0u8; 10];
        mem.read(base + 0x100, &mut name).unwrap();
        assert_eq!(&name, b"virtio_gpu");
        // GETPARAM 3D_FEATURES.
        let mut g = [0u8; 16];
        g[0..8].copy_from_slice(&1u64.to_le_bytes());
        g[8..16].copy_from_slice(&(base + 0x200).to_le_bytes());
        mem.write(base, &g).unwrap();
        assert_eq!(k.drm_ioctl(&f, iowr(0x43, 16), base, &mut mem), 0);
        assert_eq!(mem.read_u32(base + 0x200).unwrap(), 1);
        g[0..8].copy_from_slice(&99u64.to_le_bytes());
        mem.write(base, &g).unwrap();
        assert_eq!(k.drm_ioctl(&f, iowr(0x43, 16), base, &mut mem), -22);
        // GET_CAPS set 2.
        let mut c = [0u8; 24];
        c[0..4].copy_from_slice(&2u32.to_le_bytes());
        c[8..16].copy_from_slice(&(base + 0x1000).to_le_bytes());
        c[16..20].copy_from_slice(&(caps::V2_SIZE as u32).to_le_bytes());
        mem.write(base, &c).unwrap();
        assert_eq!(k.drm_ioctl(&f, iowr(0x49, 24), base, &mut mem), 0);
        assert_eq!(mem.read_u32(base + 0x1000).unwrap(), 2); // max_version
        // RESOURCE_CREATE a 4x4 BGRA texture.
        let mut rc = [0u8; 56];
        for (i, v) in [2u32, 1, 2, 4, 4, 1, 1, 0, 0, 0, 0, 0, 64, 16]
            .iter()
            .enumerate()
        {
            rc[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        mem.write(base, &rc).unwrap();
        assert_eq!(k.drm_ioctl(&f, iowr(0x44, 56), base, &mut mem), 0);
        let mut rc = [0u8; 56];
        mem.read(base, &mut rc).unwrap();
        let (bo, res) = (le32(&rc, 40), le32(&rc, 44));
        assert_eq!((bo, res), (1, 1));
        // MAP gives an offset that resolves to the object's pages.
        let mut m = [0u8; 16];
        m[8..12].copy_from_slice(&bo.to_le_bytes());
        mem.write(base, &m).unwrap();
        assert_eq!(k.drm_ioctl(&f, iowr(0x41, 16), base, &mut mem), 0);
        let off = mem.read_u64(base).unwrap();
        let (obj, pgoff) = f.mapping(off).unwrap();
        assert_eq!((obj.size(), pgoff), (PAGE_SIZE, 0));
        // TRANSFER_TO_HOST of the whole texture, then a bad box.
        let mut t = [0u8; 44];
        for (i, v) in [bo, 0, 0, 0, 4, 4, 1, 0, 0, 16, 0].iter().enumerate() {
            t[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        mem.write(base, &t).unwrap();
        assert_eq!(k.drm_ioctl(&f, iowr(0x47, 44), base, &mut mem), 0);
        t[16..20].copy_from_slice(&5u32.to_le_bytes());
        mem.write(base, &t).unwrap();
        assert_eq!(k.drm_ioctl(&f, iowr(0x47, 44), base, &mut mem), -22);
        // GEM_CLOSE, twice.
        mem.write(base, &bo.to_le_bytes()).unwrap();
        assert_eq!(k.drm_ioctl(&f, iowr(0x09, 8), base, &mut mem), 0);
        assert_eq!(k.drm_ioctl(&f, iowr(0x09, 8), base, &mut mem), -22);
        assert!(f.mapping(off).is_none());
    }

    /// The whole path a guest takes: open the node, make a buffer
    /// resource, map it, store through the mapping, and move the bytes to
    /// the host and back into another part of the object.
    #[test]
    fn render_node_through_syscalls() {
        use crate::abi::arch::Sysno;
        use crate::kernel::testutil::{BASE, call, put_str, setup};
        const MAP_SHARED: u64 = 1;
        let (k, mut mem, mut v, mut cx) = setup();
        let path = BASE;
        let args = BASE + 0x100;
        put_str(&mut mem, path, nodes::RENDER_NODE);
        let open = |k: &Kernel, cx: &mut ServiceCtx, mem: &mut GuestMemory, v: &mut _| {
            call(
                k,
                cx,
                mem,
                v,
                Sysno::Openat,
                [(-100i64).cast_unsigned(), path, 2, 0, 0, 0],
            )
        };
        // No GPU attached: the node doesn't exist.
        assert!(open(&k, &mut cx, &mut mem, &mut v) < 0);
        k.attach_gpu(GpuDevice::new(Box::new(Recorder::new(false)), false))
            .unwrap();
        let fd = open(&k, &mut cx, &mut mem, &mut v);
        assert!(fd >= 0, "open: {fd}");
        let fd = fd as u64;
        // A 64-byte buffer resource (target 0).
        let mut rc = [0u8; 56];
        for (i, w) in [0u32, 64, 16, 64, 1, 1, 1, 0, 0, 0, 0, 0, 64, 0]
            .iter()
            .enumerate()
        {
            rc[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        mem.write(args, &rc).unwrap();
        let ioctl =
            |k: &Kernel, cx: &mut ServiceCtx, mem: &mut GuestMemory, v: &mut _, req: u32| {
                call(
                    k,
                    cx,
                    mem,
                    v,
                    Sysno::Ioctl,
                    [fd, u64::from(req), args, 0, 0, 0],
                )
            };
        assert_eq!(ioctl(&k, &mut cx, &mut mem, &mut v, iowr(0x44, 56)), 0);
        let bo = mem.read_u32(args + 40).unwrap();
        let mut m = [0u8; 16];
        m[8..12].copy_from_slice(&bo.to_le_bytes());
        mem.write(args, &m).unwrap();
        assert_eq!(ioctl(&k, &mut cx, &mut mem, &mut v, iowr(0x41, 16)), 0);
        let off = mem.read_u64(args).unwrap();
        let addr = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Mmap,
            [0, 4096, 3, MAP_SHARED, fd, off],
        );
        assert!(addr > 0, "mmap: {addr}");
        let addr = addr as u64;
        mem.write(addr, b"virgl!!!").unwrap();
        // To the host: bytes 0..8; back from the host into bytes 32..40.
        let transfer = |x: u32, offset: u32| {
            let mut t = [0u8; 44];
            for (i, w) in [bo, x, 0, 0, 8, 1, 1, 0, offset, 0, 0].iter().enumerate() {
                t[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
            t
        };
        mem.write(args, &transfer(0, 0)).unwrap();
        assert_eq!(ioctl(&k, &mut cx, &mut mem, &mut v, iowr(0x47, 44)), 0);
        mem.write(args, &transfer(0, 32)).unwrap();
        assert_eq!(ioctl(&k, &mut cx, &mut mem, &mut v, iowr(0x46, 44)), 0);
        assert_eq!(mem.read_vec(addr + 32, 8).unwrap(), b"virgl!!!");
        // fstat says what libdrm checks: a char device, 226:128.
        let st = BASE + 0x400;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fstat,
                [fd, st, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u64(st + 32).unwrap(), nodes::render_rdev());
    }

    #[test]
    fn execbuffer_runs_commands_in_a_context_made_on_demand() {
        let (k, mut mem, _, _) = crate::kernel::testutil::setup();
        let f = file();
        let base = crate::kernel::testutil::BASE;
        // CREATE_SUB_CTX 1; SET_SUB_CTX 1.
        let cmds: Vec<u8> = [
            crate::gpu::virgl::cmd0(29, 0, 1),
            1,
            crate::gpu::virgl::cmd0(28, 0, 1),
            1,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        mem.write(base + 0x800, &cmds).unwrap();
        let mut eb = [0u8; 64];
        eb[4..8].copy_from_slice(&(cmds.len() as u32).to_le_bytes());
        eb[8..16].copy_from_slice(&(base + 0x800).to_le_bytes());
        mem.write(base, &eb).unwrap();
        assert_eq!(k.drm_ioctl(&f, iowr(0x42, 64), base, &mut mem), 0);
        assert!(f.0.st.lock().unwrap().ctx.is_some());
        // CONTEXT_INIT after that is EEXIST, as on Linux.
        assert_eq!(k.drm_ioctl(&f, iowr(0x4b, 16), base, &mut mem), -17);
        // Not a DRM ioctl.
        assert_eq!(k.drm_ioctl(&f, 0x5401, base, &mut mem), err(Errno::ENOTTY));
    }
}
