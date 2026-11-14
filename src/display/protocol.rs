//! The core interfaces: `wl_compositor`, `wl_surface`, `wl_region`,
//! `wl_subcompositor`/`wl_subsurface`, `wl_shm` and its pools and buffers,
//! `wl_output` (+ `zxdg_output_v1`), `wp_viewporter` and the data device
//! (the clipboard between guest clients).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::render::{self, Image};
use super::server::{Buffer, Compositor, Data, Iface, Pending, Pool, Rect, Role, SubInfo, Surface};
use super::wire::{self, Args, Ev, ProtoError};
use crate::kernel::Kernel;

fn bad_op(obj: u32) -> ProtoError {
    ProtoError::new(obj, wire::ERR_INVALID_METHOD, "bad opcode")
}

impl Compositor {
    pub(super) fn req_compositor(
        &mut self,
        cid: u32,
        version: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        let id = a.uint()?;
        match op {
            0 => {
                self.create(cid, id, Iface::Surface, version, Data::None)?;
                if let Some(c) = self.clients.get_mut(&cid) {
                    c.surfaces.insert(
                        id,
                        Surface {
                            scale: 1,
                            stack: vec![id],
                            ..Surface::default()
                        },
                    );
                    if version >= 6 {
                        c.send(Ev::new(id, 2).int(1)); // preferred_buffer_scale
                        c.send(Ev::new(id, 3).uint(0)); // preferred_buffer_transform
                    }
                }
                Ok(())
            }
            1 => self.create(cid, id, Iface::Region, 1, Data::None),
            _ => Err(bad_op(id)),
        }
    }

    pub(super) fn req_surface(
        &mut self,
        kernel: &Kernel,
        cid: u32,
        sid: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        let s = (cid, sid);
        match op {
            0 => {
                self.destroy_surface(cid, sid);
                self.destroy(cid, sid);
            }
            1 => {
                let buffer = a.uint()?;
                let (x, y) = (a.int()?, a.int()?);
                if buffer != 0 {
                    self.expect(cid, buffer, Iface::Buffer)?;
                }
                if let Some(surf) = self.surface_mut(s) {
                    surf.pending.attach = Some((buffer != 0).then_some(buffer));
                    surf.pending.offset = (x, y);
                }
            }
            2 | 9 => {
                let r = (a.int()?, a.int()?, a.int()?, a.int()?);
                if let Some(surf) = self.surface_mut(s) {
                    if op == 2 {
                        surf.pending.damage.push(r);
                    } else {
                        surf.pending.damage_buffer.push(r);
                    }
                }
            }
            3 => {
                let id = a.uint()?;
                self.create(cid, id, Iface::Callback, 1, Data::None)?;
                if let Some(surf) = self.surface_mut(s) {
                    surf.pending.frames.push(id);
                }
            }
            4 | 5 | 7 => {
                a.uint()?;
            }
            6 => self.commit(kernel, cid, sid)?,
            8 => {
                let scale = a.int()?;
                if scale < 1 {
                    return Err(ProtoError::new(sid, 0, "invalid scale"));
                }
                if let Some(surf) = self.surface_mut(s) {
                    surf.pending.scale = Some(scale);
                }
            }
            10 => {
                let (x, y) = (a.int()?, a.int()?);
                if let Some(surf) = self.surface_mut(s) {
                    surf.pending.offset = (x, y);
                }
            }
            _ => return Err(bad_op(sid)),
        }
        Ok(())
    }

    /// `wl_surface.destroy`: unmap it and forget it.
    fn destroy_surface(&mut self, cid: u32, sid: u32) {
        let s = (cid, sid);
        self.unmap(s);
        if let Some(Surface {
            role: Role::Sub(sub),
            ..
        }) = self.surface(s)
        {
            let parent = sub.parent;
            if let Some(p) = self.surface_mut((cid, parent)) {
                p.stack.retain(|&x| x != sid);
            }
        }
        if let Some(c) = self.clients.get_mut(&cid) {
            c.surfaces.remove(&sid);
        }
        if matches!(self.cursor, super::server::Cursor::Surface(c, ..) if c == s) {
            self.cursor = super::server::Cursor::Hidden;
            self.cursor_dirty = true;
        }
        if self.ptr.focus == Some(s) {
            self.ptr.focus = None;
            self.ptr.buttons.clear();
        }
        self.update_pointer();
    }

    /// Take surface `s` off the display (its role object is gone, or it
    /// committed a null buffer).
    pub(super) fn unmap(&mut self, s: super::server::SurfRef) {
        let before = self.stack.len() + self.popups.len();
        self.stack.retain(|&x| x != s);
        self.popups.retain(|&x| x != s);
        if let Some(surf) = self.surface_mut(s) {
            match &mut surf.role {
                Role::Toplevel(t) => t.mapped = false,
                Role::Popup(p) => p.mapped = false,
                _ => {}
            }
        }
        if before != self.stack.len() + self.popups.len() {
            self.scene_dirty = true;
            if self.kb_focus == Some(s) {
                self.kb_focus = None;
            }
            self.update_focus();
            self.update_pointer();
        }
    }

    /// Whether surface `s` is a synchronized subsurface, or the child of one.
    fn synchronized(&self, s: super::server::SurfRef) -> bool {
        let mut cur = s.1;
        for _ in 0..32 {
            match self.surface((s.0, cur)).map(|x| &x.role) {
                Some(Role::Sub(sub)) => {
                    if sub.sync {
                        return true;
                    }
                    cur = sub.parent;
                }
                _ => return false,
            }
        }
        false
    }

    fn commit(&mut self, kernel: &Kernel, cid: u32, sid: u32) -> Result<(), ProtoError> {
        let s = (cid, sid);
        let sync = self.synchronized(s);
        let Some(surf) = self.surface_mut(s) else {
            return Ok(());
        };
        let pending = std::mem::take(&mut surf.pending);
        if sync {
            match &mut surf.cached {
                Some(c) => c.merge(pending),
                None => surf.cached = Some(pending),
            }
            return Ok(());
        }
        let state = match surf.cached.take() {
            Some(mut c) => {
                c.merge(pending);
                c
            }
            None => pending,
        };
        self.apply(kernel, cid, sid, state, 0)
    }

    /// Make `state` surface `sid`'s current state.
    #[allow(clippy::too_many_lines)]
    fn apply(
        &mut self,
        kernel: &Kernel,
        cid: u32,
        sid: u32,
        state: Pending,
        depth: u32,
    ) -> Result<(), ProtoError> {
        let s = (cid, sid);
        // The new buffer's contents, copied now so it can be released.
        let mut released = None;
        let buffer = match state.attach {
            Some(Some(b)) => match self.object(cid, b).map(|o| &o.data) {
                Some(Data::Buffer(buf)) => Some(Some((b, buf.clone()))),
                // Destroyed between attach and commit: nothing to show.
                _ => Some(None),
            },
            Some(None) => Some(None),
            None => None,
        };
        let Some(surf) = self.surface_mut(s) else {
            return Ok(());
        };
        if let Some(scale) = state.scale {
            surf.scale = scale;
        }
        if let Some(v) = state.vp_src {
            surf.vp_src = v;
        }
        if let Some(v) = state.vp_dst {
            surf.vp_dst = v;
        }
        if let Some(g) = state.geometry {
            surf.geometry = Some(g);
        }
        match buffer {
            Some(Some((id, buf))) => {
                let damage = damage_box(surf, &state, &buf);
                upload(kernel, &buf, &mut surf.image, damage);
                surf.has_buffer = true;
                released = Some(id);
            }
            Some(None) => {
                surf.has_buffer = false;
                surf.image = Image::default();
            }
            None => {}
        }
        if matches!(surf.role, Role::Cursor) {
            let (dx, dy) = state.offset;
            if let super::server::Cursor::Surface(cs, hx, hy) = self.cursor
                && cs == s
            {
                self.cursor = super::server::Cursor::Surface(cs, hx - dx, hy - dy);
            }
        }
        let children: Vec<u32> = self
            .surface(s)
            .map(|x| x.stack.iter().copied().filter(|&c| c != sid).collect())
            .unwrap_or_default();
        if let Some(c) = self.clients.get_mut(&cid) {
            let now = super::server::now_ms();
            for f in state.frames {
                c.frames.push((f, self.presented, now));
            }
            if let Some(b) = released {
                c.send(Ev::new(b, 0)); // wl_buffer.release
            }
        }
        // The children's positions, and their cached state, go with ours.
        for child in children {
            let cached = match self.surface_mut((cid, child)) {
                Some(cs) => {
                    if let Role::Sub(sub) = &mut cs.role
                        && let Some(p) = sub.pending_pos.take()
                    {
                        sub.pos = p;
                    }
                    cs.cached.take()
                }
                None => None,
            };
            if let Some(st) = cached
                && depth < 32
            {
                self.apply(kernel, cid, child, st, depth + 1)?;
            }
        }
        self.after_commit(cid, sid);
        self.scene_dirty = true;
        Ok(())
    }

    pub(super) fn req_subcompositor(
        &mut self,
        cid: u32,
        obj: u32,
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
                let sid = a.uint()?;
                let parent = a.uint()?;
                self.expect(cid, sid, Iface::Surface)?;
                self.expect(cid, parent, Iface::Surface)?;
                if sid == parent
                    || !matches!(self.surface((cid, sid)).map(|s| &s.role), Some(Role::None))
                {
                    return Err(ProtoError::new(obj, 0, "bad subsurface"));
                }
                self.create(cid, id, Iface::Subsurface, 1, Data::Surface(sid))?;
                if let Some(surf) = self.surface_mut((cid, sid)) {
                    surf.role = Role::Sub(SubInfo {
                        parent,
                        pos: (0, 0),
                        pending_pos: None,
                        sync: true,
                    });
                }
                if let Some(p) = self.surface_mut((cid, parent)) {
                    p.stack.push(sid);
                }
                Ok(())
            }
            _ => Err(bad_op(obj)),
        }
    }

    pub(super) fn req_subsurface(
        &mut self,
        cid: u32,
        obj: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        let sid = self.target(cid, obj)?;
        let parent = match self.surface((cid, sid)).map(|s| &s.role) {
            Some(Role::Sub(sub)) => sub.parent,
            _ => 0,
        };
        match op {
            0 => {
                if let Some(p) = self.surface_mut((cid, parent)) {
                    p.stack.retain(|&x| x != sid);
                }
                if let Some(surf) = self.surface_mut((cid, sid)) {
                    surf.role = Role::None;
                }
                self.destroy(cid, obj);
                self.scene_dirty = true;
            }
            1 => {
                let p = (a.int()?, a.int()?);
                if let Some(Role::Sub(sub)) = self.surface_mut((cid, sid)).map(|s| &mut s.role) {
                    sub.pending_pos = Some(p);
                }
            }
            2 | 3 => {
                let sibling = a.uint()?;
                if let Some(p) = self.surface_mut((cid, parent))
                    && let Some(i) = p.stack.iter().position(|&x| x == sid)
                {
                    let me = p.stack.remove(i);
                    let Some(j) = p.stack.iter().position(|&x| x == sibling) else {
                        p.stack.insert(i, me);
                        return Err(ProtoError::new(obj, 0, "not a sibling"));
                    };
                    p.stack.insert(if op == 2 { j + 1 } else { j }, me);
                }
                self.scene_dirty = true;
            }
            4 | 5 => {
                if let Some(Role::Sub(sub)) = self.surface_mut((cid, sid)).map(|s| &mut s.role) {
                    sub.sync = op == 4;
                }
            }
            _ => return Err(bad_op(obj)),
        }
        Ok(())
    }

    pub(super) fn req_shm(
        &mut self,
        kernel: &Kernel,
        cid: u32,
        obj: u32,
        _version: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        match op {
            0 => {
                let id = a.uint()?;
                let fd = a.fd()?;
                let size = a.int()?;
                let Some(_) = kernel.served_file_size(&fd) else {
                    kernel.served_drop_fd(fd);
                    return Err(ProtoError::new(obj, 2, "not a shared-memory file"));
                };
                if size <= 0 {
                    kernel.served_drop_fd(fd);
                    return Err(ProtoError::new(obj, 1, "invalid pool size"));
                }
                let pool = Arc::new(Pool {
                    fd,
                    size: u64::from(size.unsigned_abs()).into(),
                });
                self.create(cid, id, Iface::ShmPool, 1, Data::Pool(pool))
            }
            1 => {
                self.destroy(cid, obj);
                Ok(())
            }
            _ => Err(bad_op(obj)),
        }
    }

    pub(super) fn req_pool(
        &mut self,
        _kernel: &Kernel,
        cid: u32,
        obj: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        let pool = match self.object(cid, obj).map(|o| &o.data) {
            Some(Data::Pool(p)) => Arc::clone(p),
            _ => return Err(bad_op(obj)),
        };
        match op {
            0 => {
                let id = a.uint()?;
                let (offset, w, h, stride) = (a.int()?, a.int()?, a.int()?, a.int()?);
                let format = a.uint()?;
                let Some(bpp) = render::bytes_per_pixel(format) else {
                    return Err(ProtoError::new(obj, 0, format!("format {format:#x}")));
                };
                let ok = offset >= 0
                    && w > 0
                    && h > 0
                    && stride > 0
                    && (stride as usize) >= (w as usize) * bpp
                    && i64::from(offset) + i64::from(stride) * i64::from(h)
                        <= i64::try_from(pool.size.load(Ordering::Relaxed)).unwrap_or(0);
                if !ok {
                    return Err(ProtoError::new(obj, 1, "buffer outside the pool"));
                }
                let buf = Buffer {
                    pool,
                    offset: u64::from(offset.unsigned_abs()),
                    width: w.unsigned_abs(),
                    height: h.unsigned_abs(),
                    stride: stride.unsigned_abs(),
                    format,
                };
                self.create(cid, id, Iface::Buffer, 1, Data::Buffer(buf))
            }
            1 => {
                self.destroy(cid, obj);
                Ok(())
            }
            2 => {
                let size = a.int()?;
                let size = u64::from(size.max(0).unsigned_abs());
                if size < pool.size.load(Ordering::Relaxed) {
                    return Err(ProtoError::new(obj, 1, "pools only grow"));
                }
                pool.size.store(size, Ordering::Relaxed);
                Ok(())
            }
            _ => Err(bad_op(obj)),
        }
    }

    /// Describe the output to `wl_output` `id`.
    pub(super) fn send_output(&mut self, cid: u32, id: u32, version: u32) {
        let (w, h) = (self.width, self.height);
        #[allow(clippy::cast_possible_wrap)]
        let (wi, hi) = (w as i32, h as i32);
        let Some(c) = self.clients.get_mut(&cid) else {
            return;
        };
        // 96 dpi.
        c.send(
            Ev::new(id, 0)
                .int(0)
                .int(0)
                .int(wi * 254 / 960)
                .int(hi * 254 / 960)
                .int(0)
                .string("nixvm")
                .string("display")
                .int(0),
        );
        c.send(Ev::new(id, 1).uint(3).int(wi).int(hi).int(60_000));
        if version >= 2 {
            c.send(Ev::new(id, 3).int(1));
        }
        if version >= 4 {
            c.send(Ev::new(id, 4).string("NIXVM-1"));
            c.send(Ev::new(id, 5).string("nixvm display"));
        }
        if version >= 2 {
            c.send(Ev::new(id, 2));
        }
    }

    /// The display changed size: tell every output.
    pub(super) fn outputs_changed(&mut self) {
        #[allow(clippy::cast_possible_wrap)]
        let (w, h) = (self.width as i32, self.height as i32);
        let ids: Vec<u32> = self.clients.keys().copied().collect();
        for cid in ids {
            let Some(c) = self.clients.get_mut(&cid) else {
                continue;
            };
            let xdg = c.ids_of(Iface::XdgOutput);
            for (id, _) in &xdg {
                c.send(Ev::new(*id, 1).int(w).int(h));
            }
            for (id, v) in c.ids_of(Iface::Output) {
                c.send(Ev::new(id, 1).uint(3).int(w).int(h).int(60_000));
                if v >= 2 {
                    c.send(Ev::new(id, 2));
                }
            }
            for (id, v) in &xdg {
                if *v < 3 {
                    c.send(Ev::new(*id, 2));
                }
            }
        }
    }

    pub(super) fn req_xdg_output_manager(
        &mut self,
        cid: u32,
        obj: u32,
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
                let output = a.uint()?;
                self.expect(cid, output, Iface::Output)?;
                let version = self.object(cid, obj).map_or(1, |o| o.version);
                let out_version = self.object(cid, output).map_or(1, |o| o.version);
                self.create(cid, id, Iface::XdgOutput, version, Data::None)?;
                #[allow(clippy::cast_possible_wrap)]
                let (w, h) = (self.width as i32, self.height as i32);
                let Some(c) = self.clients.get_mut(&cid) else {
                    return Ok(());
                };
                c.send(Ev::new(id, 0).int(0).int(0));
                c.send(Ev::new(id, 1).int(w).int(h));
                if version >= 2 {
                    c.send(Ev::new(id, 3).string("NIXVM-1"));
                    c.send(Ev::new(id, 4).string("nixvm display"));
                }
                if version >= 3 {
                    if out_version >= 2 {
                        c.send(Ev::new(output, 2));
                    }
                } else {
                    c.send(Ev::new(id, 2));
                }
                Ok(())
            }
            _ => Err(bad_op(obj)),
        }
    }

    pub(super) fn req_viewporter(
        &mut self,
        cid: u32,
        obj: u32,
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
                let sid = a.uint()?;
                self.expect(cid, sid, Iface::Surface)?;
                if self
                    .surface((cid, sid))
                    .is_some_and(|s| s.viewport.is_some())
                {
                    return Err(ProtoError::new(obj, 0, "viewport exists"));
                }
                self.create(cid, id, Iface::Viewport, 1, Data::Surface(sid))?;
                if let Some(s) = self.surface_mut((cid, sid)) {
                    s.viewport = Some(id);
                }
                Ok(())
            }
            _ => Err(bad_op(obj)),
        }
    }

    pub(super) fn req_viewport(
        &mut self,
        cid: u32,
        obj: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        let sid = self.target(cid, obj)?;
        match op {
            0 => {
                if let Some(s) = self.surface_mut((cid, sid)) {
                    s.viewport = None;
                    s.pending.vp_src = Some(None);
                    s.pending.vp_dst = Some(None);
                }
                self.destroy(cid, obj);
            }
            1 => {
                // Raw 24.8 fixed point: all -1 (-256) unsets it.
                let raw = [a.int()?, a.int()?, a.int()?, a.int()?];
                let unset = raw == [-256; 4];
                if !unset && (raw[0] < 0 || raw[1] < 0 || raw[2] <= 0 || raw[3] <= 0) {
                    return Err(ProtoError::new(obj, 0, "bad source"));
                }
                let f = |v: i32| f64::from(v) / 256.0;
                let r = (f(raw[0]), f(raw[1]), f(raw[2]), f(raw[3]));
                if let Some(s) = self.surface_mut((cid, sid)) {
                    s.pending.vp_src = Some((!unset).then_some(r));
                }
            }
            2 => {
                let (w, h) = (a.int()?, a.int()?);
                let unset = w == -1 && h == -1;
                if !unset && (w <= 0 || h <= 0) {
                    return Err(ProtoError::new(obj, 0, "bad destination"));
                }
                if let Some(s) = self.surface_mut((cid, sid)) {
                    s.pending.vp_dst = Some((!unset).then_some((w, h)));
                }
            }
            _ => return Err(bad_op(obj)),
        }
        Ok(())
    }
}

/// The bounding box of the damage `state` carries, in buffer pixels; `None`
/// for all of it.
#[allow(clippy::cast_possible_wrap)]
fn damage_box(surf: &Surface, state: &Pending, buf: &Buffer) -> Option<Rect> {
    if surf.image.width != buf.width
        || surf.image.height != buf.height
        || surf.vp_src.is_some()
        || surf.vp_dst.is_some()
        || (state.damage.is_empty() && state.damage_buffer.is_empty())
    {
        return None;
    }
    let s = surf.scale.max(1);
    let mut x0 = i32::MAX;
    let mut y0 = i32::MAX;
    let mut x1 = i32::MIN;
    let mut y1 = i32::MIN;
    let scaled = state.damage.iter().map(|&(x, y, w, h)| {
        (
            x.saturating_mul(s),
            y.saturating_mul(s),
            w.saturating_mul(s),
            h.saturating_mul(s),
        )
    });
    for (x, y, w, h) in scaled.chain(state.damage_buffer.iter().copied()) {
        if w <= 0 || h <= 0 {
            continue;
        }
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x.saturating_add(w));
        y1 = y1.max(y.saturating_add(h));
    }
    let (bw, bh) = (buf.width as i32, buf.height as i32);
    let (x0, y0, x1, y1) = (x0.max(0), y0.max(0), x1.min(bw), y1.min(bh));
    if x0 >= x1 || y0 >= y1 {
        return Some((0, 0, 0, 0));
    }
    Some((x0, y0, x1 - x0, y1 - y0))
}

/// Copy `rect` (all of it if `None`) of `buf` into `img`.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn upload(kernel: &Kernel, buf: &Buffer, img: &mut Image, rect: Option<Rect>) {
    let (w, h) = (buf.width as usize, buf.height as usize);
    if img.width != buf.width || img.height != buf.height {
        img.width = buf.width;
        img.height = buf.height;
        img.pixels = vec![0; w * h];
    }
    img.opaque = render::is_opaque(buf.format);
    let (x0, y0, rw, rh) = match rect {
        Some((x, y, rw, rh)) => (x as usize, y as usize, rw as usize, rh as usize),
        None => (0, 0, w, h),
    };
    if rw == 0 || rh == 0 {
        return;
    }
    let bpp = render::bytes_per_pixel(buf.format).unwrap_or(4);
    let stride = buf.stride as usize;
    // One read covers the damaged rows.
    let start = buf.offset + (y0 * stride + x0 * bpp) as u64;
    let len = (rh - 1) * stride + rw * bpp;
    let mut bytes = vec![0u8; len];
    kernel.served_file_read(&buf.pool.fd, start, &mut bytes);
    for row in 0..rh {
        let src = &bytes[row * stride..row * stride + rw * bpp];
        let d0 = (y0 + row) * w + x0;
        render::convert_row(buf.format, src, &mut img.pixels[d0..d0 + rw]);
    }
}
