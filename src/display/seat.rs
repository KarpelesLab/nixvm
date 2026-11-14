//! The seat: `wl_seat`, `wl_pointer`, `wl_keyboard`, focus, the user's input
//! turned into events, and the data device — a clipboard shared by the
//! guest's clients.

use super::Input;
use super::server::{Compositor, Cursor, Data, Iface, Role, SurfRef, time32};
use super::wire::{self, Args, Ev, ProtoError};
use crate::kernel::Kernel;

fn bad_op(obj: u32) -> ProtoError {
    ProtoError::new(obj, wire::ERR_INVALID_METHOD, "bad opcode")
}

/// Pixels of wheel motion per scroll step.
const SCROLL_STEP: f64 = 50.0;

impl Compositor {
    pub(super) fn req_seat(
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
                let id = a.uint()?;
                self.create(cid, id, Iface::Pointer, version, Data::None)?;
                // Already over one of its surfaces: say so.
                if let Some(f) = self.ptr.focus.filter(|f| f.0 == cid)
                    && let Some(p) = self.placed(f)
                {
                    let serial = self.next_serial();
                    let (x, y) = (self.ptr.x - f64::from(p.x), self.ptr.y - f64::from(p.y));
                    self.send(cid, Ev::new(id, 0).uint(serial).uint(f.1).fixed(x).fixed(y));
                    if version >= 5 {
                        self.send(cid, Ev::new(id, 5));
                    }
                }
            }
            1 => {
                let id = a.uint()?;
                self.create(cid, id, Iface::Keyboard, version, Data::None)?;
                let fd = kernel.served_open_file(&self.keymap_path).or_else(|| {
                    let km = super::keymap::keymap();
                    kernel
                        .served_write_file(&self.keymap_path, &km, 0o444)
                        .ok()?;
                    kernel.served_open_file(&self.keymap_path)
                });
                if let (Some(fd), Some(c)) = (fd, self.clients.get_mut(&cid)) {
                    c.send_fd(Ev::new(id, 0).uint(1).uint(self.keymap_len), fd);
                }
                if version >= 4 {
                    self.send(cid, Ev::new(id, 5).int(25).int(600));
                }
                if let Some(f) = self.kb_focus.filter(|f| f.0 == cid) {
                    let serial = self.next_serial();
                    self.keyboard_enter(cid, id, f.1, serial);
                }
            }
            2 => {
                let id = a.uint()?;
                self.create(cid, id, Iface::Touch, version, Data::None)?;
            }
            3 => {
                self.destroy(cid, obj);
            }
            _ => return Err(bad_op(obj)),
        }
        Ok(())
    }

    fn keyboard_enter(&mut self, cid: u32, kb: u32, sid: u32, serial: u32) {
        let keys = wire::u32_array(&self.keys);
        let m = self.mods;
        let Some(c) = self.clients.get_mut(&cid) else {
            return;
        };
        c.send(Ev::new(kb, 1).uint(serial).uint(sid).array(&keys));
        c.send(
            Ev::new(kb, 4)
                .uint(serial)
                .uint(m.depressed)
                .uint(m.latched)
                .uint(m.locked)
                .uint(m.group),
        );
    }

    pub(super) fn req_pointer(
        &mut self,
        cid: u32,
        obj: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        match op {
            0 => {
                let _serial = a.uint()?;
                let sid = a.uint()?;
                let (hx, hy) = (a.int()?, a.int()?);
                if self.ptr.focus.is_none_or(|f| f.0 != cid) {
                    return Ok(());
                }
                if sid == 0 {
                    self.cursor = Cursor::Hidden;
                } else {
                    self.expect(cid, sid, Iface::Surface)?;
                    let Some(surf) = self.surface_mut((cid, sid)) else {
                        return Ok(());
                    };
                    match surf.role {
                        Role::None | Role::Cursor => surf.role = Role::Cursor,
                        _ => return Err(ProtoError::new(obj, 0, "surface has another role")),
                    }
                    self.cursor = Cursor::Surface((cid, sid), hx, hy);
                }
                self.cursor_dirty = true;
            }
            1 => {
                self.destroy(cid, obj);
            }
            _ => return Err(bad_op(obj)),
        }
        Ok(())
    }

    /// The toplevel at the top of the stack: the one with keyboard focus
    /// (or whose popup has it).
    pub(super) fn window_of_focus(&self) -> Option<SurfRef> {
        self.stack.last().copied()
    }

    /// The toplevel surface `s` belongs to (through subsurface and popup
    /// parents).
    fn root_of(&self, s: SurfRef) -> Option<SurfRef> {
        let mut cur = s.1;
        for _ in 0..48 {
            match &self.surface((s.0, cur))?.role {
                Role::Sub(sub) => cur = sub.parent,
                Role::Popup(p) => cur = p.parent,
                Role::Toplevel(_) => return Some((s.0, cur)),
                _ => return None,
            }
        }
        None
    }

    /// Give keyboard focus to where it belongs now: the topmost grabbing
    /// popup, else the top toplevel. Toplevels whose activation changed are
    /// reconfigured.
    pub(super) fn update_focus(&mut self) {
        let grab =
            self.popups.iter().rev().copied().find(
                |&p| matches!(self.surface(p).map(|x| &x.role), Some(Role::Popup(i)) if i.grab),
            );
        let target = grab.or_else(|| self.stack.last().copied());
        if target != self.kb_focus {
            let old = self.kb_focus;
            self.kb_focus = target;
            if let Some(o) = old
                && self.surface(o).is_some()
            {
                let serial = self.next_serial();
                if let Some(c) = self.clients.get_mut(&o.0) {
                    for (kb, _) in c.ids_of(Iface::Keyboard) {
                        c.send(Ev::new(kb, 2).uint(serial).uint(o.1));
                    }
                }
            }
            if let Some(n) = target {
                let serial = self.next_serial();
                let kbs = self
                    .clients
                    .get(&n.0)
                    .map(|c| c.ids_of(Iface::Keyboard))
                    .unwrap_or_default();
                for (kb, _) in kbs {
                    self.keyboard_enter(n.0, kb, n.1, serial);
                }
                if old.is_none_or(|o| o.0 != n.0) {
                    self.send_selection(n.0);
                }
            }
        }
        let top = self.window_of_focus();
        let stale: Vec<SurfRef> = self
            .stack
            .iter()
            .copied()
            .filter(|&s| {
                matches!(self.surface(s).map(|x| &x.role),
                    Some(Role::Toplevel(t)) if t.configured && t.activated != (Some(s) == top))
            })
            .collect();
        for s in stale {
            self.configure_toplevel(s);
        }
    }

    /// Re-find the surface under the pointer (the scene changed under it).
    pub(super) fn update_pointer(&mut self) {
        if !self.ptr.inside || !self.ptr.buttons.is_empty() {
            return;
        }
        let hit = self.hit(self.ptr.x, self.ptr.y);
        self.set_pointer_focus(hit);
    }

    fn pointers(&self, cid: u32) -> Vec<(u32, u32)> {
        self.clients
            .get(&cid)
            .map(|c| c.ids_of(Iface::Pointer))
            .unwrap_or_default()
    }

    fn set_pointer_focus(&mut self, hit: Option<(SurfRef, f64, f64)>) {
        let new = hit.map(|h| h.0);
        if new == self.ptr.focus {
            return;
        }
        if let Some(o) = self.ptr.focus.take()
            && self.surface(o).is_some()
        {
            let serial = self.next_serial();
            for (p, v) in self.pointers(o.0) {
                self.send(o.0, Ev::new(p, 1).uint(serial).uint(o.1));
                if v >= 5 {
                    self.send(o.0, Ev::new(p, 5));
                }
            }
        }
        self.cursor = Cursor::Default;
        self.cursor_dirty = true;
        if let Some((s, x, y)) = hit {
            self.ptr.focus = Some(s);
            let serial = self.next_serial();
            for (p, v) in self.pointers(s.0) {
                self.send(s.0, Ev::new(p, 0).uint(serial).uint(s.1).fixed(x).fixed(y));
                if v >= 5 {
                    self.send(s.0, Ev::new(p, 5));
                }
            }
        }
    }

    /// Deliver one input event.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    pub(super) fn input(&mut self, ev: Input) {
        match ev {
            Input::PointerMove { x, y } => {
                self.ptr.x = x.clamp(0.0, f64::from(self.width.max(1) - 1));
                self.ptr.y = y.clamp(0.0, f64::from(self.height.max(1) - 1));
                self.ptr.inside = true;
                self.cursor_dirty = true;
                let target = if self.ptr.buttons.is_empty() {
                    let hit = self.hit(self.ptr.x, self.ptr.y);
                    self.set_pointer_focus(hit);
                    hit
                } else {
                    // Held buttons: the surface that got the press keeps it.
                    self.ptr.focus.and_then(|f| {
                        let p = self.placed(f)?;
                        Some((f, self.ptr.x - f64::from(p.x), self.ptr.y - f64::from(p.y)))
                    })
                };
                if let Some((s, lx, ly)) = target {
                    let t = time32();
                    for (p, v) in self.pointers(s.0) {
                        self.send(s.0, Ev::new(p, 2).uint(t).fixed(lx).fixed(ly));
                        if v >= 5 {
                            self.send(s.0, Ev::new(p, 5));
                        }
                    }
                }
            }
            Input::PointerButton { button, pressed } => {
                if pressed {
                    // A click outside a grabbing popup dismisses it.
                    let grabbing: Vec<SurfRef> = self
                        .popups
                        .iter()
                        .rev()
                        .copied()
                        .filter(|&p| {
                            matches!(self.surface(p).map(|x| &x.role),
                                Some(Role::Popup(i)) if i.grab)
                        })
                        .collect();
                    if !grabbing.is_empty() {
                        let inside = self.ptr.focus.is_some_and(|f| {
                            self.popups
                                .iter()
                                .any(|&p| p.0 == f.0 && self.is_within(f, p))
                        });
                        if !inside {
                            let all: Vec<SurfRef> = self.popups.iter().rev().copied().collect();
                            for p in all {
                                self.popup_done(p);
                            }
                            self.update_focus();
                            return;
                        }
                    }
                    if let Some(root) = self.ptr.focus.and_then(|f| self.root_of(f))
                        && self.stack.last() != Some(&root)
                        && let Some(i) = self.stack.iter().position(|&s| s == root)
                    {
                        let r = self.stack.remove(i);
                        self.stack.push(r);
                        self.scene_dirty = true;
                        self.update_focus();
                    }
                    if !self.ptr.buttons.contains(&button) {
                        self.ptr.buttons.push(button);
                    }
                } else {
                    self.ptr.buttons.retain(|&b| b != button);
                }
                if let Some(f) = self.ptr.focus {
                    let serial = self.next_serial();
                    let t = time32();
                    for (p, v) in self.pointers(f.0) {
                        self.send(
                            f.0,
                            Ev::new(p, 3)
                                .uint(serial)
                                .uint(t)
                                .uint(button)
                                .uint(u32::from(pressed)),
                        );
                        if v >= 5 {
                            self.send(f.0, Ev::new(p, 5));
                        }
                    }
                }
                if !pressed {
                    self.update_pointer();
                }
            }
            Input::Scroll { dx, dy } => {
                let Some(f) = self.ptr.focus else {
                    return;
                };
                self.ptr.scroll_acc.0 += dx;
                self.ptr.scroll_acc.1 += dy;
                let steps_x = (self.ptr.scroll_acc.0 / SCROLL_STEP).trunc();
                let steps_y = (self.ptr.scroll_acc.1 / SCROLL_STEP).trunc();
                self.ptr.scroll_acc.0 -= steps_x * SCROLL_STEP;
                self.ptr.scroll_acc.1 -= steps_y * SCROLL_STEP;
                if steps_x == 0.0 && steps_y == 0.0 {
                    return;
                }
                let t = time32();
                for (p, v) in self.pointers(f.0) {
                    if v >= 5 {
                        self.send(f.0, Ev::new(p, 6).uint(0)); // axis_source: wheel
                    }
                    for (axis, steps) in [(0u32, steps_y), (1, steps_x)] {
                        if steps == 0.0 {
                            continue;
                        }
                        if v >= 5 {
                            self.send(f.0, Ev::new(p, 8).uint(axis).int(steps as i32));
                        }
                        self.send(f.0, Ev::new(p, 4).uint(t).uint(axis).fixed(steps * 15.0));
                    }
                    if v >= 5 {
                        self.send(f.0, Ev::new(p, 5));
                    }
                }
            }
            Input::PointerLeave => {
                self.set_pointer_focus(None);
                self.ptr.buttons.clear();
                self.ptr.inside = false;
                self.cursor_dirty = true;
            }
            Input::Key { code, pressed } => {
                if pressed {
                    if self.keys.contains(&code) {
                        return; // the embedder's auto-repeat: clients repeat themselves
                    }
                    self.keys.push(code);
                } else {
                    if !self.keys.contains(&code) {
                        return;
                    }
                    self.keys.retain(|&k| k != code);
                }
                let keys = self.keys.clone();
                let changed = self.mods.key(code, pressed, &keys);
                let Some(f) = self.kb_focus else {
                    return;
                };
                let serial = self.next_serial();
                let t = time32();
                let m = self.mods;
                if let Some(c) = self.clients.get_mut(&f.0) {
                    for (kb, _) in c.ids_of(Iface::Keyboard) {
                        c.send(
                            Ev::new(kb, 3)
                                .uint(serial)
                                .uint(t)
                                .uint(code)
                                .uint(u32::from(pressed)),
                        );
                        if changed {
                            c.send(
                                Ev::new(kb, 4)
                                    .uint(serial)
                                    .uint(m.depressed)
                                    .uint(m.latched)
                                    .uint(m.locked)
                                    .uint(m.group),
                            );
                        }
                    }
                }
            }
            Input::Resize { width, height } => {
                let (width, height) = (width.clamp(1, 8192), height.clamp(1, 8192));
                if (width, height) == (self.width, self.height) {
                    return;
                }
                self.width = width;
                self.height = height;
                self.ptr.x = self.ptr.x.min(f64::from(width - 1));
                self.ptr.y = self.ptr.y.min(f64::from(height - 1));
                self.scene_dirty = true;
                self.outputs_changed();
                let tops: Vec<SurfRef> = self
                    .clients
                    .iter()
                    .flat_map(|(&cid, c)| {
                        c.surfaces.iter().filter_map(move |(&sid, s)| {
                            matches!(&s.role, Role::Toplevel(t) if t.configured)
                                .then_some((cid, sid))
                        })
                    })
                    .collect();
                for s in tops {
                    self.configure_toplevel(s);
                }
            }
        }
    }

    /// Whether `s` is popup `p` or one of its subsurfaces.
    fn is_within(&self, s: SurfRef, p: SurfRef) -> bool {
        let mut cur = s.1;
        for _ in 0..32 {
            if cur == p.1 {
                return true;
            }
            match self.surface((s.0, cur)).map(|x| &x.role) {
                Some(Role::Sub(sub)) => cur = sub.parent,
                _ => return false,
            }
        }
        false
    }

    pub(super) fn req_data_device_manager(
        &mut self,
        cid: u32,
        version: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        let id = a.uint()?;
        match op {
            0 => self.create(cid, id, Iface::DataSource, version, Data::Mimes(Vec::new())),
            1 => {
                a.uint()?;
                self.create(cid, id, Iface::DataDevice, version, Data::None)?;
                if self.kb_focus.is_some_and(|f| f.0 == cid) {
                    self.send_selection(cid);
                }
                Ok(())
            }
            _ => Err(bad_op(id)),
        }
    }

    pub(super) fn req_data_source(
        &mut self,
        cid: u32,
        obj: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        match op {
            0 => {
                let mime = a.string()?;
                if let Some(o) = self.object_mut(cid, obj)
                    && let Data::Mimes(m) = &mut o.data
                    && m.len() < 64
                {
                    m.push(mime);
                }
            }
            1 => {
                self.destroy(cid, obj);
                if self.selection == Some((cid, obj)) {
                    self.selection = None;
                    if let Some(f) = self.kb_focus {
                        self.send_selection(f.0);
                    }
                }
            }
            2 => {
                a.uint()?;
            }
            _ => return Err(bad_op(obj)),
        }
        Ok(())
    }

    pub(super) fn req_data_device(
        &mut self,
        cid: u32,
        obj: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        match op {
            0 => {
                // Drag and drop isn't supported: the drag is cancelled.
                let source = a.uint()?;
                a.uint()?;
                a.uint()?;
                a.uint()?;
                if source != 0 && self.object(cid, source).is_some() {
                    self.send(cid, Ev::new(source, 2));
                }
            }
            1 => {
                let source = a.uint()?;
                a.uint()?;
                if source != 0 {
                    self.expect(cid, source, Iface::DataSource)?;
                }
                let new = (source != 0).then_some((cid, source));
                if let Some(old) = self.selection
                    && Some(old) != new
                {
                    self.send(old.0, Ev::new(old.1, 2)); // cancelled
                }
                self.selection = new;
                if let Some(f) = self.kb_focus {
                    self.send_selection(f.0);
                }
            }
            2 => {
                self.destroy(cid, obj);
            }
            _ => return Err(bad_op(obj)),
        }
        Ok(())
    }

    pub(super) fn req_data_offer(
        &mut self,
        kernel: &Kernel,
        cid: u32,
        obj: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        match op {
            0 => {
                a.uint()?;
                a.opt_string()?;
            }
            1 => {
                let mime = a.string()?;
                let fd = a.fd()?;
                let src = match self.object(cid, obj).map(|o| &o.data) {
                    Some(Data::Offer(s)) => Some(*s),
                    _ => None,
                };
                // The source's client writes the data straight into the
                // receiver's pipe.
                match src.filter(|&s| self.selection == Some(s)) {
                    Some(s) if self.object(s.0, s.1).is_some() => {
                        if let Some(c) = self.clients.get_mut(&s.0) {
                            c.send_fd(Ev::new(s.1, 1).string(&mime), fd);
                        }
                    }
                    _ => kernel.served_drop_fd(fd),
                }
            }
            2 => {
                self.destroy(cid, obj);
            }
            3 => {}
            4 => {
                a.uint()?;
                a.uint()?;
            }
            _ => return Err(bad_op(obj)),
        }
        Ok(())
    }

    /// Offer the current selection to client `cid`'s data devices.
    pub(super) fn send_selection(&mut self, cid: u32) {
        let mimes = match self.selection.and_then(|s| self.object(s.0, s.1)) {
            Some(o) => match &o.data {
                Data::Mimes(m) => Some(m.clone()),
                _ => None,
            },
            None => None,
        };
        let sel = self.selection;
        let Some(c) = self.clients.get_mut(&cid) else {
            return;
        };
        for (dev, version) in c.ids_of(Iface::DataDevice) {
            match (&mimes, sel) {
                (Some(m), Some(src)) => {
                    let offer = c.alloc_id();
                    c.objects.insert(
                        offer,
                        super::server::Object {
                            iface: Iface::DataOffer,
                            version,
                            data: Data::Offer(src),
                        },
                    );
                    c.send(Ev::new(dev, 0).uint(offer));
                    for mime in m {
                        c.send(Ev::new(offer, 0).string(mime));
                    }
                    c.send(Ev::new(dev, 5).uint(offer));
                }
                _ => c.send(Ev::new(dev, 5).uint(0)),
            }
        }
    }
}
