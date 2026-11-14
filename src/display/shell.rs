//! xdg-shell (`xdg_wm_base`, `xdg_positioner`, `xdg_surface`,
//! `xdg_toplevel`, `xdg_popup`) and `zxdg_decoration_manager_v1`.
//!
//! The window policy is a kiosk's: every toplevel is maximized to fill the
//! display, the newest on top and focused; a toplevel with a parent (a
//! dialog) keeps its own size, centered. Decorations are "server-side", and
//! the server draws none: windows are edge to edge.

use super::server::{Compositor, Data, Iface, PopupInfo, Positioner, Rect, Role, SurfRef, TopInfo};
use super::wire::{self, Args, Ev, ProtoError};

fn bad_op(obj: u32) -> ProtoError {
    ProtoError::new(obj, wire::ERR_INVALID_METHOD, "bad opcode")
}

/// `xdg_positioner` anchors and gravities: which edges they touch.
fn horiz(e: u32) -> i32 {
    match e {
        3 | 5 | 6 => -1, // left, top_left, bottom_left
        4 | 7 | 8 => 1,  // right, top_right, bottom_right
        _ => 0,
    }
}

fn vert(e: u32) -> i32 {
    match e {
        1 | 5 | 7 => -1, // top, top_left, top_right
        2 | 6 | 8 => 1,  // bottom, bottom_left, bottom_right
        _ => 0,
    }
}

/// The anchor/gravity `e` mirrored horizontally or vertically.
fn flip(e: u32, horizontal: bool) -> u32 {
    let (h, v) = (horiz(e), vert(e));
    let (h, v) = if horizontal { (-h, v) } else { (h, -v) };
    match (h, v) {
        (0, 0) => 0,
        (0, -1) => 1,
        (0, 1) => 2,
        (-1, 0) => 3,
        (1, 0) => 4,
        (-1, -1) => 5,
        (-1, 1) => 6,
        (1, -1) => 7,
        _ => 8,
    }
}

/// Where `p` puts a popup along one axis: `(pos, len)`.
fn axis(rect_pos: i32, rect_len: i32, anchor: i32, gravity: i32, len: i32, offset: i32) -> i32 {
    let a = match anchor {
        -1 => rect_pos,
        1 => rect_pos + rect_len,
        _ => rect_pos + rect_len / 2,
    };
    let start = match gravity {
        -1 => a - len,
        1 => a,
        _ => a - len / 2,
    };
    start + offset
}

/// Place a popup by positioner `p`, keeping it inside `bounds` as its
/// constraint adjustments allow. Everything is relative to the parent's
/// window geometry.
pub fn place(p: &Positioner, bounds: Rect) -> Rect {
    const SLIDE_X: u32 = 1;
    const SLIDE_Y: u32 = 2;
    const FLIP_X: u32 = 4;
    const FLIP_Y: u32 = 8;
    const RESIZE_X: u32 = 16;
    const RESIZE_Y: u32 = 32;
    let (rx, ry, rw, rh) = p.anchor_rect;
    let (mut w, mut h) = p.size;
    let (bx, by, bw, bh) = bounds;
    let fits = |pos: i32, len: i32, lo: i32, span: i32| pos >= lo && pos + len <= lo + span;

    let mut x = axis(rx, rw, horiz(p.anchor), horiz(p.gravity), w, p.offset.0);
    if !fits(x, w, bx, bw) && p.constraint & FLIP_X != 0 {
        let fx = axis(
            rx,
            rw,
            horiz(flip(p.anchor, true)),
            horiz(flip(p.gravity, true)),
            w,
            -p.offset.0,
        );
        if fits(fx, w, bx, bw) {
            x = fx;
        }
    }
    if !fits(x, w, bx, bw) && p.constraint & SLIDE_X != 0 {
        x = x.min(bx + bw - w).max(bx);
    }
    if !fits(x, w, bx, bw) && p.constraint & RESIZE_X != 0 {
        let x1 = (x + w).min(bx + bw);
        x = x.max(bx);
        w = (x1 - x).max(1);
    }

    let mut y = axis(ry, rh, vert(p.anchor), vert(p.gravity), h, p.offset.1);
    if !fits(y, h, by, bh) && p.constraint & FLIP_Y != 0 {
        let fy = axis(
            ry,
            rh,
            vert(flip(p.anchor, false)),
            vert(flip(p.gravity, false)),
            h,
            -p.offset.1,
        );
        if fits(fy, h, by, bh) {
            y = fy;
        }
    }
    if !fits(y, h, by, bh) && p.constraint & SLIDE_Y != 0 {
        y = y.min(by + bh - h).max(by);
    }
    if !fits(y, h, by, bh) && p.constraint & RESIZE_Y != 0 {
        let y1 = (y + h).min(by + bh);
        y = y.max(by);
        h = (y1 - y).max(1);
    }
    (x, y, w, h)
}

impl Compositor {
    pub(super) fn req_wm_base(
        &mut self,
        cid: u32,
        obj: u32,
        version: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        match op {
            0 => {
                self.destroy(cid, obj);
            }
            1 => {
                let id = a.uint()?;
                self.create(
                    cid,
                    id,
                    Iface::Positioner,
                    version,
                    Data::Positioner(Positioner::default()),
                )?;
            }
            2 => {
                let id = a.uint()?;
                let sid = a.uint()?;
                self.expect(cid, sid, Iface::Surface)?;
                let Some(surf) = self.surface((cid, sid)) else {
                    return Ok(());
                };
                if surf.xdg.is_some() || !matches!(surf.role, Role::None) {
                    return Err(ProtoError::new(obj, 0, "surface already has a role"));
                }
                self.create(cid, id, Iface::XdgSurface, version, Data::Surface(sid))?;
                if let Some(surf) = self.surface_mut((cid, sid)) {
                    surf.xdg = Some(id);
                }
            }
            3 => {
                a.uint()?;
            }
            _ => return Err(bad_op(obj)),
        }
        Ok(())
    }

    pub(super) fn req_positioner(
        &mut self,
        cid: u32,
        obj: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        if op == 0 {
            self.destroy(cid, obj);
            return Ok(());
        }
        let mut p = match self.object(cid, obj).map(|o| &o.data) {
            Some(Data::Positioner(p)) => *p,
            _ => return Err(bad_op(obj)),
        };
        match op {
            1 => {
                p.size = (a.int()?, a.int()?);
                if p.size.0 <= 0 || p.size.1 <= 0 {
                    return Err(ProtoError::new(obj, 0, "invalid size"));
                }
            }
            2 => {
                p.anchor_rect = (a.int()?, a.int()?, a.int()?, a.int()?);
                if p.anchor_rect.2 < 0 || p.anchor_rect.3 < 0 {
                    return Err(ProtoError::new(obj, 0, "invalid anchor rect"));
                }
            }
            3 => p.anchor = a.uint()?,
            4 => p.gravity = a.uint()?,
            5 => p.constraint = a.uint()?,
            6 => p.offset = (a.int()?, a.int()?),
            7 => {}
            8 => {
                a.int()?;
                a.int()?;
            }
            9 => {
                a.uint()?;
            }
            _ => return Err(bad_op(obj)),
        }
        if let Some(o) = self.object_mut(cid, obj) {
            o.data = Data::Positioner(p);
        }
        Ok(())
    }

    pub(super) fn req_xdg_surface(
        &mut self,
        cid: u32,
        obj: u32,
        version: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        let sid = self.target(cid, obj)?;
        match op {
            0 => {
                if let Some(s) = self.surface_mut((cid, sid)) {
                    s.xdg = None;
                }
                self.destroy(cid, obj);
            }
            1 => {
                let id = a.uint()?;
                self.check_no_role(cid, sid, obj)?;
                self.create(cid, id, Iface::Toplevel, version, Data::Surface(sid))?;
                if let Some(s) = self.surface_mut((cid, sid)) {
                    s.role = Role::Toplevel(TopInfo {
                        id,
                        ..TopInfo::default()
                    });
                }
            }
            2 => {
                let id = a.uint()?;
                let parent = a.uint()?;
                let positioner = a.uint()?;
                self.check_no_role(cid, sid, obj)?;
                let parent_sid = if parent == 0 {
                    0
                } else {
                    self.expect(cid, parent, Iface::XdgSurface)?;
                    self.target(cid, parent)?
                };
                let pos = match self.object(cid, positioner).map(|o| &o.data) {
                    Some(Data::Positioner(p)) => *p,
                    _ => {
                        return Err(ProtoError::new(
                            positioner,
                            wire::ERR_INVALID_OBJECT,
                            "not a positioner",
                        ));
                    }
                };
                if pos.size.0 <= 0 {
                    return Err(ProtoError::new(obj, 0, "incomplete positioner"));
                }
                self.create(cid, id, Iface::Popup, version, Data::Surface(sid))?;
                if let Some(s) = self.surface_mut((cid, sid)) {
                    s.role = Role::Popup(PopupInfo {
                        id,
                        parent: parent_sid,
                        positioner: pos,
                        rect: (0, 0, pos.size.0, pos.size.1),
                        grab: false,
                        configured: false,
                        mapped: false,
                    });
                }
            }
            3 => {
                let g = (a.int()?, a.int()?, a.int()?, a.int()?);
                if g.2 <= 0 || g.3 <= 0 {
                    return Err(ProtoError::new(obj, 0, "invalid window geometry"));
                }
                if let Some(s) = self.surface_mut((cid, sid)) {
                    s.pending.geometry = Some(g);
                }
            }
            4 => {
                a.uint()?;
            }
            _ => return Err(bad_op(obj)),
        }
        Ok(())
    }

    fn check_no_role(&self, cid: u32, sid: u32, obj: u32) -> Result<(), ProtoError> {
        match self.surface((cid, sid)).map(|s| &s.role) {
            Some(Role::None) => Ok(()),
            _ => Err(ProtoError::new(obj, 1, "surface already has a role")),
        }
    }

    pub(super) fn req_toplevel(
        &mut self,
        cid: u32,
        obj: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        let sid = self.target(cid, obj)?;
        let s = (cid, sid);
        let mut reconfigure = false;
        match op {
            0 => {
                self.unmap(s);
                if let Some(surf) = self.surface_mut(s) {
                    surf.role = Role::None;
                }
                self.destroy(cid, obj);
                return Ok(());
            }
            1 => {
                let parent = a.uint()?;
                let p = if parent == 0 {
                    None
                } else {
                    Some(self.target(cid, parent)?)
                };
                if let Some(t) = self.toplevel_mut(s) {
                    t.parent = p;
                }
                self.scene_dirty = true;
                reconfigure = true;
            }
            2 | 3 => {
                let text = a.string()?;
                if let Some(t) = self.toplevel_mut(s) {
                    if op == 2 {
                        t.title = text;
                    } else {
                        t.app_id = text;
                    }
                }
            }
            4 => {
                a.uint()?;
                a.uint()?;
                a.int()?;
                a.int()?;
            }
            5 => {
                a.uint()?;
                a.uint()?;
            }
            6 => {
                a.uint()?;
                a.uint()?;
                a.uint()?;
            }
            7 | 8 => {
                a.int()?;
                a.int()?;
            }
            9 | 10 | 13 => reconfigure = true,
            11 | 12 => {
                if op == 11 {
                    a.uint()?;
                }
                if let Some(t) = self.toplevel_mut(s) {
                    t.fullscreen = op == 11;
                }
                reconfigure = true;
            }
            _ => return Err(bad_op(obj)),
        }
        if reconfigure && self.toplevel_mut(s).is_some_and(|t| t.configured) {
            self.configure_toplevel(s);
        }
        Ok(())
    }

    pub(super) fn toplevel_mut(&mut self, s: SurfRef) -> Option<&mut TopInfo> {
        match &mut self.surface_mut(s)?.role {
            Role::Toplevel(t) => Some(t),
            _ => None,
        }
    }

    /// Send toplevel `s` its size and states.
    pub(super) fn configure_toplevel(&mut self, s: SurfRef) {
        #[allow(clippy::cast_possible_wrap)]
        let (w, h) = (self.width as i32, self.height as i32);
        let activated = self.window_of_focus() == Some(s);
        let serial = self.next_serial();
        let Some(surf) = self.surface(s) else {
            return;
        };
        let Some(xdg) = surf.xdg else {
            return;
        };
        let Role::Toplevel(t) = &surf.role else {
            return;
        };
        let (id, first, dialog, fullscreen) =
            (t.id, !t.configured, t.parent.is_some(), t.fullscreen);
        let version = self.object(s.0, id).map_or(1, |o| o.version);
        let size = if dialog { (0, 0) } else { (w, h) };
        let mut states = Vec::new();
        if !dialog {
            states.push(if fullscreen { 2 } else { 1 });
            if version >= 2 {
                states.extend([5, 6, 7, 8]); // tiled on every edge
            }
        }
        if activated {
            states.push(4);
        }
        let Some(c) = self.clients.get_mut(&s.0) else {
            return;
        };
        if version >= 4 {
            c.send(Ev::new(id, 2).int(w).int(h)); // configure_bounds
        }
        if first && version >= 5 {
            c.send(Ev::new(id, 3).array(&[])); // wm_capabilities: none
        }
        c.send(
            Ev::new(id, 0)
                .int(size.0)
                .int(size.1)
                .array(&wire::u32_array(&states)),
        );
        c.send(Ev::new(xdg, 0).uint(serial));
        if let Some(t) = self.toplevel_mut(s) {
            t.configured = true;
            t.activated = activated;
            t.size = size;
        }
    }

    /// Place popup `s` and tell it where it is.
    fn configure_popup(&mut self, s: SurfRef, token: Option<u32>) {
        let serial = self.next_serial();
        let Some(surf) = self.surface(s) else {
            return;
        };
        let (Some(xdg), Role::Popup(p)) = (surf.xdg, &surf.role) else {
            return;
        };
        let (id, parent, pos) = (p.id, p.parent, p.positioner);
        // The display, relative to the parent's window geometry.
        let (px, py) = self.geometry_origin((s.0, parent), 0);
        #[allow(clippy::cast_possible_wrap)]
        let bounds = (-px, -py, self.width as i32, self.height as i32);
        let rect = place(&pos, bounds);
        let version = self.object(s.0, id).map_or(1, |o| o.version);
        if let Some(Role::Popup(p)) = self.surface_mut(s).map(|x| &mut x.role) {
            p.rect = rect;
            p.configured = true;
        }
        let Some(c) = self.clients.get_mut(&s.0) else {
            return;
        };
        if let Some(t) = token
            && version >= 3
        {
            c.send(Ev::new(id, 2).uint(t));
        }
        c.send(
            Ev::new(id, 0)
                .int(rect.0)
                .int(rect.1)
                .int(rect.2)
                .int(rect.3),
        );
        c.send(Ev::new(xdg, 0).uint(serial));
        self.scene_dirty = true;
    }

    pub(super) fn req_popup(
        &mut self,
        cid: u32,
        obj: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        let sid = self.target(cid, obj)?;
        let s = (cid, sid);
        match op {
            0 => {
                // Its own child popups can't outlive it.
                self.dismiss_children(s);
                self.unmap(s);
                if let Some(surf) = self.surface_mut(s) {
                    surf.role = Role::None;
                }
                self.destroy(cid, obj);
            }
            1 => {
                a.uint()?;
                a.uint()?;
                if let Some(Role::Popup(p)) = self.surface_mut(s).map(|x| &mut x.role) {
                    p.grab = true;
                }
                self.update_focus();
            }
            2 => {
                let positioner = a.uint()?;
                let token = a.uint()?;
                let pos = match self.object(cid, positioner).map(|o| &o.data) {
                    Some(Data::Positioner(p)) => *p,
                    _ => return Err(ProtoError::new(positioner, 0, "not a positioner")),
                };
                if let Some(Role::Popup(p)) = self.surface_mut(s).map(|x| &mut x.role) {
                    p.positioner = pos;
                }
                self.configure_popup(s, Some(token));
            }
            _ => return Err(bad_op(obj)),
        }
        Ok(())
    }

    /// Send `popup_done` to the mapped popups whose parent chain leads to
    /// `s`, topmost first.
    pub(super) fn dismiss_children(&mut self, s: SurfRef) {
        let doomed: Vec<SurfRef> = self
            .popups
            .iter()
            .rev()
            .copied()
            .filter(|&p| p != s && p.0 == s.0 && self.popup_descends_from(p, s.1))
            .collect();
        for p in doomed {
            self.popup_done(p);
        }
    }

    fn popup_descends_from(&self, p: SurfRef, ancestor: u32) -> bool {
        let mut cur = p.1;
        for _ in 0..16 {
            match self.surface((p.0, cur)).map(|x| &x.role) {
                Some(Role::Popup(info)) => {
                    if info.parent == ancestor {
                        return true;
                    }
                    cur = info.parent;
                }
                _ => return false,
            }
        }
        false
    }

    /// Tell popup `p` it's dismissed, and take it off the display.
    pub(super) fn popup_done(&mut self, p: SurfRef) {
        if let Some(Role::Popup(info)) = self.surface(p).map(|x| &x.role) {
            let id = info.id;
            self.send(p.0, Ev::new(id, 1));
        }
        self.unmap(p);
    }

    /// What a commit means for surface `sid`'s role.
    pub(super) fn after_commit(&mut self, cid: u32, sid: u32) {
        let s = (cid, sid);
        let Some(surf) = self.surface(s) else {
            return;
        };
        let has_buffer = surf.has_buffer;
        let needs_enter = has_buffer && !surf.entered && !matches!(surf.role, Role::Cursor);
        match &surf.role {
            Role::Toplevel(t) => {
                let (configured, mapped) = (t.configured, t.mapped);
                if !configured {
                    self.configure_toplevel(s);
                } else if has_buffer && !mapped {
                    if let Some(t) = self.toplevel_mut(s) {
                        t.mapped = true;
                    }
                    self.stack.push(s);
                    self.update_focus();
                    self.update_pointer();
                } else if !has_buffer && mapped {
                    self.unmap(s);
                }
            }
            Role::Popup(p) => {
                let (configured, mapped) = (p.configured, p.mapped);
                if !configured {
                    self.configure_popup(s, None);
                } else if has_buffer && !mapped {
                    if let Some(Role::Popup(p)) = self.surface_mut(s).map(|x| &mut x.role) {
                        p.mapped = true;
                    }
                    self.popups.push(s);
                    self.update_focus();
                    self.update_pointer();
                } else if !has_buffer && mapped {
                    self.unmap(s);
                }
            }
            Role::Cursor => self.cursor_dirty = true,
            Role::Sub(_) | Role::None => {}
        }
        if needs_enter {
            if let Some(surf) = self.surface_mut(s) {
                surf.entered = true;
            }
            if let Some(c) = self.clients.get_mut(&cid) {
                for (out, _) in c.ids_of(Iface::Output) {
                    c.send(Ev::new(sid, 0).uint(out));
                }
            }
        }
    }

    pub(super) fn req_decoration_manager(
        &mut self,
        cid: u32,
        obj: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        match op {
            0 => {
                self.destroy(cid, obj);
            }
            1 => {
                let id = a.uint()?;
                let toplevel = a.uint()?;
                self.expect(cid, toplevel, Iface::Toplevel)?;
                let sid = self.target(cid, toplevel)?;
                self.create(cid, id, Iface::Decoration, 1, Data::Surface(sid))?;
                if let Some(t) = self.toplevel_mut((cid, sid)) {
                    t.decoration = Some(id);
                }
                self.send(cid, Ev::new(id, 0).uint(2)); // server_side
            }
            _ => return Err(bad_op(obj)),
        }
        Ok(())
    }

    pub(super) fn req_decoration(
        &mut self,
        cid: u32,
        obj: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        let sid = self.target(cid, obj)?;
        match op {
            0 => {
                if let Some(t) = self.toplevel_mut((cid, sid)) {
                    t.decoration = None;
                }
                self.destroy(cid, obj);
            }
            1 | 2 => {
                if op == 1 {
                    a.uint()?;
                }
                self.send(cid, Ev::new(obj, 0).uint(2));
                if self.toplevel_mut((cid, sid)).is_some_and(|t| t.configured) {
                    self.configure_toplevel((cid, sid));
                }
            }
            _ => return Err(bad_op(obj)),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn menu(anchor_rect: Rect, size: (i32, i32)) -> Positioner {
        Positioner {
            size,
            anchor_rect,
            anchor: 6,  // bottom_left
            gravity: 8, // bottom_right
            constraint: 1 | 2 | 4 | 8,
            offset: (0, 0),
        }
    }

    #[test]
    fn a_menu_opens_below_its_button() {
        let p = menu((10, 0, 50, 20), (100, 200));
        assert_eq!(place(&p, (0, 0, 800, 600)), (10, 20, 100, 200));
    }

    #[test]
    fn a_menu_near_the_bottom_flips_up() {
        let p = menu((10, 500, 50, 20), (100, 200));
        assert_eq!(place(&p, (0, 0, 800, 600)), (10, 300, 100, 200));
    }

    #[test]
    fn a_menu_near_the_right_edge_slides_left() {
        let mut p = menu((780, 0, 20, 20), (100, 50));
        p.constraint = 1 | 2; // slide only
        assert_eq!(place(&p, (0, 0, 800, 600)), (700, 20, 100, 50));
    }

    #[test]
    fn centered_and_flipped_anchors() {
        assert_eq!(flip(6, true), 8);
        assert_eq!(flip(6, false), 5);
        assert_eq!(flip(0, true), 0);
        let p = Positioner {
            size: (10, 10),
            anchor_rect: (0, 0, 100, 100),
            ..Positioner::default()
        };
        assert_eq!(place(&p, (0, 0, 800, 600)), (45, 45, 10, 10));
    }
}
