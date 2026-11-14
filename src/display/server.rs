//! The compositor's state and its turn: connections, objects, request
//! routing, the scene and its composition. The interfaces themselves are in
//! [`super::protocol`] (core, shm, subsurfaces, outputs), [`super::shell`]
//! (xdg-shell, decorations) and [`super::seat`] (input).

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

use super::render::{self, Canvas, Image};
use super::wire::{self, Args, Ev, ProtoError};
use crate::kernel::{Kernel, ServedFd, ServedRecv};

/// A surface, by client and object id.
pub type SurfRef = (u32, u32);

/// Every interface nixvm implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Iface {
    Display,
    Registry,
    Callback,
    Compositor,
    Subcompositor,
    Surface,
    Region,
    Subsurface,
    Shm,
    ShmPool,
    Buffer,
    Seat,
    Pointer,
    Keyboard,
    Touch,
    Output,
    XdgOutputManager,
    XdgOutput,
    WmBase,
    Positioner,
    XdgSurface,
    Toplevel,
    Popup,
    DataDeviceManager,
    DataSource,
    DataDevice,
    DataOffer,
    DecorationManager,
    Decoration,
    Viewporter,
    Viewport,
}

/// The globals, in registry order: interface, its name, highest version.
pub const GLOBALS: &[(Iface, &str, u32)] = &[
    (Iface::Compositor, "wl_compositor", 6),
    (Iface::Subcompositor, "wl_subcompositor", 1),
    (Iface::Shm, "wl_shm", 2),
    (Iface::Seat, "wl_seat", 7),
    (Iface::Output, "wl_output", 4),
    (Iface::WmBase, "xdg_wm_base", 5),
    (Iface::DataDeviceManager, "wl_data_device_manager", 3),
    (Iface::DecorationManager, "zxdg_decoration_manager_v1", 1),
    (Iface::Viewporter, "wp_viewporter", 1),
    (Iface::XdgOutputManager, "zxdg_output_manager_v1", 3),
];

/// A client's shared-memory pool: the file it passed, read at commit.
#[derive(Debug)]
pub struct Pool {
    pub fd: ServedFd,
    pub size: std::sync::atomic::AtomicU64,
}

/// A `wl_buffer` in a pool.
#[derive(Debug, Clone)]
pub struct Buffer {
    pub pool: Arc<Pool>,
    pub offset: u64,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: u32,
}

/// An `xdg_positioner`'s rules.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Positioner {
    pub size: (i32, i32),
    pub anchor_rect: (i32, i32, i32, i32),
    pub anchor: u32,
    pub gravity: u32,
    pub constraint: u32,
    pub offset: (i32, i32),
}

/// What an object holds besides its interface.
#[derive(Debug, Default)]
pub enum Data {
    #[default]
    None,
    Pool(Arc<Pool>),
    Buffer(Buffer),
    Positioner(Positioner),
    /// Objects that extend a surface (`xdg_surface`, `xdg_toplevel`,
    /// `xdg_popup`, `wl_subsurface`, `wp_viewport`, the decoration): its id.
    Surface(u32),
    /// A data source's MIME types.
    Mimes(Vec<String>),
    /// A data offer: the source it offers (client, object).
    Offer(SurfRef),
}

#[derive(Debug)]
pub struct Object {
    pub iface: Iface,
    pub version: u32,
    pub data: Data,
}

/// A rectangle: x, y, width, height.
pub type Rect = (i32, i32, i32, i32);

/// A surface's double-buffered state, pending until committed. (An outer
/// `None` is "unchanged"; `Some(None)` unsets.)
#[allow(clippy::option_option)]
#[derive(Debug, Default)]
pub struct Pending {
    /// `attach` was called: the new buffer (`None` unmaps).
    pub attach: Option<Option<u32>>,
    pub offset: (i32, i32),
    /// Damage in surface coordinates, then in buffer coordinates.
    pub damage: Vec<Rect>,
    pub damage_buffer: Vec<Rect>,
    pub frames: Vec<u32>,
    pub scale: Option<i32>,
    pub vp_src: Option<Option<(f64, f64, f64, f64)>>,
    pub vp_dst: Option<Option<(i32, i32)>>,
    pub geometry: Option<Rect>,
}

impl Pending {
    /// Fold `newer` (a later commit) into this cached state.
    pub fn merge(&mut self, newer: Pending) {
        if newer.attach.is_some() {
            self.attach = newer.attach;
            self.offset = newer.offset;
        }
        self.damage.extend(newer.damage);
        self.damage_buffer.extend(newer.damage_buffer);
        self.frames.extend(newer.frames);
        if newer.scale.is_some() {
            self.scale = newer.scale;
        }
        if newer.vp_src.is_some() {
            self.vp_src = newer.vp_src;
        }
        if newer.vp_dst.is_some() {
            self.vp_dst = newer.vp_dst;
        }
        if newer.geometry.is_some() {
            self.geometry = newer.geometry;
        }
    }
}

/// A surface's role.
#[derive(Debug, Default)]
pub enum Role {
    #[default]
    None,
    Cursor,
    Sub(SubInfo),
    Toplevel(TopInfo),
    Popup(PopupInfo),
}

#[derive(Debug)]
pub struct SubInfo {
    pub parent: u32,
    pub pos: (i32, i32),
    pub pending_pos: Option<(i32, i32)>,
    pub sync: bool,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Default)]
pub struct TopInfo {
    pub id: u32,
    pub parent: Option<u32>,
    pub configured: bool,
    pub mapped: bool,
    pub fullscreen: bool,
    pub decoration: Option<u32>,
    pub title: String,
    pub app_id: String,
    /// The `activated` state last sent.
    pub activated: bool,
    /// The size last configured.
    pub size: (i32, i32),
}

#[derive(Debug)]
pub struct PopupInfo {
    pub id: u32,
    pub parent: u32,
    pub positioner: Positioner,
    /// Where it was placed, relative to the parent's window geometry.
    pub rect: Rect,
    pub grab: bool,
    pub configured: bool,
    pub mapped: bool,
}

#[derive(Debug, Default)]
pub struct Surface {
    pub pending: Pending,
    /// A synchronized subsurface's commits, waiting for its parent's.
    pub cached: Option<Pending>,
    pub image: Image,
    pub has_buffer: bool,
    pub scale: i32,
    pub vp_src: Option<(f64, f64, f64, f64)>,
    pub vp_dst: Option<(i32, i32)>,
    pub geometry: Option<Rect>,
    pub role: Role,
    /// Its `xdg_surface`, if it has one.
    pub xdg: Option<u32>,
    /// Itself and its subsurfaces, bottom to top.
    pub stack: Vec<u32>,
    pub viewport: Option<u32>,
    /// `wl_surface.enter` was sent.
    pub entered: bool,
}

impl Surface {
    /// Its size in surface (display) pixels.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn size(&self) -> (i32, i32) {
        if !self.has_buffer {
            return (0, 0);
        }
        if let Some(d) = self.vp_dst {
            return d;
        }
        if let Some((_, _, w, h)) = self.vp_src {
            return (w as i32, h as i32);
        }
        let s = self.scale.max(1);
        (self.image.width as i32 / s, self.image.height as i32 / s)
    }

    /// Its window geometry (`xdg_surface.set_window_geometry`, else its
    /// bounds).
    pub fn window_geometry(&self) -> Rect {
        self.geometry.unwrap_or_else(|| {
            let (w, h) = self.size();
            (0, 0, w, h)
        })
    }
}

/// A connected client.
pub struct Client {
    pub conn: usize,
    inbuf: Vec<u8>,
    infds: VecDeque<ServedFd>,
    out: Vec<u8>,
    outfds: Vec<ServedFd>,
    pub objects: HashMap<u32, Object>,
    pub surfaces: HashMap<u32, Surface>,
    /// Frame callbacks committed and not yet done: id, the presentation
    /// count at commit, the commit time (ms).
    pub frames: Vec<(u32, u64, u64)>,
    /// A protocol error was sent: close after flushing.
    pub doomed: bool,
    /// The next id for an object nixvm creates for it.
    next_id: u32,
}

impl Client {
    /// An id for an object the server creates.
    pub fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(SERVER_IDS);
        id
    }

    pub fn send(&mut self, ev: Ev) {
        self.out.extend_from_slice(&ev.finish());
    }

    pub fn send_fd(&mut self, ev: Ev, fd: ServedFd) {
        self.outfds.push(fd);
        self.send(ev);
    }

    /// The ids of its objects of interface `iface`.
    pub fn ids_of(&self, iface: Iface) -> Vec<(u32, u32)> {
        let mut v: Vec<(u32, u32)> = self
            .objects
            .iter()
            .filter(|(_, o)| o.iface == iface)
            .map(|(&id, o)| (id, o.version))
            .collect();
        v.sort_unstable();
        v
    }
}

/// The pointer.
#[derive(Debug, Default)]
pub struct PointerState {
    pub x: f64,
    pub y: f64,
    pub inside: bool,
    pub focus: Option<SurfRef>,
    pub buttons: Vec<u32>,
    /// Wheel motion not yet a whole step.
    pub scroll_acc: (f64, f64),
}

/// What the pointer looks like.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cursor {
    Default,
    Hidden,
    Surface(SurfRef, i32, i32),
}

/// A surface placed on the display.
#[derive(Debug, Clone, Copy)]
pub struct Placed {
    pub surf: SurfRef,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

pub struct Compositor {
    pub width: u32,
    pub height: u32,
    pub clients: BTreeMap<u32, Client>,
    next_client: u32,
    serial: u32,
    /// Mapped toplevels, bottom to top.
    pub stack: Vec<SurfRef>,
    /// Mapped popups, bottom to top.
    pub popups: Vec<SurfRef>,
    pub kb_focus: Option<SurfRef>,
    pub ptr: PointerState,
    pub cursor: Cursor,
    pub mods: super::keymap::Mods,
    pub keys: Vec<u32>,
    pub keymap_path: String,
    pub keymap_len: u32,
    /// The scene changed: composite again.
    pub scene_dirty: bool,
    /// Only the pointer moved.
    pub cursor_dirty: bool,
    scene: Vec<u32>,
    out: Vec<u32>,
    arrow: Image,
    pub last_fire_ms: u64,
    /// How many frames the embedder has taken (see [`super::Display::frame`]).
    pub presented: u64,
    /// The clipboard: the data source (client, object) that holds it.
    pub selection: Option<SurfRef>,
}

/// The background where no window covers the display.
pub const BACKGROUND: u32 = 0xff20_242c;

/// The first id the server allocates for objects it creates.
const SERVER_IDS: u32 = 0xff00_0000;

/// Monotonic milliseconds, as event timestamps want.
pub fn now_ms() -> u64 {
    u64::try_from(crate::clock::now_monotonic().as_millis()).unwrap_or(u64::MAX)
}

#[allow(clippy::cast_possible_truncation)]
pub fn time32() -> u32 {
    now_ms() as u32
}

impl Compositor {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            clients: BTreeMap::new(),
            next_client: 1,
            serial: 0,
            stack: Vec::new(),
            popups: Vec::new(),
            kb_focus: None,
            ptr: PointerState::default(),
            cursor: Cursor::Default,
            mods: super::keymap::Mods::default(),
            keys: Vec::new(),
            keymap_path: String::new(),
            keymap_len: 0,
            scene_dirty: true,
            cursor_dirty: false,
            scene: Vec::new(),
            out: Vec::new(),
            arrow: render::arrow(),
            last_fire_ms: 0,
            presented: 0,
            selection: None,
        }
    }

    pub fn next_serial(&mut self) -> u32 {
        self.serial = self.serial.wrapping_add(1);
        self.serial
    }

    pub fn send(&mut self, cid: u32, ev: Ev) {
        if let Some(c) = self.clients.get_mut(&cid) {
            c.send(ev);
        }
    }

    pub fn surface(&self, s: SurfRef) -> Option<&Surface> {
        self.clients.get(&s.0)?.surfaces.get(&s.1)
    }

    pub fn surface_mut(&mut self, s: SurfRef) -> Option<&mut Surface> {
        self.clients.get_mut(&s.0)?.surfaces.get_mut(&s.1)
    }

    pub fn object(&self, cid: u32, id: u32) -> Option<&Object> {
        self.clients.get(&cid)?.objects.get(&id)
    }

    pub fn object_mut(&mut self, cid: u32, id: u32) -> Option<&mut Object> {
        self.clients.get_mut(&cid)?.objects.get_mut(&id)
    }

    /// The surface object `id` extends (see [`Data::Surface`]).
    pub fn target(&self, cid: u32, id: u32) -> Result<u32, ProtoError> {
        match self.object(cid, id) {
            Some(Object {
                data: Data::Surface(s),
                ..
            }) => Ok(*s),
            _ => Err(ProtoError::new(id, wire::ERR_INVALID_OBJECT, "no surface")),
        }
    }

    /// Check that `id` is an object of interface `iface`.
    pub fn expect(&self, cid: u32, id: u32, iface: Iface) -> Result<(), ProtoError> {
        match self.object(cid, id) {
            Some(o) if o.iface == iface => Ok(()),
            _ => Err(ProtoError::new(
                id,
                wire::ERR_INVALID_OBJECT,
                format!("object {id} is not a {iface:?}"),
            )),
        }
    }

    /// Create object `id` for client `cid`.
    pub fn create(
        &mut self,
        cid: u32,
        id: u32,
        iface: Iface,
        version: u32,
        data: Data,
    ) -> Result<(), ProtoError> {
        let Some(c) = self.clients.get_mut(&cid) else {
            return Ok(());
        };
        if id == 0 || id >= SERVER_IDS || c.objects.contains_key(&id) {
            return Err(ProtoError::new(
                1,
                wire::ERR_INVALID_OBJECT,
                format!("invalid new id {id}"),
            ));
        }
        c.objects.insert(
            id,
            Object {
                iface,
                version,
                data,
            },
        );
        Ok(())
    }

    /// Forget object `id` and tell the client its id is free again.
    pub fn destroy(&mut self, cid: u32, id: u32) -> Option<Object> {
        let c = self.clients.get_mut(&cid)?;
        let o = c.objects.remove(&id);
        if id < SERVER_IDS {
            c.send(Ev::new(1, 1).uint(id)); // wl_display.delete_id
        }
        o
    }

    /// One turn of the compositor's I/O: accept, read and handle requests.
    pub fn pump(&mut self, kernel: &Kernel, listener: usize) {
        while let Some(conn) = kernel.served_accept(listener) {
            let cid = self.next_client;
            self.next_client += 1;
            let mut objects = HashMap::new();
            objects.insert(
                1,
                Object {
                    iface: Iface::Display,
                    version: 1,
                    data: Data::None,
                },
            );
            self.clients.insert(
                cid,
                Client {
                    conn,
                    inbuf: Vec::new(),
                    infds: VecDeque::new(),
                    out: Vec::new(),
                    outfds: Vec::new(),
                    objects,
                    surfaces: HashMap::new(),
                    frames: Vec::new(),
                    doomed: false,
                    next_id: SERVER_IDS,
                },
            );
        }
        let ids: Vec<u32> = self.clients.keys().copied().collect();
        for cid in ids {
            let closed = {
                let Some(c) = self.clients.get_mut(&cid) else {
                    continue;
                };
                let mut fds = Vec::new();
                let r = kernel.served_recv(c.conn, &mut c.inbuf, &mut fds);
                c.infds.extend(fds);
                r == ServedRecv::Closed
            };
            self.handle_requests(kernel, cid);
            if closed {
                self.disconnect(kernel, cid);
            }
        }
    }

    fn handle_requests(&mut self, kernel: &Kernel, cid: u32) {
        let mut pos = 0;
        loop {
            let Some(c) = self.clients.get_mut(&cid) else {
                return;
            };
            if c.doomed {
                break;
            }
            let (obj, op, size) = match wire::header(&c.inbuf[pos..]) {
                Ok(Some(h)) => h,
                Ok(None) => break,
                Err(e) => {
                    self.post_error(cid, &e);
                    break;
                }
            };
            let body = c.inbuf[pos + 8..pos + size].to_vec();
            pos += size;
            let mut fds = std::mem::take(&mut c.infds);
            let r = {
                let mut args = Args::new(obj, &body, &mut fds);
                self.dispatch(kernel, cid, obj, op, &mut args)
            };
            if let Some(c) = self.clients.get_mut(&cid) {
                fds.extend(c.infds.drain(..));
                c.infds = fds;
            } else {
                for f in fds {
                    kernel.served_drop_fd(f);
                }
            }
            if let Err(e) = r {
                self.post_error(cid, &e);
                break;
            }
        }
        if let Some(c) = self.clients.get_mut(&cid) {
            c.inbuf.drain(..pos);
        }
    }

    /// Send `e` to the client and close it once that's flushed.
    pub fn post_error(&mut self, cid: u32, e: &ProtoError) {
        if let Some(c) = self.clients.get_mut(&cid) {
            c.send(Ev::new(1, 0).uint(e.object).uint(e.code).string(&e.message));
            c.doomed = true;
        }
    }

    fn dispatch(
        &mut self,
        kernel: &Kernel,
        cid: u32,
        obj: u32,
        op: u16,
        a: &mut Args,
    ) -> Result<(), ProtoError> {
        let Some(o) = self.object(cid, obj) else {
            // A request on an object the client just destroyed (and we
            // haven't confirmed) is harmless; libwayland ignores it too.
            return Ok(());
        };
        let (iface, version) = (o.iface, o.version);
        let r = match iface {
            Iface::Display => self.req_display(cid, op, a),
            Iface::Registry => self.req_registry(cid, op, a),
            Iface::Compositor => self.req_compositor(cid, obj, version, op, a),
            Iface::Surface => self.req_surface(kernel, cid, obj, op, a),
            Iface::Region
            | Iface::Buffer
            | Iface::Keyboard
            | Iface::Touch
            | Iface::Output
            | Iface::XdgOutput => self.req_simple(cid, obj, op, 0),
            Iface::Subcompositor => self.req_subcompositor(cid, obj, op, a),
            Iface::Subsurface => self.req_subsurface(cid, obj, op, a),
            Iface::Shm => self.req_shm(kernel, cid, obj, version, op, a),
            Iface::ShmPool => self.req_pool(kernel, cid, obj, op, a),
            Iface::Seat => self.req_seat(kernel, cid, obj, version, op, a),
            Iface::Pointer => self.req_pointer(cid, obj, op, a),
            Iface::XdgOutputManager => self.req_xdg_output_manager(cid, obj, op, a),
            Iface::WmBase => self.req_wm_base(cid, obj, version, op, a),
            Iface::Positioner => self.req_positioner(cid, obj, op, a),
            Iface::XdgSurface => self.req_xdg_surface(cid, obj, version, op, a),
            Iface::Toplevel => self.req_toplevel(cid, obj, op, a),
            Iface::Popup => self.req_popup(cid, obj, op, a),
            Iface::DataDeviceManager => self.req_data_device_manager(cid, obj, version, op, a),
            Iface::DataSource => self.req_data_source(cid, obj, op, a),
            Iface::DataOffer => self.req_data_offer(kernel, cid, obj, op, a),
            Iface::DataDevice => self.req_data_device(cid, obj, op, a),
            Iface::DecorationManager => self.req_decoration_manager(cid, obj, op, a),
            Iface::Decoration => self.req_decoration(cid, obj, op, a),
            Iface::Viewporter => self.req_viewporter(cid, obj, op, a),
            Iface::Viewport => self.req_viewport(cid, obj, op, a),
            Iface::Callback => Err(ProtoError::new(
                obj,
                wire::ERR_INVALID_METHOD,
                "wl_callback has no requests",
            )),
        };
        // An fd argument the handler didn't consume is the client's mistake;
        // nothing to do (it stays queued for nobody and goes at disconnect).
        r
    }

    /// A request on an interface whose only request nixvm acts on is
    /// `destroy` (or `release`) at opcode `destroy_op`; the rest are accepted
    /// and ignored.
    #[allow(clippy::unnecessary_wraps)] // one shape for every handler
    pub fn req_simple(
        &mut self,
        cid: u32,
        obj: u32,
        op: u16,
        destroy_op: u16,
    ) -> Result<(), ProtoError> {
        if op == destroy_op {
            self.destroy(cid, obj);
        }
        Ok(())
    }

    fn req_display(&mut self, cid: u32, op: u16, a: &mut Args) -> Result<(), ProtoError> {
        match op {
            0 => {
                // sync
                let id = a.uint()?;
                self.create(cid, id, Iface::Callback, 1, Data::None)?;
                let serial = self.next_serial();
                self.send(cid, Ev::new(id, 0).uint(serial));
                self.destroy(cid, id);
                Ok(())
            }
            1 => {
                // get_registry
                let id = a.uint()?;
                self.create(cid, id, Iface::Registry, 1, Data::None)?;
                for (i, (_, name, version)) in GLOBALS.iter().enumerate() {
                    let gname = u32::try_from(i + 1).unwrap_or(0);
                    self.send(cid, Ev::new(id, 0).uint(gname).string(name).uint(*version));
                }
                Ok(())
            }
            _ => Err(ProtoError::new(1, wire::ERR_INVALID_METHOD, "bad opcode")),
        }
    }

    fn req_registry(&mut self, cid: u32, op: u16, a: &mut Args) -> Result<(), ProtoError> {
        if op != 0 {
            return Err(ProtoError::new(1, wire::ERR_INVALID_METHOD, "bad opcode"));
        }
        let name = a.uint()?;
        let iface_name = a.string()?;
        let version = a.uint()?;
        let id = a.uint()?;
        let Some(&(iface, gname, max)) = GLOBALS.get((name as usize).wrapping_sub(1)) else {
            return Err(ProtoError::new(
                1,
                wire::ERR_INVALID_OBJECT,
                format!("no global {name}"),
            ));
        };
        if gname != iface_name || version == 0 || version > max {
            return Err(ProtoError::new(
                1,
                wire::ERR_INVALID_OBJECT,
                format!("bad bind of {iface_name} v{version} to global {gname} v{max}"),
            ));
        }
        self.create(cid, id, iface, version, Data::None)?;
        match iface {
            Iface::Shm => {
                for f in render::FORMATS {
                    self.send(cid, Ev::new(id, 0).uint(*f));
                }
            }
            Iface::Seat => {
                self.send(cid, Ev::new(id, 0).uint(3)); // pointer | keyboard
                if version >= 2 {
                    self.send(cid, Ev::new(id, 1).string("seat0"));
                }
            }
            Iface::Output => self.send_output(cid, id, version),
            _ => {}
        }
        Ok(())
    }

    /// Drop client `cid`: its windows go, its connection is closed.
    pub fn disconnect(&mut self, kernel: &Kernel, cid: u32) {
        let Some(mut c) = self.clients.remove(&cid) else {
            return;
        };
        // Whatever it was told before the end (a protocol error) goes out.
        if !c.out.is_empty() {
            kernel.served_send(c.conn, &c.out, std::mem::take(&mut c.outfds));
        }
        for f in c.infds.drain(..) {
            kernel.served_drop_fd(f);
        }
        for f in c.outfds.drain(..) {
            kernel.served_drop_fd(f);
        }
        kernel.served_close(c.conn);
        self.stack.retain(|s| s.0 != cid);
        self.popups.retain(|s| s.0 != cid);
        if self.ptr.focus.is_some_and(|s| s.0 == cid) {
            self.ptr.focus = None;
            self.ptr.buttons.clear();
        }
        if self.kb_focus.is_some_and(|s| s.0 == cid) {
            self.kb_focus = None;
        }
        if matches!(self.cursor, Cursor::Surface((c2, _), ..) if c2 == cid) {
            self.cursor = Cursor::Default;
        }
        if self.selection.is_some_and(|s| s.0 == cid) {
            self.selection = None;
            if let Some(f) = self.kb_focus {
                self.send_selection(f.0);
            }
        }
        self.scene_dirty = true;
        self.update_focus();
        self.update_pointer();
    }

    /// Send what's queued to each client; close the doomed and the gone.
    pub fn flush(&mut self, kernel: &Kernel) {
        let mut gone = Vec::new();
        for (&cid, c) in &mut self.clients {
            if !c.out.is_empty() {
                let out = std::mem::take(&mut c.out);
                let fds = std::mem::take(&mut c.outfds);
                if !kernel.served_send(c.conn, &out, fds) {
                    gone.push(cid);
                    continue;
                }
            }
            if c.doomed {
                gone.push(cid);
            }
        }
        for cid in gone {
            self.disconnect(kernel, cid);
        }
    }

    /// Whether any client has frame callbacks waiting.
    pub fn frames_pending(&self) -> bool {
        self.clients.values().any(|c| !c.frames.is_empty())
    }

    /// Fire the frame callbacks committed before presentation `presented`,
    /// or all of them if `all`.
    pub fn fire_frames(&mut self, presented: u64, all: bool) {
        let t = time32();
        for c in self.clients.values_mut() {
            let mut keep = Vec::new();
            for (id, at, ms) in std::mem::take(&mut c.frames) {
                if all || at < presented {
                    c.send(Ev::new(id, 0).uint(t));
                    c.objects.remove(&id);
                    c.send(Ev::new(1, 1).uint(id));
                } else {
                    keep.push((id, at, ms));
                }
            }
            c.frames = keep;
        }
        self.last_fire_ms = now_ms();
    }

    /// The origin, in display pixels, of mapped toplevel or popup `s`'s
    /// window geometry.
    pub fn geometry_origin(&self, s: SurfRef, depth: u32) -> (i32, i32) {
        let Some(surf) = self.surface(s) else {
            return (0, 0);
        };
        match &surf.role {
            Role::Toplevel(t) => {
                let (_, _, gw, gh) = surf.window_geometry();
                if t.parent.is_some() {
                    // A dialog: centered, at its own size.
                    #[allow(clippy::cast_possible_wrap)]
                    let (w, h) = (self.width as i32, self.height as i32);
                    (((w - gw) / 2).max(0), ((h - gh) / 2).max(0))
                } else {
                    (0, 0)
                }
            }
            Role::Popup(p) if depth < 16 => {
                let (px, py) = self.geometry_origin((s.0, p.parent), depth + 1);
                (px + p.rect.0, py + p.rect.1)
            }
            _ => (0, 0),
        }
    }

    /// Every visible surface, bottom to top, where it is on the display.
    pub fn layout(&self) -> Vec<Placed> {
        let mut out = Vec::new();
        for &s in self.stack.iter().chain(&self.popups) {
            let Some(surf) = self.surface(s) else {
                continue;
            };
            let (gx, gy) = self.geometry_origin(s, 0);
            let (ox, oy, _, _) = surf.window_geometry();
            self.place_tree(s, gx - ox, gy - oy, 0, &mut out);
        }
        out
    }

    fn place_tree(&self, s: SurfRef, x: i32, y: i32, depth: u32, out: &mut Vec<Placed>) {
        let Some(surf) = self.surface(s) else {
            return;
        };
        if !surf.has_buffer || depth > 32 {
            return;
        }
        for &id in &surf.stack {
            if id == s.1 {
                let (w, h) = surf.size();
                out.push(Placed {
                    surf: s,
                    x,
                    y,
                    w,
                    h,
                });
            } else if let Some(Surface {
                role: Role::Sub(sub),
                ..
            }) = self.surface((s.0, id))
            {
                self.place_tree((s.0, id), x + sub.pos.0, y + sub.pos.1, depth + 1, out);
            }
        }
    }

    /// The topmost surface under `(x, y)`, and the point in its coordinates.
    pub fn hit(&self, x: f64, y: f64) -> Option<(SurfRef, f64, f64)> {
        self.layout().into_iter().rev().find_map(|p| {
            let lx = x - f64::from(p.x);
            let ly = y - f64::from(p.y);
            (lx >= 0.0 && ly >= 0.0 && lx < f64::from(p.w) && ly < f64::from(p.h))
                .then_some((p.surf, lx, ly))
        })
    }

    /// Where surface `s` is on the display, if it's visible.
    pub fn placed(&self, s: SurfRef) -> Option<Placed> {
        self.layout().into_iter().find(|p| p.surf == s)
    }

    /// Composite the scene and the pointer into `pixels` (RGBA) if anything
    /// changed, bumping `generation`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )]
    pub fn composite(&mut self, pixels: &mut Vec<u8>, generation: &mut u64) {
        if !self.scene_dirty && !self.cursor_dirty {
            return;
        }
        let n = self.width as usize * self.height as usize;
        if self.scene_dirty {
            self.scene.clear();
            self.scene.resize(n, BACKGROUND);
            let layout = self.layout();
            let mut canvas = Canvas {
                width: self.width,
                height: self.height,
                pixels: &mut self.scene,
            };
            for p in layout {
                let Some(surf) = self
                    .clients
                    .get(&p.surf.0)
                    .and_then(|c| c.surfaces.get(&p.surf.1))
                else {
                    continue;
                };
                canvas.draw(&surf.image, src_rect(surf), p.x, p.y, p.w, p.h);
            }
        }
        self.out.clear();
        self.out.extend_from_slice(&self.scene);
        if self.ptr.inside {
            let (px, py) = (self.ptr.x.floor() as i32, self.ptr.y.floor() as i32);
            let mut canvas = Canvas {
                width: self.width,
                height: self.height,
                pixels: &mut self.out,
            };
            match self.cursor {
                Cursor::Hidden => {}
                Cursor::Default => {
                    let a = &self.arrow;
                    canvas.draw(
                        a,
                        (0, 0, a.width as i32, a.height as i32),
                        px,
                        py,
                        a.width as i32,
                        a.height as i32,
                    );
                }
                Cursor::Surface(s, hx, hy) => {
                    if let Some(surf) = self
                        .clients
                        .get(&s.0)
                        .and_then(|c| c.surfaces.get(&s.1))
                        .filter(|s| s.has_buffer)
                    {
                        let (w, h) = surf.size();
                        canvas.draw(&surf.image, src_rect(surf), px - hx, py - hy, w, h);
                    }
                }
            }
        }
        render::to_rgba(&self.out, pixels);
        *generation += 1;
        self.scene_dirty = false;
        self.cursor_dirty = false;
    }
}

/// The part of `surf`'s buffer that is shown, in buffer pixels.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn src_rect(surf: &Surface) -> Rect {
    let s = surf.scale.max(1);
    match surf.vp_src {
        Some((x, y, w, h)) => (
            (x * f64::from(s)) as i32,
            (y * f64::from(s)) as i32,
            (w * f64::from(s)) as i32,
            (h * f64::from(s)) as i32,
        ),
        None => (0, 0, surf.image.width as i32, surf.image.height as i32),
    }
}
