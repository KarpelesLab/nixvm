//! The native backend: the host's own EGL + GLES 3, loaded with `dlopen`
//! (nothing links against GL, and a host without it simply has no `egl`
//! backend), on a surfaceless context — no window, no display server.
//!
//! The display is `EGL_PLATFORM_SURFACELESS_MESA` when the host's EGL offers
//! it, else the first `EGL_EXT_platform_device` device. Rendering targets
//! are FBOs, so no surface is ever needed (`EGL_KHR_surfaceless_context`).
//! The context is made current around each batch of calls ([`GpuBackend::
//! begin`]/[`end`](GpuBackend::end)) because the guest's ioctls come in on
//! whichever CPU thread runs the caller.
//!
//! `unsafe` here is the FFI: `dlopen`/`dlsym` and calls through the loaded
//! function pointers, with arguments sized by the Rust slices they come from.

#![allow(unsafe_code)]

use std::ffi::{CString, c_char, c_void};

use super::GpuBackend;

type Ptr = *const c_void;

unsafe extern "C" {
    fn dlopen(file: *const c_char, mode: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}

const RTLD_NOW: i32 = 2;

// EGL enums.
const EGL_NONE: i32 = 0x3038;
const EGL_RENDERABLE_TYPE: i32 = 0x3040;
const EGL_OPENGL_ES3_BIT: i32 = 0x40;
const EGL_OPENGL_ES_API: u32 = 0x30A0;
const EGL_CONTEXT_MAJOR_VERSION: i32 = 0x3098;
const EGL_PLATFORM_SURFACELESS_MESA: u32 = 0x31DD;
const EGL_PLATFORM_DEVICE_EXT: u32 = 0x313F;
const EGL_VENDOR: i32 = 0x3053;

// GL enums used only here.
const GL_COMPILE_STATUS: u32 = 0x8B81;
const GL_LINK_STATUS: u32 = 0x8B82;
const GL_VERTEX_SHADER: u32 = 0x8B31;
const GL_FRAGMENT_SHADER: u32 = 0x8B30;
const GL_PACK_ALIGNMENT: u32 = 0x0D05;
const GL_UNPACK_ALIGNMENT: u32 = 0x0CF5;
const GL_RENDERER: u32 = 0x1F01;
const GL_VERSION: u32 = 0x1F02;

macro_rules! fns {
    ($table:ident, $prefix:literal: $($name:ident ( $($arg:ty),* ) $(-> $ret:ty)?;)*) => {
        #[allow(non_snake_case)]
        struct $table {
            $($name: unsafe extern "C" fn($($arg),*) $(-> $ret)?,)*
        }

        impl $table {
            /// Resolve every entry point with `get`.
            fn load(get: &dyn Fn(&str) -> Ptr) -> Result<Self, String> {
                Ok(Self {
                    $($name: {
                        let name = concat!($prefix, stringify!($name));
                        let p = get(name);
                        if p.is_null() {
                            return Err(format!("{name} not found"));
                        }
                        // SAFETY: `p` is the address of the C function of
                        // this name, whose prototype the declaration above
                        // copies from the Khronos headers.
                        unsafe { std::mem::transmute::<Ptr, unsafe extern "C" fn($($arg),*) $(-> $ret)?>(p) }
                    },)*
                })
            }
        }
    };
}

fns!(EglFns, "egl":
    GetProcAddress(*const c_char) -> Ptr;
    Initialize(Ptr, *mut i32, *mut i32) -> u32;
    BindAPI(u32) -> u32;
    ChooseConfig(Ptr, *const i32, *mut Ptr, i32, *mut i32) -> u32;
    CreateContext(Ptr, Ptr, Ptr, *const i32) -> Ptr;
    MakeCurrent(Ptr, Ptr, Ptr, Ptr) -> u32;
    GetError() -> i32;
    QueryString(Ptr, i32) -> *const c_char;
);

fns!(GlFns, "gl":
    GenBuffers(i32, *mut u32);
    DeleteBuffers(i32, *const u32);
    BindBuffer(u32, u32);
    BufferData(u32, isize, Ptr, u32);
    BufferSubData(u32, isize, isize, Ptr);
    BindBufferRange(u32, u32, u32, isize, isize);
    GenTextures(i32, *mut u32);
    DeleteTextures(i32, *const u32);
    ActiveTexture(u32);
    BindTexture(u32, u32);
    TexStorage2D(u32, i32, u32, i32, i32);
    TexStorage3D(u32, i32, u32, i32, i32, i32);
    TexSubImage2D(u32, i32, i32, i32, i32, i32, u32, u32, Ptr);
    TexSubImage3D(u32, i32, i32, i32, i32, i32, i32, i32, u32, u32, Ptr);
    TexParameteri(u32, u32, i32);
    GenSamplers(i32, *mut u32);
    DeleteSamplers(i32, *const u32);
    SamplerParameteri(u32, u32, i32);
    SamplerParameterf(u32, u32, f32);
    BindSampler(u32, u32);
    GenFramebuffers(i32, *mut u32);
    DeleteFramebuffers(i32, *const u32);
    BindFramebuffer(u32, u32);
    FramebufferTexture2D(u32, u32, u32, u32, i32);
    FramebufferTextureLayer(u32, u32, u32, i32, i32);
    DrawBuffers(i32, *const u32);
    ReadBuffer(u32);
    CheckFramebufferStatus(u32) -> u32;
    ReadPixels(i32, i32, i32, i32, u32, u32, *mut c_void);
    BlitFramebuffer(i32, i32, i32, i32, i32, i32, i32, i32, u32, u32);
    ClearBufferfv(u32, i32, *const f32);
    ClearBufferiv(u32, i32, *const i32);
    ClearBufferuiv(u32, i32, *const u32);
    ClearBufferfi(u32, i32, f32, i32);
    CreateShader(u32) -> u32;
    ShaderSource(u32, i32, *const *const c_char, *const i32);
    CompileShader(u32);
    GetShaderiv(u32, u32, *mut i32);
    GetShaderInfoLog(u32, i32, *mut i32, *mut c_char);
    DeleteShader(u32);
    CreateProgram() -> u32;
    AttachShader(u32, u32);
    LinkProgram(u32);
    GetProgramiv(u32, u32, *mut i32);
    GetProgramInfoLog(u32, i32, *mut i32, *mut c_char);
    DeleteProgram(u32);
    UseProgram(u32);
    GetUniformLocation(u32, *const c_char) -> i32;
    GetUniformBlockIndex(u32, *const c_char) -> u32;
    UniformBlockBinding(u32, u32, u32);
    Uniform1i(i32, i32);
    Uniform1f(i32, f32);
    Uniform4uiv(i32, i32, *const u32);
    GenVertexArrays(i32, *mut u32);
    BindVertexArray(u32);
    EnableVertexAttribArray(u32);
    DisableVertexAttribArray(u32);
    VertexAttribPointer(u32, i32, u32, u8, i32, Ptr);
    VertexAttribIPointer(u32, i32, u32, i32, Ptr);
    VertexAttribDivisor(u32, u32);
    Enable(u32);
    Disable(u32);
    Viewport(i32, i32, i32, i32);
    DepthRangef(f32, f32);
    Scissor(i32, i32, i32, i32);
    ColorMask(u8, u8, u8, u8);
    BlendFuncSeparate(u32, u32, u32, u32);
    BlendEquationSeparate(u32, u32);
    BlendColor(f32, f32, f32, f32);
    DepthFunc(u32);
    DepthMask(u8);
    StencilFuncSeparate(u32, u32, i32, u32);
    StencilOpSeparate(u32, u32, u32, u32);
    StencilMaskSeparate(u32, u32);
    CullFace(u32);
    FrontFace(u32);
    PolygonOffset(f32, f32);
    LineWidth(f32);
    DrawArraysInstanced(u32, i32, i32, i32);
    DrawElementsInstanced(u32, i32, u32, Ptr, i32);
    Finish();
    GetError() -> u32;
    PixelStorei(u32, i32);
    GetString(u32) -> *const u8;
);

/// See the module docs.
pub struct Egl {
    api: EglFns,
    gl: GlFns,
    display: Ptr,
    context: Ptr,
    name: String,
}

// SAFETY: the EGL display and context are process-wide handles, used only
// under the device's lock and made current on one thread at a time
// (`begin`/`end`).
unsafe impl Send for Egl {}

impl std::fmt::Debug for Egl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Egl")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

fn open(names: &[&str]) -> Option<*mut c_void> {
    names.iter().find_map(|n| {
        let c = CString::new(*n).ok()?;
        // SAFETY: a NUL-terminated library name.
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
        (!h.is_null()).then_some(h)
    })
}

fn sym(lib: *mut c_void, name: &str) -> Ptr {
    let Ok(c) = CString::new(name) else {
        return std::ptr::null();
    };
    // SAFETY: `lib` came from dlopen; the name is NUL-terminated.
    unsafe { dlsym(lib, c.as_ptr()).cast_const() }
}

impl Egl {
    /// Load the host's EGL and GLES and make a surfaceless GLES 3 context.
    ///
    /// # Errors
    /// Why there's no such context on this host.
    pub fn new() -> Result<Self, String> {
        let libegl = open(&["libEGL.so.1", "libEGL.so"]).ok_or("no libEGL on this host")?;
        let libgles =
            open(&["libGLESv2.so.2", "libGLESv2.so"]).ok_or("no libGLESv2 on this host")?;
        let egl = EglFns::load(&|n| sym(libegl, n))?;
        let get_proc = egl.GetProcAddress;
        let proc_addr = |n: &str| -> Ptr {
            let Ok(c) = CString::new(n) else {
                return std::ptr::null();
            };
            // SAFETY: a NUL-terminated name.
            unsafe { get_proc(c.as_ptr()) }
        };
        let gl = GlFns::load(&|n| {
            let p = sym(libgles, n);
            if p.is_null() { proc_addr(n) } else { p }
        })?;
        type GetPlatformDisplay = unsafe extern "C" fn(u32, Ptr, *const i32) -> Ptr;
        type QueryDevices = unsafe extern "C" fn(i32, *mut Ptr, *mut i32) -> u32;
        let gpd = proc_addr("eglGetPlatformDisplayEXT");
        if gpd.is_null() {
            return Err("EGL lacks eglGetPlatformDisplayEXT".into());
        }
        // SAFETY: the address of eglGetPlatformDisplayEXT.
        let gpd: GetPlatformDisplay =
            unsafe { std::mem::transmute::<Ptr, GetPlatformDisplay>(gpd) };
        let mut major = 0;
        let mut minor = 0;
        // SAFETY: plain EGL calls with valid out-pointers.
        let mut display = unsafe {
            gpd(
                EGL_PLATFORM_SURFACELESS_MESA,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if display.is_null()
            || unsafe { (egl.Initialize)(display, &raw mut major, &raw mut minor) } == 0
        {
            let qd = proc_addr("eglQueryDevicesEXT");
            if qd.is_null() {
                return Err("EGL has neither a surfaceless platform nor devices".into());
            }
            // SAFETY: the address of eglQueryDevicesEXT.
            let qd: QueryDevices = unsafe { std::mem::transmute::<Ptr, QueryDevices>(qd) };
            let mut dev: Ptr = std::ptr::null();
            let mut n = 0;
            // SAFETY: room for one device.
            if unsafe { qd(1, &raw mut dev, &raw mut n) } == 0 || n == 0 {
                return Err("no EGL device".into());
            }
            // SAFETY: as above.
            display = unsafe { gpd(EGL_PLATFORM_DEVICE_EXT, dev, std::ptr::null()) };
            if display.is_null()
                || unsafe { (egl.Initialize)(display, &raw mut major, &raw mut minor) } == 0
            {
                // SAFETY: no arguments.
                return Err(format!("eglInitialize failed: 0x{:x}", unsafe {
                    (egl.GetError)()
                }));
            }
        }
        // SAFETY: an initialized display; attribute lists end in EGL_NONE.
        let context = unsafe {
            (egl.BindAPI)(EGL_OPENGL_ES_API);
            let attrs = [EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE];
            let mut config: Ptr = std::ptr::null();
            let mut n = 0;
            (egl.ChooseConfig)(display, attrs.as_ptr(), &raw mut config, 1, &raw mut n);
            let cattrs = [EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE];
            (egl.CreateContext)(
                display,
                if n > 0 { config } else { std::ptr::null() },
                std::ptr::null(),
                cattrs.as_ptr(),
            )
        };
        if context.is_null() {
            // SAFETY: no arguments.
            return Err(format!("no GLES 3 context: 0x{:x}", unsafe {
                (egl.GetError)()
            }));
        }
        let mut me = Self {
            api: egl,
            gl,
            display,
            context,
            name: String::new(),
        };
        me.begin();
        // SAFETY: a current context; GetString returns static strings.
        unsafe {
            (me.gl.PixelStorei)(GL_PACK_ALIGNMENT, 1);
            (me.gl.PixelStorei)(GL_UNPACK_ALIGNMENT, 1);
            let s = |p: *const u8| {
                if p.is_null() {
                    String::new()
                } else {
                    std::ffi::CStr::from_ptr(p.cast())
                        .to_string_lossy()
                        .into_owned()
                }
            };
            let vendor = s((me.api.QueryString)(display, EGL_VENDOR).cast());
            me.name = format!(
                "{} ({}, EGL {major}.{minor} {vendor})",
                s((me.gl.GetString)(GL_RENDERER)),
                s((me.gl.GetString)(GL_VERSION))
            );
        }
        me.end();
        Ok(me)
    }

    fn compile(&mut self, kind: u32, src: &str) -> Result<u32, String> {
        let c = CString::new(src).map_err(|e| e.to_string())?;
        // SAFETY: a current context; the source is NUL-terminated.
        unsafe {
            let s = (self.gl.CreateShader)(kind);
            let p = c.as_ptr();
            (self.gl.ShaderSource)(s, 1, &raw const p, std::ptr::null());
            (self.gl.CompileShader)(s);
            let mut ok = 0;
            (self.gl.GetShaderiv)(s, GL_COMPILE_STATUS, &raw mut ok);
            if ok == 0 {
                let mut log = vec![0u8; 4096];
                let mut n = 0;
                (self.gl.GetShaderInfoLog)(
                    s,
                    log.len() as i32,
                    &raw mut n,
                    log.as_mut_ptr().cast(),
                );
                (self.gl.DeleteShader)(s);
                return Err(String::from_utf8_lossy(&log[..n.max(0) as usize]).into_owned());
            }
            Ok(s)
        }
    }
}

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap_or_default()
}

// SAFETY (whole impl): each method is one GL call on the current context
// with scalar arguments or pointers into Rust slices of the size the call
// reads or writes.
#[allow(clippy::cast_possible_wrap)]
impl GpuBackend for Egl {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn begin(&mut self) {
        unsafe {
            (self.api.MakeCurrent)(
                self.display,
                std::ptr::null(),
                std::ptr::null(),
                self.context,
            );
        }
    }
    fn end(&mut self) {
        unsafe {
            (self.api.MakeCurrent)(
                self.display,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
            );
        }
    }
    fn create_buffer(&mut self) -> u32 {
        let mut b = 0;
        unsafe { (self.gl.GenBuffers)(1, &raw mut b) };
        b
    }
    fn delete_buffer(&mut self, buffer: u32) {
        unsafe { (self.gl.DeleteBuffers)(1, &raw const buffer) };
    }
    fn bind_buffer(&mut self, target: u32, buffer: u32) {
        unsafe { (self.gl.BindBuffer)(target, buffer) };
    }
    fn buffer_data(&mut self, target: u32, size: usize, data: Option<&[u8]>, usage: u32) {
        let p = data.map_or(std::ptr::null(), |d| d.as_ptr().cast());
        unsafe { (self.gl.BufferData)(target, size as isize, p, usage) };
    }
    fn buffer_sub_data(&mut self, target: u32, offset: usize, data: &[u8]) {
        unsafe {
            (self.gl.BufferSubData)(
                target,
                offset as isize,
                data.len() as isize,
                data.as_ptr().cast(),
            );
        }
    }
    fn bind_buffer_range(
        &mut self,
        target: u32,
        index: u32,
        buffer: u32,
        offset: usize,
        size: usize,
    ) {
        unsafe { (self.gl.BindBufferRange)(target, index, buffer, offset as isize, size as isize) };
    }
    fn create_texture(&mut self) -> u32 {
        let mut t = 0;
        unsafe { (self.gl.GenTextures)(1, &raw mut t) };
        t
    }
    fn delete_texture(&mut self, texture: u32) {
        unsafe { (self.gl.DeleteTextures)(1, &raw const texture) };
    }
    fn active_texture(&mut self, unit: u32) {
        unsafe { (self.gl.ActiveTexture)(unit) };
    }
    fn bind_texture(&mut self, target: u32, texture: u32) {
        unsafe { (self.gl.BindTexture)(target, texture) };
    }
    fn tex_storage(
        &mut self,
        target: u32,
        levels: u32,
        internal: u32,
        width: u32,
        height: u32,
        depth: u32,
    ) {
        unsafe {
            if depth == 0 {
                (self.gl.TexStorage2D)(
                    target,
                    levels as i32,
                    internal,
                    width as i32,
                    height as i32,
                );
            } else {
                (self.gl.TexStorage3D)(
                    target,
                    levels as i32,
                    internal,
                    width as i32,
                    height as i32,
                    depth as i32,
                );
            }
        }
    }
    fn tex_sub_image(
        &mut self,
        target: u32,
        level: u32,
        x: u32,
        y: u32,
        z: u32,
        width: u32,
        height: u32,
        depth: u32,
        format: u32,
        ty: u32,
        data: &[u8],
    ) {
        unsafe {
            if depth == 0 {
                (self.gl.TexSubImage2D)(
                    target,
                    level as i32,
                    x as i32,
                    y as i32,
                    width as i32,
                    height as i32,
                    format,
                    ty,
                    data.as_ptr().cast(),
                );
            } else {
                (self.gl.TexSubImage3D)(
                    target,
                    level as i32,
                    x as i32,
                    y as i32,
                    z as i32,
                    width as i32,
                    height as i32,
                    depth as i32,
                    format,
                    ty,
                    data.as_ptr().cast(),
                );
            }
        }
    }
    fn tex_parameter(&mut self, target: u32, pname: u32, value: i32) {
        unsafe { (self.gl.TexParameteri)(target, pname, value) };
    }
    fn create_sampler(&mut self) -> u32 {
        let mut s = 0;
        unsafe { (self.gl.GenSamplers)(1, &raw mut s) };
        s
    }
    fn delete_sampler(&mut self, sampler: u32) {
        unsafe { (self.gl.DeleteSamplers)(1, &raw const sampler) };
    }
    fn sampler_parameter_i(&mut self, sampler: u32, pname: u32, value: i32) {
        unsafe { (self.gl.SamplerParameteri)(sampler, pname, value) };
    }
    fn sampler_parameter_f(&mut self, sampler: u32, pname: u32, value: f32) {
        unsafe { (self.gl.SamplerParameterf)(sampler, pname, value) };
    }
    fn bind_sampler(&mut self, unit: u32, sampler: u32) {
        unsafe { (self.gl.BindSampler)(unit, sampler) };
    }
    fn create_framebuffer(&mut self) -> u32 {
        let mut f = 0;
        unsafe { (self.gl.GenFramebuffers)(1, &raw mut f) };
        f
    }
    fn delete_framebuffer(&mut self, fb: u32) {
        unsafe { (self.gl.DeleteFramebuffers)(1, &raw const fb) };
    }
    fn bind_framebuffer(&mut self, target: u32, fb: u32) {
        unsafe { (self.gl.BindFramebuffer)(target, fb) };
    }
    fn framebuffer_texture(
        &mut self,
        target: u32,
        attachment: u32,
        textarget: u32,
        texture: u32,
        level: u32,
        layer: Option<u32>,
    ) {
        unsafe {
            match layer {
                None => (self.gl.FramebufferTexture2D)(
                    target,
                    attachment,
                    textarget,
                    texture,
                    level as i32,
                ),
                Some(l) => (self.gl.FramebufferTextureLayer)(
                    target,
                    attachment,
                    texture,
                    level as i32,
                    l as i32,
                ),
            }
        }
    }
    fn draw_buffers(&mut self, buffers: &[u32]) {
        unsafe { (self.gl.DrawBuffers)(buffers.len() as i32, buffers.as_ptr()) };
    }
    fn read_buffer(&mut self, buffer: u32) {
        unsafe { (self.gl.ReadBuffer)(buffer) };
    }
    fn check_framebuffer_status(&mut self, target: u32) -> u32 {
        unsafe { (self.gl.CheckFramebufferStatus)(target) }
    }
    fn read_pixels(
        &mut self,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        format: u32,
        ty: u32,
        out: &mut [u8],
    ) {
        unsafe {
            (self.gl.ReadPixels)(
                x,
                y,
                width as i32,
                height as i32,
                format,
                ty,
                out.as_mut_ptr().cast(),
            );
        }
    }
    fn blit_framebuffer(&mut self, s: [i32; 4], d: [i32; 4], mask: u32, filter: u32) {
        unsafe {
            (self.gl.BlitFramebuffer)(s[0], s[1], s[2], s[3], d[0], d[1], d[2], d[3], mask, filter);
        }
    }
    fn clear_buffer_fv(&mut self, buffer: u32, drawbuffer: u32, value: [f32; 4]) {
        unsafe { (self.gl.ClearBufferfv)(buffer, drawbuffer as i32, value.as_ptr()) };
    }
    fn clear_buffer_iv(&mut self, buffer: u32, drawbuffer: u32, value: [i32; 4]) {
        unsafe { (self.gl.ClearBufferiv)(buffer, drawbuffer as i32, value.as_ptr()) };
    }
    fn clear_buffer_uiv(&mut self, buffer: u32, drawbuffer: u32, value: [u32; 4]) {
        unsafe { (self.gl.ClearBufferuiv)(buffer, drawbuffer as i32, value.as_ptr()) };
    }
    fn clear_buffer_fi(&mut self, depth: f32, stencil: i32) {
        unsafe { (self.gl.ClearBufferfi)(super::gl::DEPTH_STENCIL, 0, depth, stencil) };
    }
    fn create_program(&mut self, vs: &str, fs: &str) -> Result<u32, String> {
        let v = self
            .compile(GL_VERTEX_SHADER, vs)
            .map_err(|e| format!("vertex: {e}"))?;
        let f = match self.compile(GL_FRAGMENT_SHADER, fs) {
            Ok(f) => f,
            Err(e) => {
                unsafe { (self.gl.DeleteShader)(v) };
                return Err(format!("fragment: {e}"));
            }
        };
        unsafe {
            let p = (self.gl.CreateProgram)();
            (self.gl.AttachShader)(p, v);
            (self.gl.AttachShader)(p, f);
            (self.gl.LinkProgram)(p);
            (self.gl.DeleteShader)(v);
            (self.gl.DeleteShader)(f);
            let mut ok = 0;
            (self.gl.GetProgramiv)(p, GL_LINK_STATUS, &raw mut ok);
            if ok == 0 {
                let mut log = vec![0u8; 4096];
                let mut n = 0;
                (self.gl.GetProgramInfoLog)(
                    p,
                    log.len() as i32,
                    &raw mut n,
                    log.as_mut_ptr().cast(),
                );
                (self.gl.DeleteProgram)(p);
                return Err(format!(
                    "link: {}",
                    String::from_utf8_lossy(&log[..n.max(0) as usize])
                ));
            }
            Ok(p)
        }
    }
    fn delete_program(&mut self, program: u32) {
        unsafe { (self.gl.DeleteProgram)(program) };
    }
    fn use_program(&mut self, program: u32) {
        unsafe { (self.gl.UseProgram)(program) };
    }
    fn uniform_location(&mut self, program: u32, name: &str) -> i32 {
        let c = cstr(name);
        unsafe { (self.gl.GetUniformLocation)(program, c.as_ptr()) }
    }
    fn uniform_block_index(&mut self, program: u32, name: &str) -> u32 {
        let c = cstr(name);
        unsafe { (self.gl.GetUniformBlockIndex)(program, c.as_ptr()) }
    }
    fn uniform_block_binding(&mut self, program: u32, index: u32, binding: u32) {
        unsafe { (self.gl.UniformBlockBinding)(program, index, binding) };
    }
    fn uniform_1i(&mut self, location: i32, value: i32) {
        unsafe { (self.gl.Uniform1i)(location, value) };
    }
    fn uniform_1f(&mut self, location: i32, value: f32) {
        unsafe { (self.gl.Uniform1f)(location, value) };
    }
    fn uniform_4uiv(&mut self, location: i32, values: &[u32]) {
        unsafe { (self.gl.Uniform4uiv)(location, (values.len() / 4) as i32, values.as_ptr()) };
    }
    fn create_vertex_array(&mut self) -> u32 {
        let mut v = 0;
        unsafe { (self.gl.GenVertexArrays)(1, &raw mut v) };
        v
    }
    fn bind_vertex_array(&mut self, vao: u32) {
        unsafe { (self.gl.BindVertexArray)(vao) };
    }
    fn enable_vertex_attrib_array(&mut self, index: u32, on: bool) {
        unsafe {
            if on {
                (self.gl.EnableVertexAttribArray)(index);
            } else {
                (self.gl.DisableVertexAttribArray)(index);
            }
        }
    }
    fn vertex_attrib_pointer(
        &mut self,
        index: u32,
        size: u32,
        ty: u32,
        normalized: bool,
        integer: bool,
        stride: u32,
        offset: usize,
    ) {
        unsafe {
            if integer {
                (self.gl.VertexAttribIPointer)(
                    index,
                    size as i32,
                    ty,
                    stride as i32,
                    offset as Ptr,
                );
            } else {
                (self.gl.VertexAttribPointer)(
                    index,
                    size as i32,
                    ty,
                    u8::from(normalized),
                    stride as i32,
                    offset as Ptr,
                );
            }
        }
    }
    fn vertex_attrib_divisor(&mut self, index: u32, divisor: u32) {
        unsafe { (self.gl.VertexAttribDivisor)(index, divisor) };
    }
    fn enable(&mut self, cap: u32, on: bool) {
        unsafe {
            if on {
                (self.gl.Enable)(cap);
            } else {
                (self.gl.Disable)(cap);
            }
        }
    }
    fn viewport(&mut self, x: i32, y: i32, width: i32, height: i32) {
        unsafe { (self.gl.Viewport)(x, y, width, height) };
    }
    fn depth_range(&mut self, near: f32, far: f32) {
        unsafe { (self.gl.DepthRangef)(near, far) };
    }
    fn scissor(&mut self, x: i32, y: i32, width: i32, height: i32) {
        unsafe { (self.gl.Scissor)(x, y, width, height) };
    }
    fn color_mask(&mut self, r: bool, g: bool, b: bool, a: bool) {
        unsafe { (self.gl.ColorMask)(u8::from(r), u8::from(g), u8::from(b), u8::from(a)) };
    }
    fn blend_func_separate(&mut self, src_rgb: u32, dst_rgb: u32, src_a: u32, dst_a: u32) {
        unsafe { (self.gl.BlendFuncSeparate)(src_rgb, dst_rgb, src_a, dst_a) };
    }
    fn blend_equation_separate(&mut self, rgb: u32, alpha: u32) {
        unsafe { (self.gl.BlendEquationSeparate)(rgb, alpha) };
    }
    fn blend_color(&mut self, c: [f32; 4]) {
        unsafe { (self.gl.BlendColor)(c[0], c[1], c[2], c[3]) };
    }
    fn depth_func(&mut self, func: u32) {
        unsafe { (self.gl.DepthFunc)(func) };
    }
    fn depth_mask(&mut self, on: bool) {
        unsafe { (self.gl.DepthMask)(u8::from(on)) };
    }
    fn stencil_func_separate(&mut self, face: u32, func: u32, reference: i32, mask: u32) {
        unsafe { (self.gl.StencilFuncSeparate)(face, func, reference, mask) };
    }
    fn stencil_op_separate(&mut self, face: u32, fail: u32, zfail: u32, zpass: u32) {
        unsafe { (self.gl.StencilOpSeparate)(face, fail, zfail, zpass) };
    }
    fn stencil_mask_separate(&mut self, face: u32, mask: u32) {
        unsafe { (self.gl.StencilMaskSeparate)(face, mask) };
    }
    fn cull_face(&mut self, face: u32) {
        unsafe { (self.gl.CullFace)(face) };
    }
    fn front_face(&mut self, mode: u32) {
        unsafe { (self.gl.FrontFace)(mode) };
    }
    fn polygon_offset(&mut self, factor: f32, units: f32) {
        unsafe { (self.gl.PolygonOffset)(factor, units) };
    }
    fn line_width(&mut self, width: f32) {
        unsafe { (self.gl.LineWidth)(width) };
    }
    fn draw_arrays(&mut self, mode: u32, first: i32, count: i32, instances: i32) {
        unsafe { (self.gl.DrawArraysInstanced)(mode, first, count, instances) };
    }
    fn draw_elements(&mut self, mode: u32, count: i32, ty: u32, offset: usize, instances: i32) {
        unsafe { (self.gl.DrawElementsInstanced)(mode, count, ty, offset as Ptr, instances) };
    }
    fn finish(&mut self) {
        unsafe { (self.gl.Finish)() };
    }
    fn get_error(&mut self) -> u32 {
        unsafe { (self.gl.GetError)() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::gl;

    /// Render with the host's GL when it has one (skipped, and said so,
    /// when it doesn't): clear an RGBA8 texture and read it back.
    #[test]
    fn host_gles_clears_and_reads_back() {
        let mut e = match Egl::new() {
            Ok(e) => e,
            Err(err) => {
                eprintln!("skipping: no host EGL ({err})");
                return;
            }
        };
        e.begin();
        let t = e.create_texture();
        e.bind_texture(gl::TEXTURE_2D, t);
        e.tex_storage(gl::TEXTURE_2D, 1, gl::RGBA8, 4, 4, 0);
        let f = e.create_framebuffer();
        e.bind_framebuffer(gl::FRAMEBUFFER, f);
        e.framebuffer_texture(
            gl::FRAMEBUFFER,
            gl::COLOR_ATTACHMENT0,
            gl::TEXTURE_2D,
            t,
            0,
            None,
        );
        assert_eq!(
            e.check_framebuffer_status(gl::FRAMEBUFFER),
            gl::FRAMEBUFFER_COMPLETE
        );
        e.clear_buffer_fv(gl::COLOR, 0, [1.0, 0.5, 0.0, 1.0]);
        let mut px = [0u8; 4 * 4 * 4];
        e.read_pixels(0, 0, 4, 4, gl::RGBA, gl::UNSIGNED_BYTE, &mut px);
        e.end();
        // 0.5 may round either way.
        assert!(
            px[0] == 255 && px[1].abs_diff(128) <= 1 && px[2] == 0 && px[3] == 255,
            "{:?} from {}",
            &px[0..4],
            e.name()
        );
    }
}
