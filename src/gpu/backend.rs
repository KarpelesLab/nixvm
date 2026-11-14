//! [`GpuBackend`]: the GLES 3.0 calls nixvm's renderer makes, which a
//! backend carries out on a real GL — WebGL2 in the browser
//! (`gpu::webgl`, wasm only), host EGL + GLES natively (`gpu::egl`) — or records
//! ([`Recorder`], for tests and `NIXVM_GPU=log`).
//!
//! The surface is GLES 3.0 and nothing else, so anything here maps 1:1 to a
//! `WebGL2RenderingContext` method. Objects are named by `u32` as in GLES (a
//! WebGL backend keeps the JS objects in a table under those names); `0` is
//! "none" everywhere. Enum arguments are GLES values ([`super::gl`]).
//! Translating Gallium state into these calls is the renderer's job
//! ([`super::renderer`]), done once for every backend.

/// The GLES 3.0 calls the renderer makes. See the module docs.
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
pub trait GpuBackend: Send {
    /// What to show as the host renderer (logs, `GL_RENDERER`'s suffix).
    fn name(&self) -> String;
    /// Make the backend's context current on this thread before a batch of
    /// calls (an EGL context is bound to one thread at a time; the guest's
    /// ioctls arrive on whichever CPU thread runs the caller).
    fn begin(&mut self) {}
    /// The batch is over: release the context so another thread may take it.
    fn end(&mut self) {}

    // Buffers.
    fn create_buffer(&mut self) -> u32;
    fn delete_buffer(&mut self, buffer: u32);
    fn bind_buffer(&mut self, target: u32, buffer: u32);
    /// `glBufferData` on the buffer bound to `target`: `size` bytes, from
    /// `data` when given (else undefined contents).
    fn buffer_data(&mut self, target: u32, size: usize, data: Option<&[u8]>, usage: u32);
    fn buffer_sub_data(&mut self, target: u32, offset: usize, data: &[u8]);
    fn bind_buffer_range(
        &mut self,
        target: u32,
        index: u32,
        buffer: u32,
        offset: usize,
        size: usize,
    );

    // Textures and samplers.
    fn create_texture(&mut self) -> u32;
    fn delete_texture(&mut self, texture: u32);
    fn active_texture(&mut self, unit: u32);
    fn bind_texture(&mut self, target: u32, texture: u32);
    /// `glTexStorage2D` (`depth == 0`) or `glTexStorage3D` on the bound
    /// texture of `target`.
    fn tex_storage(
        &mut self,
        target: u32,
        levels: u32,
        internal: u32,
        width: u32,
        height: u32,
        depth: u32,
    );
    /// `glTexSubImage2D` / `3D` on the bound texture: `target` may be a
    /// cube face for 2D uploads; `depth == 0` means a 2D call.
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
    );
    fn tex_parameter(&mut self, target: u32, pname: u32, value: i32);
    fn create_sampler(&mut self) -> u32;
    fn delete_sampler(&mut self, sampler: u32);
    fn sampler_parameter_i(&mut self, sampler: u32, pname: u32, value: i32);
    fn sampler_parameter_f(&mut self, sampler: u32, pname: u32, value: f32);
    fn bind_sampler(&mut self, unit: u32, sampler: u32);

    // Framebuffers.
    fn create_framebuffer(&mut self) -> u32;
    fn delete_framebuffer(&mut self, fb: u32);
    fn bind_framebuffer(&mut self, target: u32, fb: u32);
    /// `glFramebufferTexture2D` (`layer == None`) or
    /// `glFramebufferTextureLayer` on the framebuffer bound to `target`.
    fn framebuffer_texture(
        &mut self,
        target: u32,
        attachment: u32,
        textarget: u32,
        texture: u32,
        level: u32,
        layer: Option<u32>,
    );
    fn draw_buffers(&mut self, buffers: &[u32]);
    fn read_buffer(&mut self, buffer: u32);
    fn check_framebuffer_status(&mut self, target: u32) -> u32;
    /// `glReadPixels` from the bound read framebuffer into `out`.
    fn read_pixels(
        &mut self,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        format: u32,
        ty: u32,
        out: &mut [u8],
    );
    fn blit_framebuffer(&mut self, src: [i32; 4], dst: [i32; 4], mask: u32, filter: u32);
    fn clear_buffer_fv(&mut self, buffer: u32, drawbuffer: u32, value: [f32; 4]);
    fn clear_buffer_iv(&mut self, buffer: u32, drawbuffer: u32, value: [i32; 4]);
    fn clear_buffer_uiv(&mut self, buffer: u32, drawbuffer: u32, value: [u32; 4]);
    fn clear_buffer_fi(&mut self, depth: f32, stencil: i32);

    // Programs.
    /// Compile and link a program from GLSL ES 3.00 sources; the info log
    /// on failure.
    fn create_program(&mut self, vs: &str, fs: &str) -> Result<u32, String>;
    fn delete_program(&mut self, program: u32);
    fn use_program(&mut self, program: u32);
    /// A uniform's location in `program`, `-1` if it has none (unused or
    /// absent). Backends without integer locations map them to a table.
    fn uniform_location(&mut self, program: u32, name: &str) -> i32;
    fn uniform_block_index(&mut self, program: u32, name: &str) -> u32;
    fn uniform_block_binding(&mut self, program: u32, index: u32, binding: u32);
    fn uniform_1i(&mut self, location: i32, value: i32);
    fn uniform_1f(&mut self, location: i32, value: f32);
    fn uniform_4uiv(&mut self, location: i32, values: &[u32]);

    // Vertex input.
    fn create_vertex_array(&mut self) -> u32;
    fn bind_vertex_array(&mut self, vao: u32);
    fn enable_vertex_attrib_array(&mut self, index: u32, on: bool);
    /// `glVertexAttribPointer` (or `IPointer` when `integer`) against the
    /// buffer bound to `ARRAY_BUFFER`.
    fn vertex_attrib_pointer(
        &mut self,
        index: u32,
        size: u32,
        ty: u32,
        normalized: bool,
        integer: bool,
        stride: u32,
        offset: usize,
    );
    fn vertex_attrib_divisor(&mut self, index: u32, divisor: u32);

    // Fixed-function state.
    fn enable(&mut self, cap: u32, on: bool);
    fn viewport(&mut self, x: i32, y: i32, width: i32, height: i32);
    fn depth_range(&mut self, near: f32, far: f32);
    fn scissor(&mut self, x: i32, y: i32, width: i32, height: i32);
    fn color_mask(&mut self, r: bool, g: bool, b: bool, a: bool);
    fn blend_func_separate(&mut self, src_rgb: u32, dst_rgb: u32, src_a: u32, dst_a: u32);
    fn blend_equation_separate(&mut self, rgb: u32, alpha: u32);
    fn blend_color(&mut self, rgba: [f32; 4]);
    fn depth_func(&mut self, func: u32);
    fn depth_mask(&mut self, on: bool);
    fn stencil_func_separate(&mut self, face: u32, func: u32, reference: i32, mask: u32);
    fn stencil_op_separate(&mut self, face: u32, fail: u32, zfail: u32, zpass: u32);
    fn stencil_mask_separate(&mut self, face: u32, mask: u32);
    fn cull_face(&mut self, face: u32);
    fn front_face(&mut self, mode: u32);
    fn polygon_offset(&mut self, factor: f32, units: f32);
    fn line_width(&mut self, width: f32);

    // Drawing.
    fn draw_arrays(&mut self, mode: u32, first: i32, count: i32, instances: i32);
    fn draw_elements(&mut self, mode: u32, count: i32, ty: u32, offset: usize, instances: i32);
    fn finish(&mut self);
    fn get_error(&mut self) -> u32;
}

/// A backend that only records the calls made to it, by name, and makes up
/// object names: the decoder's and renderer's tests watch it, and
/// `NIXVM_GPU=log` runs a guest against it to see what reaches the host.
#[derive(Debug, Default)]
pub struct Recorder {
    /// Every call, rendered like `draw_arrays(4, 0, 3, 1)`. Shared, so a
    /// test keeps a handle after the device takes the recorder.
    pub calls: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    /// Programs created: their (vertex, fragment) sources.
    pub programs: std::sync::Arc<std::sync::Mutex<Vec<(String, String)>>>,
    /// Print each call to stderr as well.
    pub echo: bool,
    next: u32,
}

impl Recorder {
    #[must_use]
    pub fn new(echo: bool) -> Self {
        Self {
            echo,
            ..Self::default()
        }
    }

    fn log(&mut self, s: String) {
        if self.echo {
            eprintln!("[gpu] {s}");
        }
        self.calls.lock().unwrap().push(s);
    }

    fn name_obj(&mut self, what: &str) -> u32 {
        self.next += 1;
        let n = self.next;
        self.log(format!("{what}() = {n}"));
        n
    }
}

macro_rules! rec {
    ($self:ident, $name:literal $(, $a:expr)*) => {{
        let args: Vec<String> = vec![$(format!("{:?}", $a)),*];
        $self.log(format!("{}({})", $name, args.join(", ")));
    }};
}

impl GpuBackend for Recorder {
    fn name(&self) -> String {
        "recorder".into()
    }
    fn create_buffer(&mut self) -> u32 {
        self.name_obj("create_buffer")
    }
    fn delete_buffer(&mut self, buffer: u32) {
        rec!(self, "delete_buffer", buffer);
    }
    fn bind_buffer(&mut self, target: u32, buffer: u32) {
        rec!(self, "bind_buffer", target, buffer);
    }
    fn buffer_data(&mut self, target: u32, size: usize, data: Option<&[u8]>, usage: u32) {
        rec!(
            self,
            "buffer_data",
            target,
            size,
            data.map(<[u8]>::len),
            usage
        );
    }
    fn buffer_sub_data(&mut self, target: u32, offset: usize, data: &[u8]) {
        rec!(self, "buffer_sub_data", target, offset, data.len());
    }
    fn bind_buffer_range(
        &mut self,
        target: u32,
        index: u32,
        buffer: u32,
        offset: usize,
        size: usize,
    ) {
        rec!(
            self,
            "bind_buffer_range",
            target,
            index,
            buffer,
            offset,
            size
        );
    }
    fn create_texture(&mut self) -> u32 {
        self.name_obj("create_texture")
    }
    fn delete_texture(&mut self, texture: u32) {
        rec!(self, "delete_texture", texture);
    }
    fn active_texture(&mut self, unit: u32) {
        rec!(self, "active_texture", unit);
    }
    fn bind_texture(&mut self, target: u32, texture: u32) {
        rec!(self, "bind_texture", target, texture);
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
        rec!(
            self,
            "tex_storage",
            target,
            levels,
            internal,
            width,
            height,
            depth
        );
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
        rec!(
            self,
            "tex_sub_image",
            target,
            level,
            x,
            y,
            z,
            width,
            height,
            depth,
            format,
            ty,
            data.len()
        );
    }
    fn tex_parameter(&mut self, target: u32, pname: u32, value: i32) {
        rec!(self, "tex_parameter", target, pname, value);
    }
    fn create_sampler(&mut self) -> u32 {
        self.name_obj("create_sampler")
    }
    fn delete_sampler(&mut self, sampler: u32) {
        rec!(self, "delete_sampler", sampler);
    }
    fn sampler_parameter_i(&mut self, sampler: u32, pname: u32, value: i32) {
        rec!(self, "sampler_parameter_i", sampler, pname, value);
    }
    fn sampler_parameter_f(&mut self, sampler: u32, pname: u32, value: f32) {
        rec!(self, "sampler_parameter_f", sampler, pname, value);
    }
    fn bind_sampler(&mut self, unit: u32, sampler: u32) {
        rec!(self, "bind_sampler", unit, sampler);
    }
    fn create_framebuffer(&mut self) -> u32 {
        self.name_obj("create_framebuffer")
    }
    fn delete_framebuffer(&mut self, fb: u32) {
        rec!(self, "delete_framebuffer", fb);
    }
    fn bind_framebuffer(&mut self, target: u32, fb: u32) {
        rec!(self, "bind_framebuffer", target, fb);
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
        rec!(
            self,
            "framebuffer_texture",
            target,
            attachment,
            textarget,
            texture,
            level,
            layer
        );
    }
    fn draw_buffers(&mut self, buffers: &[u32]) {
        rec!(self, "draw_buffers", buffers);
    }
    fn read_buffer(&mut self, buffer: u32) {
        rec!(self, "read_buffer", buffer);
    }
    fn check_framebuffer_status(&mut self, target: u32) -> u32 {
        rec!(self, "check_framebuffer_status", target);
        super::gl::FRAMEBUFFER_COMPLETE
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
        rec!(
            self,
            "read_pixels",
            x,
            y,
            width,
            height,
            format,
            ty,
            out.len()
        );
    }
    fn blit_framebuffer(&mut self, src: [i32; 4], dst: [i32; 4], mask: u32, filter: u32) {
        rec!(self, "blit_framebuffer", src, dst, mask, filter);
    }
    fn clear_buffer_fv(&mut self, buffer: u32, drawbuffer: u32, value: [f32; 4]) {
        rec!(self, "clear_buffer_fv", buffer, drawbuffer, value);
    }
    fn clear_buffer_iv(&mut self, buffer: u32, drawbuffer: u32, value: [i32; 4]) {
        rec!(self, "clear_buffer_iv", buffer, drawbuffer, value);
    }
    fn clear_buffer_uiv(&mut self, buffer: u32, drawbuffer: u32, value: [u32; 4]) {
        rec!(self, "clear_buffer_uiv", buffer, drawbuffer, value);
    }
    fn clear_buffer_fi(&mut self, depth: f32, stencil: i32) {
        rec!(self, "clear_buffer_fi", depth, stencil);
    }
    fn create_program(&mut self, vs: &str, fs: &str) -> Result<u32, String> {
        self.programs
            .lock()
            .unwrap()
            .push((vs.to_string(), fs.to_string()));
        if self.echo {
            eprintln!("[gpu] vertex shader:\n{vs}\n[gpu] fragment shader:\n{fs}");
        }
        Ok(self.name_obj("create_program"))
    }
    fn delete_program(&mut self, program: u32) {
        rec!(self, "delete_program", program);
    }
    fn use_program(&mut self, program: u32) {
        rec!(self, "use_program", program);
    }
    fn uniform_location(&mut self, program: u32, name: &str) -> i32 {
        rec!(self, "uniform_location", program, name);
        // Distinct per name, so tests can tell uniforms apart.
        name.bytes()
            .fold(7i32, |h, b| h.wrapping_mul(31).wrapping_add(i32::from(b)))
            & 0xffff
    }
    fn uniform_block_index(&mut self, program: u32, name: &str) -> u32 {
        rec!(self, "uniform_block_index", program, name);
        0
    }
    fn uniform_block_binding(&mut self, program: u32, index: u32, binding: u32) {
        rec!(self, "uniform_block_binding", program, index, binding);
    }
    fn uniform_1i(&mut self, location: i32, value: i32) {
        rec!(self, "uniform_1i", location, value);
    }
    fn uniform_1f(&mut self, location: i32, value: f32) {
        rec!(self, "uniform_1f", location, value);
    }
    fn uniform_4uiv(&mut self, location: i32, values: &[u32]) {
        rec!(self, "uniform_4uiv", location, values.len());
    }
    fn create_vertex_array(&mut self) -> u32 {
        self.name_obj("create_vertex_array")
    }
    fn bind_vertex_array(&mut self, vao: u32) {
        rec!(self, "bind_vertex_array", vao);
    }
    fn enable_vertex_attrib_array(&mut self, index: u32, on: bool) {
        rec!(self, "enable_vertex_attrib_array", index, on);
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
        rec!(
            self,
            "vertex_attrib_pointer",
            index,
            size,
            ty,
            normalized,
            integer,
            stride,
            offset
        );
    }
    fn vertex_attrib_divisor(&mut self, index: u32, divisor: u32) {
        rec!(self, "vertex_attrib_divisor", index, divisor);
    }
    fn enable(&mut self, cap: u32, on: bool) {
        rec!(self, "enable", cap, on);
    }
    fn viewport(&mut self, x: i32, y: i32, width: i32, height: i32) {
        rec!(self, "viewport", x, y, width, height);
    }
    fn depth_range(&mut self, near: f32, far: f32) {
        rec!(self, "depth_range", near, far);
    }
    fn scissor(&mut self, x: i32, y: i32, width: i32, height: i32) {
        rec!(self, "scissor", x, y, width, height);
    }
    fn color_mask(&mut self, r: bool, g: bool, b: bool, a: bool) {
        rec!(self, "color_mask", r, g, b, a);
    }
    fn blend_func_separate(&mut self, src_rgb: u32, dst_rgb: u32, src_a: u32, dst_a: u32) {
        rec!(self, "blend_func_separate", src_rgb, dst_rgb, src_a, dst_a);
    }
    fn blend_equation_separate(&mut self, rgb: u32, alpha: u32) {
        rec!(self, "blend_equation_separate", rgb, alpha);
    }
    fn blend_color(&mut self, rgba: [f32; 4]) {
        rec!(self, "blend_color", rgba);
    }
    fn depth_func(&mut self, func: u32) {
        rec!(self, "depth_func", func);
    }
    fn depth_mask(&mut self, on: bool) {
        rec!(self, "depth_mask", on);
    }
    fn stencil_func_separate(&mut self, face: u32, func: u32, reference: i32, mask: u32) {
        rec!(self, "stencil_func_separate", face, func, reference, mask);
    }
    fn stencil_op_separate(&mut self, face: u32, fail: u32, zfail: u32, zpass: u32) {
        rec!(self, "stencil_op_separate", face, fail, zfail, zpass);
    }
    fn stencil_mask_separate(&mut self, face: u32, mask: u32) {
        rec!(self, "stencil_mask_separate", face, mask);
    }
    fn cull_face(&mut self, face: u32) {
        rec!(self, "cull_face", face);
    }
    fn front_face(&mut self, mode: u32) {
        rec!(self, "front_face", mode);
    }
    fn polygon_offset(&mut self, factor: f32, units: f32) {
        rec!(self, "polygon_offset", factor, units);
    }
    fn line_width(&mut self, width: f32) {
        rec!(self, "line_width", width);
    }
    fn draw_arrays(&mut self, mode: u32, first: i32, count: i32, instances: i32) {
        rec!(self, "draw_arrays", mode, first, count, instances);
    }
    fn draw_elements(&mut self, mode: u32, count: i32, ty: u32, offset: usize, instances: i32) {
        rec!(self, "draw_elements", mode, count, ty, offset, instances);
    }
    fn finish(&mut self) {
        rec!(self, "finish");
    }
    fn get_error(&mut self) -> u32 {
        0
    }
}
