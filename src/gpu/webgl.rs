//! The browser backend: WebGL2, through the JS shim `webgl.js` beside this
//! file (wasm-bindgen ships it with the package as a snippet, so the page
//! needs no glue of its own beyond handing over a context).
//!
//! The embedder passes a `WebGL2RenderingContext` — from an on-screen or
//! offscreen canvas — to [`set_context`] (the wasm `Terminal` does it in
//! `attach_webgl2`); every [`GpuBackend`] call then runs on it, synchronously,
//! on the thread running the guest. GL objects stay in a table on the JS
//! side and cross as integers, so nothing here holds a JS value.

use wasm_bindgen::prelude::*;

use super::GpuBackend;

#[wasm_bindgen(module = "/src/gpu/webgl.js")]
extern "C" {
    fn nx_gl_set_context(gl: JsValue);
    fn nx_gl_has_context() -> bool;
    fn nx_gl_name() -> String;
    fn nx_gl_create_buffer() -> u32;
    fn nx_gl_delete_buffer(b: u32);
    fn nx_gl_bind_buffer(t: u32, b: u32);
    fn nx_gl_buffer_data_size(t: u32, size: f64, usage: u32);
    fn nx_gl_buffer_data(t: u32, data: &[u8], usage: u32);
    fn nx_gl_buffer_sub_data(t: u32, off: f64, data: &[u8]);
    fn nx_gl_bind_buffer_range(t: u32, i: u32, b: u32, off: f64, size: f64);
    fn nx_gl_create_texture() -> u32;
    fn nx_gl_delete_texture(t: u32);
    fn nx_gl_active_texture(u: u32);
    fn nx_gl_bind_texture(t: u32, tex: u32);
    fn nx_gl_tex_storage(t: u32, levels: u32, internal: u32, w: u32, h: u32, d: u32);
    #[allow(clippy::too_many_arguments)]
    fn nx_gl_tex_sub_image(
        t: u32,
        level: u32,
        x: u32,
        y: u32,
        z: u32,
        w: u32,
        h: u32,
        d: u32,
        format: u32,
        ty: u32,
        data: &[u8],
    );
    fn nx_gl_tex_parameter(t: u32, p: u32, v: i32);
    fn nx_gl_create_sampler() -> u32;
    fn nx_gl_delete_sampler(s: u32);
    fn nx_gl_sampler_parameter_i(s: u32, p: u32, v: i32);
    fn nx_gl_sampler_parameter_f(s: u32, p: u32, v: f32);
    fn nx_gl_bind_sampler(u: u32, s: u32);
    fn nx_gl_create_framebuffer() -> u32;
    fn nx_gl_delete_framebuffer(f: u32);
    fn nx_gl_bind_framebuffer(t: u32, f: u32);
    fn nx_gl_framebuffer_texture_2d(t: u32, att: u32, textarget: u32, tex: u32, level: u32);
    fn nx_gl_framebuffer_texture_layer(t: u32, att: u32, tex: u32, level: u32, layer: u32);
    fn nx_gl_draw_buffers(bufs: &[u32]);
    fn nx_gl_read_buffer(b: u32);
    fn nx_gl_check_framebuffer_status(t: u32) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn nx_gl_read_pixels(x: i32, y: i32, w: u32, h: u32, format: u32, ty: u32, out: &mut [u8]);
    fn nx_gl_blit_framebuffer(s: &[i32], d: &[i32], mask: u32, filter: u32);
    fn nx_gl_clear_buffer_fv(b: u32, db: u32, v: &[f32]);
    fn nx_gl_clear_buffer_iv(b: u32, db: u32, v: &[i32]);
    fn nx_gl_clear_buffer_uiv(b: u32, db: u32, v: &[u32]);
    fn nx_gl_clear_buffer_fi(depth: f32, stencil: i32);
    fn nx_gl_create_program(vs: &str, fs: &str) -> u32;
    fn nx_gl_last_log() -> String;
    fn nx_gl_delete_program(p: u32);
    fn nx_gl_use_program(p: u32);
    fn nx_gl_uniform_location(p: u32, name: &str) -> i32;
    fn nx_gl_uniform_block_index(p: u32, name: &str) -> u32;
    fn nx_gl_uniform_block_binding(p: u32, i: u32, b: u32);
    fn nx_gl_uniform_1i(l: i32, v: i32);
    fn nx_gl_uniform_1f(l: i32, v: f32);
    fn nx_gl_uniform_4uiv(l: i32, v: &[u32]);
    fn nx_gl_create_vertex_array() -> u32;
    fn nx_gl_bind_vertex_array(v: u32);
    fn nx_gl_enable_vertex_attrib_array(i: u32, on: bool);
    #[allow(clippy::too_many_arguments)]
    fn nx_gl_vertex_attrib_pointer(
        i: u32,
        size: u32,
        ty: u32,
        norm: bool,
        integer: bool,
        stride: u32,
        off: f64,
    );
    fn nx_gl_vertex_attrib_divisor(i: u32, d: u32);
    fn nx_gl_enable(cap: u32, on: bool);
    fn nx_gl_viewport(x: i32, y: i32, w: i32, h: i32);
    fn nx_gl_depth_range(n: f32, f: f32);
    fn nx_gl_scissor(x: i32, y: i32, w: i32, h: i32);
    fn nx_gl_color_mask(r: bool, g: bool, b: bool, a: bool);
    fn nx_gl_blend_func_separate(a: u32, b: u32, c: u32, d: u32);
    fn nx_gl_blend_equation_separate(a: u32, b: u32);
    fn nx_gl_blend_color(r: f32, g: f32, b: f32, a: f32);
    fn nx_gl_depth_func(f: u32);
    fn nx_gl_depth_mask(on: bool);
    fn nx_gl_stencil_func_separate(face: u32, f: u32, r: i32, m: u32);
    fn nx_gl_stencil_op_separate(face: u32, a: u32, b: u32, c: u32);
    fn nx_gl_stencil_mask_separate(face: u32, m: u32);
    fn nx_gl_cull_face(f: u32);
    fn nx_gl_front_face(m: u32);
    fn nx_gl_polygon_offset(f: f32, u: f32);
    fn nx_gl_line_width(w: f32);
    fn nx_gl_draw_arrays(mode: u32, first: i32, count: i32, instances: i32);
    fn nx_gl_draw_elements(mode: u32, count: i32, ty: u32, off: f64, instances: i32);
    fn nx_gl_finish();
    fn nx_gl_get_error() -> u32;
}

/// Hand the backend the page's `WebGL2RenderingContext`.
pub fn set_context(gl: JsValue) {
    nx_gl_set_context(gl);
}

/// The WebGL2 backend. See the module docs.
#[derive(Debug, Default)]
pub struct WebGl;

impl WebGl {
    /// The backend, once [`set_context`] has given it a context.
    ///
    /// # Errors
    /// If no context has been set.
    pub fn new() -> Result<Self, String> {
        if nx_gl_has_context() {
            Ok(Self)
        } else {
            Err("no WebGL2 context set".into())
        }
    }
}

// Byte offsets and sizes cross as JS numbers (f64): exact up to 2^53.
#[allow(clippy::cast_precision_loss)]
fn n(v: usize) -> f64 {
    v as f64
}

impl GpuBackend for WebGl {
    fn name(&self) -> String {
        nx_gl_name()
    }
    fn create_buffer(&mut self) -> u32 {
        nx_gl_create_buffer()
    }
    fn delete_buffer(&mut self, buffer: u32) {
        nx_gl_delete_buffer(buffer);
    }
    fn bind_buffer(&mut self, target: u32, buffer: u32) {
        nx_gl_bind_buffer(target, buffer);
    }
    fn buffer_data(&mut self, target: u32, size: usize, data: Option<&[u8]>, usage: u32) {
        match data {
            Some(d) => nx_gl_buffer_data(target, d, usage),
            None => nx_gl_buffer_data_size(target, n(size), usage),
        }
    }
    fn buffer_sub_data(&mut self, target: u32, offset: usize, data: &[u8]) {
        nx_gl_buffer_sub_data(target, n(offset), data);
    }
    fn bind_buffer_range(
        &mut self,
        target: u32,
        index: u32,
        buffer: u32,
        offset: usize,
        size: usize,
    ) {
        nx_gl_bind_buffer_range(target, index, buffer, n(offset), n(size));
    }
    fn create_texture(&mut self) -> u32 {
        nx_gl_create_texture()
    }
    fn delete_texture(&mut self, texture: u32) {
        nx_gl_delete_texture(texture);
    }
    fn active_texture(&mut self, unit: u32) {
        nx_gl_active_texture(unit);
    }
    fn bind_texture(&mut self, target: u32, texture: u32) {
        nx_gl_bind_texture(target, texture);
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
        nx_gl_tex_storage(target, levels, internal, width, height, depth);
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
        nx_gl_tex_sub_image(
            target, level, x, y, z, width, height, depth, format, ty, data,
        );
    }
    fn tex_parameter(&mut self, target: u32, pname: u32, value: i32) {
        nx_gl_tex_parameter(target, pname, value);
    }
    fn create_sampler(&mut self) -> u32 {
        nx_gl_create_sampler()
    }
    fn delete_sampler(&mut self, sampler: u32) {
        nx_gl_delete_sampler(sampler);
    }
    fn sampler_parameter_i(&mut self, sampler: u32, pname: u32, value: i32) {
        nx_gl_sampler_parameter_i(sampler, pname, value);
    }
    fn sampler_parameter_f(&mut self, sampler: u32, pname: u32, value: f32) {
        nx_gl_sampler_parameter_f(sampler, pname, value);
    }
    fn bind_sampler(&mut self, unit: u32, sampler: u32) {
        nx_gl_bind_sampler(unit, sampler);
    }
    fn create_framebuffer(&mut self) -> u32 {
        nx_gl_create_framebuffer()
    }
    fn delete_framebuffer(&mut self, fb: u32) {
        nx_gl_delete_framebuffer(fb);
    }
    fn bind_framebuffer(&mut self, target: u32, fb: u32) {
        nx_gl_bind_framebuffer(target, fb);
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
        match layer {
            None => nx_gl_framebuffer_texture_2d(target, attachment, textarget, texture, level),
            Some(l) => nx_gl_framebuffer_texture_layer(target, attachment, texture, level, l),
        }
    }
    fn draw_buffers(&mut self, buffers: &[u32]) {
        nx_gl_draw_buffers(buffers);
    }
    fn read_buffer(&mut self, buffer: u32) {
        nx_gl_read_buffer(buffer);
    }
    fn check_framebuffer_status(&mut self, target: u32) -> u32 {
        nx_gl_check_framebuffer_status(target)
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
        nx_gl_read_pixels(x, y, width, height, format, ty, out);
    }
    fn blit_framebuffer(&mut self, src: [i32; 4], dst: [i32; 4], mask: u32, filter: u32) {
        nx_gl_blit_framebuffer(&src, &dst, mask, filter);
    }
    fn clear_buffer_fv(&mut self, buffer: u32, drawbuffer: u32, value: [f32; 4]) {
        nx_gl_clear_buffer_fv(buffer, drawbuffer, &value);
    }
    fn clear_buffer_iv(&mut self, buffer: u32, drawbuffer: u32, value: [i32; 4]) {
        nx_gl_clear_buffer_iv(buffer, drawbuffer, &value);
    }
    fn clear_buffer_uiv(&mut self, buffer: u32, drawbuffer: u32, value: [u32; 4]) {
        nx_gl_clear_buffer_uiv(buffer, drawbuffer, &value);
    }
    fn clear_buffer_fi(&mut self, depth: f32, stencil: i32) {
        nx_gl_clear_buffer_fi(depth, stencil);
    }
    fn create_program(&mut self, vs: &str, fs: &str) -> Result<u32, String> {
        match nx_gl_create_program(vs, fs) {
            0 => Err(nx_gl_last_log()),
            p => Ok(p),
        }
    }
    fn delete_program(&mut self, program: u32) {
        nx_gl_delete_program(program);
    }
    fn use_program(&mut self, program: u32) {
        nx_gl_use_program(program);
    }
    fn uniform_location(&mut self, program: u32, name: &str) -> i32 {
        nx_gl_uniform_location(program, name)
    }
    fn uniform_block_index(&mut self, program: u32, name: &str) -> u32 {
        nx_gl_uniform_block_index(program, name)
    }
    fn uniform_block_binding(&mut self, program: u32, index: u32, binding: u32) {
        nx_gl_uniform_block_binding(program, index, binding);
    }
    fn uniform_1i(&mut self, location: i32, value: i32) {
        nx_gl_uniform_1i(location, value);
    }
    fn uniform_1f(&mut self, location: i32, value: f32) {
        nx_gl_uniform_1f(location, value);
    }
    fn uniform_4uiv(&mut self, location: i32, values: &[u32]) {
        nx_gl_uniform_4uiv(location, values);
    }
    fn create_vertex_array(&mut self) -> u32 {
        nx_gl_create_vertex_array()
    }
    fn bind_vertex_array(&mut self, vao: u32) {
        nx_gl_bind_vertex_array(vao);
    }
    fn enable_vertex_attrib_array(&mut self, index: u32, on: bool) {
        nx_gl_enable_vertex_attrib_array(index, on);
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
        nx_gl_vertex_attrib_pointer(index, size, ty, normalized, integer, stride, n(offset));
    }
    fn vertex_attrib_divisor(&mut self, index: u32, divisor: u32) {
        nx_gl_vertex_attrib_divisor(index, divisor);
    }
    fn enable(&mut self, cap: u32, on: bool) {
        nx_gl_enable(cap, on);
    }
    fn viewport(&mut self, x: i32, y: i32, width: i32, height: i32) {
        nx_gl_viewport(x, y, width, height);
    }
    fn depth_range(&mut self, near: f32, far: f32) {
        nx_gl_depth_range(near, far);
    }
    fn scissor(&mut self, x: i32, y: i32, width: i32, height: i32) {
        nx_gl_scissor(x, y, width, height);
    }
    fn color_mask(&mut self, r: bool, g: bool, b: bool, a: bool) {
        nx_gl_color_mask(r, g, b, a);
    }
    fn blend_func_separate(&mut self, src_rgb: u32, dst_rgb: u32, src_a: u32, dst_a: u32) {
        nx_gl_blend_func_separate(src_rgb, dst_rgb, src_a, dst_a);
    }
    fn blend_equation_separate(&mut self, rgb: u32, alpha: u32) {
        nx_gl_blend_equation_separate(rgb, alpha);
    }
    fn blend_color(&mut self, c: [f32; 4]) {
        nx_gl_blend_color(c[0], c[1], c[2], c[3]);
    }
    fn depth_func(&mut self, func: u32) {
        nx_gl_depth_func(func);
    }
    fn depth_mask(&mut self, on: bool) {
        nx_gl_depth_mask(on);
    }
    fn stencil_func_separate(&mut self, face: u32, func: u32, reference: i32, mask: u32) {
        nx_gl_stencil_func_separate(face, func, reference, mask);
    }
    fn stencil_op_separate(&mut self, face: u32, fail: u32, zfail: u32, zpass: u32) {
        nx_gl_stencil_op_separate(face, fail, zfail, zpass);
    }
    fn stencil_mask_separate(&mut self, face: u32, mask: u32) {
        nx_gl_stencil_mask_separate(face, mask);
    }
    fn cull_face(&mut self, face: u32) {
        nx_gl_cull_face(face);
    }
    fn front_face(&mut self, mode: u32) {
        nx_gl_front_face(mode);
    }
    fn polygon_offset(&mut self, factor: f32, units: f32) {
        nx_gl_polygon_offset(factor, units);
    }
    fn line_width(&mut self, width: f32) {
        nx_gl_line_width(width);
    }
    fn draw_arrays(&mut self, mode: u32, first: i32, count: i32, instances: i32) {
        nx_gl_draw_arrays(mode, first, count, instances);
    }
    fn draw_elements(&mut self, mode: u32, count: i32, ty: u32, offset: usize, instances: i32) {
        nx_gl_draw_elements(mode, count, ty, n(offset), instances);
    }
    fn finish(&mut self) {
        nx_gl_finish();
    }
    fn get_error(&mut self) -> u32 {
        nx_gl_get_error()
    }
}
