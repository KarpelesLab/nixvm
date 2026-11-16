//! A GL state cache in front of a [`GpuBackend`]: calls that would set
//! state to what it already is are dropped before they reach the host.
//!
//! The renderer applies the whole Gallium state at every draw (simple, and
//! contexts sharing the host's one GL context never see each other's
//! leftovers); this layer makes that cheap. It mirrors what GLES 3.0 keeps:
//! global fixed-function state, the bound program, VAO, framebuffers and
//! buffers, per-unit texture and sampler bindings, per-texture parameters,
//! per-framebuffer attachments and draw/read buffers, per-VAO vertex
//! attributes and per-program uniform values. Everything else passes
//! through untouched.
//!
//! The cache is only right if every GL call goes through it, which holds:
//! the renderer and the display's compositor ([`super::GpuDevice::with_gl`])
//! both get this wrapper. Deleting an object forgets whatever named it
//! (GL reuses names); [`StateCache::invalidate`] forgets everything.

use std::hash::{BuildHasherDefault, Hasher};

use super::GpuBackend;
use super::backend::HostCaps;
use super::gl;

/// The cache sits on every GL call: its maps hash small integer keys
/// with a multiply-xor (FxHash's) rather than SipHash.
#[derive(Default)]
pub(crate) struct FxHasher(u64);

impl Hasher for FxHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.write_u64(u64::from(*b));
        }
    }
    fn write_u8(&mut self, v: u8) {
        self.write_u64(u64::from(v));
    }
    fn write_u32(&mut self, v: u32) {
        self.write_u64(u64::from(v));
    }
    fn write_i32(&mut self, v: i32) {
        self.write_u64(u64::from(v.cast_unsigned()));
    }
    fn write_usize(&mut self, v: usize) {
        self.write_u64(v as u64);
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

/// A `HashMap` hashed with [`FxHasher`].
pub(crate) type FxMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// A framebuffer attachment as last set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Attach {
    /// `(textarget, texture, level, layer)`.
    Texture(u32, u32, u32, Option<u32>),
    Renderbuffer(u32),
}

impl Attach {
    fn names(self, texture: bool, name: u32) -> bool {
        match self {
            Self::Texture(_, t, ..) => texture && t == name,
            Self::Renderbuffer(r) => !texture && r == name,
        }
    }
}

/// A vertex attribute's pointer as last set: the `ARRAY_BUFFER` it read
/// and its layout.
type Pointer = (u32, u32, u32, bool, bool, u32, usize);

/// Blend state of one colour buffer (or all, for the global calls).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct BufBlend {
    on: Option<bool>,
    func: Option<[u32; 4]>,
    eq: Option<[u32; 2]>,
    mask: Option<[bool; 4]>,
}

/// The GL state as last set (`None`/absent: unknown).
#[derive(Default)]
struct State {
    enabled: FxMap<u32, bool>,
    viewport: Option<[i32; 4]>,
    scissor: Option<[i32; 4]>,
    depth_range: Option<[u32; 2]>,
    blend: BufBlend,
    /// Per colour buffer, when set with the indexed calls.
    blend_i: FxMap<u32, BufBlend>,
    blend_color: Option<[u32; 4]>,
    depth_func: Option<u32>,
    depth_mask: Option<bool>,
    /// Per face (0 front, 1 back).
    stencil_func: [Option<(u32, i32, u32)>; 2],
    stencil_op: [Option<(u32, u32, u32)>; 2],
    stencil_mask: [Option<u32>; 2],
    cull_face: Option<u32>,
    front_face: Option<u32>,
    polygon_offset: Option<[u32; 2]>,
    line_width: Option<u32>,
    program: Option<u32>,
    vao: Option<u32>,
    draw_fbo: Option<u32>,
    read_fbo: Option<u32>,
    /// Per (framebuffer, attachment).
    attachments: FxMap<(u32, u32), Attach>,
    draw_buffers: FxMap<u32, Vec<u32>>,
    read_buffers: FxMap<u32, u32>,
    /// Generic buffer bindings by target (not `ELEMENT_ARRAY_BUFFER`,
    /// which belongs to the VAO).
    buffers: FxMap<u32, u32>,
    /// Indexed buffer bindings: (target, index) → (buffer, offset, size).
    ranges: FxMap<(u32, u32), (u32, usize, usize)>,
    active_unit: Option<u32>,
    /// (unit, target) → texture.
    textures: FxMap<(u32, u32), u32>,
    /// (texture, pname) → value.
    tex_params: FxMap<(u32, u32), i32>,
    samplers: FxMap<u32, u32>,
    /// Per (VAO, attribute index).
    attrib_on: FxMap<(u32, u32), bool>,
    attrib_ptr: FxMap<(u32, u32), Pointer>,
    attrib_div: FxMap<(u32, u32), u32>,
    /// (program, location) → the value's words.
    uniforms: FxMap<(u32, i32), Vec<u32>>,
}

/// See the module docs.
pub struct StateCache {
    inner: Box<dyn GpuBackend>,
    s: State,
}

impl std::fmt::Debug for StateCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateCache").finish_non_exhaustive()
    }
}

impl StateCache {
    /// A cache in front of `inner`, knowing nothing yet.
    #[must_use]
    pub fn new(inner: Box<dyn GpuBackend>) -> Self {
        Self {
            inner,
            s: State::default(),
        }
    }

    /// Forget everything: the next call of each kind reaches the host.
    pub fn invalidate(&mut self) {
        self.s = State::default();
    }

    /// The backend behind the cache.
    pub fn inner(&mut self) -> &mut dyn GpuBackend {
        &mut *self.inner
    }

    /// The framebuffer bound at `target`.
    fn fbo(&self, target: u32) -> Option<u32> {
        if target == gl::READ_FRAMEBUFFER {
            self.s.read_fbo
        } else {
            self.s.draw_fbo
        }
    }

    fn faces(face: u32) -> &'static [usize] {
        match face {
            gl::FRONT => &[0],
            gl::BACK => &[1],
            _ => &[0, 1],
        }
    }

    /// Set `slot` to `v`: whether it changed (and the call must be made).
    fn set<T: PartialEq>(slot: &mut Option<T>, v: T) -> bool {
        if slot.as_ref() == Some(&v) {
            false
        } else {
            *slot = Some(v);
            true
        }
    }

    fn uniform(&mut self, location: i32, words: Vec<u32>) -> bool {
        let Some(p) = self.s.program else {
            return true;
        };
        if location < 0 {
            return false;
        }
        match self.s.uniforms.get(&(p, location)) {
            Some(v) if *v == words => false,
            _ => {
                self.s.uniforms.insert((p, location), words);
                true
            }
        }
    }

    /// Forget the attachments naming texture/renderbuffer `name`.
    fn forget_attachments(&mut self, texture: bool, name: u32) {
        self.s.attachments.retain(|_, a| !a.names(texture, name));
    }

    /// Record an attachment of the framebuffer at `target`: whether the call
    /// must be made. `DEPTH_STENCIL_ATTACHMENT` is the depth and stencil
    /// attachments both.
    fn attach(&mut self, target: u32, attachment: u32, a: Attach) -> bool {
        let Some(fbo) = self.fbo(target) else {
            return true;
        };
        let points: &[u32] = if attachment == gl::DEPTH_STENCIL_ATTACHMENT {
            &[gl::DEPTH_ATTACHMENT, gl::STENCIL_ATTACHMENT]
        } else {
            &[attachment]
        };
        let same = points
            .iter()
            .all(|p| self.s.attachments.get(&(fbo, *p)) == Some(&a));
        for p in points {
            self.s.attachments.insert((fbo, *p), a);
        }
        !same
    }
}

macro_rules! pass {
    ($($name:ident($($arg:ident: $ty:ty),*) $(-> $ret:ty)?;)*) => {
        $(fn $name(&mut self, $($arg: $ty),*) $(-> $ret)? {
            self.inner.$name($($arg),*)
        })*
    };
}

#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
impl GpuBackend for StateCache {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn host_caps(&mut self) -> HostCaps {
        self.inner.host_caps()
    }
    fn begin(&mut self) {
        self.inner.begin();
    }
    fn end(&mut self) {
        self.inner.end();
    }

    pass! {
        create_buffer() -> u32;
        buffer_data(target: u32, size: usize, data: Option<&[u8]>, usage: u32);
        buffer_sub_data(target: u32, offset: usize, data: &[u8]);
        get_buffer_sub_data(target: u32, offset: usize, out: &mut [u8]);
        create_texture() -> u32;
        tex_storage(target: u32, levels: u32, internal: u32, width: u32, height: u32, depth: u32);
        tex_sub_image(target: u32, level: u32, x: u32, y: u32, z: u32, width: u32, height: u32,
            depth: u32, format: u32, ty: u32, data: &[u8]);
        create_sampler() -> u32;
        sampler_parameter_i(sampler: u32, pname: u32, value: i32);
        sampler_parameter_f(sampler: u32, pname: u32, value: f32);
        create_framebuffer() -> u32;
        create_renderbuffer() -> u32;
        renderbuffer_storage(rb: u32, samples: u32, internal: u32, w: u32, h: u32);
        check_framebuffer_status(target: u32) -> u32;
        read_pixels(x: i32, y: i32, width: u32, height: u32, format: u32, ty: u32, out: &mut [u8]);
        blit_framebuffer(src: [i32; 4], dst: [i32; 4], mask: u32, filter: u32);
        clear_buffer_fv(buffer: u32, drawbuffer: u32, value: [f32; 4]);
        clear_buffer_iv(buffer: u32, drawbuffer: u32, value: [i32; 4]);
        clear_buffer_uiv(buffer: u32, drawbuffer: u32, value: [u32; 4]);
        clear_buffer_fi(depth: f32, stencil: i32);
        uniform_location(program: u32, name: &str) -> i32;
        uniform_block_index(program: u32, name: &str) -> u32;
        uniform_block_binding(program: u32, index: u32, binding: u32);
        create_vertex_array() -> u32;
        draw_arrays(mode: u32, first: i32, count: i32, instances: i32);
        draw_elements(mode: u32, count: i32, ty: u32, offset: usize, instances: i32);
        begin_transform_feedback(mode: u32);
        end_transform_feedback();
        finish();
        get_error() -> u32;
    }

    fn create_program_xfb(
        &mut self,
        vs: &str,
        fs: &str,
        xfb: Option<(&[String], bool)>,
    ) -> Result<u32, String> {
        let p = self.inner.create_program_xfb(vs, fs, xfb)?;
        self.s.uniforms.retain(|k, _| k.0 != p);
        Ok(p)
    }
    fn delete_program(&mut self, program: u32) {
        self.s.uniforms.retain(|k, _| k.0 != program);
        if self.s.program == Some(program) {
            self.s.program = None;
        }
        self.inner.delete_program(program);
    }
    fn use_program(&mut self, program: u32) {
        if Self::set(&mut self.s.program, program) {
            self.inner.use_program(program);
        }
    }
    fn uniform_1i(&mut self, location: i32, value: i32) {
        if self.uniform(location, vec![value.cast_unsigned()]) {
            self.inner.uniform_1i(location, value);
        }
    }
    fn uniform_1f(&mut self, location: i32, value: f32) {
        if self.uniform(location, vec![value.to_bits()]) {
            self.inner.uniform_1f(location, value);
        }
    }
    fn uniform_4uiv(&mut self, location: i32, values: &[u32]) {
        if self.uniform(location, values.to_vec()) {
            self.inner.uniform_4uiv(location, values);
        }
    }
    fn uniform_4fv(&mut self, location: i32, values: &[f32]) {
        if self.uniform(location, values.iter().map(|v| v.to_bits()).collect()) {
            self.inner.uniform_4fv(location, values);
        }
    }

    fn delete_buffer(&mut self, buffer: u32) {
        self.s.buffers.retain(|_, b| *b != buffer);
        self.s.ranges.retain(|_, r| r.0 != buffer);
        self.s.attrib_ptr.retain(|_, p| p.0 != buffer);
        self.inner.delete_buffer(buffer);
    }
    fn bind_buffer(&mut self, target: u32, buffer: u32) {
        if target == gl::ELEMENT_ARRAY_BUFFER || self.s.buffers.get(&target) != Some(&buffer) {
            if target != gl::ELEMENT_ARRAY_BUFFER {
                self.s.buffers.insert(target, buffer);
            }
            self.inner.bind_buffer(target, buffer);
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
        // Also the target's generic binding.
        self.s.buffers.insert(target, buffer);
        if self.s.ranges.get(&(target, index)) != Some(&(buffer, offset, size)) {
            self.s
                .ranges
                .insert((target, index), (buffer, offset, size));
            self.inner
                .bind_buffer_range(target, index, buffer, offset, size);
        }
    }
    fn bind_buffer_base(&mut self, target: u32, index: u32, buffer: u32) {
        self.s.buffers.insert(target, buffer);
        self.s.ranges.remove(&(target, index));
        self.inner.bind_buffer_base(target, index, buffer);
    }

    fn delete_texture(&mut self, texture: u32) {
        self.s.textures.retain(|_, t| *t != texture);
        self.s.tex_params.retain(|k, _| k.0 != texture);
        self.forget_attachments(true, texture);
        self.inner.delete_texture(texture);
    }
    fn active_texture(&mut self, unit: u32) {
        if Self::set(&mut self.s.active_unit, unit) {
            self.inner.active_texture(unit);
        }
    }
    fn bind_texture(&mut self, target: u32, texture: u32) {
        let Some(unit) = self.s.active_unit else {
            self.inner.bind_texture(target, texture);
            return;
        };
        if self.s.textures.get(&(unit, target)) != Some(&texture) {
            self.s.textures.insert((unit, target), texture);
            self.inner.bind_texture(target, texture);
        }
    }
    fn tex_parameter(&mut self, target: u32, pname: u32, value: i32) {
        let bound = self
            .s
            .active_unit
            .and_then(|u| self.s.textures.get(&(u, target)).copied());
        match bound {
            Some(t) if t != 0 => {
                if self.s.tex_params.get(&(t, pname)) != Some(&value) {
                    self.s.tex_params.insert((t, pname), value);
                    self.inner.tex_parameter(target, pname, value);
                }
            }
            _ => self.inner.tex_parameter(target, pname, value),
        }
    }
    fn delete_sampler(&mut self, sampler: u32) {
        self.s.samplers.retain(|_, s| *s != sampler);
        self.inner.delete_sampler(sampler);
    }
    fn bind_sampler(&mut self, unit: u32, sampler: u32) {
        if self.s.samplers.get(&unit) != Some(&sampler) {
            self.s.samplers.insert(unit, sampler);
            self.inner.bind_sampler(unit, sampler);
        }
    }

    fn delete_framebuffer(&mut self, fb: u32) {
        self.s.attachments.retain(|k, _| k.0 != fb);
        self.s.draw_buffers.remove(&fb);
        self.s.read_buffers.remove(&fb);
        if self.s.draw_fbo == Some(fb) {
            self.s.draw_fbo = None;
        }
        if self.s.read_fbo == Some(fb) {
            self.s.read_fbo = None;
        }
        self.inner.delete_framebuffer(fb);
    }
    fn bind_framebuffer(&mut self, target: u32, fb: u32) {
        let changed = match target {
            gl::DRAW_FRAMEBUFFER => Self::set(&mut self.s.draw_fbo, fb),
            gl::READ_FRAMEBUFFER => Self::set(&mut self.s.read_fbo, fb),
            _ => {
                let d = Self::set(&mut self.s.draw_fbo, fb);
                Self::set(&mut self.s.read_fbo, fb) || d
            }
        };
        if changed {
            self.inner.bind_framebuffer(target, fb);
        }
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
        // Every detachment looks the same, whatever it detached.
        let a = if texture == 0 {
            Attach::Texture(0, 0, 0, None)
        } else {
            Attach::Texture(textarget, texture, level, layer)
        };
        if self.attach(target, attachment, a) {
            self.inner
                .framebuffer_texture(target, attachment, textarget, texture, level, layer);
        }
    }
    fn framebuffer_renderbuffer(&mut self, target: u32, attachment: u32, rb: u32) {
        let a = if rb == 0 {
            Attach::Texture(0, 0, 0, None)
        } else {
            Attach::Renderbuffer(rb)
        };
        if self.attach(target, attachment, a) {
            self.inner.framebuffer_renderbuffer(target, attachment, rb);
        }
    }
    fn delete_renderbuffer(&mut self, rb: u32) {
        self.forget_attachments(false, rb);
        self.inner.delete_renderbuffer(rb);
    }
    fn draw_buffers(&mut self, buffers: &[u32]) {
        let Some(fbo) = self.s.draw_fbo else {
            self.inner.draw_buffers(buffers);
            return;
        };
        if self.s.draw_buffers.get(&fbo).map(Vec::as_slice) != Some(buffers) {
            self.s.draw_buffers.insert(fbo, buffers.to_vec());
            self.inner.draw_buffers(buffers);
        }
    }
    fn read_buffer(&mut self, buffer: u32) {
        let Some(fbo) = self.s.read_fbo else {
            self.inner.read_buffer(buffer);
            return;
        };
        if self.s.read_buffers.get(&fbo) != Some(&buffer) {
            self.s.read_buffers.insert(fbo, buffer);
            self.inner.read_buffer(buffer);
        }
    }

    fn bind_vertex_array(&mut self, vao: u32) {
        if Self::set(&mut self.s.vao, vao) {
            self.inner.bind_vertex_array(vao);
        }
    }
    fn enable_vertex_attrib_array(&mut self, index: u32, on: bool) {
        let Some(vao) = self.s.vao else {
            self.inner.enable_vertex_attrib_array(index, on);
            return;
        };
        if self.s.attrib_on.get(&(vao, index)) != Some(&on) {
            self.s.attrib_on.insert((vao, index), on);
            self.inner.enable_vertex_attrib_array(index, on);
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
        let (Some(vao), Some(&buf)) = (self.s.vao, self.s.buffers.get(&gl::ARRAY_BUFFER)) else {
            self.inner
                .vertex_attrib_pointer(index, size, ty, normalized, integer, stride, offset);
            return;
        };
        let p = (buf, size, ty, normalized, integer, stride, offset);
        if self.s.attrib_ptr.get(&(vao, index)) != Some(&p) {
            self.s.attrib_ptr.insert((vao, index), p);
            self.inner
                .vertex_attrib_pointer(index, size, ty, normalized, integer, stride, offset);
        }
    }
    fn vertex_attrib_divisor(&mut self, index: u32, divisor: u32) {
        let Some(vao) = self.s.vao else {
            self.inner.vertex_attrib_divisor(index, divisor);
            return;
        };
        if self.s.attrib_div.get(&(vao, index)) != Some(&divisor) {
            self.s.attrib_div.insert((vao, index), divisor);
            self.inner.vertex_attrib_divisor(index, divisor);
        }
    }

    fn enable(&mut self, cap: u32, on: bool) {
        if cap == gl::BLEND {
            // Every buffer's.
            self.s.blend_i.values_mut().for_each(|b| b.on = None);
            if Self::set(&mut self.s.blend.on, on) {
                self.inner.enable(cap, on);
            }
            return;
        }
        if self.s.enabled.get(&cap) != Some(&on) {
            self.s.enabled.insert(cap, on);
            self.inner.enable(cap, on);
        }
    }
    fn enable_i(&mut self, cap: u32, index: u32, on: bool) {
        self.s.blend.on = None;
        if Self::set(&mut self.s.blend_i.entry(index).or_default().on, on) {
            self.inner.enable_i(cap, index, on);
        }
    }
    fn blend_func_separate_i(
        &mut self,
        buf: u32,
        src_rgb: u32,
        dst_rgb: u32,
        src_a: u32,
        dst_a: u32,
    ) {
        self.s.blend.func = None;
        let v = [src_rgb, dst_rgb, src_a, dst_a];
        if Self::set(&mut self.s.blend_i.entry(buf).or_default().func, v) {
            self.inner
                .blend_func_separate_i(buf, src_rgb, dst_rgb, src_a, dst_a);
        }
    }
    fn blend_equation_separate_i(&mut self, buf: u32, rgb: u32, alpha: u32) {
        self.s.blend.eq = None;
        if Self::set(&mut self.s.blend_i.entry(buf).or_default().eq, [rgb, alpha]) {
            self.inner.blend_equation_separate_i(buf, rgb, alpha);
        }
    }
    fn color_mask_i(&mut self, buf: u32, r: bool, g: bool, b: bool, a: bool) {
        self.s.blend.mask = None;
        if Self::set(
            &mut self.s.blend_i.entry(buf).or_default().mask,
            [r, g, b, a],
        ) {
            self.inner.color_mask_i(buf, r, g, b, a);
        }
    }
    fn color_mask(&mut self, r: bool, g: bool, b: bool, a: bool) {
        self.s.blend_i.values_mut().for_each(|b| b.mask = None);
        if Self::set(&mut self.s.blend.mask, [r, g, b, a]) {
            self.inner.color_mask(r, g, b, a);
        }
    }
    fn blend_func_separate(&mut self, src_rgb: u32, dst_rgb: u32, src_a: u32, dst_a: u32) {
        self.s.blend_i.values_mut().for_each(|b| b.func = None);
        if Self::set(&mut self.s.blend.func, [src_rgb, dst_rgb, src_a, dst_a]) {
            self.inner
                .blend_func_separate(src_rgb, dst_rgb, src_a, dst_a);
        }
    }
    fn blend_equation_separate(&mut self, rgb: u32, alpha: u32) {
        self.s.blend_i.values_mut().for_each(|b| b.eq = None);
        if Self::set(&mut self.s.blend.eq, [rgb, alpha]) {
            self.inner.blend_equation_separate(rgb, alpha);
        }
    }
    fn blend_color(&mut self, rgba: [f32; 4]) {
        if Self::set(&mut self.s.blend_color, rgba.map(f32::to_bits)) {
            self.inner.blend_color(rgba);
        }
    }
    fn viewport(&mut self, x: i32, y: i32, width: i32, height: i32) {
        if Self::set(&mut self.s.viewport, [x, y, width, height]) {
            self.inner.viewport(x, y, width, height);
        }
    }
    fn depth_range(&mut self, near: f32, far: f32) {
        if Self::set(&mut self.s.depth_range, [near.to_bits(), far.to_bits()]) {
            self.inner.depth_range(near, far);
        }
    }
    fn scissor(&mut self, x: i32, y: i32, width: i32, height: i32) {
        if Self::set(&mut self.s.scissor, [x, y, width, height]) {
            self.inner.scissor(x, y, width, height);
        }
    }
    fn depth_func(&mut self, func: u32) {
        if Self::set(&mut self.s.depth_func, func) {
            self.inner.depth_func(func);
        }
    }
    fn depth_mask(&mut self, on: bool) {
        if Self::set(&mut self.s.depth_mask, on) {
            self.inner.depth_mask(on);
        }
    }
    fn stencil_func_separate(&mut self, face: u32, func: u32, reference: i32, mask: u32) {
        let mut changed = false;
        for &f in Self::faces(face) {
            changed |= Self::set(&mut self.s.stencil_func[f], (func, reference, mask));
        }
        if changed {
            self.inner
                .stencil_func_separate(face, func, reference, mask);
        }
    }
    fn stencil_op_separate(&mut self, face: u32, fail: u32, zfail: u32, zpass: u32) {
        let mut changed = false;
        for &f in Self::faces(face) {
            changed |= Self::set(&mut self.s.stencil_op[f], (fail, zfail, zpass));
        }
        if changed {
            self.inner.stencil_op_separate(face, fail, zfail, zpass);
        }
    }
    fn stencil_mask_separate(&mut self, face: u32, mask: u32) {
        let mut changed = false;
        for &f in Self::faces(face) {
            changed |= Self::set(&mut self.s.stencil_mask[f], mask);
        }
        if changed {
            self.inner.stencil_mask_separate(face, mask);
        }
    }
    fn cull_face(&mut self, face: u32) {
        if Self::set(&mut self.s.cull_face, face) {
            self.inner.cull_face(face);
        }
    }
    fn front_face(&mut self, mode: u32) {
        if Self::set(&mut self.s.front_face, mode) {
            self.inner.front_face(mode);
        }
    }
    fn polygon_offset(&mut self, factor: f32, units: f32) {
        if Self::set(
            &mut self.s.polygon_offset,
            [factor.to_bits(), units.to_bits()],
        ) {
            self.inner.polygon_offset(factor, units);
        }
    }
    fn line_width(&mut self, width: f32) {
        if Self::set(&mut self.s.line_width, width.to_bits()) {
            self.inner.line_width(width);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::backend::Recorder;

    fn cache() -> (StateCache, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        let rec = Recorder::new(false);
        let calls = std::sync::Arc::clone(&rec.calls);
        (StateCache::new(Box::new(rec)), calls)
    }

    #[test]
    fn repeated_state_is_dropped() {
        let (mut c, calls) = cache();
        for _ in 0..3 {
            c.enable(gl::DEPTH_TEST, true);
            c.viewport(0, 0, 64, 64);
            c.use_program(5);
            c.uniform_1f(3, 1.5);
        }
        c.uniform_1f(3, 2.0);
        c.use_program(6);
        c.uniform_1f(3, 2.0); // another program's uniform
        assert_eq!(
            *calls.lock().unwrap(),
            [
                "enable(2929, true)",
                "viewport(0, 0, 64, 64)",
                "use_program(5)",
                "uniform_1f(3, 1.5)",
                "uniform_1f(3, 2.0)",
                "use_program(6)",
                "uniform_1f(3, 2.0)",
            ]
        );
    }

    #[test]
    fn global_and_indexed_blend_state_invalidate_each_other() {
        let (mut c, calls) = cache();
        c.enable(gl::BLEND, true);
        c.enable_i(gl::BLEND, 1, false);
        // Buffer 1 differs now: the global call must be made again.
        c.enable(gl::BLEND, true);
        c.enable_i(gl::BLEND, 1, true);
        c.enable_i(gl::BLEND, 1, true);
        assert_eq!(calls.lock().unwrap().len(), 4);
    }

    #[test]
    fn deleted_names_are_forgotten() {
        let (mut c, calls) = cache();
        c.bind_framebuffer(gl::DRAW_FRAMEBUFFER, 1);
        c.framebuffer_texture(
            gl::DRAW_FRAMEBUFFER,
            gl::COLOR_ATTACHMENT0,
            gl::TEXTURE_2D,
            7,
            0,
            None,
        );
        c.framebuffer_texture(
            gl::DRAW_FRAMEBUFFER,
            gl::COLOR_ATTACHMENT0,
            gl::TEXTURE_2D,
            7,
            0,
            None,
        );
        c.active_texture(gl::TEXTURE0);
        c.bind_texture(gl::TEXTURE_2D, 7);
        c.tex_parameter(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        c.delete_texture(7);
        // A new texture under the same name: everything is set again.
        c.bind_texture(gl::TEXTURE_2D, 7);
        c.tex_parameter(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        c.framebuffer_texture(
            gl::DRAW_FRAMEBUFFER,
            gl::COLOR_ATTACHMENT0,
            gl::TEXTURE_2D,
            7,
            0,
            None,
        );
        let calls = calls.lock().unwrap();
        assert_eq!(
            calls
                .iter()
                .filter(|c| c.starts_with("framebuffer_texture"))
                .count(),
            2
        );
        assert_eq!(
            calls
                .iter()
                .filter(|c| c.starts_with("tex_parameter"))
                .count(),
            2
        );
        assert_eq!(
            calls
                .iter()
                .filter(|c| c.starts_with("bind_texture"))
                .count(),
            2
        );
    }

    #[test]
    fn depth_stencil_attachment_is_both() {
        let (mut c, calls) = cache();
        c.bind_framebuffer(gl::FRAMEBUFFER, 2);
        c.framebuffer_texture(
            gl::DRAW_FRAMEBUFFER,
            gl::DEPTH_STENCIL_ATTACHMENT,
            gl::TEXTURE_2D,
            0,
            0,
            None,
        );
        // Detached already, as part of depth-stencil.
        c.framebuffer_texture(
            gl::DRAW_FRAMEBUFFER,
            gl::DEPTH_ATTACHMENT,
            gl::TEXTURE_2D,
            0,
            0,
            None,
        );
        c.framebuffer_texture(
            gl::DRAW_FRAMEBUFFER,
            gl::DEPTH_ATTACHMENT,
            gl::TEXTURE_2D,
            4,
            0,
            None,
        );
        // Stencil is still detached but depth isn't: the DS call is made.
        c.framebuffer_texture(
            gl::DRAW_FRAMEBUFFER,
            gl::DEPTH_STENCIL_ATTACHMENT,
            gl::TEXTURE_2D,
            0,
            0,
            None,
        );
        let n = calls
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.starts_with("framebuffer_texture"))
            .count();
        assert_eq!(n, 3);
    }
}
