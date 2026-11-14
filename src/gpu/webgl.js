// nixvm's WebGL2 GPU backend, JS side (src/gpu/webgl.rs is the Rust side).
// wasm-bindgen copies this module into its output as a snippet; the page
// hands nixvm a WebGL2RenderingContext (Terminal.attach_webgl2) and every
// call below runs on it. GL objects live in a table here and cross into
// Rust as small integers, as GLES names them, so Rust never holds a JS
// object. Uniform locations get a table of their own.

let gl = null;
const objs = [null];
const free = [];
const locs = [null];

function put(o) {
  if (!o) return 0;
  const i = free.length ? free.pop() : objs.length;
  objs[i] = o;
  return i;
}
function get(i) {
  return i ? objs[i] : null;
}
function drop(i) {
  const o = objs[i];
  if (i) {
    objs[i] = null;
    free.push(i);
  }
  return o;
}

export function nx_gl_set_context(g) {
  gl = g;
  objs.length = 1;
  free.length = 0;
  locs.length = 1;
  gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
}
export function nx_gl_has_context() {
  return gl !== null;
}
export function nx_gl_name() {
  if (!gl) return "no WebGL2 context";
  const ext = gl.getExtension("WEBGL_debug_renderer_info");
  const r = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  return `WebGL2 (${r})`;
}

export function nx_gl_create_buffer() { return put(gl.createBuffer()); }
export function nx_gl_delete_buffer(b) { gl.deleteBuffer(drop(b)); }
export function nx_gl_bind_buffer(t, b) { gl.bindBuffer(t, get(b)); }
export function nx_gl_buffer_data_size(t, size, usage) { gl.bufferData(t, size, usage); }
export function nx_gl_buffer_data(t, data, usage) { gl.bufferData(t, data, usage); }
export function nx_gl_buffer_sub_data(t, off, data) { gl.bufferSubData(t, off, data); }
export function nx_gl_bind_buffer_range(t, i, b, off, size) { gl.bindBufferRange(t, i, get(b), off, size); }

export function nx_gl_create_texture() { return put(gl.createTexture()); }
export function nx_gl_delete_texture(t) { gl.deleteTexture(drop(t)); }
export function nx_gl_active_texture(u) { gl.activeTexture(u); }
export function nx_gl_bind_texture(t, tex) { gl.bindTexture(t, get(tex)); }
export function nx_gl_tex_storage(t, levels, internal, w, h, d) {
  if (d === 0) gl.texStorage2D(t, levels, internal, w, h);
  else gl.texStorage3D(t, levels, internal, w, h, d);
}
export function nx_gl_tex_sub_image(t, level, x, y, z, w, h, d, format, type, data) {
  if (d === 0) gl.texSubImage2D(t, level, x, y, w, h, format, type, data);
  else gl.texSubImage3D(t, level, x, y, z, w, h, d, format, type, data);
}
export function nx_gl_tex_parameter(t, p, v) { gl.texParameteri(t, p, v); }
export function nx_gl_create_sampler() { return put(gl.createSampler()); }
export function nx_gl_delete_sampler(s) { gl.deleteSampler(drop(s)); }
export function nx_gl_sampler_parameter_i(s, p, v) { gl.samplerParameteri(get(s), p, v); }
export function nx_gl_sampler_parameter_f(s, p, v) { gl.samplerParameterf(get(s), p, v); }
export function nx_gl_bind_sampler(u, s) { gl.bindSampler(u, get(s)); }

export function nx_gl_create_framebuffer() { return put(gl.createFramebuffer()); }
export function nx_gl_delete_framebuffer(f) { gl.deleteFramebuffer(drop(f)); }
export function nx_gl_bind_framebuffer(t, f) { gl.bindFramebuffer(t, get(f)); }
export function nx_gl_framebuffer_texture_2d(t, att, textarget, tex, level) {
  gl.framebufferTexture2D(t, att, textarget, get(tex), level);
}
export function nx_gl_framebuffer_texture_layer(t, att, tex, level, layer) {
  gl.framebufferTextureLayer(t, att, get(tex), level, layer);
}
export function nx_gl_draw_buffers(bufs) { gl.drawBuffers(Array.from(bufs)); }
export function nx_gl_read_buffer(b) { gl.readBuffer(b); }
export function nx_gl_check_framebuffer_status(t) { return gl.checkFramebufferStatus(t); }
export function nx_gl_read_pixels(x, y, w, h, format, type, out) {
  // WebGL wants the typed array that matches `type`; read bytes through
  // a view of the same memory.
  let view = out;
  if (type === gl.FLOAT) view = new Float32Array(out.buffer, out.byteOffset, out.byteLength / 4);
  else if (type === gl.UNSIGNED_INT || type === gl.UNSIGNED_INT_2_10_10_10_REV ||
           type === gl.UNSIGNED_INT_10F_11F_11F_REV || type === gl.UNSIGNED_INT_5_9_9_9_REV)
    view = new Uint32Array(out.buffer, out.byteOffset, out.byteLength / 4);
  else if (type === gl.INT) view = new Int32Array(out.buffer, out.byteOffset, out.byteLength / 4);
  else if (type === gl.UNSIGNED_SHORT || type === gl.UNSIGNED_SHORT_5_6_5 || type === gl.HALF_FLOAT)
    view = new Uint16Array(out.buffer, out.byteOffset, out.byteLength / 2);
  else if (type === gl.SHORT) view = new Int16Array(out.buffer, out.byteOffset, out.byteLength / 2);
  else if (type === gl.BYTE) view = new Int8Array(out.buffer, out.byteOffset, out.byteLength);
  gl.readPixels(x, y, w, h, format, type, view);
}
export function nx_gl_blit_framebuffer(s, d, mask, filter) {
  gl.blitFramebuffer(s[0], s[1], s[2], s[3], d[0], d[1], d[2], d[3], mask, filter);
}
export function nx_gl_clear_buffer_fv(b, db, v) { gl.clearBufferfv(b, db, v); }
export function nx_gl_clear_buffer_iv(b, db, v) { gl.clearBufferiv(b, db, v); }
export function nx_gl_clear_buffer_uiv(b, db, v) { gl.clearBufferuiv(b, db, v); }
export function nx_gl_clear_buffer_fi(depth, stencil) { gl.clearBufferfi(gl.DEPTH_STENCIL, 0, depth, stencil); }

// Compile and link; returns the program's name, or 0 with the log kept for
// nx_gl_last_log.
let lastLog = "";
function compile(kind, src) {
  const s = gl.createShader(kind);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    lastLog = (kind === gl.VERTEX_SHADER ? "vertex: " : "fragment: ") + gl.getShaderInfoLog(s);
    gl.deleteShader(s);
    return null;
  }
  return s;
}
export function nx_gl_create_program(vs, fs) {
  const v = compile(gl.VERTEX_SHADER, vs);
  if (!v) return 0;
  const f = compile(gl.FRAGMENT_SHADER, fs);
  if (!f) { gl.deleteShader(v); return 0; }
  const p = gl.createProgram();
  gl.attachShader(p, v);
  gl.attachShader(p, f);
  gl.linkProgram(p);
  gl.deleteShader(v);
  gl.deleteShader(f);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    lastLog = "link: " + gl.getProgramInfoLog(p);
    gl.deleteProgram(p);
    return 0;
  }
  return put(p);
}
export function nx_gl_last_log() { return lastLog; }
export function nx_gl_delete_program(p) { gl.deleteProgram(drop(p)); }
export function nx_gl_use_program(p) { gl.useProgram(get(p)); }
export function nx_gl_uniform_location(p, name) {
  const l = gl.getUniformLocation(get(p), name);
  if (!l) return -1;
  locs.push(l);
  return locs.length - 1;
}
export function nx_gl_uniform_block_index(p, name) { return gl.getUniformBlockIndex(get(p), name) >>> 0; }
export function nx_gl_uniform_block_binding(p, i, b) { gl.uniformBlockBinding(get(p), i, b); }
export function nx_gl_uniform_1i(l, v) { if (l > 0) gl.uniform1i(locs[l], v); }
export function nx_gl_uniform_1f(l, v) { if (l > 0) gl.uniform1f(locs[l], v); }
export function nx_gl_uniform_4uiv(l, v) { if (l > 0) gl.uniform4uiv(locs[l], v); }

export function nx_gl_create_vertex_array() { return put(gl.createVertexArray()); }
export function nx_gl_bind_vertex_array(v) { gl.bindVertexArray(get(v)); }
export function nx_gl_enable_vertex_attrib_array(i, on) {
  if (on) gl.enableVertexAttribArray(i); else gl.disableVertexAttribArray(i);
}
export function nx_gl_vertex_attrib_pointer(i, size, type, norm, integer, stride, off) {
  if (integer) gl.vertexAttribIPointer(i, size, type, stride, off);
  else gl.vertexAttribPointer(i, size, type, norm, stride, off);
}
export function nx_gl_vertex_attrib_divisor(i, d) { gl.vertexAttribDivisor(i, d); }

export function nx_gl_enable(cap, on) { if (on) gl.enable(cap); else gl.disable(cap); }
export function nx_gl_viewport(x, y, w, h) { gl.viewport(x, y, w, h); }
export function nx_gl_depth_range(n, f) { gl.depthRange(n, f); }
export function nx_gl_scissor(x, y, w, h) { gl.scissor(x, y, w, h); }
export function nx_gl_color_mask(r, g, b, a) { gl.colorMask(r, g, b, a); }
export function nx_gl_blend_func_separate(a, b, c, d) { gl.blendFuncSeparate(a, b, c, d); }
export function nx_gl_blend_equation_separate(a, b) { gl.blendEquationSeparate(a, b); }
export function nx_gl_blend_color(r, g, b, a) { gl.blendColor(r, g, b, a); }
export function nx_gl_depth_func(f) { gl.depthFunc(f); }
export function nx_gl_depth_mask(on) { gl.depthMask(on); }
export function nx_gl_stencil_func_separate(face, f, r, m) { gl.stencilFuncSeparate(face, f, r, m); }
export function nx_gl_stencil_op_separate(face, a, b, c) { gl.stencilOpSeparate(face, a, b, c); }
export function nx_gl_stencil_mask_separate(face, m) { gl.stencilMaskSeparate(face, m); }
export function nx_gl_cull_face(f) { gl.cullFace(f); }
export function nx_gl_front_face(m) { gl.frontFace(m); }
export function nx_gl_polygon_offset(f, u) { gl.polygonOffset(f, u); }
export function nx_gl_line_width(w) { gl.lineWidth(w); }
export function nx_gl_draw_arrays(mode, first, count, instances) {
  gl.drawArraysInstanced(mode, first, count, instances);
}
export function nx_gl_draw_elements(mode, count, type, off, instances) {
  gl.drawElementsInstanced(mode, count, type, off, instances);
}
export function nx_gl_finish() { gl.finish(); }
export function nx_gl_get_error() { return gl.getError(); }
