//! TGSI → GLSL ES 3.00: what virglrenderer's `vrend_shader.c` does for a GLES
//! host, for the subset of TGSI that GLES 3.0-class shaders compile to.
//!
//! The model follows TGSI's: every register is a `vec4` of 32-bit words.
//! Floats are stored as floats; integer results are stored by bit pattern
//! (`intBitsToFloat`) and read back the same way (`floatBitsToInt`), so an
//! untyped `MOV` copies bits and each instruction picks the interpretation it
//! needs, as in TGSI. Constants are `uvec4` uniforms for the same reason.
//!
//! A vertex and a fragment shader are translated together ([`link`]): GLSL ES
//! needs a varying's interpolation qualifier to match on both sides, and some
//! choices (flat-shaded colours, integer vertex inputs, how many colour
//! buffers a "write all" shader fills) depend on draw-time state, which the
//! renderer passes in [`Options`] and keys its program cache on.
//!
//! Names: vertex inputs `nx_a<n>`, temporaries `nx_t<n>` (arrays
//! `nx_ta<id>`), outputs collect in `nx_o<n>` and are copied to `gl_Position`
//! / varyings `nx_v_<semantic><index>` / colour outputs `nx_out<n>` at the
//! end; constant buffer 0 is `uniform uvec4 nx_<s>c[]`, buffer `k` the
//! uniform block `nx_<s>ub<k>` with member `nx_<s>u<k>[]`, samplers
//! `nx_<s>s<n>` — `<s>` being `vs` or `fs`. `nx_flip_y` (vertex) flips the
//! output vertically for a Gallium viewport with negative y scale (a window
//! system buffer), `nx_fb_height` (fragment) serves lower-left
//! `gl_FragCoord`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use super::tgsi::{Decl, File, ImmType, Inst, Reg, Shader, Src, Stage};

/// Draw-time state a program depends on. See the module docs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Options {
    /// Vertex inputs fed as unsigned / signed integers (bit `n`: input `n`).
    pub uint_inputs: u32,
    pub sint_inputs: u32,
    /// Flat shading: `COLOR` varyings with `COLOR` interpolation go `flat`.
    pub flatshade: bool,
    /// Colour buffers bound: how many a `FS_COLOR0_WRITES_ALL_CBUFS` shader
    /// writes.
    pub nr_cbufs: u32,
    /// Colour buffers holding unsigned / signed integers (bit `n`: buffer `n`).
    pub uint_cbufs: u32,
    pub sint_cbufs: u32,
}

/// What a linked program declares, for the renderer to feed it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Layout {
    /// Size in `uvec4`s of constant buffer 0, per stage (vertex, fragment).
    pub consts: [u32; 2],
    /// Uniform blocks used (buffer index ≥ 1), per stage.
    pub ubos: [Vec<u32>; 2],
    /// Sampler slots used, per stage.
    pub samplers: [Vec<u32>; 2],
}

/// The GLSL of a linked program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub vs: String,
    pub fs: String,
    pub layout: Layout,
}

/// Translate a vertex and fragment shader pair.
///
/// # Errors
/// What in the TGSI this translator doesn't support.
pub fn link(vs: &Shader, fs: &Shader, opts: &Options) -> Result<Program, String> {
    if vs.stage != Stage::Vertex || fs.stage != Stage::Fragment {
        return Err("only vertex + fragment programs are supported".into());
    }
    // Varyings: the fragment shader's inputs decide the qualifiers.
    let mut varyings: BTreeMap<String, String> = BTreeMap::new();
    for d in fs.decls.iter().filter(|d| d.file == File::Input) {
        if let Some(name) = varying_name(d) {
            varyings.insert(name, interp_qualifier(d, opts).to_string());
        }
    }
    let written: BTreeSet<String> = vs
        .decls
        .iter()
        .filter(|d| d.file == File::Output)
        .filter_map(varying_name)
        .collect();
    let mut layout = Layout::default();
    let vsrc = Gen::new(vs, opts, &varyings, &written).run(&mut layout)?;
    let fsrc = Gen::new(fs, opts, &varyings, &written).run(&mut layout)?;
    Ok(Program {
        vs: vsrc,
        fs: fsrc,
        layout,
    })
}

/// The varying a semantic travels as, if it's one that does.
fn varying_name(d: &Decl) -> Option<String> {
    let (sem, idx) = d.semantic.as_ref()?;
    let short = match sem.as_str() {
        "GENERIC" => "g",
        "TEXCOORD" => "t",
        "COLOR" => "c",
        "BCOLOR" => "b",
        "FOG" => "f",
        "PRIM_ID" => "p",
        _ => return None,
    };
    Some(format!("nx_v_{short}{idx}"))
}

fn interp_qualifier(d: &Decl, opts: &Options) -> &'static str {
    let centroid = d.location.as_deref() == Some("CENTROID");
    match d.interp.as_deref() {
        Some("CONSTANT") => "flat ",
        Some("COLOR") if opts.flatshade => "flat ",
        _ if d.semantic.as_ref().is_some_and(|(s, _)| s == "PRIM_ID") => "flat ",
        _ if centroid => "centroid ",
        _ => "",
    }
}

const SWZ: [char; 4] = ['x', 'y', 'z', 'w'];

fn mask_str(m: u8) -> String {
    (0..4)
        .filter(|i| m & (1 << i) != 0)
        .map(|i| SWZ[i])
        .collect()
}

/// How an instruction's operands and result are read.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ty {
    F,
    I,
    U,
}

struct Gen<'a> {
    sh: &'a Shader,
    opts: &'a Options,
    varyings: &'a BTreeMap<String, String>,
    /// The varyings the vertex shader writes.
    written: &'a BTreeSet<String>,
    st: &'static str,
    out: String,
    body: String,
    indent: usize,
    /// Sampler slot → (GLSL type, TGSI target).
    samplers: BTreeMap<u32, (String, String)>,
    /// Temp arrays: id → (first register, length).
    temp_arrays: BTreeMap<u32, (u32, u32)>,
}

impl<'a> Gen<'a> {
    fn new(
        sh: &'a Shader,
        opts: &'a Options,
        varyings: &'a BTreeMap<String, String>,
        written: &'a BTreeSet<String>,
    ) -> Self {
        Self {
            sh,
            opts,
            varyings,
            written,
            st: if sh.stage == Stage::Vertex {
                "vs"
            } else {
                "fs"
            },
            out: String::new(),
            body: String::new(),
            indent: 1,
            samplers: BTreeMap::new(),
            temp_arrays: BTreeMap::new(),
        }
    }

    fn is_vs(&self) -> bool {
        self.sh.stage == Stage::Vertex
    }

    fn stage_ix(&self) -> usize {
        usize::from(!self.is_vs())
    }

    fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.body.push_str("  ");
        }
        self.body.push_str(s);
        self.body.push('\n');
    }

    fn run(mut self, layout: &mut Layout) -> Result<String, String> {
        self.declare_samplers()?;
        let mut head =
            String::from("#version 300 es\nprecision highp float;\nprecision highp int;\n");
        let mut types: Vec<&str> = self.samplers.values().map(|(t, _)| t.as_str()).collect();
        types.sort_unstable();
        types.dedup();
        for t in types {
            let _ = writeln!(head, "precision highp {t};");
        }
        let six = self.stage_ix();
        let st = self.st;
        // Declarations.
        let mut ubos = Vec::new();
        for d in &self.sh.decls {
            match d.file {
                File::Const => {
                    let size = d.last + 1;
                    match d.dim.unwrap_or(0) {
                        0 => layout.consts[six] = layout.consts[six].max(size),
                        k => ubos.push((k, size)),
                    }
                }
                File::Temp => {
                    if let Some(id) = d.array {
                        self.temp_arrays.insert(id, (d.first, d.last - d.first + 1));
                        let _ = writeln!(self.out, "vec4 nx_ta{id}[{}];", d.last - d.first + 1);
                    } else {
                        for i in d.first..=d.last {
                            let _ = writeln!(self.out, "vec4 nx_t{i} = vec4(0.0);");
                        }
                    }
                }
                File::Address => {
                    for i in d.first..=d.last {
                        let _ = writeln!(self.out, "ivec4 nx_addr{i} = ivec4(0);");
                    }
                }
                File::Output => {
                    for i in d.first..=d.last {
                        let _ = writeln!(self.out, "vec4 nx_o{i} = vec4(0.0);");
                    }
                }
                File::Input if self.is_vs() => {
                    for i in d.first..=d.last {
                        let ty = if self.opts.uint_inputs & (1 << i) != 0 {
                            "uvec4"
                        } else if self.opts.sint_inputs & (1 << i) != 0 {
                            "ivec4"
                        } else {
                            "vec4"
                        };
                        let _ = writeln!(head, "layout(location = {i}) in {ty} nx_a{i};");
                    }
                }
                File::Input => self.declare_fs_input(d, &mut head)?,
                File::Image | File::Buffer | File::Memory | File::HwAtomic => {
                    return Err(format!("{:?} registers are not supported", d.file));
                }
                _ => {}
            }
        }
        if layout.consts[six] > 0 {
            let _ = writeln!(head, "uniform uvec4 nx_{st}c[{}];", layout.consts[six]);
        }
        ubos.sort_unstable();
        ubos.dedup_by_key(|u| u.0);
        for (k, size) in &ubos {
            let _ = writeln!(
                head,
                "layout(std140) uniform nx_{st}ub{k} {{ uvec4 nx_{st}u{k}[{size}]; }};"
            );
            layout.ubos[six].push(*k);
        }
        for (slot, (ty, _)) in &self.samplers {
            let _ = writeln!(head, "uniform {ty} nx_{st}s{slot};");
            layout.samplers[six].push(*slot);
        }
        // Outputs.
        if self.is_vs() {
            head.push_str("uniform float nx_flip_y;\n");
            for d in self.sh.decls.iter().filter(|d| d.file == File::Output) {
                if let Some(name) = varying_name(d) {
                    let q = self.varyings.get(&name).map_or("", String::as_str);
                    let _ = writeln!(head, "{q}out vec4 {name};");
                }
            }
        } else {
            for n in 0..self.fs_color_outputs() {
                let ty = self.cbuf_type(n);
                let _ = writeln!(head, "layout(location = {n}) out {ty} nx_out{n};");
            }
            if self.needs_fb_height() {
                head.push_str("uniform float nx_fb_height;\n");
            }
        }
        for (i, imm) in self.sh.imms.iter().enumerate() {
            let vals: Vec<String> = (0..4).map(|c| imm_literal(imm.ty, imm.words[c])).collect();
            let _ = writeln!(
                self.out,
                "const vec4 nx_imm{i} = vec4({});",
                vals.join(", ")
            );
        }
        // Body.
        let insts = self.sh.insts.clone();
        for inst in &insts {
            self.inst(inst)?;
        }
        let mut src = head;
        src.push_str(&self.out);
        src.push_str("void main() {\n");
        src.push_str(&self.body);
        src.push_str("}\n");
        Ok(src)
    }

    fn needs_fb_height(&self) -> bool {
        self.sh
            .prop("FS_COORD_ORIGIN")
            .is_some_and(|v| v.first().is_some_and(|o| o == "LOWER_LEFT"))
    }

    /// How many colour outputs the fragment shader declares.
    fn fs_color_outputs(&self) -> u32 {
        let declared = self
            .sh
            .decls
            .iter()
            .filter(|d| d.file == File::Output)
            .filter_map(|d| d.semantic.as_ref())
            .filter(|(s, _)| s == "COLOR")
            .map(|(_, i)| i + 1)
            .max()
            .unwrap_or(0);
        if self.writes_all() {
            declared.max(self.opts.nr_cbufs)
        } else {
            declared
        }
    }

    fn writes_all(&self) -> bool {
        self.sh
            .prop("FS_COLOR0_WRITES_ALL_CBUFS")
            .is_some_and(|v| v.first().is_some_and(|x| x != "0"))
    }

    fn cbuf_type(&self, n: u32) -> &'static str {
        if self.opts.uint_cbufs & (1 << n) != 0 {
            "uvec4"
        } else if self.opts.sint_cbufs & (1 << n) != 0 {
            "ivec4"
        } else {
            "vec4"
        }
    }

    fn declare_fs_input(&mut self, d: &Decl, head: &mut String) -> Result<(), String> {
        if d.first != d.last {
            return Err("fragment input ranges are not supported".into());
        }
        if let Some(name) = varying_name(d) {
            if self.written.contains(&name) {
                let q = self.varyings.get(&name).map_or("", String::as_str);
                let _ = writeln!(head, "{q}in vec4 {name};");
            } else {
                // Nothing feeds it: GL's default for an unwritten varying.
                let _ = writeln!(head, "vec4 {name} = vec4(0.0, 0.0, 0.0, 1.0);");
            }
        }
        Ok(())
    }

    fn declare_samplers(&mut self) -> Result<(), String> {
        // The target and return type come from the SVIEW declaration when
        // there is one, else from the first instruction sampling the slot.
        let mut targets: BTreeMap<u32, (String, String)> = BTreeMap::new();
        for d in self.sh.decls.iter().filter(|d| d.file == File::SamplerView) {
            if let Some((t, r)) = &d.sview {
                for i in d.first..=d.last {
                    targets.insert(i, (t.clone(), r.clone()));
                }
            }
        }
        for inst in &self.sh.insts {
            if let (Some(t), Some(s)) = (
                &inst.tex,
                inst.src.iter().find(|s| s.reg.file == File::Sampler),
            ) {
                let slot = s.reg.index as u32;
                let e = targets
                    .entry(slot)
                    .or_insert_with(|| (t.clone(), "FLOAT".into()));
                // A shadow lookup decides the sampler's type.
                if t.starts_with("SHADOW") {
                    e.0.clone_from(t);
                }
            }
        }
        for (slot, (t, ret)) in targets {
            let base = match t.as_str() {
                "1D" | "2D" | "RECT" => "sampler2D",
                "3D" => "sampler3D",
                "CUBE" => "samplerCube",
                "1D_ARRAY" | "2D_ARRAY" => "sampler2DArray",
                "SHADOW1D" | "SHADOW2D" | "SHADOWRECT" => "sampler2DShadow",
                "SHADOWCUBE" => "samplerCubeShadow",
                "SHADOW1D_ARRAY" | "SHADOW2D_ARRAY" => "sampler2DArrayShadow",
                _ => return Err(format!("texture target {t} is not supported")),
            };
            let prefix = match ret.as_str() {
                "UINT" if !base.contains("Shadow") => "u",
                "SINT" if !base.contains("Shadow") => "i",
                _ => "",
            };
            self.samplers.insert(slot, (format!("{prefix}{base}"), t));
        }
        Ok(())
    }

    /// `+ addrN.c` for an indirect index, else nothing.
    fn indirect(ind: Option<super::tgsi::Indirect>) -> Result<String, String> {
        match ind {
            None => Ok(String::new()),
            Some(i) if i.file == File::Address => {
                Ok(format!(" + nx_addr{}.{}", i.index, SWZ[i.comp as usize]))
            }
            Some(i) => Err(format!("indirect through {:?} is not supported", i.file)),
        }
    }

    /// A register as a `vec4` expression.
    fn reg(&self, r: &Reg) -> Result<String, String> {
        let st = self.st;
        let ind = Self::indirect(r.indirect)?;
        Ok(match r.file {
            File::Temp => {
                let arr = self.temp_arrays.iter().find(|(_, (first, len))| {
                    r.index as u32 >= *first && (r.index as u32) < first + len
                });
                match (arr, ind.is_empty()) {
                    (Some((id, (first, _))), _) => {
                        format!("nx_ta{id}[{}{ind}]", r.index as u32 - first)
                    }
                    (None, true) => format!("nx_t{}", r.index),
                    (None, false) => return Err("indirect temp outside an array".into()),
                }
            }
            File::Immediate => format!("nx_imm{}", r.index),
            File::Output => format!("nx_o{}", r.index),
            File::Address => format!("vec4(nx_addr{})", r.index),
            File::Const => {
                if r.dim_indirect.is_some() {
                    return Err("indirect constant buffer selection is not supported".into());
                }
                match r.dim.unwrap_or(0) {
                    0 => format!("uintBitsToFloat(nx_{st}c[{}{ind}])", r.index),
                    k => format!("uintBitsToFloat(nx_{st}u{k}[{}{ind}])", r.index),
                }
            }
            File::Input if self.is_vs() => {
                let i = r.index as u32;
                if self.opts.uint_inputs & (1 << i) != 0 {
                    format!("uintBitsToFloat(nx_a{i})")
                } else if self.opts.sint_inputs & (1 << i) != 0 {
                    format!("intBitsToFloat(nx_a{i})")
                } else {
                    format!("nx_a{i}")
                }
            }
            File::Input => {
                let d = self
                    .sh
                    .decl(File::Input, r.index as u32)
                    .ok_or_else(|| format!("undeclared IN[{}]", r.index))?;
                self.fs_input(d)?
            }
            File::SystemValue => {
                let d = self
                    .sh
                    .decl(File::SystemValue, r.index as u32)
                    .ok_or_else(|| format!("undeclared SV[{}]", r.index))?;
                let sem = d.semantic.as_ref().map_or("", |(s, _)| s.as_str());
                match sem {
                    "VERTEXID" | "VERTEXID_NOBASE" => "intBitsToFloat(ivec4(gl_VertexID))".into(),
                    "INSTANCEID" => "intBitsToFloat(ivec4(gl_InstanceID))".into(),
                    "BASEVERTEX" | "BASEINSTANCE" | "DRAWID" => "intBitsToFloat(ivec4(0))".into(),
                    "FACE" => "vec4(gl_FrontFacing ? 1.0 : -1.0)".into(),
                    "POSITION" => self.frag_coord(),
                    _ => return Err(format!("system value {sem} is not supported")),
                }
            }
            f => return Err(format!("{f:?} as an operand is not supported")),
        })
    }

    fn frag_coord(&self) -> String {
        let mut e = String::from("gl_FragCoord");
        if self.needs_fb_height() {
            e = format!("vec4({e}.x, nx_fb_height - {e}.y, {e}.zw)");
        }
        if self
            .sh
            .prop("FS_COORD_PIXEL_CENTER")
            .is_some_and(|v| v.first().is_some_and(|c| c == "INTEGER"))
        {
            e = format!("({e} - vec4(0.5, 0.5, 0.0, 0.0))");
        }
        e
    }

    fn fs_input(&self, d: &Decl) -> Result<String, String> {
        let (sem, _) = d
            .semantic
            .as_ref()
            .ok_or("fragment input without a semantic")?;
        if let Some(name) = varying_name(d) {
            return Ok(name);
        }
        Ok(match sem.as_str() {
            "POSITION" => self.frag_coord(),
            "FACE" => "vec4(gl_FrontFacing ? 1.0 : -1.0)".into(),
            "PCOORD" => "vec4(gl_PointCoord, 0.0, 1.0)".into(),
            s => return Err(format!("fragment input {s} is not supported")),
        })
    }

    fn swizzled(&self, s: &Src) -> Result<String, String> {
        let base = self.reg(&s.reg)?;
        if s.swizzle == [0, 1, 2, 3] {
            Ok(base)
        } else {
            let sw: String = s.swizzle.iter().map(|&c| SWZ[c as usize]).collect();
            Ok(format!("{base}.{sw}"))
        }
    }

    /// Source `i` read as `ty` (`vec4`, `ivec4`, `uvec4`), modifiers applied.
    fn src(&self, inst: &Inst, i: usize, ty: Ty) -> Result<String, String> {
        let s = inst
            .src
            .get(i)
            .ok_or_else(|| format!("{} needs operand {}", inst.op, i + 1))?;
        let e = self.swizzled(s)?;
        let mut e = match ty {
            Ty::F => e,
            Ty::I => format!("floatBitsToInt({e})"),
            Ty::U => format!("floatBitsToUint({e})"),
        };
        if s.abs {
            e = format!("abs({e})");
        }
        if s.negate {
            e = match ty {
                Ty::U => format!("uvec4(-ivec4({e}))"),
                _ => format!("(-{e})"),
            };
        }
        Ok(e)
    }

    /// Source `i`'s x component read as `ty`.
    fn src_x(&self, inst: &Inst, i: usize, ty: Ty) -> Result<String, String> {
        Ok(format!("{}.x", self.src(inst, i, ty)?))
    }

    /// Store `expr` (of `ty`'s vector type) into the destination.
    fn store(&mut self, inst: &Inst, expr: &str, ty: Ty) -> Result<(), String> {
        let d = inst
            .dst
            .first()
            .ok_or_else(|| format!("{} needs a destination", inst.op))?;
        if d.reg.file == File::Address {
            let m = mask_str(d.mask);
            let v = match ty {
                Ty::F => format!("ivec4(floor({expr}))"),
                Ty::I => expr.to_string(),
                Ty::U => format!("ivec4({expr})"),
            };
            self.line(&format!("nx_addr{}.{m} = ({v}).{m};", d.reg.index));
            return Ok(());
        }
        let mut v = match ty {
            Ty::F => expr.to_string(),
            Ty::I => format!("intBitsToFloat({expr})"),
            Ty::U => format!("uintBitsToFloat({expr})"),
        };
        if inst.sat {
            v = format!("clamp({v}, 0.0, 1.0)");
        }
        let target = match d.reg.file {
            File::Temp | File::Output => self.reg(&d.reg)?,
            f => return Err(format!("writing {f:?} is not supported")),
        };
        if d.mask == 0xf {
            self.line(&format!("{target} = {v};"));
        } else {
            let m = mask_str(d.mask);
            self.line(&format!("{target}.{m} = ({v}).{m};"));
        }
        Ok(())
    }

    fn epilogue(&mut self) {
        let outs: Vec<Decl> = self
            .sh
            .decls
            .iter()
            .filter(|d| d.file == File::Output)
            .cloned()
            .collect();
        for d in &outs {
            for i in d.first..=d.last {
                let sem = d.semantic.as_ref().map_or("", |(s, _)| s.as_str());
                let idx = d.semantic.as_ref().map_or(0, |(_, n)| *n);
                if self.is_vs() {
                    match sem {
                        "POSITION" => {
                            self.line(&format!("gl_Position = nx_o{i};"));
                            self.line("gl_Position.y *= nx_flip_y;");
                        }
                        "PSIZE" => self.line(&format!("gl_PointSize = nx_o{i}.x;")),
                        _ => {
                            if let Some(name) = varying_name(d) {
                                self.line(&format!("{name} = nx_o{i};"));
                            }
                        }
                    }
                } else {
                    match sem {
                        "COLOR" => {
                            let targets: Vec<u32> = if self.writes_all() && idx == 0 {
                                (0..self.fs_color_outputs()).collect()
                            } else {
                                vec![idx]
                            };
                            for n in targets {
                                let v = match self.cbuf_type(n) {
                                    "uvec4" => format!("floatBitsToUint(nx_o{i})"),
                                    "ivec4" => format!("floatBitsToInt(nx_o{i})"),
                                    _ => format!("nx_o{i}"),
                                };
                                self.line(&format!("nx_out{n} = {v};"));
                            }
                        }
                        "POSITION" => self.line(&format!("gl_FragDepth = nx_o{i}.z;")),
                        _ => {}
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn inst(&mut self, inst: &Inst) -> Result<(), String> {
        use Ty::{F, I, U};
        let op = inst.op.as_str();
        let s = |g: &Self, i: usize, t: Ty| g.src(inst, i, t);
        let x = |g: &Self, i: usize, t: Ty| g.src_x(inst, i, t);
        // Component-wise binary/unary forms.
        let bin = |g: &Self, f: &str, t: Ty| -> Result<String, String> {
            Ok(format!("{f}({}, {})", s(g, 0, t)?, s(g, 1, t)?))
        };
        let infix = |g: &Self, o: &str, t: Ty| -> Result<String, String> {
            Ok(format!("({} {o} {})", s(g, 0, t)?, s(g, 1, t)?))
        };
        let un = |g: &Self, f: &str, t: Ty| -> Result<String, String> {
            Ok(format!("{f}({})", s(g, 0, t)?))
        };
        let cmp_f = |g: &Self, f: &str| -> Result<String, String> {
            Ok(format!("vec4({}({}, {}))", f, s(g, 0, F)?, s(g, 1, F)?))
        };
        let cmp_bits = |g: &Self, f: &str, t: Ty| -> Result<String, String> {
            Ok(format!(
                "(uvec4({}({}, {})) * 0xffffffffu)",
                f,
                s(g, 0, t)?,
                s(g, 1, t)?
            ))
        };
        match op {
            "MOV" => {
                let e = s(self, 0, F)?;
                self.store(inst, &e, F)
            }
            "ADD" => {
                let e = infix(self, "+", F)?;
                self.store(inst, &e, F)
            }
            "MUL" => {
                let e = infix(self, "*", F)?;
                self.store(inst, &e, F)
            }
            "DIV" => {
                let e = infix(self, "/", F)?;
                self.store(inst, &e, F)
            }
            "MAD" | "FMA" => {
                let e = format!(
                    "({} * {} + {})",
                    s(self, 0, F)?,
                    s(self, 1, F)?,
                    s(self, 2, F)?
                );
                self.store(inst, &e, F)
            }
            "LRP" => {
                let e = format!(
                    "mix({}, {}, {})",
                    s(self, 2, F)?,
                    s(self, 1, F)?,
                    s(self, 0, F)?
                );
                self.store(inst, &e, F)
            }
            "DP2" | "DP3" | "DP4" => {
                let n = match op {
                    "DP2" => "xy",
                    "DP3" => "xyz",
                    _ => "",
                };
                let (a, b) = (s(self, 0, F)?, s(self, 1, F)?);
                let e = if n.is_empty() {
                    format!("vec4(dot({a}, {b}))")
                } else {
                    format!("vec4(dot(({a}).{n}, ({b}).{n}))")
                };
                self.store(inst, &e, F)
            }
            "RCP" => {
                let e = format!("vec4(1.0 / {})", x(self, 0, F)?);
                self.store(inst, &e, F)
            }
            "RSQ" => {
                let e = format!("vec4(inversesqrt({}))", x(self, 0, F)?);
                self.store(inst, &e, F)
            }
            "SQRT" => {
                let e = format!("vec4(sqrt({}))", x(self, 0, F)?);
                self.store(inst, &e, F)
            }
            "EX2" | "LG2" | "SIN" | "COS" => {
                let f = match op {
                    "EX2" => "exp2",
                    "LG2" => "log2",
                    "SIN" => "sin",
                    _ => "cos",
                };
                let e = format!("vec4({f}({}))", x(self, 0, F)?);
                self.store(inst, &e, F)
            }
            "POW" => {
                let e = format!("vec4(pow({}, {}))", x(self, 0, F)?, x(self, 1, F)?);
                self.store(inst, &e, F)
            }
            "MIN" | "MAX" => {
                let e = bin(self, &op.to_lowercase(), F)?;
                self.store(inst, &e, F)
            }
            "FRC" | "FLR" | "CEIL" | "TRUNC" | "ROUND" | "SSG" | "DDX" | "DDY" | "DDX_FINE"
            | "DDY_FINE" => {
                let f = match op {
                    "FRC" => "fract",
                    "FLR" => "floor",
                    "CEIL" => "ceil",
                    "TRUNC" => "trunc",
                    "ROUND" => "roundEven",
                    "SSG" => "sign",
                    "DDX" | "DDX_FINE" => "dFdx",
                    _ => "dFdy",
                };
                let e = un(self, f, F)?;
                self.store(inst, &e, F)
            }
            "LDEXP" => {
                let e = format!("({} * exp2(vec4({})))", s(self, 0, F)?, s(self, 1, I)?);
                self.store(inst, &e, F)
            }
            "SLT" | "SGE" | "SEQ" | "SNE" | "SLE" | "SGT" => {
                let f = match op {
                    "SLT" => "lessThan",
                    "SGE" => "greaterThanEqual",
                    "SEQ" => "equal",
                    "SNE" => "notEqual",
                    "SLE" => "lessThanEqual",
                    _ => "greaterThan",
                };
                let e = cmp_f(self, f)?;
                self.store(inst, &e, F)
            }
            "FSLT" | "FSGE" | "FSEQ" | "FSNE" => {
                let f = match op {
                    "FSLT" => "lessThan",
                    "FSGE" => "greaterThanEqual",
                    "FSEQ" => "equal",
                    _ => "notEqual",
                };
                let e = cmp_bits(self, f, F)?;
                self.store(inst, &e, U)
            }
            "USEQ" | "USNE" | "USLT" | "USGE" => {
                let f = match op {
                    "USEQ" => "equal",
                    "USNE" => "notEqual",
                    "USLT" => "lessThan",
                    _ => "greaterThanEqual",
                };
                let e = cmp_bits(self, f, U)?;
                self.store(inst, &e, U)
            }
            "ISLT" | "ISGE" => {
                let f = if op == "ISLT" {
                    "lessThan"
                } else {
                    "greaterThanEqual"
                };
                let e = cmp_bits(self, f, I)?;
                self.store(inst, &e, U)
            }
            "CMP" => {
                let e = format!(
                    "mix({}, {}, lessThan({}, vec4(0.0)))",
                    s(self, 2, F)?,
                    s(self, 1, F)?,
                    s(self, 0, F)?
                );
                self.store(inst, &e, F)
            }
            "UCMP" => {
                let e = format!(
                    "mix({}, {}, notEqual({}, uvec4(0u)))",
                    s(self, 2, F)?,
                    s(self, 1, F)?,
                    s(self, 0, U)?
                );
                self.store(inst, &e, F)
            }
            "I2F" => {
                let e = format!("vec4({})", s(self, 0, I)?);
                self.store(inst, &e, F)
            }
            "U2F" => {
                let e = format!("vec4({})", s(self, 0, U)?);
                self.store(inst, &e, F)
            }
            "F2I" => {
                let e = format!("ivec4({})", s(self, 0, F)?);
                self.store(inst, &e, I)
            }
            "F2U" => {
                let e = format!("uvec4({})", s(self, 0, F)?);
                self.store(inst, &e, U)
            }
            "UADD" => {
                let e = infix(self, "+", U)?;
                self.store(inst, &e, U)
            }
            "UMUL" => {
                let e = infix(self, "*", U)?;
                self.store(inst, &e, U)
            }
            "UMAD" => {
                let e = format!(
                    "({} * {} + {})",
                    s(self, 0, U)?,
                    s(self, 1, U)?,
                    s(self, 2, U)?
                );
                self.store(inst, &e, U)
            }
            "UDIV" => {
                // Division by zero gives all ones, as TGSI defines.
                let (a, b) = (s(self, 0, U)?, s(self, 1, U)?);
                let e = format!(
                    "mix({a} / max({b}, uvec4(1u)), uvec4(0xffffffffu), equal({b}, uvec4(0u)))"
                );
                self.store(inst, &e, U)
            }
            "UMOD" => {
                let (a, b) = (s(self, 0, U)?, s(self, 1, U)?);
                let e = format!(
                    "mix({a} % max({b}, uvec4(1u)), uvec4(0xffffffffu), equal({b}, uvec4(0u)))"
                );
                self.store(inst, &e, U)
            }
            "IDIV" => {
                let (a, b) = (s(self, 0, I)?, s(self, 1, I)?);
                let e = format!(
                    "mix({a} / mix({b}, ivec4(1), equal({b}, ivec4(0))), ivec4(0), equal({b}, ivec4(0)))"
                );
                self.store(inst, &e, I)
            }
            "MOD" => {
                let (a, b) = (s(self, 0, I)?, s(self, 1, I)?);
                let e = format!(
                    "mix({a} % mix({b}, ivec4(1), equal({b}, ivec4(0))), ivec4(0), equal({b}, ivec4(0)))"
                );
                self.store(inst, &e, I)
            }
            "IMAX" | "IMIN" => {
                let e = bin(self, if op == "IMAX" { "max" } else { "min" }, I)?;
                self.store(inst, &e, I)
            }
            "UMAX" | "UMIN" => {
                let e = bin(self, if op == "UMAX" { "max" } else { "min" }, U)?;
                self.store(inst, &e, U)
            }
            "INEG" => {
                let e = format!("(-{})", s(self, 0, I)?);
                self.store(inst, &e, I)
            }
            "IABS" => {
                let e = un(self, "abs", I)?;
                self.store(inst, &e, I)
            }
            "ISSG" => {
                let e = un(self, "sign", I)?;
                self.store(inst, &e, I)
            }
            "AND" => {
                let e = infix(self, "&", U)?;
                self.store(inst, &e, U)
            }
            "OR" => {
                let e = infix(self, "|", U)?;
                self.store(inst, &e, U)
            }
            "XOR" => {
                let e = infix(self, "^", U)?;
                self.store(inst, &e, U)
            }
            "NOT" => {
                let e = format!("(~{})", s(self, 0, U)?);
                self.store(inst, &e, U)
            }
            "SHL" => {
                let e = format!("({} << ({} & uvec4(31u)))", s(self, 0, U)?, s(self, 1, U)?);
                self.store(inst, &e, U)
            }
            "USHR" => {
                let e = format!("({} >> ({} & uvec4(31u)))", s(self, 0, U)?, s(self, 1, U)?);
                self.store(inst, &e, U)
            }
            "ISHR" => {
                let e = format!("({} >> ({} & ivec4(31)))", s(self, 0, I)?, s(self, 1, I)?);
                self.store(inst, &e, I)
            }
            "ARL" => {
                let e = un(self, "floor", F)?;
                self.store(inst, &e, F)
            }
            "UARL" => {
                let e = s(self, 0, I)?;
                self.store(inst, &e, I)
            }
            "KILL" => {
                self.line("discard;");
                Ok(())
            }
            "KILL_IF" => {
                let e = s(self, 0, F)?;
                self.line(&format!("if (any(lessThan({e}, vec4(0.0)))) discard;"));
                Ok(())
            }
            "IF" => {
                let e = x(self, 0, F)?;
                self.line(&format!("if ({e} != 0.0) {{"));
                self.indent += 1;
                Ok(())
            }
            "UIF" => {
                let e = x(self, 0, U)?;
                self.line(&format!("if ({e} != 0u) {{"));
                self.indent += 1;
                Ok(())
            }
            "ELSE" => {
                self.indent = self.indent.saturating_sub(1).max(1);
                self.line("} else {");
                self.indent += 1;
                Ok(())
            }
            "ENDIF" | "ENDLOOP" => {
                self.indent = self.indent.saturating_sub(1).max(1);
                self.line("}");
                Ok(())
            }
            "BGNLOOP" => {
                self.line("while (true) {");
                self.indent += 1;
                Ok(())
            }
            "BRK" => {
                self.line("break;");
                Ok(())
            }
            "CONT" => {
                self.line("continue;");
                Ok(())
            }
            "RET" => {
                self.epilogue();
                self.line("return;");
                Ok(())
            }
            "END" => {
                self.epilogue();
                Ok(())
            }
            "NOP" => Ok(()),
            "TEX" | "TXP" | "TXB" | "TXL" | "TXD" | "TXF" | "TXQ" => self.tex(inst),
            _ => Err(format!("TGSI opcode {op} is not supported")),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn tex(&mut self, inst: &Inst) -> Result<(), String> {
        let samp = inst
            .src
            .iter()
            .find(|s| s.reg.file == File::Sampler)
            .ok_or("texturing without a sampler")?;
        let slot = samp.reg.index as u32;
        let (ty, target) = self
            .samplers
            .get(&slot)
            .cloned()
            .ok_or("undeclared sampler")?;
        let target = inst.tex.clone().unwrap_or(target);
        let sname = format!("nx_{}s{slot}", self.st);
        let c = self.src(inst, 0, Ty::F)?;
        let shadow = ty.contains("Shadow");
        let result_ty = if ty.starts_with('u') {
            Ty::U
        } else if ty.starts_with('i') {
            Ty::I
        } else {
            Ty::F
        };
        // Coordinates (with the comparison value for shadow lookups) and the
        // derivative / fetch dimensionality.
        let (coord, dims): (String, usize) = match target.as_str() {
            "1D" => (format!("vec2(({c}).x, 0.5)"), 2),
            "2D" => (format!("({c}).xy"), 2),
            "RECT" => (format!("(({c}).xy / vec2(textureSize({sname}, 0)))"), 2),
            "3D" | "CUBE" | "2D_ARRAY" => (
                format!("({c}).xyz"),
                if target == "2D_ARRAY" { 2 } else { 3 },
            ),
            "1D_ARRAY" => (format!("vec3(({c}).x, 0.5, ({c}).y)"), 2),
            "SHADOW1D" => (format!("vec3(({c}).x, 0.5, ({c}).z)"), 2),
            "SHADOW2D" => (format!("({c}).xyz"), 2),
            "SHADOWRECT" => (
                format!("vec3(({c}).xy / vec2(textureSize({sname}, 0)), ({c}).z)"),
                2,
            ),
            "SHADOWCUBE" | "SHADOW2D_ARRAY" => {
                (format!("({c})"), if target == "SHADOWCUBE" { 3 } else { 2 })
            }
            "SHADOW1D_ARRAY" => (format!("vec4(({c}).x, 0.5, ({c}).y, ({c}).z)"), 2),
            t => return Err(format!("texture target {t} is not supported")),
        };
        let offsets = if let Some(o) = inst.tex_offsets.first() {
            let comps: Vec<String> = (0..dims)
                .map(|i| self.const_int(o, i))
                .collect::<Result<_, _>>()?;
            Some(format!("ivec{dims}({})", comps.join(", ")))
        } else {
            None
        };
        let call = match inst.op.as_str() {
            "TEX" => match &offsets {
                Some(o) => format!("textureOffset({sname}, {coord}, {o})"),
                None => format!("texture({sname}, {coord})"),
            },
            "TXP" => {
                let proj = format!("(({c}) / ({c}).w)");
                let coord = coord.replace(&c, &proj);
                format!("texture({sname}, {coord})")
            }
            "TXB" => {
                let bias = format!("({c}).w");
                format!("texture({sname}, {coord}, {bias})")
            }
            "TXL" => {
                let lod = format!("({c}).w");
                match &offsets {
                    Some(o) => format!("textureLodOffset({sname}, {coord}, {lod}, {o})"),
                    None => format!("textureLod({sname}, {coord}, {lod})"),
                }
            }
            "TXD" => {
                let sw = ["", "x", "xy", "xyz"][dims];
                let (dx, dy) = (self.src(inst, 1, Ty::F)?, self.src(inst, 2, Ty::F)?);
                format!("textureGrad({sname}, {coord}, ({dx}).{sw}, ({dy}).{sw})")
            }
            "TXF" => {
                let ci = self.src(inst, 0, Ty::I)?;
                let (p, n) = match target.as_str() {
                    "2D" | "RECT" => (format!("({ci}).xy"), 2),
                    "1D" => (format!("ivec2(({ci}).x, 0)"), 2),
                    "3D" | "2D_ARRAY" => (format!("({ci}).xyz"), 3),
                    "1D_ARRAY" => (format!("ivec3(({ci}).x, 0, ({ci}).y)"), 3),
                    t => return Err(format!("TXF on {t} is not supported")),
                };
                let _ = n;
                let lod = format!("({ci}).w");
                match &offsets {
                    Some(o) => format!("texelFetchOffset({sname}, {p}, {lod}, {o})"),
                    None => format!("texelFetch({sname}, {p}, {lod})"),
                }
            }
            "TXQ" => {
                let lod = self.src_x(inst, 0, Ty::I)?;
                let sz = format!("textureSize({sname}, {lod})");
                let v = match target.as_str() {
                    "3D" | "2D_ARRAY" | "SHADOW2D_ARRAY" | "1D_ARRAY" | "SHADOW1D_ARRAY" => {
                        format!("ivec4({sz}, 1)")
                    }
                    _ => format!("ivec4({sz}, 0, 1)"),
                };
                return self.store(inst, &v, Ty::I);
            }
            o => return Err(format!("{o} is not supported")),
        };
        if shadow {
            let e = format!("vec4({call})");
            return self.store(inst, &e, Ty::F);
        }
        self.store(inst, &call, result_ty)
    }

    /// Component `i` of an immediate operand, as an integer literal.
    fn const_int(&self, o: &Src, i: usize) -> Result<String, String> {
        if o.reg.file != File::Immediate {
            return Err("texel offsets must be immediates".into());
        }
        let imm = self
            .sh
            .imms
            .get(o.reg.index as usize)
            .ok_or("undeclared immediate")?;
        Ok((imm.words[o.swizzle[i] as usize].cast_signed()).to_string())
    }
}

/// An immediate word as a GLSL float expression of the same bits.
fn imm_literal(ty: ImmType, w: u32) -> String {
    let f = f32::from_bits(w);
    if ty == ImmType::Float && f.is_finite() && (f != 0.0 || w == 0) {
        let s = format!("{f:?}");
        if s.contains(['.', 'e']) {
            s
        } else {
            format!("{s}.0")
        }
    } else {
        format!("uintBitsToFloat({w}u)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::tgsi::parse;

    const VS: &str = "VERT
DCL IN[0]
DCL IN[1]
DCL OUT[0], POSITION
DCL OUT[1], GENERIC[0]
DCL CONST[0][0..3]
DCL TEMP[0]
IMM[0] FLT32 { 0x3f800000, 0x00000000, 0x80000000, 0x7fc00000}
  0: MUL TEMP[0], IN[0].xxxx, CONST[0][0]
  1: MAD TEMP[0], IN[0].yyyy, CONST[0][1], TEMP[0]
  2: ADD OUT[0], TEMP[0], CONST[0][3]
  3: MOV OUT[1], IN[1]
  4: END
";

    const FS: &str = "FRAG
PROPERTY FS_COLOR0_WRITES_ALL_CBUFS 1
DCL IN[0], GENERIC[0], CONSTANT
DCL OUT[0], COLOR
DCL CONST[0][0]
DCL SAMP[0]
DCL SVIEW[0], 2D, FLOAT
DCL TEMP[0]
  0: TEX TEMP[0], IN[0].xyyy, SAMP[0], 2D
  1: MUL_SAT OUT[0], TEMP[0], CONST[0][0].xxxx
  2: END
";

    #[test]
    fn links_a_textured_pair() {
        let vs = parse(VS).unwrap();
        let fs = parse(FS).unwrap();
        let p = link(
            &vs,
            &fs,
            &Options {
                nr_cbufs: 2,
                ..Options::default()
            },
        )
        .unwrap();
        assert!(p.vs.starts_with("#version 300 es\n"));
        assert!(p.vs.contains("layout(location = 0) in vec4 nx_a0;"));
        assert!(p.vs.contains("uniform uvec4 nx_vsc[4];"));
        // The varying is flat on both sides, as the fragment shader asks.
        assert!(p.vs.contains("flat out vec4 nx_v_g0;"));
        assert!(p.fs.contains("flat in vec4 nx_v_g0;"));
        assert!(p.vs.contains("gl_Position = nx_o0;"));
        assert!(p.vs.contains("gl_Position.y *= nx_flip_y;"));
        assert!(p.vs.contains("nx_t0 = (nx_a0.xxxx * uintBitsToFloat(nx_vsc[0]));"));
        // -0.0 and NaN immediates keep their bits.
        assert!(p.vs.contains(
            "vec4(1.0, 0.0, uintBitsToFloat(2147483648u), uintBitsToFloat(2143289344u))"
        ));
        assert!(p.fs.contains("uniform sampler2D nx_fss0;"));
        assert!(p.fs.contains("texture(nx_fss0, (nx_v_g0.xyyy).xy)"));
        assert!(p.fs.contains("clamp("));
        // Colour 0 goes to both bound buffers.
        assert!(p.fs.contains("nx_out0 = nx_o0;") && p.fs.contains("nx_out1 = nx_o0;"));
        assert_eq!(p.layout.consts, [4, 1]);
        assert_eq!(p.layout.samplers, [vec![], vec![0]]);
    }

    #[test]
    fn integer_ops_go_through_bit_casts() {
        let vs = parse(
            "VERT
DCL IN[0]
DCL OUT[0], POSITION
DCL TEMP[0..1]
IMM[0] UINT32 {1, 2, 3, 4}
  0: UADD TEMP[0], IMM[0], IMM[0].yyyy
  1: USEQ TEMP[1], TEMP[0], IMM[0]
  2: UIF TEMP[1].xxxx :0
  3:   I2F TEMP[0], TEMP[0]
  4: ENDIF
  5: MOV OUT[0], TEMP[0]
  6: END
",
        )
        .unwrap();
        let fs = parse("FRAG\nDCL OUT[0], COLOR\n  0: END\n").unwrap();
        let p = link(&vs, &fs, &Options::default()).unwrap();
        assert!(p.vs.contains(
            "uintBitsToFloat((floatBitsToUint(nx_imm0) + floatBitsToUint(nx_imm0.yyyy)))"
        ));
        assert!(
            p.vs.contains("if (floatBitsToUint(nx_t1.xxxx).x != 0u) {"),
            "{}",
            p.vs
        );
        assert!(p.vs.contains("vec4(floatBitsToInt(nx_t0))"));
    }

    #[test]
    fn unsupported_things_are_errors() {
        let vs = parse("VERT\nDCL OUT[0], POSITION\n  0: BARRIER\n  1: END\n").unwrap();
        let fs = parse("FRAG\n  0: END\n").unwrap();
        assert!(link(&vs, &fs, &Options::default()).is_err());
        assert!(link(&fs, &vs, &Options::default()).is_err());
    }
}
