//! TGSI, as Mesa's `virgl` driver sends shaders: the text `tgsi_dump_str`
//! writes (with `TGSI_DUMP_FLOAT_AS_HEX`), parsed into a small IR that
//! [`super::glsl`] turns into GLSL ES 3.00.
//!
//! ```text
//! VERT
//! DCL IN[0]
//! DCL OUT[0], POSITION
//! DCL CONST[0][0..3]
//! DCL TEMP[0], LOCAL
//! IMM[0] FLT32 { 0x3f800000, 0x00000000, 0x00000000, 0x00000000}
//!   0: MUL TEMP[0], IN[0].xxxx, CONST[0][0]
//!   1: MOV OUT[0], TEMP[0]
//!   2: END
//! ```

use std::fmt;

/// A shader stage, as TGSI's header line names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
    Geometry,
    TessCtrl,
    TessEval,
    Compute,
}

impl Stage {
    /// The virgl / Gallium shader type number (`PIPE_SHADER_*`).
    #[must_use]
    pub fn from_pipe(n: u32) -> Option<Self> {
        Some(match n {
            0 => Self::Vertex,
            1 => Self::Fragment,
            2 => Self::Geometry,
            3 => Self::TessCtrl,
            4 => Self::TessEval,
            5 => Self::Compute,
            _ => return None,
        })
    }
}

/// A register file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum File {
    Null,
    Const,
    Input,
    Output,
    Temp,
    Sampler,
    Address,
    Immediate,
    SystemValue,
    Image,
    SamplerView,
    Buffer,
    Memory,
    HwAtomic,
}

impl File {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "NULL" => Self::Null,
            "CONST" => Self::Const,
            "IN" => Self::Input,
            "OUT" => Self::Output,
            "TEMP" => Self::Temp,
            "SAMP" => Self::Sampler,
            "ADDR" => Self::Address,
            "IMM" => Self::Immediate,
            "SV" => Self::SystemValue,
            "IMAGE" => Self::Image,
            "SVIEW" => Self::SamplerView,
            "BUFFER" => Self::Buffer,
            "MEMORY" => Self::Memory,
            "HWATOMIC" => Self::HwAtomic,
            _ => return None,
        })
    }
}

/// `FILE[INDEX].c` used as an index: `ADDR[0].x`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Indirect {
    pub file: File,
    pub index: u32,
    /// The component (0..=3).
    pub comp: u8,
}

/// A register reference: `CONST[1][ADDR[0].x+4]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reg {
    pub file: File,
    /// The (base) index; with `indirect`, added to the address value.
    pub index: i32,
    pub indirect: Option<Indirect>,
    /// The second dimension written first (`CONST[1][…]`: buffer 1).
    pub dim: Option<i32>,
    pub dim_indirect: Option<Indirect>,
}

/// A source operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Src {
    pub reg: Reg,
    /// Component selected for x, y, z, w (0..=3).
    pub swizzle: [u8; 4],
    pub negate: bool,
    pub abs: bool,
}

/// A destination operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dst {
    pub reg: Reg,
    /// Bits 0..=3: x, y, z, w.
    pub mask: u8,
}

/// An instruction.
#[derive(Debug, Clone, PartialEq)]
pub struct Inst {
    pub op: String,
    pub sat: bool,
    pub dst: Vec<Dst>,
    pub src: Vec<Src>,
    /// The texture target of a texturing instruction (`2D`, `SHADOW2D` …).
    pub tex: Option<String>,
    /// Texel offsets (`IMM[0].xyz` …) of a texturing instruction.
    pub tex_offsets: Vec<Src>,
}

/// A declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decl {
    pub file: File,
    pub first: u32,
    pub last: u32,
    /// The second dimension (`DCL CONST[1][0..3]`: 1).
    pub dim: Option<u32>,
    pub array: Option<u32>,
    /// Semantic name and index (`GENERIC[3]`).
    pub semantic: Option<(String, u32)>,
    /// Fragment input interpolation (`PERSPECTIVE`, `LINEAR`, `CONSTANT`,
    /// `COLOR`) and location (`CENTROID`, `SAMPLE`).
    pub interp: Option<String>,
    pub location: Option<String>,
    /// A sampler view's texture target and return type.
    pub sview: Option<(String, String)>,
    pub invariant: bool,
}

/// An immediate: four words and how to read them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Imm {
    pub ty: ImmType,
    pub words: [u32; 4],
    /// How many of the words were given.
    pub len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImmType {
    Float,
    Uint,
    Int,
}

/// A parsed shader.
#[derive(Debug, Clone, PartialEq)]
pub struct Shader {
    pub stage: Stage,
    pub decls: Vec<Decl>,
    pub imms: Vec<Imm>,
    pub insts: Vec<Inst>,
    /// `PROPERTY NAME value[, value…]`.
    pub props: Vec<(String, Vec<String>)>,
}

/// Why a shader didn't parse: the line and what was wrong with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub msg: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TGSI line {}: {}", self.line, self.msg)
    }
}

impl Shader {
    /// The value of property `name`, if declared.
    #[must_use]
    pub fn prop(&self, name: &str) -> Option<&[String]> {
        self.props
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
    }

    /// The declaration covering register `index` of `file` (first dimension
    /// `dim` when given).
    #[must_use]
    pub fn decl(&self, file: File, index: u32) -> Option<&Decl> {
        self.decls
            .iter()
            .find(|d| d.file == file && d.first <= index && index <= d.last)
    }
}

/// A cursor over one line.
struct Lex<'a> {
    s: &'a str,
}

impl<'a> Lex<'a> {
    fn ws(&mut self) {
        self.s = self.s.trim_start();
    }
    fn eat(&mut self, p: &str) -> bool {
        self.ws();
        if let Some(r) = self.s.strip_prefix(p) {
            self.s = r;
            true
        } else {
            false
        }
    }
    fn word(&mut self) -> &'a str {
        self.ws();
        let end = self
            .s
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(self.s.len());
        let (w, r) = self.s.split_at(end);
        self.s = r;
        w
    }
    fn int(&mut self) -> Result<i64, String> {
        self.ws();
        let neg = self.eat("-");
        let w = self.word();
        let v = if let Some(h) = w.strip_prefix("0x") {
            i64::from_str_radix(h, 16)
        } else {
            w.parse::<i64>()
        }
        .map_err(|_| format!("bad number {w:?}"))?;
        Ok(if neg { -v } else { v })
    }
    fn expect(&mut self, p: &str) -> Result<(), String> {
        if self.eat(p) {
            Ok(())
        } else {
            Err(format!("expected {p:?} at {:?}", self.s))
        }
    }
    fn done(&mut self) -> bool {
        self.ws();
        self.s.is_empty()
    }
}

fn comp(c: char) -> Option<u8> {
    Some(match c {
        'x' => 0,
        'y' => 1,
        'z' => 2,
        'w' => 3,
        _ => return None,
    })
}

/// `FILE[IDX].c` inside brackets.
fn indirect(lx: &mut Lex) -> Result<Indirect, String> {
    let f = lx.word();
    let file = File::parse(f).ok_or_else(|| format!("bad file {f:?}"))?;
    lx.expect("[")?;
    let index = lx.int()? as u32;
    lx.expect("]")?;
    lx.expect(".")?;
    let c = lx.word();
    let comp = c
        .chars()
        .next()
        .and_then(comp)
        .ok_or_else(|| format!("bad swizzle {c:?}"))?;
    Ok(Indirect { file, index, comp })
}

/// One `[…]` index: a number, or `ADDR[0].x+3`, then an optional `(array)`.
fn index(lx: &mut Lex) -> Result<(i32, Option<Indirect>), String> {
    lx.expect("[")?;
    lx.ws();
    let r = if lx.s.starts_with(|c: char| c.is_ascii_uppercase()) {
        let ind = indirect(lx)?;
        // `+n`, or `-n` (which `int` reads with its sign).
        let off = if lx.eat("+") || lx.s.trim_start().starts_with('-') {
            lx.int()? as i32
        } else {
            0
        };
        (off, Some(ind))
    } else {
        (lx.int()? as i32, None)
    };
    lx.expect("]")?;
    if lx.s.starts_with('(') {
        lx.expect("(")?;
        lx.int()?;
        lx.expect(")")?;
    }
    Ok(r)
}

fn reg(lx: &mut Lex) -> Result<Reg, String> {
    let f = lx.word();
    let file = File::parse(f).ok_or_else(|| format!("bad file {f:?}"))?;
    let (a, ai) = index(lx)?;
    lx.ws();
    if lx.s.starts_with('[') {
        let (b, bi) = index(lx)?;
        Ok(Reg {
            file,
            index: b,
            indirect: bi,
            dim: Some(a),
            dim_indirect: ai,
        })
    } else {
        Ok(Reg {
            file,
            index: a,
            indirect: ai,
            dim: None,
            dim_indirect: None,
        })
    }
}

fn mask(lx: &mut Lex) -> Result<u8, String> {
    if !lx.s.starts_with('.') {
        return Ok(0xf);
    }
    lx.expect(".")?;
    let w = lx.word();
    let mut m = 0;
    for c in w.chars() {
        m |= 1 << comp(c).ok_or_else(|| format!("bad writemask {w:?}"))?;
    }
    Ok(m)
}

fn src(lx: &mut Lex) -> Result<Src, String> {
    let negate = lx.eat("-");
    let abs = lx.eat("|");
    let reg = reg(lx)?;
    let mut swizzle = [0, 1, 2, 3];
    if lx.s.starts_with('.') {
        lx.expect(".")?;
        let w = lx.word();
        let cs: Vec<u8> = w.chars().filter_map(comp).collect();
        if cs.is_empty() || cs.len() != w.len() {
            return Err(format!("bad swizzle {w:?}"));
        }
        for (i, s) in swizzle.iter_mut().enumerate() {
            *s = cs[i.min(cs.len() - 1)];
        }
    }
    if abs {
        lx.expect("|")?;
    }
    Ok(Src {
        reg,
        swizzle,
        negate,
        abs,
    })
}

const TEX_TARGETS: &[&str] = &[
    "BUFFER",
    "1D",
    "2D",
    "3D",
    "CUBE",
    "RECT",
    "SHADOW1D",
    "SHADOW2D",
    "SHADOWRECT",
    "1D_ARRAY",
    "2D_ARRAY",
    "SHADOW1D_ARRAY",
    "SHADOW2D_ARRAY",
    "SHADOWCUBE",
    "2D_MSAA",
    "2D_ARRAY_MSAA",
    "CUBEARRAY",
    "SHADOWCUBEARRAY",
    "UNKNOWN",
];

fn decl(lx: &mut Lex) -> Result<Decl, String> {
    let f = lx.word();
    let file = File::parse(f).ok_or_else(|| format!("bad file {f:?}"))?;
    // Geometry/tessellation inputs carry an empty `[]` first.
    lx.eat("[]");
    lx.expect("[")?;
    let a = lx.int()? as u32;
    let (mut first, mut last) = (a, a);
    if lx.eat("..") {
        last = lx.int()? as u32;
    }
    lx.expect("]")?;
    let mut dim = None;
    if lx.s.starts_with('[') {
        dim = Some(first);
        lx.expect("[")?;
        first = lx.int()? as u32;
        last = first;
        if lx.eat("..") {
            last = lx.int()? as u32;
        }
        lx.expect("]")?;
    }
    mask(lx)?;
    let mut d = Decl {
        file,
        first,
        last,
        dim,
        array: None,
        semantic: None,
        interp: None,
        location: None,
        sview: None,
        invariant: false,
    };
    while lx.eat(",") {
        let w = lx.word();
        match w {
            "ARRAY" => {
                lx.expect("(")?;
                d.array = Some(lx.int()? as u32);
                lx.expect(")")?;
            }
            "LOCAL" | "WR" | "RAW" | "ATOMIC" | "GLOBAL" | "SHARED" | "PRIVATE" | "INPUT" => {}
            "INVARIANT" => d.invariant = true,
            // `COLOR` is a semantic first, an interpolation mode after one.
            "CONSTANT" | "LINEAR" | "PERSPECTIVE" | "COLOR"
                if w != "COLOR" || d.semantic.is_some() =>
            {
                d.interp = Some(w.into());
            }
            "CENTER" | "CENTROID" | "SAMPLE" => d.location = Some(w.into()),
            "STREAM" => {
                lx.expect("(")?;
                while !lx.eat(")") {
                    lx.word();
                    lx.eat(",");
                    if lx.done() {
                        break;
                    }
                }
            }
            _ if file == File::SamplerView && TEX_TARGETS.contains(&w) => {
                lx.expect(",")?;
                let ret = lx.word().to_string();
                // Per-channel return types repeat; the first serves.
                while lx.s.trim_start().starts_with(',') {
                    let save = lx.s;
                    lx.eat(",");
                    let w2 = lx.word();
                    if !["UNORM", "SNORM", "SINT", "UINT", "FLOAT"].contains(&w2) {
                        lx.s = save;
                        break;
                    }
                }
                d.sview = Some((w.into(), ret));
            }
            _ if file == File::Image => {
                // IMAGE: target, format — kept as an unsupported decl.
                while lx.eat(",") {
                    lx.word();
                }
            }
            _ if !w.is_empty() => {
                let mut idx = 0;
                if lx.s.starts_with('[') {
                    lx.expect("[")?;
                    idx = lx.int()? as u32;
                    lx.expect("]")?;
                }
                d.semantic = Some((w.into(), idx));
            }
            _ => return Err(format!("bad declaration at {:?}", lx.s)),
        }
    }
    Ok(d)
}

fn imm(lx: &mut Lex) -> Result<Imm, String> {
    lx.expect("[")?;
    lx.int()?;
    lx.expect("]")?;
    let ty = match lx.word() {
        "FLT32" => ImmType::Float,
        "UINT32" => ImmType::Uint,
        "INT32" => ImmType::Int,
        t => return Err(format!("unsupported immediate type {t}")),
    };
    lx.expect("{")?;
    let mut words = [0u32; 4];
    let mut len = 0;
    while !lx.eat("}") {
        if len == 4 {
            return Err("too many immediate words".into());
        }
        lx.ws();
        let v = if ty == ImmType::Float && !lx.s.starts_with("0x") {
            // A decimal float (a dump without FLOAT_AS_HEX).
            let end = lx.s.find([',', '}']).unwrap_or(lx.s.len());
            let (t, r) = lx.s.split_at(end);
            lx.s = r;
            t.trim()
                .parse::<f32>()
                .map_err(|_| format!("bad float {t:?}"))?
                .to_bits()
        } else {
            lx.int()? as u32
        };
        words[len] = v;
        len += 1;
        lx.eat(",");
    }
    Ok(Imm { ty, words, len })
}

fn inst(lx: &mut Lex) -> Result<Inst, String> {
    let mut op = lx.word().to_string();
    let mut sat = false;
    if let Some(o) = op.strip_suffix("_PRECISE") {
        op = o.to_string();
    }
    if let Some(o) = op.strip_suffix("_SAT") {
        op = o.to_string();
        sat = true;
    }
    let mut inst = Inst {
        op,
        sat,
        dst: Vec::new(),
        src: Vec::new(),
        tex: None,
        tex_offsets: Vec::new(),
    };
    let (ndst, _) = arity(&inst.op);
    let mut first = true;
    loop {
        lx.ws();
        if lx.s.is_empty() || lx.s.starts_with(':') {
            break;
        }
        if !first {
            lx.expect(",")?;
        }
        first = false;
        lx.ws();
        let save = lx.s;
        let w = lx.word();
        if TEX_TARGETS.contains(&w) && !lx.s.starts_with('[') {
            inst.tex = Some(w.into());
            continue;
        }
        if ["COHERENT", "RESTRICT", "VOLATILE"].contains(&w) {
            continue;
        }
        lx.s = save;
        if inst.dst.len() < ndst {
            let reg = reg(lx)?;
            let mask = mask(lx)?;
            inst.dst.push(Dst { reg, mask });
        } else if inst.tex.is_some() {
            inst.tex_offsets.push(src(lx)?);
        } else {
            inst.src.push(src(lx)?);
        }
    }
    Ok(inst)
}

/// How many destination and source operands opcode `op` takes.
#[must_use]
pub fn arity(op: &str) -> (usize, usize) {
    match op {
        "END" | "RET" | "BRK" | "CONT" | "ELSE" | "ENDIF" | "BGNLOOP" | "ENDLOOP" | "KILL"
        | "NOP" | "BGNSUB" | "ENDSUB" | "DEFAULT" | "ENDSWITCH" | "BARRIER" | "DEMOTE" => (0, 0),
        "IF" | "UIF" | "KILL_IF" | "SWITCH" | "CASE" | "EMIT" | "ENDPRIM" | "MEMBAR" => (0, 1),
        "STORE" => (1, 2),
        _ => (1, 4),
    }
}

/// Parse TGSI text.
///
/// # Errors
/// The first line that isn't TGSI this parser knows.
pub fn parse(text: &str) -> Result<Shader, ParseError> {
    let mut lines = text
        .split('\n')
        .map(|l| l.trim_end_matches('\0').trim())
        .enumerate()
        .filter(|(_, l)| !l.is_empty());
    let (_, head) = lines.next().ok_or(ParseError {
        line: 1,
        msg: "empty shader".into(),
    })?;
    let stage = match head {
        "VERT" => Stage::Vertex,
        "FRAG" => Stage::Fragment,
        "GEOM" => Stage::Geometry,
        "TESS_CTRL" => Stage::TessCtrl,
        "TESS_EVAL" => Stage::TessEval,
        "COMP" => Stage::Compute,
        _ => {
            return Err(ParseError {
                line: 1,
                msg: format!("unknown processor {head:?}"),
            });
        }
    };
    let mut sh = Shader {
        stage,
        decls: Vec::new(),
        imms: Vec::new(),
        insts: Vec::new(),
        props: Vec::new(),
    };
    for (n, line) in lines {
        let mut lx = Lex { s: line };
        let r: Result<(), String> = (|| {
            if lx.eat("DCL") {
                sh.decls.push(decl(&mut lx)?);
            } else if lx.eat("IMM") {
                sh.imms.push(imm(&mut lx)?);
            } else if lx.eat("PROPERTY") {
                let name = lx.word().to_string();
                let vals =
                    lx.s.split(',')
                        .map(|v| v.trim().to_string())
                        .filter(|v| !v.is_empty())
                        .collect();
                sh.props.push((name, vals));
            } else if lx.s.starts_with(|c: char| c.is_ascii_digit()) {
                lx.int()?;
                lx.expect(":")?;
                sh.insts.push(inst(&mut lx)?);
            } else {
                return Err(format!("unexpected {line:?}"));
            }
            Ok(())
        })();
        r.map_err(|msg| ParseError { line: n + 1, msg })?;
    }
    Ok(sh)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VS: &str = "VERT
DCL IN[0]
DCL IN[1]
DCL OUT[0], POSITION
DCL OUT[1], GENERIC[0]
DCL CONST[0][0..3]
DCL CONST[1][0..1], ARRAY(1)
DCL TEMP[0..1], LOCAL
DCL ADDR[0]
IMM[0] FLT32 {    0x3f800000,    0x00000000,    0x00000000,    0x00000000}
IMM[1] UINT32 {1, 4294967295, 0, 0}
PROPERTY NEXT_SHADER FRAG
  0: MUL TEMP[0], IN[0].xxxx, CONST[0][0]
  1: MAD_SAT TEMP[0], IN[0].yyyy, CONST[0][1], -|TEMP[0]|
  2: UARL ADDR[0].x, IMM[1].xxxx
  3: MOV TEMP[1].xy, CONST[0][ADDR[0].x+2].wzyx
  4: MOV OUT[0], TEMP[0]
  5: MOV OUT[1], IN[1]
  6: END
";

    #[test]
    fn parses_a_vertex_shader() {
        let sh = parse(VS).unwrap();
        assert_eq!(sh.stage, Stage::Vertex);
        assert_eq!(sh.decls.len(), 8);
        let out1 = sh.decl(File::Output, 1).unwrap();
        assert_eq!(out1.semantic, Some(("GENERIC".into(), 0)));
        let c1 = &sh.decls[4];
        assert_eq!(
            (c1.file, c1.dim, c1.first, c1.last),
            (File::Const, Some(0), 0, 3)
        );
        assert_eq!(sh.decls[5].array, Some(1));
        assert_eq!(sh.decls[5].dim, Some(1));
        assert_eq!(sh.imms[0].words[0], 0x3f80_0000);
        assert_eq!(sh.imms[1].words[1], u32::MAX);
        assert_eq!(sh.prop("NEXT_SHADER"), Some(&["FRAG".to_string()][..]));
        assert_eq!(sh.insts.len(), 7);
        let mad = &sh.insts[1];
        assert!(mad.sat);
        assert_eq!(mad.op, "MAD");
        assert!(mad.src[2].negate && mad.src[2].abs);
        let mov = &sh.insts[3];
        assert_eq!(mov.dst[0].mask, 0b0011);
        let r = mov.src[0].reg;
        assert_eq!((r.file, r.dim, r.index), (File::Const, Some(0), 2));
        assert_eq!(
            r.indirect,
            Some(Indirect {
                file: File::Address,
                index: 0,
                comp: 0
            })
        );
        assert_eq!(mov.src[0].swizzle, [3, 2, 1, 0]);
    }

    #[test]
    fn parses_fragment_declarations_and_texturing() {
        let fs = "FRAG
PROPERTY FS_COLOR0_WRITES_ALL_CBUFS 1
DCL IN[0], GENERIC[0], PERSPECTIVE
DCL IN[1], COLOR, COLOR, CENTROID
DCL OUT[0], COLOR
DCL SAMP[0]
DCL SVIEW[0], 2D, FLOAT
  0: TEX TEMP[0], IN[0].xyyy, SAMP[0], 2D
  1: TXF TEMP[0], IN[0], SAMP[0], 2D, IMM[0].xyz
  2: KILL_IF -IN[0].xxxx
  3: UIF TEMP[0].xxxx :0
  4: ELSE :0
  5: ENDIF
  6: MOV OUT[0], TEMP[0]
  7: END
";
        let sh = parse(fs).unwrap();
        assert_eq!(sh.stage, Stage::Fragment);
        let in0 = sh.decl(File::Input, 0).unwrap();
        assert_eq!(in0.interp.as_deref(), Some("PERSPECTIVE"));
        let in1 = sh.decl(File::Input, 1).unwrap();
        assert_eq!(in1.semantic, Some(("COLOR".into(), 0)));
        assert_eq!(in1.interp.as_deref(), Some("COLOR"));
        assert_eq!(in1.location.as_deref(), Some("CENTROID"));
        let sv = sh.decl(File::SamplerView, 0).unwrap();
        assert_eq!(sv.sview, Some(("2D".into(), "FLOAT".into())));
        assert_eq!(sh.insts[0].tex.as_deref(), Some("2D"));
        assert_eq!(sh.insts[0].src.len(), 2);
        assert_eq!(sh.insts[1].tex_offsets.len(), 1);
        assert_eq!(sh.insts[2].src[0].swizzle, [0, 0, 0, 0]);
        assert_eq!(sh.insts[3].op, "UIF");
        assert_eq!(sh.insts[3].src.len(), 1);
    }

    #[test]
    fn reports_bad_lines() {
        let e = parse("VERT\nDCL FOO[0]\n").unwrap_err();
        assert_eq!(e.line, 2);
        assert!(parse("").is_err());
    }
}
