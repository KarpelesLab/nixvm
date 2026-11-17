//! The embedding API (`Sandbox::execute`): a chosen working directory, binds
//! at arbitrary guest paths without the implicit `/work`, stdio as the
//! caller's readers and writers, and a `RunOutcome` instead of anything
//! printed.
//!
//! The always-run tests use tiny hand-assembled aarch64 programs on the
//! interpreter (host independent). With `NIXVM_ALPINE_TAR` set, a real shell
//! runs `echo hi; pwd` too.

use std::path::{Path, PathBuf};
use std::time::Duration;

use nixvm::abi::Arch;
use nixvm::kernel::notice::Notice;
use nixvm::sandbox::{Capture, LimitHit, Network, Sandbox, SandboxBuilder};

const EHDR_LEN: usize = 64;
const PHDR_LEN: usize = 56;
const VADDR: u64 = 0x1_0000;
const BODY_OFF: u64 = (EHDR_LEN + PHDR_LEN) as u64;

/// A static aarch64 ET_EXEC: one RWX `PT_LOAD` at [`VADDR`] covering the
/// headers and `body`, entry at the start of `body`.
fn build_elf(body: &[u8]) -> Vec<u8> {
    let mut f = vec![0u8; EHDR_LEN + PHDR_LEN];
    f[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    f[4] = 2;
    f[5] = 1;
    f[6] = 1;
    f[16..18].copy_from_slice(&2u16.to_le_bytes());
    f[18..20].copy_from_slice(&183u16.to_le_bytes()); // EM_AARCH64
    f[20..24].copy_from_slice(&1u32.to_le_bytes());
    f[24..32].copy_from_slice(&(VADDR + BODY_OFF).to_le_bytes());
    f[32..40].copy_from_slice(&(EHDR_LEN as u64).to_le_bytes());
    f[52..54].copy_from_slice(&(EHDR_LEN as u16).to_le_bytes());
    f[54..56].copy_from_slice(&(PHDR_LEN as u16).to_le_bytes());
    f[56..58].copy_from_slice(&1u16.to_le_bytes());
    let total = BODY_OFF + body.len() as u64;
    let p = EHDR_LEN;
    f[p..p + 4].copy_from_slice(&1u32.to_le_bytes()); // PT_LOAD
    f[p + 4..p + 8].copy_from_slice(&7u32.to_le_bytes()); // RWX
    f[p + 16..p + 24].copy_from_slice(&VADDR.to_le_bytes());
    f[p + 24..p + 32].copy_from_slice(&VADDR.to_le_bytes());
    f[p + 32..p + 40].copy_from_slice(&total.to_le_bytes());
    f[p + 40..p + 48].copy_from_slice(&total.to_le_bytes());
    f[p + 48..p + 56].copy_from_slice(&4096u64.to_le_bytes());
    f.extend_from_slice(body);
    f
}

/// Just enough of an aarch64 assembler for the programs below: code, then
/// data placed after it.
#[derive(Default)]
struct Asm {
    code: Vec<u32>,
    /// `(code index of a movz/movk pair, data offset)` to patch.
    fixups: Vec<(usize, usize)>,
    data: Vec<u8>,
}

impl Asm {
    fn movz(&mut self, rd: u32, imm: u16) {
        self.code.push(0xD280_0000 | (u32::from(imm) << 5) | rd);
    }
    /// `movn xd, #imm`: `xd = !imm`.
    fn movn(&mut self, rd: u32, imm: u16) {
        self.code.push(0x9280_0000 | (u32::from(imm) << 5) | rd);
    }
    /// `xd = address of data at `off``, as a movz/movk pair.
    fn adr_data(&mut self, rd: u32, off: usize) {
        self.fixups.push((self.code.len(), off));
        self.code.push(0xD280_0000 | rd);
        self.code.push(0xF2A0_0000 | rd);
    }
    /// `mov xd, x0`.
    fn mov_from_x0(&mut self, rd: u32) {
        self.code.push(0xAA00_03E0 | rd);
    }
    /// `sub xd, xn, #imm`.
    fn sub_imm(&mut self, rd: u32, rn: u32, imm: u32) {
        self.code.push(0xD100_0000 | (imm << 10) | (rn << 5) | rd);
    }
    fn syscall(&mut self, nr: u16) {
        self.movz(8, nr);
        self.code.push(0xD400_0001); // svc #0
    }
    fn data(&mut self, bytes: &[u8]) -> usize {
        let off = self.data.len();
        self.data.extend_from_slice(bytes);
        off
    }
    fn finish(mut self) -> Vec<u8> {
        let data_addr = VADDR + BODY_OFF + self.code.len() as u64 * 4;
        for (at, off) in std::mem::take(&mut self.fixups) {
            let addr = data_addr + off as u64;
            assert!(addr < 1 << 32);
            self.code[at] |= ((addr & 0xffff) as u32) << 5;
            self.code[at + 1] |= ((addr >> 16) as u32 & 0xffff) << 5;
        }
        let mut body: Vec<u8> = self.code.iter().flat_map(|w| w.to_le_bytes()).collect();
        body.extend_from_slice(&self.data);
        build_elf(&body)
    }
}

// aarch64 syscall numbers.
const GETCWD: u16 = 17;
const OPENAT: u16 = 56;
const READ: u16 = 63;
const WRITE: u16 = 64;
const EXIT: u16 = 93;

/// A program that writes its working directory and a newline to stdout,
/// `hi\n` into `hello.txt` (relative to that directory), copies what it
/// reads from stdin to stderr, and exits 7.
fn cwd_program() -> Vec<u8> {
    let mut a = Asm::default();
    let path = a.data(b"hello.txt\0");
    let hi = a.data(b"hi\n");
    let nl = a.data(b"\n");
    let buf = a.data(&[0; 256]);
    // n = getcwd(buf, 256) (counts the NUL); write(1, buf, n - 1)
    a.adr_data(0, buf);
    a.movz(1, 256);
    a.syscall(GETCWD);
    a.sub_imm(2, 0, 1);
    a.adr_data(1, buf);
    a.movz(0, 1);
    a.syscall(WRITE);
    a.adr_data(1, nl);
    a.movz(2, 1);
    a.movz(0, 1);
    a.syscall(WRITE);
    // fd = openat(AT_FDCWD, "hello.txt", O_WRONLY|O_CREAT|O_TRUNC, 0644)
    a.movn(0, 99);
    a.adr_data(1, path);
    a.movz(2, 0x241);
    a.movz(3, 0o644);
    a.syscall(OPENAT);
    // write(fd, "hi\n", 3)
    a.adr_data(1, hi);
    a.movz(2, 3);
    a.syscall(WRITE);
    // n = read(0, buf, 256); write(2, buf, n)
    a.movz(0, 0);
    a.adr_data(1, buf);
    a.movz(2, 256);
    a.syscall(READ);
    a.mov_from_x0(2);
    a.adr_data(1, buf);
    a.movz(0, 2);
    a.syscall(WRITE);
    a.movz(0, 7);
    a.syscall(EXIT);
    a.finish()
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nixvm-embed-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A root holding `program` as `/prog`.
fn root_with(tag: &str, program: &[u8]) -> PathBuf {
    let root = temp_dir(tag);
    std::fs::write(root.join("prog"), program).unwrap();
    root
}

fn aarch64(root: &Path) -> SandboxBuilder {
    Sandbox::builder()
        .arch(Arch::Aarch64)
        .prefer_interp(true)
        .mem_bytes(64 << 20)
        .root_dir(root)
        .command(["/prog"])
}

#[test]
fn a_command_runs_in_the_chosen_directory_with_the_callers_stdio() {
    // Whatever this process's NIXVM_* say, an embedder that installs
    // nothing gets none of it.
    let root = root_with("cwd", &cwd_program());
    let project = temp_dir("cwd-project");
    let (out, err) = (Capture::new(), Capture::new());
    let outcome = aarch64(&root)
        .no_work_dir()
        .bind(&project, "/home/me/project")
        .cwd("/home/me/project")
        .network(Network::None)
        .stdin(&b"from stdin"[..])
        .stdout(out.writer())
        .stderr(err.writer())
        .execute()
        .unwrap();
    assert_eq!(out.text(), "/home/me/project\n");
    assert_eq!(err.text(), "from stdin");
    assert_eq!(
        std::fs::read_to_string(project.join("hello.txt")).unwrap(),
        "hi\n",
        "the guest wrote through the bind, relative to its working directory"
    );
    assert_eq!(outcome.exit_code, 7);
    assert_eq!(outcome.signal, None);
    assert_eq!(outcome.limit, None);
    assert!(!outcome.success());
    assert_eq!(outcome.backend, "interp");
    assert!(outcome.notices.is_empty(), "{:?}", outcome.notices);
    std::fs::remove_dir_all(&root).ok();
    std::fs::remove_dir_all(&project).ok();
}

#[test]
fn work_is_mounted_only_when_asked_and_a_missing_cwd_is_made() {
    let root = root_with("nowork", &cwd_program());
    let work = temp_dir("nowork-work");
    let out = Capture::new();
    let outcome = aarch64(&root)
        .work_dir(&work)
        .no_work_dir()
        .cwd("/work")
        .stdout(out.writer())
        .execute()
        .unwrap();
    assert_eq!(outcome.exit_code, 7);
    assert_eq!(out.text(), "/work\n");
    assert!(
        !work.join("hello.txt").exists(),
        "/work is the guest's own directory, not the host's"
    );
    // With it, the default: the host directory at /work.
    let outcome = aarch64(&root)
        .work_dir(&work)
        .cwd("/work")
        .execute()
        .unwrap();
    assert_eq!(outcome.exit_code, 7);
    assert!(work.join("hello.txt").exists());
    std::fs::remove_dir_all(&root).ok();
    std::fs::remove_dir_all(&work).ok();
}

#[test]
fn a_fault_is_a_notice_in_the_outcome_not_a_line_on_stderr() {
    // udf #0: an illegal instruction.
    let mut a = Asm::default();
    a.code.push(0);
    let root = root_with("fault", &a.finish());
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&seen);
    let outcome = aarch64(&root)
        .no_work_dir()
        .on_notice(move |n| sink.lock().unwrap().push(n.to_string()))
        .execute()
        .unwrap();
    assert_eq!(outcome.signal, Some(4), "SIGILL");
    assert_eq!(outcome.exit_code, 128 + 4);
    assert!(
        matches!(
            outcome.notices.as_slice(),
            [Notice::Fault {
                pid: 1,
                signal: 4,
                ..
            }]
        ),
        "{:?}",
        outcome.notices
    );
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert!(
        seen[0].starts_with("[fault] pid 1 illegal instruction"),
        "{seen:?}"
    );
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn the_wall_clock_limit_ends_the_run_and_says_so() {
    // b . : spin forever.
    let mut a = Asm::default();
    a.code.push(0x1400_0000);
    let root = root_with("spin", &a.finish());
    let outcome = aarch64(&root)
        .no_work_dir()
        .wall_time(Duration::from_millis(300))
        .execute()
        .unwrap();
    assert_eq!(outcome.limit, Some(LimitHit::WallTime));
    assert_eq!(outcome.signal, Some(9));
    std::fs::remove_dir_all(&root).ok();
}

/// `echo hi; pwd` in a real shell, with a bind as the working directory.
/// Needs `NIXVM_ALPINE_TAR` (an Alpine minirootfs tar for this host's
/// architecture) and a `tar` to unpack it.
#[test]
fn a_shell_command_with_a_bind_as_its_working_directory() {
    let Some(tar) = std::env::var_os("NIXVM_ALPINE_TAR") else {
        eprintln!("NIXVM_ALPINE_TAR not set; skipping the shell test");
        return;
    };
    let root = temp_dir("alpine");
    let ok = std::process::Command::new("tar")
        .arg("-xf")
        .arg(&tar)
        .arg("-C")
        .arg(&root)
        .status()
        .is_ok_and(|s| s.success());
    assert!(ok, "unpack {tar:?}");
    let project = temp_dir("alpine-project");
    std::fs::write(project.join("README"), "hello from the host\n").unwrap();
    let (out, err) = (Capture::new(), Capture::new());
    let outcome = Sandbox::builder()
        .root_dir(&root)
        .no_work_dir()
        .bind(&project, "/src/project")
        .cwd("/src/project")
        .network(Network::None)
        .stdout(out.writer())
        .stderr(err.writer())
        .shell("echo hi; pwd; cat README; echo made > out.txt; ls /")
        .execute()
        .unwrap();
    assert!(outcome.success(), "{outcome:?}\nstderr: {}", err.text());
    let text = out.text();
    assert!(
        text.starts_with("hi\n/src/project\nhello from the host\n"),
        "{text}"
    );
    assert!(!text.contains("work"), "no implicit /work: {text}");
    assert_eq!(
        std::fs::read_to_string(project.join("out.txt")).unwrap(),
        "made\n"
    );
    std::fs::remove_dir_all(&root).ok();
    std::fs::remove_dir_all(&project).ok();
}
