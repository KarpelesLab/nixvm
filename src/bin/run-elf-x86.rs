//! `run-elf-x86` — a development harness: load a host static x86-64 ELF and
//! run it on the best available x86-64 backend (KVM on Linux, else the
//! software interpreter), printing the exit status, any fault, and the
//! unsupported-syscall ledger. Mirrors `run-elf` (the aarch64 harness) but
//! selects the x86-64 guest arch.
//!
//! ```text
//! cargo run --bin run-elf-x86 -- <host-elf> [guest-argv...]
//! ```

use std::path::PathBuf;

use nixvm::abi::Arch;
use nixvm::fs::{DevFs, MountTable, Overlay, Passthrough, ProcFs, SysFs, TmpFs};
use nixvm::kernel::Kernel;
use nixvm::loader::{ProcessSpec, load_static};
use nixvm::vcpu::GuestMemory;
use nixvm::vcpu::mem::PAGE_SIZE;

const GUEST_BASE: u64 = 0x1_0000;
const MEM_BYTES: u64 = 512 * 1024 * 1024;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(host) = args.first() else {
        eprintln!("usage: run-elf-x86 <host-elf> [guest-argv...]");
        std::process::exit(2);
    };
    let elf = match std::fs::read(host) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("run-elf-x86: cannot read {host}: {e}");
            std::process::exit(2);
        }
    };

    let argv: Vec<String> = if args.len() > 1 {
        args[1..].to_vec()
    } else {
        vec![basename(host)]
    };
    let envp = vec![
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".to_string(),
        "HOME=/root".to_string(),
        "TERM=xterm".to_string(),
        "PWD=/work".to_string(),
    ];

    let mut mem = GuestMemory::new(GUEST_BASE, MEM_BYTES);
    let spec = ProcessSpec { argv, envp };
    let img = match load_static(&mut mem, &elf, &spec) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("run-elf-x86: load failed: {e}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "run-elf-x86: entry={:#x} sp={:#x} brk={:#x}",
        img.entry, img.stack_pointer, img.program_break
    );

    let mid = page_down(img.program_break + (img.stack_bottom - img.program_break) / 2);

    // `vcpu::select` prefers KVM on a Linux/x86-64 host (falling back to the
    // dedicated x86-64 software interpreter; NIXVM_INTERP=1 forces the latter).
    let backend = nixvm::vcpu::select(Arch::X86_64).unwrap();
    let vcpu = backend.new_vcpu(img.entry, img.stack_pointer).unwrap();

    let mut mounts = MountTable::new();
    // NIXVM_ROOT points `/` at a host rootfs (read-only lower) with a tmpfs
    // upper (copy-on-write) — the real multi-instance layout.
    if let Ok(root) = std::env::var("NIXVM_ROOT") {
        let lower = Box::new(Passthrough::read_only(root));
        mounts.mount("/", Box::new(Overlay::new(lower, Box::new(TmpFs::new()))));
    } else {
        mounts.mount("/", Box::new(TmpFs::new()));
    }
    mounts.mount("/tmp", Box::new(TmpFs::new()));
    mounts.mount("/dev", Box::new(DevFs::new()));
    mounts.mount("/proc", Box::new(ProcFs::new()));
    mounts.mount("/sys", Box::new(SysFs::new()));
    if let Ok(cwd) = std::env::current_dir() {
        mounts.mount("/work", Box::new(Passthrough::new(cwd)));
    }

    let mut kernel = Kernel::new(Arch::X86_64, mounts);
    kernel.set_cwd("/work");
    kernel.set_heap(img.program_break, mid);
    kernel.set_mmap_area(img.stack_bottom, mid);

    let result = kernel.run(vcpu, mem);
    eprintln!("\nrun-elf-x86: result = {result:?}");
    let unsupported = kernel.unsupported();
    if unsupported.is_empty() {
        eprintln!("run-elf-x86: no unsupported syscalls");
    } else {
        eprintln!("run-elf-x86: unsupported syscalls (raw nr -> count):");
        for (nr, count) in unsupported {
            eprintln!("    {nr:>4} x{count}");
        }
    }
    match result {
        Ok(code) => std::process::exit(code & 0xff),
        Err(_) => std::process::exit(1),
    }
}

fn basename(p: &str) -> String {
    PathBuf::from(p)
        .file_name()
        .map_or_else(|| p.to_string(), |n| n.to_string_lossy().into_owned())
}

fn page_down(v: u64) -> u64 {
    v - v % PAGE_SIZE
}
