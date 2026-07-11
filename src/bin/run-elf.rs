//! `run-elf` — a development harness: load a host static ELF and run it on the
//! software interpreter, printing the exit status, any fault, and the
//! unsupported-syscall ledger. Used to bring up real binaries (Alpine busybox).
//!
//! ```text
//! cargo run --bin run-elf -- <host-elf> [guest-argv...]
//! ```

use std::path::PathBuf;

use nixvm::abi::Arch;
use nixvm::fs::{DevFs, MountTable, Overlay, Passthrough, ProcFs, SysFs, TmpFs};
use nixvm::kernel::Kernel;
use nixvm::loader::{ProcessSpec, interp_path, load_dynamic, load_static};
use nixvm::vcpu::Backend;
use nixvm::vcpu::GuestMemory;
use nixvm::vcpu::interp::InterpBackend;
use nixvm::vcpu::mem::PAGE_SIZE;

/// Read an entire file out of the mount table (for the dynamic linker lookup).
fn read_mount_file(mounts: &mut MountTable, path: &str) -> Option<Vec<u8>> {
    let size = mounts.stat(path)?.size as usize;
    let mut buf = vec![0u8; size];
    let mut off = 0;
    while off < size {
        match mounts.read_at(path, off as u64, &mut buf[off..]) {
            Ok(0) => break,
            Ok(n) => off += n,
            Err(_) => return None,
        }
    }
    buf.truncate(off);
    Some(buf)
}

const GUEST_BASE: u64 = 0x1_0000;
const MEM_BYTES: u64 = 512 * 1024 * 1024;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(host) = args.first() else {
        eprintln!("usage: run-elf <host-elf> [guest-argv...]");
        std::process::exit(2);
    };
    let elf = match std::fs::read(host) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("run-elf: cannot read {host}: {e}");
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

    // Build the mount table first so the loader can read the dynamic linker
    // (PT_INTERP → ld-musl) out of the guest root.
    let mut mounts = MountTable::new();
    // NIXVM_ROOT points `/` at a host Alpine rootfs (read-only lower) with a
    // tmpfs upper (copy-on-write) — the real multi-instance layout.
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

    let mut mem = GuestMemory::new(GUEST_BASE, MEM_BYTES);
    let spec = ProcessSpec { argv, envp };
    let loaded = if let Some(interp) = interp_path(&elf) {
        eprintln!("run-elf: dynamic executable, interpreter {interp}");
        let Some(interp_elf) = read_mount_file(&mut mounts, &interp) else {
            eprintln!("run-elf: interpreter {interp} not found in the root");
            std::process::exit(1);
        };
        load_dynamic(&mut mem, &elf, &interp_elf, &spec)
    } else {
        load_static(&mut mem, &elf, &spec)
    };
    let img = match loaded {
        Ok(i) => i,
        Err(e) => {
            eprintln!("run-elf: load failed: {e}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "run-elf: entry={:#x} sp={:#x} brk={:#x}",
        img.entry, img.stack_pointer, img.program_break
    );

    let mid = page_down(img.program_break + (img.stack_bottom - img.program_break) / 2);

    let backend = InterpBackend::new(Arch::Aarch64).unwrap();
    let vcpu = backend.new_vcpu(img.entry, img.stack_pointer).unwrap();

    let mut kernel = Kernel::new(Arch::Aarch64, mounts);
    // NIXVM_CPUS=N runs guest compute on N host worker threads (SMP); default 1.
    if let Some(n) = std::env::var("NIXVM_CPUS")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        eprintln!("run-elf: ncpus={n}");
        kernel.set_ncpus(n);
    }
    kernel.set_cwd("/work");
    kernel.set_heap(img.program_break, mid);
    kernel.set_mmap_area(img.stack_bottom, mid);

    let result = kernel.run(vcpu, mem);
    eprintln!("\nrun-elf: result = {result:?}");
    let unsupported = kernel.unsupported();
    if unsupported.is_empty() {
        eprintln!("run-elf: no unsupported syscalls");
    } else {
        eprintln!("run-elf: unsupported syscalls (raw nr -> count):");
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
