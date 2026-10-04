//! Shared scaffolding for the syscall unit tests of the newer kernel modules:
//! a no-op vcpu, a kernel over a fresh tmpfs root with a small mapped guest
//! region, and a `call` helper that drives one syscall through the real
//! dispatcher (so each test exercises the same lock routing the guest sees).

use super::{Arena, Kernel, ServiceCtx};
use crate::abi::Arch;
use crate::abi::arch::Sysno;
use crate::fs::{MountTable, TmpFs};
use crate::vcpu::mem::{PAGE_SIZE, Prot};
use crate::vcpu::{Exit, GuestMemory, Vcpu, VcpuError};

/// Base of the test guest region; the first [`MAPPED_PAGES`] pages are mapped
/// read-write, the rest are left for `mmap`/`shmat` to place mappings in.
pub(super) const BASE: u64 = 0x1_0000;
pub(super) const MAPPED_PAGES: u64 = 8;
const TOTAL_PAGES: u64 = 64;

/// A vcpu that never runs guest code (tests call handlers directly).
#[derive(Clone)]
pub(super) struct DummyVcpu;
impl Vcpu for DummyVcpu {
    fn run(&mut self, _m: &mut GuestMemory) -> Result<Exit, VcpuError> {
        Ok(Exit::Halt)
    }
    fn syscall_nr(&self) -> u64 {
        0
    }
    fn syscall_args(&self) -> [u64; 6] {
        [0; 6]
    }
    fn set_syscall_ret(&mut self, _v: u64) {}
    fn reg(&self, _i: usize) -> u64 {
        0
    }
    fn set_reg(&mut self, _i: usize, _v: u64) {}
    fn pc(&self) -> u64 {
        0
    }
    fn set_pc(&mut self, _v: u64) {}
    fn sp(&self) -> u64 {
        0
    }
    fn set_sp(&mut self, _v: u64) {}
    fn set_tls(&mut self, _v: u64) {}
    fn fork(&self) -> Box<dyn Vcpu> {
        Box::new(self.clone())
    }
    fn reset(&mut self, _e: u64, _s: u64) {}
}

/// A kernel rooted at a tmpfs, the running task as pid/tgid 1 on mm 0 (with an
/// mmap arena over the unmapped tail of the region), and the guest memory.
pub(super) fn setup() -> (Kernel, GuestMemory, DummyVcpu, ServiceCtx) {
    let mut mounts = MountTable::new();
    mounts.mount("/", Box::new(TmpFs::new()));
    let mut kernel = Kernel::new(Arch::Aarch64, mounts);
    let mut cx = ServiceCtx::for_test();
    cx.cur.pid = 1;
    cx.cur.tgid = 1;
    cx.cur.mm = 0;
    let mem = GuestMemory::new(BASE, TOTAL_PAGES * PAGE_SIZE);
    let mut mem = mem;
    mem.map(BASE, MAPPED_PAGES * PAGE_SIZE, Prot::rw()).unwrap();
    {
        let sh = kernel.shared.get_mut().unwrap();
        sh.mmap_areas.push(Arena::new(
            BASE + TOTAL_PAGES * PAGE_SIZE,
            BASE + MAPPED_PAGES * PAGE_SIZE,
        ));
    }
    (kernel, mem, DummyVcpu, cx)
}

/// Drive syscall `s` with `a` through the real dispatcher.
pub(super) fn call(
    k: &Kernel,
    cx: &mut ServiceCtx,
    mem: &mut GuestMemory,
    v: &mut DummyVcpu,
    s: Sysno,
    a: [u64; 6],
) -> i64 {
    k.dispatch(cx, s, 0, &a, v, mem)
}

/// Write a NUL-terminated string at `addr` (bypassing page protection).
pub(super) fn put_str(mem: &mut GuestMemory, addr: u64, s: &str) {
    let mut b = s.as_bytes().to_vec();
    b.push(0);
    mem.write_init(addr, &b).unwrap();
}

/// `-errno` as a syscall return.
pub(super) fn e(x: crate::abi::errno::Errno) -> i64 {
    -i64::from(x.0)
}
