//! A buffer rendered on the virtual GPU, shown on the built-in display: the
//! raw-wire client (`tests/guest/wayland_client.c`, in its `gpu` mode — no
//! Mesa, no libwayland) clears a texture on `/dev/dri/renderD128` with a
//! virgl command buffer, exports it as a dma-buf (PRIME) and shares it
//! through `zwp_linux_dmabuf_v1`; the compositor, compositing on the GPU,
//! copies it GPU-to-GPU and reads the frame back when it is taken. The
//! test checks the frame's pixels, then presses a key, which the client
//! must receive (after its buffer was released and its frame callback
//! fired) to exit successfully. The client's shared-memory mode goes
//! through the same GL compositor.
//!
//! Needs the `gpu` feature, a host EGL with GLES 3 (`NIXVM_GPU=egl`'s
//! backend) and `gcc` with static libraries on an x86-64 Linux host; it
//! skips itself otherwise, printing why.
#![cfg(all(feature = "gpu", target_os = "linux", target_arch = "x86_64"))]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use nixvm::Sandbox;
use nixvm::display::{Display, Input};
use nixvm::gpu::GpuDevice;

/// A guest root holding the client, `None` without a compiler.
fn root() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("gpu_display_root");
        std::fs::create_dir_all(dir.join("tmp")).expect("create guest root");
        let out = Command::new("gcc")
            .args(["-static", "-O1", "-w", "-o"])
            .arg(dir.join("wayland_client"))
            .arg("tests/guest/wayland_client.c")
            .output();
        match out {
            Ok(o) if o.status.success() => Some(dir),
            Ok(o) => {
                eprintln!(
                    "can't build the client, skipping: {}",
                    String::from_utf8_lossy(&o.stderr)
                );
                None
            }
            Err(e) => {
                eprintln!("no gcc, skipping: {e}");
                None
            }
        }
    })
    .as_deref()
}

/// The host GPU, `None` (with a note) if this host has no EGL to offer.
fn gpu() -> Option<GpuDevice> {
    let dev = GpuDevice::from_spec("egl", false);
    if dev.is_none() {
        eprintln!("no host EGL/GLES 3, skipping");
    }
    dev
}

/// The frame, if it changed since the last look.
fn frame(display: &Display) -> Option<(Vec<u8>, u32, u32)> {
    let mut got = None;
    display.frame(|rgba, w, h| got = Some((rgba.to_vec(), w, h)));
    got
}

/// The client's window, in the GPU buffer's colour.
const GPU_COLOR: [u8; 4] = [0x20, 0x80, 0xc0, 0xff];
/// In its shared-memory mode's.
const SHM_COLOR: [u8; 4] = [0xc0, 0x30, 0x60, 0xff];

fn shows_a_gpu_buffer(ncpus: usize) {
    shows(
        ncpus,
        false,
        &["/wayland_client", "gpu", "2080c0"],
        GPU_COLOR,
    );
}

/// Run `command` (the client) on a 320×240 display with the GPU: its
/// window fills the display in `color`, composited on the GPU; then a key
/// press ends it.
fn shows(ncpus: usize, interp: bool, command: &[&str], color: [u8; 4]) {
    let Some(root) = root() else { return };
    let Some(dev) = gpu() else { return };
    let sandbox = Sandbox::builder()
        .root_dir(root)
        .display(320, 240)
        .gpu(dev)
        .ncpus(ncpus)
        .prefer_interp(interp)
        .command(command.iter().map(|s| (*s).to_string()))
        .build();
    let display = sandbox.display().expect("a display");
    let run = std::thread::spawn(move || sandbox.run());

    let deadline = Instant::now() + Duration::from_secs(60);
    let mut center = None;
    let rgba = loop {
        if let Some((rgba, w, h)) = frame(&display) {
            assert_eq!((w, h), (320, 240));
            let i = ((120 * w + 160) * 4) as usize;
            center = Some(rgba[i..i + 4].to_vec());
            if rgba[i..i + 4] == color {
                break rgba;
            }
        }
        assert!(
            !run.is_finished(),
            "the guest exited before its window showed: {:?}",
            run.join()
        );
        assert!(
            Instant::now() < deadline,
            "no window after 60s; the center pixel is {center:?}"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    // The window fills the display, every pixel the GPU's clear colour.
    assert!(
        rgba.as_chunks::<4>().0.iter().all(|p| *p == color),
        "the window covers the whole display"
    );
    assert_eq!(display.compositor(), "gl", "the GPU composites the frame");

    display.input(Input::Key {
        code: 30,
        pressed: true,
    });
    display.input(Input::Key {
        code: 30,
        pressed: false,
    });
    let deadline = Instant::now() + Duration::from_secs(60);
    while !run.is_finished() {
        assert!(Instant::now() < deadline, "the key press never arrived");
        std::thread::sleep(Duration::from_millis(20));
    }
    let code = run.join().expect("the VM thread").expect("the VM ran");
    assert_eq!(
        code, 0,
        "the client saw its buffer released, its frame callback and its key"
    );
}

#[test]
fn a_gpu_buffer_shows_on_the_display() {
    shows_a_gpu_buffer(1);
}

#[test]
fn a_gpu_buffer_shows_on_the_display_with_several_cpus() {
    shows_a_gpu_buffer(4);
}

#[test]
fn a_gpu_buffer_shows_on_the_display_on_the_interpreter() {
    shows(1, true, &["/wayland_client", "gpu", "2080c0"], GPU_COLOR);
}

/// With a GPU the display composites on it whatever the buffers are:
/// shared memory is uploaded to a texture, and shows the same.
#[test]
fn a_shared_memory_buffer_shows_through_the_gl_compositor() {
    shows(1, false, &["/wayland_client", "c03060"], SHM_COLOR);
}
