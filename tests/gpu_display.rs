//! A buffer rendered on the virtual GPU, shown on the built-in display: the
//! raw-wire client (`tests/guest/wayland_client.c`, in its `gpu` mode — no
//! Mesa, no libwayland) clears a texture on `/dev/dri/renderD128` with a
//! virgl command buffer, exports it as a dma-buf (PRIME) and shares it
//! through `zwp_linux_dmabuf_v1`; the compositor reads it back from the
//! host GPU. The test checks the frame's pixels, then presses a key, which
//! the client must receive (after its buffer was released and its frame
//! callback fired) to exit successfully.
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

fn shows_a_gpu_buffer(ncpus: usize) {
    let Some(root) = root() else { return };
    let Some(dev) = gpu() else { return };
    const COLOR: [u8; 4] = [0x20, 0x80, 0xc0, 0xff];
    let sandbox = Sandbox::builder()
        .root_dir(root)
        .display(320, 240)
        .gpu(dev)
        .ncpus(ncpus)
        .command(["/wayland_client", "gpu", "2080c0"])
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
            if rgba[i..i + 4] == COLOR {
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
        rgba.as_chunks::<4>().0.iter().all(|p| *p == COLOR),
        "the GPU buffer covers the whole display"
    );

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
