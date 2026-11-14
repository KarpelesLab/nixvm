//! nixvm's built-in Wayland display, end to end: a guest client speaking the
//! wire protocol by hand (`tests/guest/wayland_client.c`, no libwayland)
//! maps a window of one color from a memfd-backed `wl_shm` buffer; the test
//! reads the frame through the [`Display`] handle, checks its pixels, then
//! presses a key, which the client must receive to exit successfully.
//!
//! Needs `gcc` with static libraries on an x86-64 Linux host; it skips
//! itself otherwise, printing why.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use nixvm::Sandbox;
use nixvm::display::{Display, Input};

/// A guest root holding the client, `None` without a compiler.
fn root() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("display_root");
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

const COLOR: [u8; 4] = [0xc0, 0x30, 0x60, 0xff];

/// The frame, if it changed since the last look.
fn frame(display: &Display) -> Option<(Vec<u8>, u32, u32)> {
    let mut got = None;
    display.frame(|rgba, w, h| got = Some((rgba.to_vec(), w, h)));
    got
}

/// Run the client on a 320×240 display with `configure` applied to the
/// sandbox; check its window shows up, then end it with a key press.
fn shows_a_window(
    configure: impl FnOnce(nixvm::sandbox::SandboxBuilder) -> nixvm::sandbox::SandboxBuilder,
) {
    let Some(root) = root() else { return };
    let sandbox = configure(
        Sandbox::builder()
            .root_dir(root)
            .display(320, 240)
            .command(["/wayland_client", "c03060"]),
    )
    .build();
    let display = sandbox.display().expect("a display");
    let run = std::thread::spawn(move || sandbox.run());

    let deadline = Instant::now() + Duration::from_secs(60);
    let mut center = None;
    let (rgba, w, h) = loop {
        if let Some((rgba, w, h)) = frame(&display) {
            let i = ((120 * w + 160) * 4) as usize;
            center = Some(rgba[i..i + 4].to_vec());
            if rgba[i..i + 4] == COLOR {
                break (rgba, w, h);
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
    // The window was configured to fill the display, so every pixel is its.
    assert_eq!((w, h), (320, 240));
    assert!(
        rgba.as_chunks::<4>().0.iter().all(|p| *p == COLOR),
        "the window covers the whole display"
    );
    assert_eq!(&display.png()[1..4], b"PNG");

    // A key press reaches the client (KEY_A), which then exits 0.
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
        "the client saw its key press and its frame callback"
    );
}

#[test]
fn a_raw_wire_client_shows_a_window_and_gets_a_key() {
    shows_a_window(|b| b.ncpus(1));
}

#[test]
fn a_raw_wire_client_on_the_interpreter() {
    shows_a_window(|b| b.ncpus(1).prefer_interp(true));
}

#[test]
fn a_raw_wire_client_with_several_cpus() {
    shows_a_window(|b| b.ncpus(4));
}
