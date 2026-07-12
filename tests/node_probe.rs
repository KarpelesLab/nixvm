#![cfg(feature = "fstool")]
use nixvm::vm::Vm;
use std::io::Write;
#[test]
fn node_probe() {
    let Ok(tar_path) = std::env::var("NIXVM_NODE_TAR") else { return; };
    let tar = std::fs::read(&tar_path).unwrap();
    let mut vm = Vm::boot_squashfs(&tar, vec!["/usr/bin/node".into(), "--version".into()], 2048u64*1024*1024).unwrap();
    let start = std::time::Instant::now();
    let mut idle = 0;
    loop {
        let step = vm.pump().unwrap();
        let got = !step.stdout.is_empty() || !step.stderr.is_empty();
        if got { std::io::stderr().write_all(&step.stdout).ok(); std::io::stderr().write_all(&step.stderr).ok(); std::io::stderr().flush().ok(); idle=0; }
        if let Some(code) = step.exit_code { eprintln!("\n[NODE EXIT {code} after {}s]", start.elapsed().as_secs()); break; }
        if !got { idle+=1; if idle>8000 { eprintln!("\n[quiescent {}s]", start.elapsed().as_secs()); break; } std::thread::sleep(std::time::Duration::from_millis(1)); }
        if start.elapsed().as_secs() > 900 { eprintln!("\n[cap 900s]"); break; }
    }
}
