#![cfg(feature = "fstool")]
use nixvm::vm::Vm;

// Default node: concurrent TurboFan JIT with background compile threads. This is
// the path that exercised the per-thread mmap arena bug (overlapping code
// allocations at teardown). Expect a clean [EXIT 0] with the right sum.
#[test]
fn nodejit_default() {
    let Ok(tar) = std::env::var("NIXVM_NODE_TAR") else {
        return;
    };
    let tar = std::fs::read(&tar).unwrap();
    let mut vm = Vm::boot_squashfs(
        &tar,
        vec![
            "/usr/bin/node".into(),
            "-e".into(),
            "let n=0;for(let i=0;i<1e6;i++)n+=i;console.log('sum',n)".into(),
        ],
        2048u64 * 1024 * 1024,
    )
    .unwrap();
    let s = std::time::Instant::now();
    let mut o = Vec::new();
    let mut code = None;
    loop {
        let st = vm.pump().unwrap();
        o.extend(st.stdout);
        if let Some(c) = st.exit_code {
            code = Some(c);
            eprintln!("[EXIT {c}]");
            break;
        }
        if s.elapsed().as_secs() > 120 {
            eprintln!("[timeout]");
            break;
        }
    }
    eprintln!("[stdout] {:?}", String::from_utf8_lossy(&o));
    assert_eq!(code, Some(0), "node exited non-zero");
    assert!(
        String::from_utf8_lossy(&o).contains("sum 499999500000"),
        "wrong output"
    );
}
