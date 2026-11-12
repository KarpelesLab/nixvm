//! The `nixvm` command-line tool.
//!
//! A deliberately small, std-only argument handler for the scaffold. A richer
//! parser (clap) and structured logging arrive with the `cli` feature deps in a
//! later phase. Usage:
//!
//! ```text
//! nixvm run [--mem <bytes>] [--workdir <dir>] [--root <dir>] [--arch <arch>] -- <cmd> [args...]
//! nixvm run --image <name|archive> [--entrypoint <prog>] [--user <user>] [-- <args>...]
//! nixvm pull <name> [--arch <arch>]
//! nixvm shell
//! nixvm version
//! ```

use std::process::ExitCode;

use nixvm::{Arch, Sandbox};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => cmd_run(&args[1..]),
        Some("pull") => cmd_pull(&args[1..]),
        Some("shell") => cmd_run(&["--".into(), "/bin/sh".into()]),
        Some("version" | "--version" | "-V") => {
            println!("nixvm {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some("help" | "--help" | "-h") | None => {
            print_usage();
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("nixvm: unknown subcommand `{other}`\n");
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn cmd_run(args: &[String]) -> ExitCode {
    let mut builder = Sandbox::builder();
    let mut image = false;
    let mut i = 0;
    // Parse options until the `--` separator, then the rest is the command.
    let command: Vec<String> = loop {
        match args.get(i).map(String::as_str) {
            Some("--") => break args[i + 1..].to_vec(),
            Some("--mem") => {
                let Some(v) = args.get(i + 1).and_then(|s| parse_size(s)) else {
                    eprintln!("nixvm: --mem needs a size (e.g. 512M, 2G)");
                    return ExitCode::FAILURE;
                };
                builder = builder.mem_bytes(v);
                i += 2;
            }
            Some("--workdir") => {
                let Some(dir) = args.get(i + 1) else {
                    eprintln!("nixvm: --workdir needs a path");
                    return ExitCode::FAILURE;
                };
                builder = builder.work_dir(dir);
                i += 2;
            }
            Some("--root") => {
                let Some(dir) = args.get(i + 1) else {
                    eprintln!("nixvm: --root needs a directory");
                    return ExitCode::FAILURE;
                };
                builder = builder.root_dir(dir);
                i += 2;
            }
            Some("--arch") => {
                let Some(arch) = args.get(i + 1).and_then(|s| Arch::from_name(s)) else {
                    eprintln!("nixvm: --arch needs aarch64 (arm64) or x86_64 (amd64)");
                    return ExitCode::FAILURE;
                };
                builder = builder.arch(arch);
                i += 2;
            }
            Some("--env" | "-e") => {
                let Some(kv) = args.get(i + 1) else {
                    eprintln!("nixvm: --env needs KEY=VALUE");
                    return ExitCode::FAILURE;
                };
                builder = builder.env(kv);
                i += 2;
            }
            Some("--image") => {
                let Some(name) = args.get(i + 1) else {
                    eprintln!("nixvm: --image needs an image name or archive path");
                    return ExitCode::FAILURE;
                };
                builder = builder.image(name);
                image = true;
                i += 2;
            }
            Some("--entrypoint") => {
                let Some(prog) = args.get(i + 1) else {
                    eprintln!("nixvm: --entrypoint needs a program");
                    return ExitCode::FAILURE;
                };
                builder = builder.entrypoint([prog.clone()]);
                i += 2;
            }
            Some("--user" | "-u") => {
                let Some(user) = args.get(i + 1) else {
                    eprintln!("nixvm: --user needs a user (name, uid, user:group)");
                    return ExitCode::FAILURE;
                };
                builder = builder.user(user);
                i += 2;
            }
            Some(_) => break args[i..].to_vec(), // no `--`: rest is the command
            None => break Vec::new(),
        }
    };

    // An image brings its own command; anything given replaces its Cmd.
    if command.is_empty() && !image {
        eprintln!("nixvm run: no command given");
        return ExitCode::FAILURE;
    }

    match builder.command(command).run() {
        Ok(code) => ExitCode::from(u8::try_from(code & 0xff).unwrap_or(1)),
        Err(e) => {
            eprintln!("nixvm: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `nixvm pull <name>`: fetch an image into the cache.
fn cmd_pull(args: &[String]) -> ExitCode {
    #[cfg(all(feature = "oci", feature = "fetch"))]
    {
        let mut name = None;
        let mut arch = nixvm::Arch::host_native().unwrap_or(nixvm::Arch::X86_64);
        let mut i = 0;
        while let Some(a) = args.get(i) {
            if a == "--arch" {
                let Some(x) = args.get(i + 1).and_then(|s| nixvm::Arch::from_name(s)) else {
                    eprintln!("nixvm: --arch needs aarch64 (arm64) or x86_64 (amd64)");
                    return ExitCode::FAILURE;
                };
                arch = x;
                i += 2;
            } else {
                name = Some(a.clone());
                i += 1;
            }
        }
        let Some(name) = name else {
            eprintln!("nixvm pull: no image named");
            return ExitCode::FAILURE;
        };
        match nixvm::oci::pull(&name, arch) {
            Ok(img) => {
                let bytes: usize = img.layers.iter().map(Vec::len).sum();
                println!(
                    "{name}: {} ({}, {} layers, {} MiB uncompressed)",
                    img.digest.as_deref().unwrap_or("?"),
                    img.config.architecture,
                    img.layers.len(),
                    bytes >> 20
                );
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("nixvm: {e}");
                ExitCode::FAILURE
            }
        }
    }
    #[cfg(not(all(feature = "oci", feature = "fetch")))]
    {
        let _ = args;
        eprintln!("nixvm pull: this nixvm was built without the `oci` and `fetch` features");
        ExitCode::FAILURE
    }
}

/// Parse a byte size with an optional K/M/G suffix.
fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    let (num, mult) = match s.chars().last()? {
        'k' | 'K' => (&s[..s.len() - 1], 1024),
        'm' | 'M' => (&s[..s.len() - 1], 1024 * 1024),
        'g' | 'G' => (&s[..s.len() - 1], 1024 * 1024 * 1024),
        _ => (s, 1),
    };
    num.trim().parse::<u64>().ok().map(|n| n * mult)
}

fn print_usage() {
    eprintln!(
        "nixvm — a portable Linux syscall sandbox\n\n\
         USAGE:\n    \
         nixvm run [--mem <size>] [--workdir <dir>] [--root <dir>] [--arch <arch>]\n              \
                   [--env KEY=VAL]... -- <cmd> [args...]\n    \
         nixvm run --image <name|archive> [--entrypoint <prog>] [--user <user>]\n              \
                   [--env KEY=VAL]... [-- <args>...]\n    \
         nixvm pull <name> [--arch <arch>]\n    \
         nixvm shell\n    \
         nixvm version\n\n\
         The current directory is exposed inside the sandbox at /work.\n\
         --root uses an extracted host rootfs directory as the guest root.\n\
         The guest architecture follows the command's ELF binary (an x86_64\n\
         root runs as x86_64 on any host); --arch aarch64|x86_64 pins it.\n\
         --image runs a Docker/OCI image (a registry name such as alpine:3.20, or a\n\
         `docker save` / OCI layout archive) as `docker run` would: arguments\n\
         replace its Cmd. Set NIXVM_NET=host for network access."
    );
}
