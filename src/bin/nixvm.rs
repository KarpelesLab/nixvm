//! The `nixvm` command-line tool.
//!
//! A deliberately small, std-only argument handler for the scaffold. A richer
//! parser (clap) and structured logging arrive with the `cli` feature deps in a
//! later phase. Usage:
//!
//! ```text
//! nixvm run [--mem <bytes>] [--workdir <dir>] -- <cmd> [args...]
//! nixvm shell
//! nixvm version
//! ```

use std::process::ExitCode;

use nixvm::Sandbox;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => cmd_run(&args[1..]),
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
            Some("--env" | "-e") => {
                let Some(kv) = args.get(i + 1) else {
                    eprintln!("nixvm: --env needs KEY=VALUE");
                    return ExitCode::FAILURE;
                };
                builder = builder.env(kv);
                i += 2;
            }
            Some(_) => break args[i..].to_vec(), // no `--`: rest is the command
            None => break Vec::new(),
        }
    };

    if command.is_empty() {
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
         nixvm run [--mem <size>] [--workdir <dir>] [--root <dir>]\n              \
                   [--env KEY=VAL]... -- <cmd> [args...]\n    \
         nixvm shell\n    \
         nixvm version\n\n\
         The current directory is exposed inside the sandbox at /work.\n\
         --root uses an extracted host rootfs directory as the guest root."
    );
}
