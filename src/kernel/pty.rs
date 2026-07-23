//! Pseudo-terminals (`/dev/ptmx` + `/dev/pts/N`).
//!
//! Opening `/dev/ptmx` allocates a [`Pty`] and hands back a **master** fd; the
//! guest reads its number with `TIOCGPTN`, unlocks it with `TIOCSPTLCK`, and
//! opens `/dev/pts/N` for the matching **slave** fd. Bytes written to the master
//! are the terminal *input* (run through the input line discipline, then read by
//! the slave); bytes written to the slave are terminal *output* (run through
//! output post-processing, then read by the master). This is what `script`,
//! `expect`, terminal multiplexers, and test harnesses that need a real tty use.
//!
//! The line discipline implements the load-bearing `N_TTY` behaviour: canonical
//! line buffering (`ICANON`) with erase (`VERASE`), echo (`ECHO`/`ECHOE`),
//! CR→NL input translation (`ICRNL`), and NL→CRNL output translation
//! (`OPOST`/`ONLCR`). Signal generation on `VINTR`/`VQUIT` (`ISIG`) is not yet
//! wired to the guest's signal delivery — a documented gap.

use std::collections::VecDeque;

/// `struct termios` (kernel, x86-64): `c_iflag c_oflag c_cflag c_lflag` (4×4),
/// `c_line` (1), `c_cc[19]`. 36 bytes, byte-identical to what the guest passes.
pub const TERMIOS_LEN: usize = 36;
/// `struct winsize`: `ws_row ws_col ws_xpixel ws_ypixel` (4×u16). 8 bytes.
pub const WINSIZE_LEN: usize = 8;

// c_iflag / c_oflag / c_lflag bits (asm-generic/termbits.h).
const ICRNL: u32 = 0o000400;
const OPOST: u32 = 0o000001;
const ONLCR: u32 = 0o000004;
// ISIG (0o1) — VINTR/VQUIT→signal generation is a documented gap, not wired.
const ICANON: u32 = 0o000002;
const ECHO: u32 = 0o000010;
const ECHOE: u32 = 0o000020;
// c_cc indices.
const VINTR: usize = 0;
const VERASE: usize = 2;
const VEOF: usize = 4;

/// The kernel's default `N_TTY` termios for a fresh pty (cooked mode): `ICRNL|
/// IXON` input, `OPOST|ONLCR` output, `CREAD|CS8|B38400` control,
/// `ISIG|ICANON|ECHO|ECHOE|ECHOK|ECHOCTL|ECHOKE|IEXTEN` local, standard `c_cc`.
fn default_termios() -> [u8; TERMIOS_LEN] {
    let mut t = [0u8; TERMIOS_LEN];
    let put = |t: &mut [u8; TERMIOS_LEN], off: usize, v: u32| t[off..off + 4].copy_from_slice(&v.to_le_bytes());
    put(&mut t, 0, ICRNL | 0o002000); // c_iflag = ICRNL | IXON
    put(&mut t, 4, OPOST | ONLCR); // c_oflag
    put(&mut t, 8, 0o000277); // c_cflag = B38400|CS8|CREAD (0xbf)
    put(&mut t, 12, 0x8a3b); // c_lflag = ISIG|ICANON|ECHO|ECHOE|ECHOK|ECHOCTL|ECHOKE|IEXTEN
    // c_cc (the control chars start at offset 17, after c_line at 16).
    let cc = &mut t[17..36];
    cc[VINTR] = 3; // ^C
    cc[1] = 0o34; // VQUIT ^\
    cc[VERASE] = 0o177; // DEL
    cc[3] = 0o25; // VKILL ^U
    cc[VEOF] = 4; // ^D
    cc[6] = 1; // VMIN
    cc[8] = 0o21; // VSTART ^Q
    cc[9] = 0o23; // VSTOP ^S
    cc[10] = 0o32; // VSUSP ^Z
    cc[12] = 0o22; // VREPRINT ^R
    cc[13] = 0o17; // VDISCARD ^O
    cc[14] = 0o27; // VWERASE ^W
    cc[15] = 0o26; // VLNEXT ^V
    t
}

/// One pseudo-terminal.
#[derive(Debug)]
pub(super) struct Pty {
    termios: [u8; TERMIOS_LEN],
    winsize: [u8; WINSIZE_LEN],
    /// `TIOCSPTLCK` lock: the slave cannot be opened until unlocked (int 0).
    locked: bool,
    /// Terminal input, ready for the *slave* to read (master writes land here
    /// after the input line discipline; canonical lines only appear once
    /// complete).
    input: VecDeque<u8>,
    /// The in-progress canonical line, not yet visible to the slave.
    canon: Vec<u8>,
    /// Terminal output, ready for the *master* to read (slave writes + echo,
    /// after output post-processing).
    output: VecDeque<u8>,
    /// The master fd is open. When it closes, the slave sees EOF.
    master_open: bool,
    /// Count of open slave fds. When it reaches 0, the master sees EOF/`POLLHUP`.
    slave_refs: usize,
    /// `O_NONBLOCK` on each end (set via `fcntl(F_SETFL)`/`ioctl(FIONBIO)`).
    master_nonblock: bool,
    slave_nonblock: bool,
}

impl Pty {
    fn new() -> Self {
        Self {
            termios: default_termios(),
            winsize: [0u8; WINSIZE_LEN],
            locked: true,
            input: VecDeque::new(),
            canon: Vec::new(),
            output: VecDeque::new(),
            master_open: true,
            slave_refs: 0,
            master_nonblock: false,
            slave_nonblock: false,
        }
    }

    fn iflag(&self) -> u32 {
        u32::from_le_bytes(self.termios[0..4].try_into().unwrap())
    }
    fn oflag(&self) -> u32 {
        u32::from_le_bytes(self.termios[4..8].try_into().unwrap())
    }
    fn lflag(&self) -> u32 {
        u32::from_le_bytes(self.termios[12..16].try_into().unwrap())
    }
    fn cc(&self, i: usize) -> u8 {
        self.termios[17 + i]
    }

    /// Append `byte` to the master-side output, applying `OPOST`/`ONLCR`
    /// (NL → CR-NL) when post-processing is on.
    fn out_byte(&mut self, byte: u8) {
        if byte == b'\n' && self.oflag() & OPOST != 0 && self.oflag() & ONLCR != 0 {
            self.output.push_back(b'\r');
        }
        self.output.push_back(byte);
    }

    /// The slave wrote `data` (terminal output): post-process and queue it for
    /// the master to read.
    fn slave_write(&mut self, data: &[u8]) {
        for &b in data {
            self.out_byte(b);
        }
    }

    /// The master wrote `data` (terminal input): run the input line discipline
    /// (CR→NL, echo, canonical buffering with erase) and feed the slave.
    fn master_write(&mut self, data: &[u8]) {
        let (canon, echo) = (self.lflag() & ICANON != 0, self.lflag() & ECHO != 0);
        let icrnl = self.iflag() & ICRNL != 0;
        let (verase, veof) = (self.cc(VERASE), self.cc(VEOF));
        for &raw in data {
            let b = if raw == b'\r' && icrnl { b'\n' } else { raw };
            if canon {
                if b == verase && verase != 0 {
                    if self.canon.pop().is_some() && echo && self.lflag() & ECHOE != 0 {
                        // Erase: backspace, space, backspace.
                        for &e in b"\x08 \x08" {
                            self.out_byte(e);
                        }
                    }
                    continue;
                }
                if b == veof && veof != 0 {
                    // End-of-file / end-of-line: flush the pending line (an EOF on
                    // an empty line yields a zero-length read = EOF for the slave).
                    let line: Vec<u8> = self.canon.drain(..).collect();
                    self.input.extend(line);
                    continue;
                }
                if echo {
                    self.out_byte(b);
                }
                self.canon.push(b);
                if b == b'\n' {
                    let line: Vec<u8> = self.canon.drain(..).collect();
                    self.input.extend(line);
                }
            } else {
                if echo {
                    self.out_byte(b);
                }
                self.input.push_back(b);
            }
        }
    }
}

/// The pty table (behind [`Kernel::ptys`](super::Kernel)). Slots are never
/// removed (indices are stable fd payloads); a fully-closed pty just sits empty.
#[derive(Debug, Default)]
pub(super) struct Ptys {
    table: Vec<Pty>,
}

impl Ptys {
    /// Allocate a fresh pty (its master just opened, slave locked) and return its
    /// index — the number `TIOCGPTN` reports and the `N` in `/dev/pts/N`.
    pub(super) fn alloc(&mut self) -> usize {
        // Reuse a fully-closed slot if one exists, else grow.
        if let Some(i) = self.table.iter().position(|p| !p.master_open && p.slave_refs == 0) {
            self.table[i] = Pty::new();
            i
        } else {
            self.table.push(Pty::new());
            self.table.len() - 1
        }
    }

    /// Whether slave `n` may be opened (`/dev/pts/N`): it exists, its master is
    /// open, and it has been unlocked with `TIOCSPTLCK`.
    pub(super) fn slave_openable(&self, n: usize) -> bool {
        self.table.get(n).is_some_and(|p| p.master_open && !p.locked)
    }
    pub(super) fn open_slave(&mut self, n: usize) {
        if let Some(p) = self.table.get_mut(n) {
            p.slave_refs += 1;
        }
    }

    pub(super) fn close_master(&mut self, n: usize) {
        if let Some(p) = self.table.get_mut(n) {
            p.master_open = false;
        }
    }
    pub(super) fn close_slave(&mut self, n: usize) {
        if let Some(p) = self.table.get_mut(n) {
            p.slave_refs = p.slave_refs.saturating_sub(1);
        }
    }

    // ---- data path -------------------------------------------------------

    pub(super) fn master_write(&mut self, n: usize, data: &[u8]) {
        if let Some(p) = self.table.get_mut(n) {
            p.master_write(data);
        }
    }
    pub(super) fn slave_write(&mut self, n: usize, data: &[u8]) {
        if let Some(p) = self.table.get_mut(n) {
            p.slave_write(data);
        }
    }
    /// Master read: drain up to `cap` bytes of terminal output. `None` = would
    /// block (no data yet, slave still open); `Some(vec)` may be empty on EOF
    /// (all slaves closed).
    pub(super) fn master_read(&mut self, n: usize, cap: usize) -> Option<Vec<u8>> {
        let p = self.table.get_mut(n)?;
        if p.output.is_empty() {
            return if p.slave_refs == 0 { Some(Vec::new()) } else { None };
        }
        let k = cap.min(p.output.len());
        Some(p.output.drain(..k).collect())
    }
    /// Slave read: drain up to `cap` bytes of terminal input. `None` = would
    /// block; `Some(vec)` may be empty on EOF (master closed).
    pub(super) fn slave_read(&mut self, n: usize, cap: usize) -> Option<Vec<u8>> {
        let p = self.table.get_mut(n)?;
        if p.input.is_empty() {
            return if p.master_open { None } else { Some(Vec::new()) };
        }
        let k = cap.min(p.input.len());
        Some(p.input.drain(..k).collect())
    }

    // ---- readiness (poll/epoll) -----------------------------------------

    /// `POLLIN`/`POLLOUT`/`POLLHUP` bits for the master end.
    pub(super) fn master_ready(&self, n: usize) -> u32 {
        const IN: u32 = 0x1;
        const OUT: u32 = 0x4;
        const HUP: u32 = 0x10;
        self.table.get(n).map_or(0, |p| {
            let mut m = OUT;
            if !p.output.is_empty() {
                m |= IN;
            }
            if p.slave_refs == 0 {
                m |= IN | HUP; // EOF is readable
            }
            m
        })
    }
    /// `POLLIN`/`POLLOUT`/`POLLHUP` bits for a slave end.
    pub(super) fn slave_ready(&self, n: usize) -> u32 {
        const IN: u32 = 0x1;
        const OUT: u32 = 0x4;
        const HUP: u32 = 0x10;
        self.table.get(n).map_or(0, |p| {
            let mut m = OUT;
            if !p.input.is_empty() {
                m |= IN;
            }
            if !p.master_open {
                m |= IN | HUP;
            }
            m
        })
    }
    /// Bytes available to read (for `FIONREAD`) on the master/slave end.
    pub(super) fn master_avail(&self, n: usize) -> u64 {
        self.table.get(n).map_or(0, |p| p.output.len() as u64)
    }
    pub(super) fn slave_avail(&self, n: usize) -> u64 {
        self.table.get(n).map_or(0, |p| p.input.len() as u64)
    }

    // ---- termios / winsize ioctls ---------------------------------------

    pub(super) fn get_termios(&self, n: usize) -> Option<[u8; TERMIOS_LEN]> {
        self.table.get(n).map(|p| p.termios)
    }
    pub(super) fn set_termios(&mut self, n: usize, t: [u8; TERMIOS_LEN]) {
        if let Some(p) = self.table.get_mut(n) {
            p.termios = t;
        }
    }
    pub(super) fn get_winsize(&self, n: usize) -> Option<[u8; WINSIZE_LEN]> {
        self.table.get(n).map(|p| p.winsize)
    }
    pub(super) fn set_winsize(&mut self, n: usize, w: [u8; WINSIZE_LEN]) {
        if let Some(p) = self.table.get_mut(n) {
            p.winsize = w;
        }
    }
    /// `TIOCSPTLCK`: lock (`*arg != 0`) or unlock (`0`) slave opening.
    pub(super) fn set_lock(&mut self, n: usize, locked: bool) {
        if let Some(p) = self.table.get_mut(n) {
            p.locked = locked;
        }
    }

    /// `O_NONBLOCK` for one end (`master = true` selects the master fd).
    pub(super) fn set_nonblock(&mut self, n: usize, master: bool, nb: bool) {
        if let Some(p) = self.table.get_mut(n) {
            if master {
                p.master_nonblock = nb;
            } else {
                p.slave_nonblock = nb;
            }
        }
    }
    pub(super) fn is_nonblock(&self, n: usize, master: bool) -> bool {
        self.table.get(n).is_some_and(|p| if master { p.master_nonblock } else { p.slave_nonblock })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_line_discipline_echoes_and_buffers() {
        let mut t = Ptys::default();
        let n = t.alloc();
        t.open_slave(n); // pretend the slave is open
        // Master writes "hi" then Enter: nothing readable by the slave until \n,
        // and each char is echoed back to the master (ECHO on by default).
        t.master_write(n, b"hi");
        assert_eq!(t.slave_read(n, 64), None, "no full line yet");
        t.master_write(n, b"\r"); // ICRNL: CR becomes NL, completing the line
        assert_eq!(t.slave_read(n, 64).unwrap(), b"hi\n");
        // Echo (with ONLCR on the \n) is queued for the master.
        assert_eq!(t.master_read(n, 64).unwrap(), b"hi\r\n");
    }

    #[test]
    fn raw_mode_passes_bytes_through_without_echo() {
        let mut t = Ptys::default();
        let n = t.alloc();
        t.open_slave(n);
        // Clear ICANON and ECHO (raw-ish).
        let mut tm = t.get_termios(n).unwrap();
        let lflag = u32::from_le_bytes(tm[12..16].try_into().unwrap()) & !(ICANON | ECHO);
        tm[12..16].copy_from_slice(&lflag.to_le_bytes());
        t.set_termios(n, tm);
        t.master_write(n, b"ab");
        assert_eq!(t.slave_read(n, 64).unwrap(), b"ab", "raw: available immediately");
        assert_eq!(t.master_read(n, 64), None, "raw: no echo");
    }

    #[test]
    fn erase_removes_the_last_char() {
        let mut t = Ptys::default();
        let n = t.alloc();
        t.open_slave(n);
        t.master_write(n, b"ax\x7fb\n"); // 'a','x', DEL erases 'x', 'b', Enter
        assert_eq!(t.slave_read(n, 64).unwrap(), b"ab\n");
    }

    #[test]
    fn master_sees_eof_when_all_slaves_close() {
        let mut t = Ptys::default();
        let n = t.alloc();
        t.open_slave(n);
        t.close_slave(n);
        assert_eq!(t.master_read(n, 64).unwrap(), Vec::<u8>::new(), "EOF, not block");
        assert!(t.master_ready(n) & 0x10 != 0, "POLLHUP set");
    }
}
