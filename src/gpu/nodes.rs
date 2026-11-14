//! The filesystem nodes a guest's libdrm and Mesa look for to find a GPU:
//! `/dev/dri/renderD128`, and the sysfs entries that say it is a virtio-gpu
//! (PCI `1af4:1050`, driver `virtio_gpu`) so Mesa's loader picks `virgl`.
//!
//! libdrm (`drmGetDevices2`, `drmGetDevice2`) walks
//! `/sys/dev/char/226:128` → `/sys/devices/pci0000:00/0000:00:02.0/drm/
//! renderD128`, reads `device/subsystem` (a link ending in `pci`), the PCI
//! function's `vendor`/`device`/… files, and lists `device/drm`. These are
//! small fixed trees mounted at their places when a GPU is attached
//! ([`mounts`]), so a guest without one sees exactly what it did before.

use std::io;

use crate::fs::{Attrs, DirEntry, MountFs, NodeKind};

/// `DRM_MAJOR`.
pub const DRM_MAJOR: u64 = 226;
/// The render node's minor.
pub const RENDER_MINOR: u64 = 128;
/// The render node's guest path.
pub const RENDER_NODE: &str = "/dev/dri/renderD128";
/// The PCI function the device claims to be.
pub const PCI_SLOT: &str = "0000:00:02.0";
/// virtio's PCI vendor and the modern virtio-gpu device id.
pub const PCI_VENDOR: u16 = 0x1af4;
pub const PCI_DEVICE: u16 = 0x1050;

/// `st_rdev` of the render node (glibc's `makedev` encoding).
#[must_use]
pub const fn render_rdev() -> u64 {
    ((DRM_MAJOR & 0xfff) << 8) | (RENDER_MINOR & 0xff)
}

#[derive(Debug, Clone)]
enum Node {
    Dir,
    File(Vec<u8>),
    Link(&'static str),
    Char(u64),
}

/// A fixed tree of nodes, by mount-relative path.
#[derive(Debug)]
pub struct Tree {
    nodes: Vec<(String, Node)>,
    base_inode: u64,
}

impl Tree {
    fn new(base_inode: u64, nodes: Vec<(&str, Node)>) -> Self {
        let mut v: Vec<(String, Node)> = vec![(String::new(), Node::Dir)];
        v.extend(nodes.into_iter().map(|(p, n)| (p.to_string(), n)));
        Self {
            nodes: v,
            base_inode,
        }
    }

    fn find(&self, rel: &str) -> Option<(usize, &Node)> {
        self.nodes
            .iter()
            .enumerate()
            .find(|(_, (p, _))| p == rel)
            .map(|(i, (_, n))| (i, n))
    }
}

fn enoent() -> io::Error {
    io::Error::from_raw_os_error(2)
}

impl MountFs for Tree {
    fn fs_type(&self) -> &'static str {
        if self.base_inode == DEV_INODES {
            "devtmpfs"
        } else {
            "sysfs"
        }
    }

    fn stat(&mut self, rel: &str) -> Option<Attrs> {
        let (i, n) = self.find(rel)?;
        let (kind, mode, size, rdev) = match n {
            Node::Dir => (NodeKind::Dir, 0o040_755, 0, 0),
            Node::File(d) => (NodeKind::File, 0o100_444, d.len() as u64, 0),
            Node::Link(t) => (NodeKind::Symlink, 0o120_777, t.len() as u64, 0),
            Node::Char(r) => (NodeKind::CharDevice, 0o020_666, 0, *r),
        };
        Some(Attrs {
            kind,
            size,
            mode,
            uid: 0,
            gid: 0,
            atime: 0,
            mtime: 0,
            inode: self.base_inode + i as u64,
            nlink: 1,
            rdev,
            dev: 0,
        })
    }

    fn read_at(&mut self, rel: &str, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        match self.find(rel) {
            Some((_, Node::File(d))) => {
                let off = (off as usize).min(d.len());
                let n = buf.len().min(d.len() - off);
                buf[..n].copy_from_slice(&d[off..off + n]);
                Ok(n)
            }
            Some((_, Node::Dir)) => Err(io::Error::from_raw_os_error(21)),
            Some((_, Node::Char(_))) => Ok(0),
            Some((_, Node::Link(_))) => Err(io::Error::from_raw_os_error(22)),
            None => Err(enoent()),
        }
    }

    fn readdir(&mut self, rel: &str) -> io::Result<Vec<DirEntry>> {
        match self.find(rel) {
            Some((_, Node::Dir)) => {}
            Some(_) => return Err(io::Error::from_raw_os_error(20)),
            None => return Err(enoent()),
        }
        let prefix = if rel.is_empty() {
            String::new()
        } else {
            format!("{rel}/")
        };
        Ok(self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, (p, n))| {
                let name = p.strip_prefix(&prefix)?;
                if name.is_empty() || name.contains('/') || p.is_empty() {
                    return None;
                }
                Some(DirEntry {
                    name: name.to_string(),
                    kind: match n {
                        Node::Dir => NodeKind::Dir,
                        Node::File(_) => NodeKind::File,
                        Node::Link(_) => NodeKind::Symlink,
                        Node::Char(_) => NodeKind::CharDevice,
                    },
                    inode: self.base_inode + i as u64,
                })
            })
            .collect())
    }

    fn readlink(&mut self, rel: &str) -> io::Result<String> {
        match self.find(rel) {
            Some((_, Node::Link(t))) => Ok((*t).to_string()),
            Some(_) => Err(io::Error::from_raw_os_error(22)),
            None => Err(enoent()),
        }
    }

    // The render node is opened for writing; the kernel never writes to it
    // through the filesystem (ioctls carry everything), so accept and drop.
    fn write_at(&mut self, rel: &str, _off: u64, buf: &[u8]) -> io::Result<usize> {
        match self.find(rel) {
            Some((_, Node::Char(_))) => Ok(buf.len()),
            Some(_) => Err(io::Error::from_raw_os_error(30)),
            None => Err(enoent()),
        }
    }

    fn read_only(&self) -> bool {
        false
    }
}

const DEV_INODES: u64 = 0x4752_0000;
const SYS_INODES: u64 = 0x4752_1000;

/// The PCI function's first 64 bytes of config space: what libdrm reads
/// when it wants the revision too.
fn pci_config() -> Vec<u8> {
    let mut c = vec![0u8; 64];
    c[0..2].copy_from_slice(&PCI_VENDOR.to_le_bytes());
    c[2..4].copy_from_slice(&PCI_DEVICE.to_le_bytes());
    c[8] = 1; // revision
    c[9..12].copy_from_slice(&[0x00, 0x00, 0x03]); // class: display controller
    c[44..46].copy_from_slice(&PCI_VENDOR.to_le_bytes());
    c[46..48].copy_from_slice(&0x1100u16.to_le_bytes());
    c
}

fn text(s: &str) -> Node {
    Node::File(s.as_bytes().to_vec())
}

/// The trees to mount for a GPU, with their guest mount points.
#[must_use]
pub fn mounts() -> Vec<(&'static str, Box<dyn MountFs>)> {
    let dev = Tree::new(
        DEV_INODES,
        vec![
            ("renderD128", Node::Char(render_rdev())),
            ("by-path", Node::Dir),
            (
                "by-path/pci-0000:00:02.0-render",
                Node::Link("../renderD128"),
            ),
        ],
    );
    let dev_char = Tree::new(
        SYS_INODES,
        vec![(
            "226:128",
            Node::Link("../../devices/pci0000:00/0000:00:02.0/drm/renderD128"),
        )],
    );
    let uevent = format!(
        "DRIVER=virtio-pci\nPCI_CLASS=30000\nPCI_ID={PCI_VENDOR:04X}:{PCI_DEVICE:04X}\n\
         PCI_SUBSYS_ID={PCI_VENDOR:04X}:1100\nPCI_SLOT_NAME={PCI_SLOT}\n"
    );
    let devices = Tree::new(
        SYS_INODES + 0x100,
        vec![
            (PCI_SLOT, Node::Dir),
            ("0000:00:02.0/vendor", text("0x1af4\n")),
            ("0000:00:02.0/device", text("0x1050\n")),
            ("0000:00:02.0/subsystem_vendor", text("0x1af4\n")),
            ("0000:00:02.0/subsystem_device", text("0x1100\n")),
            ("0000:00:02.0/revision", text("0x01\n")),
            ("0000:00:02.0/class", text("0x030000\n")),
            ("0000:00:02.0/boot_vga", text("1\n")),
            ("0000:00:02.0/config", Node::File(pci_config())),
            ("0000:00:02.0/uevent", Node::File(uevent.into_bytes())),
            ("0000:00:02.0/subsystem", Node::Link("../../../bus/pci")),
            ("0000:00:02.0/drm", Node::Dir),
            ("0000:00:02.0/drm/renderD128", Node::Dir),
            ("0000:00:02.0/drm/renderD128/dev", text("226:128\n")),
            (
                "0000:00:02.0/drm/renderD128/uevent",
                text("MAJOR=226\nMINOR=128\nDEVNAME=dri/renderD128\nDEVTYPE=drm_minor\n"),
            ),
            (
                "0000:00:02.0/drm/renderD128/device",
                Node::Link("../../../0000:00:02.0"),
            ),
            (
                "0000:00:02.0/drm/renderD128/subsystem",
                Node::Link("../../../../../class/drm"),
            ),
        ],
    );
    let bus = Tree::new(
        SYS_INODES + 0x200,
        vec![
            ("devices", Node::Dir),
            (
                "devices/0000:00:02.0",
                Node::Link("../../../devices/pci0000:00/0000:00:02.0"),
            ),
            ("drivers", Node::Dir),
        ],
    );
    let class = Tree::new(
        SYS_INODES + 0x300,
        vec![
            (
                "renderD128",
                Node::Link("../../devices/pci0000:00/0000:00:02.0/drm/renderD128"),
            ),
            ("version", text("drm 1.1.0 20060810\n")),
        ],
    );
    vec![
        ("/dev/dri", Box::new(dev) as Box<dyn MountFs>),
        ("/sys/dev/char", Box::new(dev_char)),
        ("/sys/devices/pci0000:00", Box::new(devices)),
        ("/sys/bus/pci", Box::new(bus)),
        ("/sys/class/drm", Box::new(class)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trees_answer_what_libdrm_reads() {
        let mut m = mounts();
        let (_, dev) = &mut m[0];
        let a = dev.stat("renderD128").unwrap();
        assert_eq!(a.kind, NodeKind::CharDevice);
        assert_eq!(a.rdev, 0xe280);
        let names: Vec<String> = dev
            .readdir("")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(names, ["renderD128", "by-path"]);
        let (_, devices) = &mut m[2];
        let mut buf = [0u8; 16];
        let n = devices.read_at("0000:00:02.0/vendor", 0, &mut buf).unwrap();
        assert_eq!(&buf[..n], b"0x1af4\n");
        assert!(
            devices
                .readlink("0000:00:02.0/subsystem")
                .unwrap()
                .ends_with("/pci")
        );
        let drm: Vec<String> = devices
            .readdir("0000:00:02.0/drm")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(drm, ["renderD128"]);
        let mut cfg = [0u8; 64];
        devices.read_at("0000:00:02.0/config", 0, &mut cfg).unwrap();
        assert_eq!(&cfg[0..4], &[0xf4, 0x1a, 0x50, 0x10]);
    }
}
