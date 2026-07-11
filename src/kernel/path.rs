//! Absolute guest-path normalization.
//!
//! Collapses `.`/`..`/empty components and duplicate slashes into a clean
//! absolute path. This is lexical only (it does not follow symlinks — that
//! happens in the VFS lookup); `..` at the root stays at the root.

/// Normalize `p` (treated as absolute) into a clean `/a/b/c` path.
#[must_use]
pub fn normalize(p: &str) -> String {
    let mut stack: Vec<&str> = Vec::new();
    for comp in p.split('/') {
        match comp {
            "" | "." => {}
            ".." => {
                stack.pop();
            }
            other => stack.push(other),
        }
    }
    if stack.is_empty() {
        "/".to_string()
    } else {
        let mut out = String::new();
        for c in stack {
            out.push('/');
            out.push_str(c);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::normalize;

    #[test]
    fn normalizes_paths() {
        assert_eq!(normalize("/"), "/");
        assert_eq!(normalize("/a/b/c"), "/a/b/c");
        assert_eq!(normalize("/a//b/./c"), "/a/b/c");
        assert_eq!(normalize("/a/b/../c"), "/a/c");
        assert_eq!(normalize("/a/../../b"), "/b");
        assert_eq!(normalize("/work/./x/../y"), "/work/y");
        assert_eq!(normalize(""), "/");
    }
}
