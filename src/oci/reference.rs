//! Image references, read the way `docker pull` reads them.

use super::OciError;

/// `registry/repository:tag` or `registry/repository@digest`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    /// The registry as named: `docker.io`, `ghcr.io`, `localhost:5000`.
    pub registry: String,
    /// The repository: `library/alpine`, `owner/image`.
    pub repository: String,
    /// A tag (`3.20`, `latest`) or a digest (`sha256:…`).
    pub reference: String,
}

impl Reference {
    /// Parse `name`: a missing registry is Docker Hub, a one-part Hub name is
    /// under `library/`, and a missing tag is `latest`.
    ///
    /// # Errors
    /// [`OciError::Reference`] for an empty or malformed name.
    pub fn parse(name: &str) -> Result<Self, OciError> {
        let bad = |why: &str| OciError::Reference(format!("{name:?}: {why}"));
        let name = name.trim();
        let name = name.strip_prefix("docker://").unwrap_or(name);
        if name.is_empty() {
            return Err(bad("empty"));
        }
        let (rest, digest) = match name.split_once('@') {
            Some((r, d)) => {
                let Some((algo, hex)) = d.split_once(':') else {
                    return Err(bad("a digest is algorithm:hex"));
                };
                if algo != "sha256"
                    || hex.len() != 64
                    || !hex.bytes().all(|b| b.is_ascii_hexdigit())
                {
                    return Err(bad("only sha256 digests are supported"));
                }
                (r, Some(d.to_ascii_lowercase()))
            }
            None => (name, None),
        };
        // The tag follows the last ':' when no '/' comes after it (a ':'
        // before a '/' is a registry's port).
        let (path, tag) = match rest.rfind(':') {
            Some(i) if !rest[i..].contains('/') => (&rest[..i], Some(&rest[i + 1..])),
            _ => (rest, None),
        };
        let (registry, repository) = match path.split_once('/') {
            Some((first, more))
                if first.contains('.') || first.contains(':') || first == "localhost" =>
            {
                (first.to_string(), more.to_string())
            }
            _ => ("docker.io".to_string(), path.to_string()),
        };
        let registry = match registry.as_str() {
            "index.docker.io" | "registry-1.docker.io" => "docker.io".to_string(),
            _ => registry,
        };
        let repository = if registry == "docker.io" && !repository.contains('/') {
            format!("library/{repository}")
        } else {
            repository
        };
        let valid = |s: &str| {
            !s.is_empty()
                && s.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"._-/".contains(&b))
        };
        if !valid(&repository) || repository.starts_with('/') || repository.ends_with('/') {
            return Err(bad(
                "a repository is lowercase letters, digits, '.', '_', '-' and '/'",
            ));
        }
        let reference = match (digest, tag) {
            (Some(d), _) => d,
            (None, Some(t)) if !t.is_empty() && t.len() <= 128 => t.to_string(),
            (None, Some(_)) => return Err(bad("bad tag")),
            (None, None) => "latest".to_string(),
        };
        Ok(Self {
            registry,
            repository,
            reference,
        })
    }

    /// The registry's API root: `https://registry-1.docker.io` for Docker
    /// Hub, plain `http` for `localhost`.
    #[must_use]
    pub fn api_base(&self) -> String {
        match self.registry.as_str() {
            "docker.io" => "https://registry-1.docker.io".to_string(),
            r if r == "localhost" || r.starts_with("localhost:") || r.starts_with("127.") => {
                format!("http://{r}")
            }
            r => format!("https://{r}"),
        }
    }

    /// Whether this names a digest rather than a tag.
    #[must_use]
    pub fn is_digest(&self) -> bool {
        self.reference.starts_with("sha256:")
    }
}

impl core::fmt::Display for Reference {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let sep = if self.is_digest() { '@' } else { ':' };
        write!(
            f,
            "{}/{}{sep}{}",
            self.registry, self.repository, self.reference
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> (String, String, String) {
        let r = Reference::parse(s).unwrap();
        (r.registry, r.repository, r.reference)
    }

    #[test]
    fn names_read_as_docker_reads_them() {
        let hub = |repo: &str, r: &str| ("docker.io".to_string(), repo.to_string(), r.to_string());
        assert_eq!(p("alpine"), hub("library/alpine", "latest"));
        assert_eq!(p("alpine:3.20"), hub("library/alpine", "3.20"));
        assert_eq!(p("docker.io/library/alpine:3"), hub("library/alpine", "3"));
        assert_eq!(p("docker://nginx"), hub("library/nginx", "latest"));
        assert_eq!(p("bitnami/redis:7.2"), hub("bitnami/redis", "7.2"));
        assert_eq!(
            p("ghcr.io/owner/img:v1"),
            ("ghcr.io".into(), "owner/img".into(), "v1".into())
        );
        assert_eq!(
            p("localhost:5000/a/b"),
            ("localhost:5000".into(), "a/b".into(), "latest".into())
        );
        let d = format!("sha256:{}", "a".repeat(64));
        assert_eq!(p(&format!("alpine:3@{d}")), hub("library/alpine", &d));
        assert!(Reference::parse("").is_err());
        assert!(Reference::parse("Alpine").is_err());
        assert!(Reference::parse("alpine@sha256:12").is_err());
        let r = Reference::parse("alpine").unwrap();
        assert_eq!(r.api_base(), "https://registry-1.docker.io");
        assert_eq!(r.to_string(), "docker.io/library/alpine:latest");
        assert_eq!(
            Reference::parse("localhost:5000/x").unwrap().api_base(),
            "http://localhost:5000"
        );
    }
}
