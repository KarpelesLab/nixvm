//! The `NIXVM_*` settings, read from the process environment only on request.
//!
//! nixvm has a number of switches named by environment variables: host
//! networking (`NIXVM_NET=host`), the syscall deny-list
//! (`NIXVM_DENY_SYSCALLS`), resource limits (`NIXVM_LIMIT_*`), the number of
//! CPUs (`NIXVM_CPUS`), forcing the interpreter (`NIXVM_INTERP=1`), the GPU
//! (`NIXVM_GPU`), the image cache and registry credentials, and debugging
//! traces (`NIXVM_TRACE`, `NIXVM_SCHEDTRACE`, `NIXVM_SIGTRACE`, …).
//!
//! The library never looks at the process environment for them by itself.
//! Every place that honours one reads it from the process-wide [`Env`]
//! [installed](Env::install) here, which is empty until someone installs one.
//! The `nixvm` binaries install [`Env::from_process`] first thing, so their
//! command-line behaviour is what the variables say; an embedder that installs
//! nothing gets the defaults whatever its environment holds, and one that
//! wants a switch sets it explicitly:
//!
//! ```
//! use nixvm::env::Env;
//!
//! // Honour the caller's NIXVM_* variables, as the `nixvm` binary does…
//! Env::from_process().install();
//! // …or only chosen ones (here: trace every syscall).
//! Env::new().with("NIXVM_TRACE", "1").install();
//! // …or none at all again.
//! Env::new().install();
//! ```
//!
//! What a variable means is documented where it's honoured; [`Env`] only
//! carries the strings. For library unit tests (`cfg(test)`) nothing
//! installed means the process's own variables, so `NIXVM_TRACE=1 cargo test`
//! keeps working; integration tests install [`Env::from_process`] themselves.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

/// A set of `NIXVM_*` settings (name → value). See the [module docs](self).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Env {
    vars: BTreeMap<String, String>,
}

/// The installed settings; `None` until [`Env::install`] runs.
static INSTALLED: RwLock<Option<Arc<Env>>> = RwLock::new(None);

impl Env {
    /// No settings: every switch at its default.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Every `NIXVM_*` variable of this process's environment (non-UTF-8
    /// values are skipped).
    #[must_use]
    pub fn from_process() -> Self {
        Self {
            vars: std::env::vars()
                .filter(|(k, _)| k.starts_with("NIXVM_"))
                .collect(),
        }
    }

    /// These settings plus `name=value` (replacing an earlier value).
    #[must_use]
    pub fn with(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.set(name, value);
        self
    }

    /// Set `name` to `value`.
    pub fn set(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.vars.insert(name.into(), value.into());
    }

    /// Remove `name`.
    pub fn remove(&mut self, name: &str) {
        self.vars.remove(name);
    }

    /// The value of `name`, if set.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&str> {
        self.vars.get(name).map(String::as_str)
    }

    /// Every setting, by name.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.vars.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Make these the settings every part of nixvm in this process reads
    /// from here on, replacing what was installed before. Read once at the
    /// start of what they govern (a kernel reads its traces and syscall
    /// policy when it's created), so install before building a guest.
    pub fn install(self) {
        *INSTALLED
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::new(self));
    }

    /// The settings installed now (empty if none were; see the
    /// [module docs](self) for unit tests).
    #[must_use]
    pub fn installed() -> Arc<Env> {
        if let Some(env) = &*INSTALLED
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            return Arc::clone(env);
        }
        #[cfg(test)]
        {
            static FROM_TESTS: std::sync::OnceLock<Arc<Env>> = std::sync::OnceLock::new();
            Arc::clone(FROM_TESTS.get_or_init(|| Arc::new(Env::from_process())))
        }
        #[cfg(not(test))]
        {
            static EMPTY: std::sync::OnceLock<Arc<Env>> = std::sync::OnceLock::new();
            Arc::clone(EMPTY.get_or_init(|| Arc::new(Env::new())))
        }
    }
}

/// The installed value of `name` (see [`Env::installed`]).
pub(crate) fn var(name: &str) -> Option<String> {
    Env::installed().get(name).map(str::to_owned)
}

/// Whether `name` is set at all in the installed settings.
pub(crate) fn is_set(name: &str) -> bool {
    Env::installed().get(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_are_only_what_was_given() {
        let env = Env::new()
            .with("NIXVM_NET", "host")
            .with("NIXVM_NET", "none");
        assert_eq!(env.get("NIXVM_NET"), Some("none"));
        assert_eq!(env.get("NIXVM_TRACE"), None);
        assert_eq!(env.iter().count(), 1);
    }
}
