//! Module ID.

use core::{any::Any, fmt};

/// FIXME
pub trait ModuleId: Any + fmt::Display + Send + Sync {
    /// FIXME
    fn clone_boxed(&self) -> Box<dyn ModuleId>;
}

impl fmt::Debug for dyn ModuleId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "ModuleId({})", self)
    }
}

impl ModuleId for &'static str {
    fn clone_boxed(&self) -> Box<dyn ModuleId> {
        Box::new(*self)
    }
}

/// FIXME
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WildcardId;

impl ModuleId for WildcardId {
    fn clone_boxed(&self) -> Box<dyn ModuleId> {
        Box::new(*self)
    }
}

impl fmt::Display for WildcardId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("*")
    }
}
