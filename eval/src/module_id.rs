//! Module ID.

use core::{
    any::{Any, TypeId},
    fmt,
};

/// FIXME
pub trait ModuleId: Any + fmt::Display + Send + Sync {
    /// FIXME
    fn clone_boxed(&self) -> Box<dyn ModuleId>;
}

impl dyn ModuleId {
    /// FIXME
    #[inline]
    pub fn is<T: Any>(&self) -> bool {
        let t = TypeId::of::<T>();
        let concrete = self.type_id();
        t == concrete
    }

    /// FIXME
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        // Same code as for `<dyn Any>::downcast_ref()`.
        if self.is::<T>() {
            unsafe { Some(&*(self as *const dyn ModuleId as *const T)) }
        } else {
            None
        }
    }
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
