//! Module ID.

use core::{
    any::{Any, TypeId},
    fmt,
};

/// Identifier of an [`ExecutableModule`](crate::ExecutableModule). This is usually a "small" type,
/// such as an integer or a string.
///
/// The ID is provided when [creating](crate::ExecutableModule::new()) a module. It is displayed
/// in error messages (using `Display::fmt`). `ModuleId` is also associated with some types
/// (e.g., [`InterpretedFn`] and [`LocationInModule`]), which allows to obtain module info.
/// This can be particularly useful for outputting rich error information.
///
/// A `ModuleId` can be downcast to a specific type, similarly to [`Any`].
///
/// [`InterpretedFn`]: crate::InterpretedFn
/// [`LocationInModule`]: crate::error::LocationInModule
pub trait ModuleId: Any + fmt::Display + Send + Sync {}

impl dyn ModuleId {
    /// Returns `true` if the boxed type is the same as `T`.
    ///
    /// This method is effectively a carbon copy of [`<dyn Any>::is`]. Such a copy is necessary
    /// because `&dyn ModuleId` cannot be converted to `&dyn Any`, despite `ModuleId` having `Any`
    /// as a super-trait.
    ///
    /// [`<dyn Any>::is`]: https://doc.rust-lang.org/std/any/trait.Any.html#method.is
    #[inline]
    pub fn is<T: ModuleId>(&self) -> bool {
        let t = TypeId::of::<T>();
        let concrete = self.type_id();
        t == concrete
    }

    /// Returns a reference to the boxed value if it is of type `T`, or `None` if it isn't.
    ///
    /// This method is effectively a carbon copy of [`<dyn Any>::downcast_ref`]. Such a copy
    /// is necessary because `&dyn ModuleId` cannot be converted to `&dyn Any`, despite `ModuleId`
    /// having `Any` as a super-trait.
    ///
    /// [`<dyn Any>::downcast_ref`]: https://doc.rust-lang.org/std/any/trait.Any.html#method.downcast_ref
    pub fn downcast_ref<T: ModuleId>(&self) -> Option<&T> {
        if self.is::<T>() {
            // SAFETY: Same code as for `<dyn Any>::downcast_ref()`.
            unsafe { Some(&*(self as *const dyn ModuleId).cast::<T>()) }
        } else {
            None
        }
    }
}

impl fmt::Debug for dyn ModuleId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "ModuleId({self})")
    }
}

impl ModuleId for &'static str {}

/// Module identifier that has a single possible value, which is displayed as `*`.
///
/// This type is a `ModuleId`-compatible replacement of `()`; `()` does not implement `Display`
/// and thus cannot implement `ModuleId` directly.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WildcardId;

impl ModuleId for WildcardId {}

impl fmt::Display for WildcardId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("*")
    }
}

/// Indexed module ID containing a prefix part (e.g., `snippet`).
///
/// The ID is `Display`ed as `{prefix} #{index + 1}`:
///
/// ```
/// # use arithmetic_eval::exec::IndexedId;
/// let module_id = IndexedId::new("snippet", 4);
/// assert_eq!(module_id.to_string(), "snippet #5");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndexedId {
    /// Prefix that can identify the nature of the module, such as `snippet`.
    pub prefix: &'static str,
    /// 0-based index of the module.
    pub index: usize,
}

impl IndexedId {
    /// Creates a new ID instance.
    pub const fn new(prefix: &'static str, index: usize) -> Self {
        Self { prefix, index }
    }
}

impl fmt::Display for IndexedId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} #{}", self.prefix, self.index + 1)
    }
}

impl ModuleId for IndexedId {}
