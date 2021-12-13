//! `VariableMap` trait and implementations.

#![allow(renamed_and_removed_lints, clippy::unknown_clippy_lints)]
// ^ `missing_panics_doc` is newer than MSRV, and `clippy::unknown_clippy_lints` is removed
// since Rust 1.51.

use arithmetic_parser::{grammars::Grammar, Block};

use core::{cmp::Ordering, fmt};

use crate::{
    exec::{ExecutableModule, ModuleId, ModuleImports},
    fns, Environment, Error, Object, StandardPrototypes, Value,
};

/// Encapsulates read access to named variables and, optionally, [`StandardPrototypes`].
pub trait VariableMap<'a, T> {
    /// Returns value of the named variable, or `None` if it is not defined.
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>>;

    /// Gets [`Prototype`]s for standard types defined by this map, or `None` if this map
    /// does not define such prototypes.
    ///
    /// The default implementation returns [`StandardPrototypes::new()`] (i.e., empty prototypes
    /// for all standard types).
    ///
    /// [`Prototype`]: crate::Prototype
    fn get_prototypes(&self) -> StandardPrototypes<T> {
        StandardPrototypes::new()
    }

    /// Creates a module based on imports solely from this map.
    ///
    /// The default implementation is reasonable for most cases.
    fn compile_module<Id, G>(
        &self,
        id: Id,
        block: &Block<'a, G>,
    ) -> Result<ExecutableModule<'a, T>, Error<'a>>
    where
        Id: ModuleId,
        G: Grammar<'a, Lit = T>,
        T: 'static + Clone,
    {
        ExecutableModule::builder(id, block)?
            .with_imports_from(self)
            .try_build()
    }
}

impl<'a, T: Clone> VariableMap<'a, T> for Environment<'a, T> {
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>> {
        self.get(name).cloned()
    }

    fn get_prototypes(&self) -> StandardPrototypes<T> {
        self.prototypes().clone()
    }
}

impl<'a, T: Clone> VariableMap<'a, T> for ModuleImports<'a, T> {
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>> {
        self.get(name).cloned()
    }

    fn get_prototypes(&self) -> StandardPrototypes<T> {
        self.prototypes().clone()
    }
}

/// [`VariableMap`] implementation that fills specified variables with a constant value.
/// Useful when building an [`ExecutableModule`] and some variables need to be set
/// later.
///
/// [`ExecutableModule`]: crate::ExecutableModule
#[derive(Debug, Clone)]
pub struct Filler<'s, 'a, T> {
    value: Value<'a, T>,
    var_names: &'s [&'s str],
}

impl<'s, T> Filler<'s, '_, T> {
    /// Creates a `Filler` that returns [`Value::void()`] for all variables in `var_names`.
    pub const fn void(var_names: &'s [&'s str]) -> Self {
        Self {
            value: Value::void(),
            var_names,
        }
    }
}

impl<'a, T: Clone> VariableMap<'a, T> for Filler<'_, 'a, T> {
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>> {
        if self.var_names.contains(&name) {
            Some(self.value.clone())
        } else {
            None
        }
    }
}

/// Commonly used constants and functions from the [`fns` module](fns).
///
/// # Contents
///
/// - All functions from the `fns` module, except for [`Compare`](fns::Compare)
///   (contained in [`Comparisons`]) and assertion functions (contained in [`Assertions`]).
///   All functions are named in lowercase, e.g., `if`, `map`.
/// - `true` and `false` Boolean constants.
#[derive(Debug, Clone, Copy, Default)]
pub struct Prelude;

#[allow(clippy::missing_panics_doc)] // false positive; `unwrap()` never panics
impl Prelude {
    /// Creates an iterator over contained values and the corresponding names.
    pub fn iter<T>(self) -> impl Iterator<Item = (&'static str, Value<'static, T>)>
    where
        T: 'static + Clone,
    {
        const VAR_NAMES: &[&str] = &[
            "false", "true", "if", "loop", "while", "map", "filter", "fold", "push", "merge",
            "impl", "defer",
        ];

        VAR_NAMES
            .iter()
            .map(move |&var_name| (var_name, self.get_variable(var_name).unwrap()))
    }

    /// Returns standard prototypes corresponding to the contained functions.
    ///
    /// Currently, this only sets a prototype for tuples / arrays containing
    /// `map`, `filter`, `fold`, `push` and `merge` functions.
    pub fn prototypes<T>(self) -> StandardPrototypes<T>
    where
        T: 'static + Clone,
    {
        const ARRAY_FNS: &[&str] = &["map", "filter", "fold", "push", "merge"];

        let array_proto: Object<_> = ARRAY_FNS
            .iter()
            .map(|&var_name| (var_name, self.get_variable(var_name).unwrap()))
            .collect();
        StandardPrototypes::new().with_array_proto(array_proto)
    }
}

impl<'a, T: 'static + Clone> VariableMap<'a, T> for Prelude {
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>> {
        Some(match name {
            "false" => Value::Bool(false),
            "true" => Value::Bool(true),
            "impl" => Value::native_fn(fns::CreatePrototype),
            "defer" => Value::native_fn(fns::Defer),
            "if" => Value::native_fn(fns::If),
            "loop" => Value::native_fn(fns::Loop),
            "while" => Value::native_fn(fns::While),
            "map" => Value::native_fn(fns::Map),
            "filter" => Value::native_fn(fns::Filter),
            "fold" => Value::native_fn(fns::Fold),
            "push" => Value::native_fn(fns::Push),
            "merge" => Value::native_fn(fns::Merge),
            _ => return None,
        })
    }

    fn get_prototypes(&self) -> StandardPrototypes<T> {
        self.prototypes()
    }
}

/// Container for assertion functions: `assert` and `assert_eq`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Assertions;

impl Assertions {
    /// Creates an iterator over contained values and the corresponding names.
    #[allow(clippy::missing_panics_doc)] // false positive; `unwrap()` never panics
    pub fn iter<T>(self) -> impl Iterator<Item = (&'static str, Value<'static, T>)>
    where
        T: 'static + Clone + fmt::Display,
    {
        const VAR_NAMES: &[&str] = &["assert", "assert_eq", "assert_fails"];

        VAR_NAMES
            .iter()
            .map(move |&var_name| (var_name, self.get_variable(var_name).unwrap()))
    }
}

impl<'a, T> VariableMap<'a, T> for Assertions
where
    T: 'static + Clone + fmt::Display,
{
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>> {
        Some(match name {
            "assert" => Value::native_fn(fns::Assert),
            "assert_eq" => Value::native_fn(fns::AssertEq),
            "assert_fails" => Value::native_fn(fns::AssertFails::default()),
            _ => return None,
        })
    }
}

/// Container with the comparison functions: `cmp`, `min` and `max`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Comparisons;

impl<'a, T> VariableMap<'a, T> for Comparisons {
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>> {
        Some(match name {
            "LESS" => Value::opaque_ref(Ordering::Less),
            "EQUAL" => Value::opaque_ref(Ordering::Equal),
            "GREATER" => Value::opaque_ref(Ordering::Greater),
            "cmp" => Value::native_fn(fns::Compare::Raw),
            "min" => Value::native_fn(fns::Compare::Min),
            "max" => Value::native_fn(fns::Compare::Max),
            _ => return None,
        })
    }
}

impl Comparisons {
    /// Creates an iterator over contained values and the corresponding names.
    #[allow(clippy::missing_panics_doc)] // false positive; `unwrap()` never panics
    pub fn iter<T>(self) -> impl Iterator<Item = (&'static str, Value<'static, T>)> {
        const VAR_NAMES: &[&str] = &["LESS", "EQUAL", "GREATER", "cmp", "min", "max"];

        VAR_NAMES
            .iter()
            .map(move |&var_name| (var_name, self.get_variable(var_name).unwrap()))
    }
}
