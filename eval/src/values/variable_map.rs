//! `VariableMap` trait and implementations.

use arithmetic_parser::{grammars::Grammar, Block};

use core::{cmp::Ordering, fmt};

use crate::{fns, Environment, Error, ExecutableModule, ModuleId, ModuleImports, Value};

/// Encapsulates read access to named variables.
pub trait VariableMap<'a, T> {
    /// Returns value of the named variable, or `None` if it is not defined.
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>>;

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
        G: Grammar<Lit = T>,
        T: Clone + fmt::Debug,
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
}

impl<'a, T: Clone> VariableMap<'a, T> for ModuleImports<'a, T> {
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>> {
        self.get(name).cloned()
    }
}

/// Commonly used constants and functions from the [`fns` module](fns).
///
/// # Contents
///
/// - All functions from the `fns` module, except for [`Compare`](fns::Compare) (since it requires
///   `PartialOrd` implementation for numbers). All functions are named in lowercase,
///   e.g., `if`, `map`.
/// - `true` and `false` Boolean constants.
#[derive(Debug, Clone, Copy)]
pub struct Prelude;

impl Prelude {
    /// Creates an iterator over contained values and the corresponding names.
    pub fn iter<T: Clone>(self) -> impl Iterator<Item = (&'static str, Value<'static, T>)> {
        #[rustfmt::skip]
        const VAR_NAMES: &[&str] = &[
            "false", "true", "assert", "assert_eq", "if", "loop", "while", "map", "filter", "fold",
            "push", "merge",
        ];

        VAR_NAMES
            .iter()
            .map(move |&var_name| (var_name, self.get_variable(var_name).unwrap()))
    }
}

impl<'a, T: Clone> VariableMap<'a, T> for Prelude {
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>> {
        Some(match name {
            "false" => Value::Bool(false),
            "true" => Value::Bool(true),
            "assert" => Value::native_fn(fns::Assert),
            "assert_eq" => Value::native_fn(fns::AssertEq),
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
}

/// Container with the comparison functions: `cmp`, `min` and `max`.
#[derive(Debug, Clone, Copy)]
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
    pub fn iter<T>(self) -> impl Iterator<Item = (&'static str, Value<'static, T>)> {
        const VAR_NAMES: &[&str] = &["LESS", "EQUAL", "GREATER", "cmp", "min", "max"];

        VAR_NAMES
            .iter()
            .map(move |&var_name| (var_name, self.get_variable(var_name).unwrap()))
    }
}
