//! `VariableMap` trait and implementations.

use arithmetic_parser::{Block, Grammar};

use crate::{fns, Environment, Error, ExecutableModule, ModuleId, ModuleImports, Number, Value};

/// Encapsulates read access to named variables.
pub trait VariableMap<'a, T: Grammar> {
    /// Returns value of the named variable, or `None` if it is not defined.
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>>;

    /// Creates a module based on imports solely from this map.
    ///
    /// The default implementation is reasonable for most cases.
    fn compile_module<Id: ModuleId>(
        &self,
        id: Id,
        block: &Block<'a, T>,
    ) -> Result<ExecutableModule<'a, T>, Error<'a>> {
        ExecutableModule::builder(id, block)?
            .with_imports_from(self)
            .try_build()
    }
}

impl<'a, T: Grammar> VariableMap<'a, T> for Environment<'a, T> {
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>> {
        self.get(name).cloned()
    }
}

impl<'a, T: Grammar> VariableMap<'a, T> for ModuleImports<'a, T> {
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>> {
        self.get(name).cloned()
    }
}

/// Commonly used constants and functions from the [`fns` module].
///
/// # Contents
///
/// - All functions from the `fns` module, except for [`Compare`] (since it requires
///   `PartialOrd` implementation for numbers). All functions are named in lowercase,
///   e.g., `if`, `map`.
/// - `true` and `false` Boolean constants.
///
/// [`fns` module]: fns/index.html
/// [`Compare`]: fns/struct.Compare.html
#[derive(Debug, Clone, Copy)]
pub struct Prelude;

impl Prelude {
    /// Creates an iterator over contained values and the corresponding names.
    pub fn iter<T>(self) -> impl Iterator<Item = (&'static str, Value<'static, T>)>
    where
        T: Grammar,
        T::Lit: Number,
    {
        const VAR_NAMES: &[&str] = &[
            "false", "true", "assert", "if", "loop", "while", "map", "filter", "fold", "push",
            "merge",
        ];

        VAR_NAMES
            .iter()
            .map(move |&var_name| (var_name, self.get_variable(var_name).unwrap()))
    }
}

impl<'a, T> VariableMap<'a, T> for Prelude
where
    T: Grammar,
    T::Lit: Number,
{
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>> {
        Some(match name {
            "false" => Value::Bool(false),
            "true" => Value::Bool(true),
            "assert" => Value::native_fn(fns::Assert),
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

impl Comparisons {
    fn min<T: PartialOrd>(x: T, y: T) -> T {
        if x < y {
            x
        } else {
            y
        }
    }

    fn max<T: PartialOrd>(x: T, y: T) -> T {
        if x > y {
            x
        } else {
            y
        }
    }
}

impl<'a, T> VariableMap<'a, T> for Comparisons
where
    T: Grammar,
    T::Lit: Number + PartialOrd,
{
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>> {
        Some(match name {
            "cmp" => Value::native_fn(fns::Compare),
            "min" => Value::native_fn(fns::Binary::new(Self::min::<T::Lit>)),
            "max" => Value::native_fn(fns::Binary::new(Self::max::<T::Lit>)),
            _ => return None,
        })
    }
}

impl Comparisons {
    /// Creates an iterator over contained values and the corresponding names.
    pub fn iter<T>(self) -> impl Iterator<Item = (&'static str, Value<'static, T>)>
    where
        T: Grammar,
        T::Lit: Number + PartialOrd,
    {
        const VAR_NAMES: &[&str] = &["cmp", "min", "max"];

        VAR_NAMES
            .iter()
            .map(move |&var_name| (var_name, self.get_variable(var_name).unwrap()))
    }
}
