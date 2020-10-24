//! `VariableMap` trait and implementations.

use arithmetic_parser::Grammar;

use crate::{fns, Environment, ModuleImports, Number, Value};

/// Encapsulates read access to named variables.
pub trait VariableMap<'a, T: Grammar> {
    /// Returns value of the named variable, or `None` if it is not defined.
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>>;

    /// Lists all variables in this container.
    // Boxing is required because the lifetime of the returned iterator depends on `&self`;
    // thus, an ordinary associated type won't do. (A type generic by the lifetime would
    // be necessary, but the corresponding feature, GAT, is not stable.)
    fn variables(&self) -> Box<dyn Iterator<Item = (&str, Value<'a, T>)> + '_>;
}

impl<'a, T: Grammar> VariableMap<'a, T> for Environment<'a, T> {
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>> {
        self.get_var(name).cloned()
    }

    fn variables(&self) -> Box<dyn Iterator<Item = (&str, Value<'a, T>)> + '_> {
        let iter = self
            .variables()
            .map(|(name, value)| (name, value.to_owned()));
        Box::new(iter)
    }
}

impl<'a, T: Grammar> VariableMap<'a, T> for ModuleImports<'a, T> {
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>> {
        self.get(name).cloned()
    }

    fn variables(&self) -> Box<dyn Iterator<Item = (&str, Value<'a, T>)> + '_> {
        let iter = self.iter().map(|(name, value)| (name, value.to_owned()));
        Box::new(iter)
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

    fn variables(&self) -> Box<dyn Iterator<Item = (&str, Value<'a, T>)> + '_> {
        const ALL_NAMES: &[&str] = &[
            "false", "true", "assert", "if", "loop", "while", "map", "filter", "fold", "push",
            "merge",
        ];

        let iter = ALL_NAMES
            .iter()
            .map(move |&name| (name, self.get_variable(name).unwrap()));
        Box::new(iter)
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

    fn variables(&self) -> Box<dyn Iterator<Item = (&str, Value<'a, T>)> + '_> {
        const ALL_NAMES: &[&str] = &["cmp", "min", "max"];

        let iter = ALL_NAMES
            .iter()
            .map(move |&name| (name, self.get_variable(name).unwrap()));
        Box::new(iter)
    }
}
