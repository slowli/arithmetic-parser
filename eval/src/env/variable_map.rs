//! Standard collections of variables.

use core::{cmp::Ordering, fmt};

use crate::{fns, Value};

/// Commonly used constants and functions from the [`fns` module](fns).
///
/// # Contents
///
/// - `true` and `false` Boolean constants.
/// - Prototype-related functions: [`impl`](fns::CreatePrototype), [`proto`](fns::GetPrototype).
/// - Deferred initialization: [`defer`](fns::Defer).
/// - Control flow functions: [`if`](fns::If), [`while`](fns::While).
#[derive(Debug, Clone, Copy, Default)]
pub struct Prelude;

impl Prelude {
    /// Creates an iterator over contained values and the corresponding names.
    pub fn vars<T>() -> impl Iterator<Item = (&'static str, Value<'static, T>)>
    where
        T: 'static + Clone,
    {
        IntoIterator::into_iter([
            ("false", Value::Bool(false)),
            ("true", Value::Bool(true)),
            ("defer", Value::native_fn(fns::Defer)),
            ("if", Value::native_fn(fns::If)),
            ("while", Value::native_fn(fns::While)),
            // Array functions (other than `array` and `len`)
            ("all", Value::native_fn(fns::All)),
            ("any", Value::native_fn(fns::Any)),
            ("filter", Value::native_fn(fns::Filter)),
            ("fold", Value::native_fn(fns::Fold)),
            ("map", Value::native_fn(fns::Map)),
            ("merge", Value::native_fn(fns::Merge)),
            ("push", Value::native_fn(fns::Push)),
        ])
    }
}

/// Container for assertion functions: `assert`, `assert_eq` and `assert_fails`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Assertions;

impl Assertions {
    /// Creates an iterator over contained values and the corresponding names.
    pub fn vars<T>() -> impl Iterator<Item = (&'static str, Value<'static, T>)>
    where
        T: 'static + Clone + fmt::Display,
    {
        IntoIterator::into_iter([
            ("assert", Value::native_fn(fns::Assert)),
            ("assert_eq", Value::native_fn(fns::AssertEq)),
            (
                "assert_fails",
                Value::native_fn(fns::AssertFails::default()),
            ),
        ])
    }
}

/// Container with the comparison functions: `cmp`, `min` and `max`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Comparisons;

impl Comparisons {
    /// Creates an iterator over contained values and the corresponding names.
    pub fn vars<T>() -> impl Iterator<Item = (&'static str, Value<'static, T>)> {
        IntoIterator::into_iter([
            ("LESS", Value::opaque_ref(Ordering::Less)),
            ("EQUAL", Value::opaque_ref(Ordering::Equal)),
            ("GREATER", Value::opaque_ref(Ordering::Greater)),
            ("cmp", Value::native_fn(fns::Compare::Raw)),
            ("min", Value::native_fn(fns::Compare::Min)),
            ("max", Value::native_fn(fns::Compare::Max)),
        ])
    }
}
