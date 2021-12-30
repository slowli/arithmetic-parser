//! Standard collections of variables.

use core::{array, cmp::Ordering, fmt};

use crate::{fns, PrototypeField, Value};

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

impl Prelude {
    /// Creates an iterator over contained values and the corresponding names.
    pub fn vars<T>() -> impl Iterator<Item = (&'static str, Value<'static, T>)>
    where
        T: 'static + Clone,
    {
        // TODO: only expose array functions within `Array` prototype?
        array::IntoIter::new([
            ("false", Value::Bool(false)),
            ("true", Value::Bool(true)),
            ("impl", Value::native_fn(fns::CreatePrototype)),
            ("defer", Value::native_fn(fns::Defer)),
            ("if", Value::native_fn(fns::If)),
            ("loop", Value::native_fn(fns::Loop)),
            ("while", Value::native_fn(fns::While)),
            ("map", Value::native_fn(fns::Map)),
            ("filter", Value::native_fn(fns::Filter)),
            ("fold", Value::native_fn(fns::Fold)),
            ("push", Value::native_fn(fns::Push)),
            ("merge", Value::native_fn(fns::Merge)),
            ("all", Value::native_fn(fns::All)),
            ("any", Value::native_fn(fns::Any)),
        ])
    }

    /// Returns standard prototypes corresponding to the contained functions.
    ///
    /// Currently, this only sets a prototype for tuples / arrays containing
    /// `map`, `filter`, `fold`, `all`, `any`, `push` and `merge` functions.
    pub fn prototypes<T>() -> impl Iterator<Item = (PrototypeField, Value<'static, T>)>
    where
        T: 'static + Clone,
    {
        let array_proto = PrototypeField::array;
        array::IntoIter::new([
            (array_proto("map"), Value::native_fn(fns::Map)),
            (array_proto("filter"), Value::native_fn(fns::Filter)),
            (array_proto("fold"), Value::native_fn(fns::Fold)),
            (array_proto("push"), Value::native_fn(fns::Push)),
            (array_proto("merge"), Value::native_fn(fns::Merge)),
            (array_proto("all"), Value::native_fn(fns::All)),
            (array_proto("any"), Value::native_fn(fns::Any)),
        ])
    }
}

/// Container for assertion functions: `assert` and `assert_eq`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Assertions;

impl Assertions {
    /// Creates an iterator over contained values and the corresponding names.
    pub fn vars<T>() -> impl Iterator<Item = (&'static str, Value<'static, T>)>
    where
        T: 'static + Clone + fmt::Display,
    {
        array::IntoIter::new([
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
        array::IntoIter::new([
            ("LESS", Value::opaque_ref(Ordering::Less)),
            ("EQUAL", Value::opaque_ref(Ordering::Equal)),
            ("GREATER", Value::opaque_ref(Ordering::Greater)),
            ("cmp", Value::native_fn(fns::Compare::Raw)),
            ("min", Value::native_fn(fns::Compare::Min)),
            ("max", Value::native_fn(fns::Compare::Max)),
        ])
    }
}
