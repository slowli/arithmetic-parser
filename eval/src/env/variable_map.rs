//! Standard collections of variables.

#![allow(clippy::unused_self)] // FIXME: remove

use core::{cmp::Ordering, fmt};

use crate::{alloc::vec, fns, Object, StandardPrototypes, Value};

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
    pub fn iter<T>(self) -> impl Iterator<Item = (&'static str, Value<'static, T>)>
    where
        T: 'static + Clone,
    {
        let variables = vec![
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
        ];
        variables.into_iter()
    }

    /// Returns standard prototypes corresponding to the contained functions.
    ///
    /// Currently, this only sets a prototype for tuples / arrays containing
    /// `map`, `filter`, `fold`, `push` and `merge` functions.
    pub fn prototypes<T>(self) -> StandardPrototypes<T>
    where
        T: 'static + Clone,
    {
        let mut array_proto = Object::default();
        array_proto.insert("map", Value::native_fn(fns::Map));
        array_proto.insert("filter", Value::native_fn(fns::Filter));
        array_proto.insert("fold", Value::native_fn(fns::Fold));
        array_proto.insert("push", Value::native_fn(fns::Push));
        array_proto.insert("merge", Value::native_fn(fns::Merge));

        StandardPrototypes::new().with_array_proto(array_proto)
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
        let variables = vec![
            ("assert", Value::native_fn(fns::Assert)),
            ("assert_eq", Value::native_fn(fns::AssertEq)),
            (
                "assert_fails",
                Value::native_fn(fns::AssertFails::default()),
            ),
        ];
        variables.into_iter()
    }
}

/// Container with the comparison functions: `cmp`, `min` and `max`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Comparisons;

impl Comparisons {
    /// Creates an iterator over contained values and the corresponding names.
    pub fn iter<T>(self) -> impl Iterator<Item = (&'static str, Value<'static, T>)> {
        let variables = vec![
            ("LESS", Value::opaque_ref(Ordering::Less)),
            ("EQUAL", Value::opaque_ref(Ordering::Equal)),
            ("GREATER", Value::opaque_ref(Ordering::Greater)),
            ("cmp", Value::native_fn(fns::Compare::Raw)),
            ("min", Value::native_fn(fns::Compare::Min)),
            ("max", Value::native_fn(fns::Compare::Max)),
        ];
        variables.into_iter()
    }
}
