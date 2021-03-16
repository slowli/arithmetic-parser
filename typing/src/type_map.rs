//! `TypeMap` trait and standard implementations.

use std::iter;

use crate::{FnType, LiteralType, TupleLength, ValueType};

/// Map containing type definitions for all variables from `Prelude` in the eval crate,
/// except for `loop` function.
///
/// # Contents
///
/// - `true` and `false` Boolean constants
/// - `if`, `while`, `map`, `filter`, `fold`, `push` and `merge` functions
///
/// `push` and `merge` functions have somewhat imprecise typing; their return values
/// are dynamically-sized slices.
///
/// # Examples
///
/// Function counting number of zeros in a slice:
///
/// ```
/// use arithmetic_parser::grammars::{Parse, Typed};
/// use arithmetic_typing::{NumGrammar, Prelude, TypeEnvironment, ValueType};
///
/// # fn main() -> anyhow::Result<()> {
/// let code = "|xs| xs.fold(0, |acc, x| if(x == 0, acc + 1, acc))";
/// let ast = Typed::<NumGrammar<f32>>::parse_statements(code)?;
///
/// let mut env: TypeEnvironment = Prelude::iter().collect();
/// let count_zeros_fn = env.process_statements(&ast)?;
/// assert_eq!(count_zeros_fn.to_string(), "fn<const N>([Num; N]) -> Num");
/// # Ok(())
/// # }
/// ```
///
/// Function that reverses a slice:
///
/// ```
/// # use arithmetic_parser::grammars::{Parse, Typed};
/// # use arithmetic_typing::{NumGrammar, Prelude, TypeEnvironment, ValueType};
/// # fn main() -> anyhow::Result<()> {
/// let code = r#"
///     empty: [Num] = ();
///     // ^ necessary to infer accumulator type as [Num], not as `()`
///     |xs| xs.fold(empty, |acc, x| (x,).merge(acc))
/// "#;
/// let ast = Typed::<NumGrammar<f32>>::parse_statements(code)?;
///
/// let mut env: TypeEnvironment = Prelude::iter().collect();
/// let reverse_fn = env.process_statements(&ast)?;
/// assert_eq!(reverse_fn.to_string(), "fn<const N>([Num; N]) -> [Num]");
/// # Ok(())
/// # }
/// ```
///
/// Limitations of `push` / `merge`:
///
/// ```
/// # use arithmetic_parser::grammars::{Parse, Typed};
/// # use arithmetic_typing::{NumGrammar, Prelude, TypeEnvironment, ValueType, TypeErrorKind};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let code = r#"
///     len = |xs| xs.fold(0, |acc, _| acc + 1);
///     slice = (1, 2).push(3);
///     slice.len(); // methods working on slices are applicable
///     (_, _, z) = slice; // but destructring is not
/// "#;
/// let ast = Typed::<NumGrammar<f32>>::parse_statements(code)?;
///
/// let mut env: TypeEnvironment = Prelude::iter().collect();
/// let err = env.process_statements(&ast).unwrap_err();
/// assert_eq!(*err.span().fragment(), "(_, _, z) = slice");
/// # assert_matches!(err.kind(), TypeErrorKind::IncompatibleLengths(_, _));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Prelude;

impl Prelude {
    /// Gets type definition by `name`.
    pub fn get_type<Lit: LiteralType>(name: &str) -> Option<ValueType<Lit>> {
        Some(match name {
            "false" | "true" => ValueType::Bool,
            "if" => Self::if_type().into(),
            "while" => Self::while_type().into(),
            "map" => Self::map_type().into(),
            "filter" => Self::filter_type().into(),
            "fold" => Self::fold_type().into(),
            "push" => Self::push_type().into(),
            "merge" => Self::merge_type().into(),
            _ => return None,
        })
    }

    /// Returns type of the `if` function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arithmetic_typing::{Num, Prelude};
    /// assert_eq!(
    ///     Prelude::if_type::<Num>().to_string(),
    ///     "fn<T: ?Lin>(Bool, T, T) -> T"
    /// );
    /// ```
    pub fn if_type<Lit: LiteralType>() -> FnType<Lit> {
        FnType::builder()
            .with_type_params(iter::once(0), false)
            .with_arg(ValueType::Bool)
            .with_arg(ValueType::Param(0))
            .with_arg(ValueType::Param(0))
            .returning(ValueType::Param(0))
    }

    /// Returns type of the `while` function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arithmetic_typing::{Num, Prelude};
    /// assert_eq!(
    ///     Prelude::while_type::<Num>().to_string(),
    ///     "fn<T: ?Lin>(T, fn(T) -> Bool, fn(T) -> T) -> T"
    /// );
    /// ```
    pub fn while_type<Lit: LiteralType>() -> FnType<Lit> {
        let condition_fn = FnType::builder()
            .with_arg(ValueType::Param(0))
            .returning(ValueType::Bool);
        let iter_fn = FnType::builder()
            .with_arg(ValueType::Param(0))
            .returning(ValueType::Param(0));

        FnType::builder()
            .with_type_params(iter::once(0), false)
            .with_arg(ValueType::Param(0)) // state
            .with_arg(condition_fn)
            .with_arg(iter_fn)
            .returning(ValueType::Param(0))
    }

    /// Returns type of the `map` function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arithmetic_typing::{Num, Prelude};
    /// assert_eq!(
    ///     Prelude::map_type::<Num>().to_string(),
    ///     "fn<const N; T: ?Lin, U: ?Lin>([T; N], fn(T) -> U) -> [U; N]"
    /// );
    /// ```
    pub fn map_type<Lit: LiteralType>() -> FnType<Lit> {
        let map_arg = FnType::builder()
            .with_arg(ValueType::Param(0))
            .returning(ValueType::Param(1));

        FnType::builder()
            .with_const_params(iter::once(0))
            .with_type_params(0..=1, false)
            .with_arg(ValueType::Param(0).repeat(TupleLength::Param(0)))
            .with_arg(map_arg)
            .returning(ValueType::Param(1).repeat(TupleLength::Param(0)))
    }

    /// Returns type of the `filter` function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arithmetic_typing::{Num, Prelude};
    /// assert_eq!(
    ///     Prelude::filter_type::<Num>().to_string(),
    ///     "fn<const N; T: ?Lin>([T; N], fn(T) -> Bool) -> [T]"
    /// );
    /// ```
    pub fn filter_type<Lit: LiteralType>() -> FnType<Lit> {
        let predicate_arg = FnType::builder()
            .with_arg(ValueType::Param(0))
            .returning(ValueType::Bool);

        FnType::builder()
            .with_const_params(iter::once(0))
            .with_type_params(iter::once(0), false)
            .with_arg(ValueType::Param(0).repeat(TupleLength::Param(0)))
            .with_arg(predicate_arg)
            .returning(ValueType::Param(0).repeat(TupleLength::Dynamic))
    }

    /// Returns type of the `fold` function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arithmetic_typing::{Num, Prelude};
    /// assert_eq!(
    ///     Prelude::fold_type::<Num>().to_string(),
    ///     "fn<const N; T: ?Lin, U: ?Lin>([T; N], U, fn(U, T) -> U) -> U"
    /// );
    /// ```
    pub fn fold_type<Lit: LiteralType>() -> FnType<Lit> {
        // 0th type param is slice element, 1st is accumulator
        let fold_arg = FnType::builder()
            .with_arg(ValueType::Param(1))
            .with_arg(ValueType::Param(0))
            .returning(ValueType::Param(1));

        FnType::builder()
            .with_const_params(iter::once(0))
            .with_type_params(0..=1, false)
            .with_arg(ValueType::Param(0).repeat(TupleLength::Param(0)))
            .with_arg(ValueType::Param(1))
            .with_arg(fold_arg)
            .returning(ValueType::Param(1))
    }

    /// Returns type of the `push` function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arithmetic_typing::{Num, Prelude};
    /// assert_eq!(
    ///     Prelude::push_type::<Num>().to_string(),
    ///     "fn<const N; T: ?Lin>([T; N], T) -> [T]"
    /// );
    /// ```
    pub fn push_type<Lit: LiteralType>() -> FnType<Lit> {
        FnType::builder()
            .with_const_params(iter::once(0))
            .with_type_params(iter::once(0), false)
            .with_arg(ValueType::Param(0).repeat(TupleLength::Param(0)))
            .with_arg(ValueType::Param(0))
            .returning(ValueType::Param(0).repeat(TupleLength::Dynamic))
    }

    /// Returns type of the `merge` function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arithmetic_typing::{Num, Prelude};
    /// assert_eq!(
    ///     Prelude::merge_type::<Num>().to_string(),
    ///     "fn<const N, M; T: ?Lin>([T; N], [T; M]) -> [T]"
    /// );
    /// ```
    pub fn merge_type<Lit: LiteralType>() -> FnType<Lit> {
        FnType::builder()
            .with_const_params(0..=1)
            .with_type_params(iter::once(0), false)
            .with_arg(ValueType::Param(0).repeat(TupleLength::Param(0)))
            .with_arg(ValueType::Param(0).repeat(TupleLength::Param(1)))
            .returning(ValueType::Param(0).repeat(TupleLength::Dynamic))
    }

    /// Returns an iterator over all type definitions in the `Prelude`.
    pub fn iter<Lit: LiteralType>() -> impl Iterator<Item = (&'static str, ValueType<Lit>)> {
        const VAR_NAMES: &[&str] = &[
            "false", "true", "if", "while", "map", "filter", "fold", "push", "merge",
        ];

        VAR_NAMES
            .iter()
            .map(move |&var_name| (var_name, Self::get_type(var_name).unwrap()))
    }
}

/// Definitions for `assert` and `assert_eq` functions.
#[derive(Debug, Clone, Copy, Default)]
pub struct Assertions;

impl Assertions {
    /// Gets type definition by `name`.
    pub fn get_type<Lit: LiteralType>(name: &str) -> Option<ValueType<Lit>> {
        Some(match name {
            "assert" => Self::assert_type().into(),
            "assert_eq" => Self::assert_eq_type().into(),
            _ => return None,
        })
    }

    /// Returns type of the `assert` function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arithmetic_typing::{Assertions, Num};
    /// assert_eq!(
    ///     Assertions::assert_type::<Num>().to_string(),
    ///     "fn(Bool)"
    /// );
    /// ```
    pub fn assert_type<Lit: LiteralType>() -> FnType<Lit> {
        FnType::builder()
            .with_arg(ValueType::Bool)
            .returning(ValueType::void())
    }

    /// Returns type of the `assert_eq` function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arithmetic_typing::{Assertions, Num};
    /// assert_eq!(
    ///     Assertions::assert_eq_type::<Num>().to_string(),
    ///     "fn<T: ?Lin>(T, T)"
    /// );
    /// ```
    pub fn assert_eq_type<Lit: LiteralType>() -> FnType<Lit> {
        FnType::builder()
            .with_type_params(iter::once(0), false)
            .with_arg(ValueType::Param(0))
            .with_arg(ValueType::Param(0))
            .returning(ValueType::void())
    }

    /// Returns an iterator over all type definitions in `Assertions`.
    pub fn iter<Lit: LiteralType>() -> impl Iterator<Item = (&'static str, ValueType<Lit>)> {
        const VAR_NAMES: &[&str] = &["assert", "assert_eq"];

        VAR_NAMES
            .iter()
            .map(move |&var_name| (var_name, Self::get_type(var_name).unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Num;

    use std::collections::HashSet;

    const EXPECTED_PRELUDE_TYPES: &[(&str, &str)] = &[
        ("false", "Bool"),
        ("true", "Bool"),
        ("if", "fn<T: ?Lin>(Bool, T, T) -> T"),
        ("while", "fn<T: ?Lin>(T, fn(T) -> Bool, fn(T) -> T) -> T"),
        (
            "map",
            "fn<const N; T: ?Lin, U: ?Lin>([T; N], fn(T) -> U) -> [U; N]",
        ),
        (
            "filter",
            "fn<const N; T: ?Lin>([T; N], fn(T) -> Bool) -> [T]",
        ),
        (
            "fold",
            "fn<const N; T: ?Lin, U: ?Lin>([T; N], U, fn(U, T) -> U) -> U",
        ),
        ("push", "fn<const N; T: ?Lin>([T; N], T) -> [T]"),
        ("merge", "fn<const N, M; T: ?Lin>([T; N], [T; M]) -> [T]"),
    ];

    #[test]
    fn string_presentations_of_prelude_types() {
        for &(name, str_presentation) in EXPECTED_PRELUDE_TYPES {
            assert_eq!(
                Prelude::get_type::<Num>(name).unwrap().to_string(),
                str_presentation
            );
        }
        let expected_names: HashSet<_> = EXPECTED_PRELUDE_TYPES
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(
            Prelude::iter::<Num>()
                .map(|(name, _)| name)
                .collect::<HashSet<_>>(),
            expected_names
        );
    }
}
