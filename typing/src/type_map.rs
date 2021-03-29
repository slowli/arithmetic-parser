//! `TypeMap` trait and standard implementations.

use std::iter;

use crate::{arith::WithBoolean, FnType, PrimitiveType, TupleLength, ValueType};

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
/// use arithmetic_parser::grammars::{NumGrammar, Parse, Typed};
/// use arithmetic_typing::{Annotated, Prelude, TypeEnvironment, ValueType};
///
/// # fn main() -> anyhow::Result<()> {
/// type Parser = Typed<Annotated<NumGrammar<f32>>>;
/// let code = "|xs| xs.fold(0, |acc, x| if(x == 0, acc + 1, acc))";
/// let ast = Parser::parse_statements(code)?;
///
/// let mut env: TypeEnvironment = Prelude::iter().collect();
/// let count_zeros_fn = env.process_statements(&ast)?;
/// assert_eq!(count_zeros_fn.to_string(), "fn<len N>([Num; N]) -> Num");
/// # Ok(())
/// # }
/// ```
///
/// Limitations of `push` / `merge`:
///
/// ```
/// # use arithmetic_parser::grammars::{NumGrammar, Parse, Typed};
/// # use arithmetic_typing::{Annotated, Prelude, TypeEnvironment, ValueType, TypeErrorKind};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// type Parser = Typed<Annotated<NumGrammar<f32>>>;
/// let code = r#"
///     len = |xs| xs.fold(0, |acc, _| acc + 1);
///     slice = (1, 2).push(3);
///     slice.len(); // methods working on slices are applicable
///     (_, _, z) = slice; // but destructring is not
/// "#;
/// let ast = Parser::parse_statements(code)?;
///
/// let mut env: TypeEnvironment = Prelude::iter().collect();
/// let err = env.process_statements(&ast).unwrap_err();
/// assert_eq!(*err.span().fragment(), "(_, _, z) = slice");
/// # assert_matches!(err.kind(), TypeErrorKind::TupleLenMismatch { .. });
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Prelude;

impl Prelude {
    /// Gets type definition by `name`.
    pub fn get_type<Prim: WithBoolean>(name: &str) -> Option<ValueType<Prim>> {
        Some(match name {
            "false" | "true" => ValueType::BOOL,
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
    ///     "fn<T>(Bool, T, T) -> T"
    /// );
    /// ```
    pub fn if_type<Prim: WithBoolean>() -> FnType<Prim> {
        FnType::builder()
            .with_type_params(iter::once(0))
            .with_arg(ValueType::BOOL)
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
    ///     "fn<T>(T, fn(T) -> Bool, fn(T) -> T) -> T"
    /// );
    /// ```
    pub fn while_type<Prim: WithBoolean>() -> FnType<Prim> {
        let condition_fn = FnType::builder()
            .with_arg(ValueType::Param(0))
            .returning(ValueType::BOOL);
        let iter_fn = FnType::builder()
            .with_arg(ValueType::Param(0))
            .returning(ValueType::Param(0));

        FnType::builder()
            .with_type_params(iter::once(0))
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
    ///     "fn<len N; T, U>([T; N], fn(T) -> U) -> [U; N]"
    /// );
    /// ```
    pub fn map_type<Prim: PrimitiveType>() -> FnType<Prim> {
        let map_arg = FnType::builder()
            .with_arg(ValueType::Param(0))
            .returning(ValueType::Param(1));

        FnType::builder()
            .with_len_params(iter::once(0))
            .with_type_params(0..=1)
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
    ///     "fn<len N, M*; T>([T; N], fn(T) -> Bool) -> [T; M]"
    /// );
    /// ```
    pub fn filter_type<Prim: WithBoolean>() -> FnType<Prim> {
        let predicate_arg = FnType::builder()
            .with_arg(ValueType::Param(0))
            .returning(ValueType::BOOL);

        FnType::builder()
            .with_len_params(iter::once(0))
            .with_dyn_len_params(iter::once(1))
            .with_type_params(iter::once(0))
            .with_arg(ValueType::Param(0).repeat(TupleLength::Param(0)))
            .with_arg(predicate_arg)
            .returning(ValueType::Param(0).repeat(TupleLength::Param(1)))
    }

    /// Returns type of the `fold` function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arithmetic_typing::{Num, Prelude};
    /// assert_eq!(
    ///     Prelude::fold_type::<Num>().to_string(),
    ///     "fn<len N; T, U>([T; N], U, fn(U, T) -> U) -> U"
    /// );
    /// ```
    pub fn fold_type<Prim: PrimitiveType>() -> FnType<Prim> {
        // 0th type param is slice element, 1st is accumulator
        let fold_arg = FnType::builder()
            .with_arg(ValueType::Param(1))
            .with_arg(ValueType::Param(0))
            .returning(ValueType::Param(1));

        FnType::builder()
            .with_len_params(iter::once(0))
            .with_type_params(0..=1)
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
    ///     "fn<len N, M*; T>([T; N], T) -> [T; M]"
    /// );
    /// ```
    pub fn push_type<Prim: PrimitiveType>() -> FnType<Prim> {
        FnType::builder()
            .with_len_params(iter::once(0))
            .with_dyn_len_params(iter::once(1))
            .with_type_params(iter::once(0))
            .with_arg(ValueType::Param(0).repeat(TupleLength::Param(0)))
            .with_arg(ValueType::Param(0))
            .returning(ValueType::Param(0).repeat(TupleLength::Param(1)))
    }

    /// Returns type of the `merge` function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arithmetic_typing::{Num, Prelude};
    /// assert_eq!(
    ///     Prelude::merge_type::<Num>().to_string(),
    ///     "fn<len N, M, L*; T>([T; N], [T; M]) -> [T; L]"
    /// );
    /// ```
    pub fn merge_type<Prim: PrimitiveType>() -> FnType<Prim> {
        FnType::builder()
            .with_len_params(0..=1)
            .with_dyn_len_params(iter::once(2))
            .with_type_params(iter::once(0))
            .with_arg(ValueType::Param(0).repeat(TupleLength::Param(0)))
            .with_arg(ValueType::Param(0).repeat(TupleLength::Param(1)))
            .returning(ValueType::Param(0).repeat(TupleLength::Param(2)))
    }

    /// Returns an iterator over all type definitions in the `Prelude`.
    pub fn iter<Prim: WithBoolean>() -> impl Iterator<Item = (&'static str, ValueType<Prim>)> {
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
    pub fn get_type<Prim: WithBoolean>(name: &str) -> Option<ValueType<Prim>> {
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
    pub fn assert_type<Prim: WithBoolean>() -> FnType<Prim> {
        FnType::builder()
            .with_arg(ValueType::BOOL)
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
    ///     "fn<T>(T, T)"
    /// );
    /// ```
    pub fn assert_eq_type<Prim: PrimitiveType>() -> FnType<Prim> {
        FnType::builder()
            .with_type_params(iter::once(0))
            .with_arg(ValueType::Param(0))
            .with_arg(ValueType::Param(0))
            .returning(ValueType::void())
    }

    /// Returns an iterator over all type definitions in `Assertions`.
    pub fn iter<Prim: WithBoolean>() -> impl Iterator<Item = (&'static str, ValueType<Prim>)> {
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
        ("if", "fn<T>(Bool, T, T) -> T"),
        ("while", "fn<T>(T, fn(T) -> Bool, fn(T) -> T) -> T"),
        ("map", "fn<len N; T, U>([T; N], fn(T) -> U) -> [U; N]"),
        (
            "filter",
            "fn<len N, M*; T>([T; N], fn(T) -> Bool) -> [T; M]",
        ),
        ("fold", "fn<len N; T, U>([T; N], U, fn(U, T) -> U) -> U"),
        ("push", "fn<len N, M*; T>([T; N], T) -> [T; M]"),
        ("merge", "fn<len N, M, L*; T>([T; N], [T; M]) -> [T; L]"),
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
