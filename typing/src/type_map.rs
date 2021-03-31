//! `TypeMap` trait and standard implementations.

use std::iter;

use crate::{arith::WithBoolean, FnType, TupleLen, ValueType};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Prelude {
    /// `false` type (Boolean).
    False,
    /// `true` type (Boolean).
    True,
    /// Type of the `if` function.
    If,
    /// Type of the `while` function.
    While,
    /// Type of the `map` function.
    Map,
    /// Type of the `filter` function.
    Filter,
    /// Type of the `fold` function.
    Fold,
    /// Type of the `push` function.
    Push,
    /// Type of the `merge` function.
    Merge,
}

impl<Prim: WithBoolean> From<Prelude> for ValueType<Prim> {
    fn from(value: Prelude) -> Self {
        match value {
            Prelude::True | Prelude::False => ValueType::BOOL,

            Prelude::If => FnType::builder()
                .with_type_params(iter::once(0))
                .with_arg(ValueType::BOOL)
                .with_arg(ValueType::Param(0))
                .with_arg(ValueType::Param(0))
                .returning(ValueType::Param(0))
                .into(),

            Prelude::While => {
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
                    .into()
            }

            Prelude::Map => {
                let map_arg = FnType::builder()
                    .with_arg(ValueType::Param(0))
                    .returning(ValueType::Param(1));

                FnType::builder()
                    .with_len_params(iter::once(0))
                    .with_type_params(0..=1)
                    .with_arg(ValueType::Param(0).repeat(TupleLen::Param(0)))
                    .with_arg(map_arg)
                    .returning(ValueType::Param(1).repeat(TupleLen::Param(0)))
                    .into()
            }

            Prelude::Filter => {
                let predicate_arg = FnType::builder()
                    .with_arg(ValueType::Param(0))
                    .returning(ValueType::BOOL);

                FnType::builder()
                    .with_len_params(iter::once(0))
                    .with_dyn_len_params(iter::once(1))
                    .with_type_params(iter::once(0))
                    .with_arg(ValueType::Param(0).repeat(TupleLen::Param(0)))
                    .with_arg(predicate_arg)
                    .returning(ValueType::Param(0).repeat(TupleLen::Param(1)))
                    .into()
            }

            Prelude::Fold => {
                // 0th type param is slice element, 1st is accumulator
                let fold_arg = FnType::builder()
                    .with_arg(ValueType::Param(1))
                    .with_arg(ValueType::Param(0))
                    .returning(ValueType::Param(1));

                FnType::builder()
                    .with_len_params(iter::once(0))
                    .with_type_params(0..=1)
                    .with_arg(ValueType::Param(0).repeat(TupleLen::Param(0)))
                    .with_arg(ValueType::Param(1))
                    .with_arg(fold_arg)
                    .returning(ValueType::Param(1))
                    .into()
            }

            Prelude::Push => FnType::builder()
                .with_len_params(iter::once(0))
                .with_dyn_len_params(iter::once(1))
                .with_type_params(iter::once(0))
                .with_arg(ValueType::Param(0).repeat(TupleLen::Param(0)))
                .with_arg(ValueType::Param(0))
                .returning(ValueType::Param(0).repeat(TupleLen::Param(1)))
                .into(),

            Prelude::Merge => FnType::builder()
                .with_len_params(0..=1)
                .with_dyn_len_params(iter::once(2))
                .with_type_params(iter::once(0))
                .with_arg(ValueType::Param(0).repeat(TupleLen::Param(0)))
                .with_arg(ValueType::Param(0).repeat(TupleLen::Param(1)))
                .returning(ValueType::Param(0).repeat(TupleLen::Param(2)))
                .into(),
        }
    }
}

impl Prelude {
    const VALUES: &'static [Self] = &[
        Self::True,
        Self::False,
        Self::If,
        Self::While,
        Self::Map,
        Self::Filter,
        Self::Fold,
        Self::Push,
        Self::Merge,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::True => "true",
            Self::False => "false",
            Self::If => "if",
            Self::While => "while",
            Self::Map => "map",
            Self::Filter => "filter",
            Self::Fold => "fold",
            Self::Push => "push",
            Self::Merge => "merge",
        }
    }

    /// Returns an iterator over all type definitions in the `Prelude`.
    pub fn iter<Prim: WithBoolean>() -> impl Iterator<Item = (&'static str, ValueType<Prim>)> {
        Self::VALUES
            .iter()
            .map(|&value| (value.as_str(), value.into()))
    }
}

/// Definitions for `assert` and `assert_eq` functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Assertions {
    /// Type of the `assert` function.
    Assert,
    /// Type of the `assert_eq` function.
    AssertEq,
}

impl<Prim: WithBoolean> From<Assertions> for ValueType<Prim> {
    fn from(value: Assertions) -> Self {
        match value {
            Assertions::Assert => FnType::builder()
                .with_arg(ValueType::BOOL)
                .returning(ValueType::void())
                .into(),
            Assertions::AssertEq => FnType::builder()
                .with_type_params(iter::once(0))
                .with_arg(ValueType::Param(0))
                .with_arg(ValueType::Param(0))
                .returning(ValueType::void())
                .into(),
        }
    }
}

impl Assertions {
    const VALUES: &'static [Self] = &[Self::Assert, Self::AssertEq];

    fn as_str(self) -> &'static str {
        match self {
            Self::Assert => "assert",
            Self::AssertEq => "assert_eq",
        }
    }

    /// Returns an iterator over all type definitions in `Assertions`.
    pub fn iter<Prim: WithBoolean>() -> impl Iterator<Item = (&'static str, ValueType<Prim>)> {
        Self::VALUES.iter().map(|&val| (val.as_str(), val.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Num;

    use std::collections::{HashMap, HashSet};

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
        let expected_types: HashMap<_, _> = EXPECTED_PRELUDE_TYPES.iter().copied().collect();

        for (name, ty) in Prelude::iter::<Num>() {
            assert_eq!(ty.to_string(), expected_types[name]);
        }

        assert_eq!(
            Prelude::iter::<Num>()
                .map(|(name, _)| name)
                .collect::<HashSet<_>>(),
            expected_types.keys().copied().collect::<HashSet<_>>()
        );
    }
}
