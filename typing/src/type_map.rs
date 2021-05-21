//! `TypeMap` trait and standard implementations.

use crate::{arith::WithBoolean, Function, PrimitiveType, Type, UnknownLen};

/// Map containing type definitions for all variables from `Prelude` in the eval crate,
/// except for `loop` function.
///
/// # Contents
///
/// - `true` and `false` Boolean constants
/// - `if`, `while`, `map`, `filter`, `fold`, `push` and `merge` functions
///
/// The `merge` function has somewhat imprecise typing; its return value is
/// a dynamically-sized slice.
///
/// The `array` function is available separately via [`Self::array()`].
///
/// # Examples
///
/// Function counting number of zeros in a slice:
///
/// ```
/// use arithmetic_parser::grammars::{F32Grammar, Parse};
/// use arithmetic_typing::{Annotated, Prelude, TypeEnvironment, Type};
///
/// # fn main() -> anyhow::Result<()> {
/// let code = "|xs| xs.fold(0, |acc, x| if(x == 0, acc + 1, acc))";
/// let ast = Annotated::<F32Grammar>::parse_statements(code)?;
///
/// let mut env: TypeEnvironment = Prelude::iter().collect();
/// let count_zeros_fn = env.process_statements(&ast)?;
/// assert_eq!(count_zeros_fn.to_string(), "([Num; N]) -> Num");
/// # Ok(())
/// # }
/// ```
///
/// Limitations of `merge`:
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse};
/// # use arithmetic_typing::{ErrorKind, Annotated, Prelude, TypeEnvironment, Type};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let code = r#"
///     len = |xs| xs.fold(0, |acc, _| acc + 1);
///     slice = (1, 2).merge((3, 4));
///     slice.len(); // methods working on slices are applicable
///     (_, _, _, z) = slice; // but destructring is not
/// "#;
/// let ast = Annotated::<F32Grammar>::parse_statements(code)?;
///
/// let mut env: TypeEnvironment = Prelude::iter().collect();
/// let errors = env.process_statements(&ast).unwrap_err();
/// assert_eq!(errors.len(), 1);
/// let err = errors.iter().next().unwrap();
/// assert_eq!(*err.main_span().fragment(), "(_, _, _, z)");
/// # assert_matches!(err.kind(), ErrorKind::TupleLenMismatch { .. });
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

impl<Prim: WithBoolean> From<Prelude> for Type<Prim> {
    fn from(value: Prelude) -> Self {
        match value {
            Prelude::True | Prelude::False => Type::BOOL,

            Prelude::If => Function::builder()
                .with_arg(Type::BOOL)
                .with_arg(Type::param(0))
                .with_arg(Type::param(0))
                .returning(Type::param(0))
                .into(),

            Prelude::While => {
                let condition_fn = Function::builder()
                    .with_arg(Type::param(0))
                    .returning(Type::BOOL);
                let iter_fn = Function::builder()
                    .with_arg(Type::param(0))
                    .returning(Type::param(0));

                Function::builder()
                    .with_arg(Type::param(0)) // state
                    .with_arg(condition_fn)
                    .with_arg(iter_fn)
                    .returning(Type::param(0))
                    .into()
            }

            Prelude::Map => {
                let map_arg = Function::builder()
                    .with_arg(Type::param(0))
                    .returning(Type::param(1));

                Function::builder()
                    .with_arg(Type::param(0).repeat(UnknownLen::param(0)))
                    .with_arg(map_arg)
                    .returning(Type::param(1).repeat(UnknownLen::param(0)))
                    .into()
            }

            Prelude::Filter => {
                let predicate_arg = Function::builder()
                    .with_arg(Type::param(0))
                    .returning(Type::BOOL);

                Function::builder()
                    .with_arg(Type::param(0).repeat(UnknownLen::Dynamic))
                    .with_arg(predicate_arg)
                    .returning(Type::param(0).repeat(UnknownLen::Dynamic))
                    .into()
            }

            Prelude::Fold => {
                // 0th type param is slice element, 1st is accumulator
                let fold_arg = Function::builder()
                    .with_arg(Type::param(1))
                    .with_arg(Type::param(0))
                    .returning(Type::param(1));

                Function::builder()
                    .with_arg(Type::param(0).repeat(UnknownLen::Dynamic))
                    .with_arg(Type::param(1))
                    .with_arg(fold_arg)
                    .returning(Type::param(1))
                    .into()
            }

            Prelude::Push => Function::builder()
                .with_arg(Type::param(0).repeat(UnknownLen::param(0)))
                .with_arg(Type::param(0))
                .returning(Type::param(0).repeat(UnknownLen::param(0) + 1))
                .into(),

            Prelude::Merge => Function::builder()
                .with_arg(Type::param(0).repeat(UnknownLen::Dynamic))
                .with_arg(Type::param(0).repeat(UnknownLen::Dynamic))
                .returning(Type::param(0).repeat(UnknownLen::Dynamic))
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

    /// Returns the type of the `array` generation function from the eval crate.
    ///
    /// The `array` function is **not** included into [`Self::iter()`] because in the general case
    /// we don't know the type of indexes.
    pub fn array<T: PrimitiveType>(index_type: T) -> Function<T> {
        Function::builder()
            .with_arg(Type::Prim(index_type.clone()))
            .with_arg(
                Function::builder()
                    .with_arg(Type::Prim(index_type))
                    .returning(Type::param(0)),
            )
            .returning(Type::param(0).repeat(UnknownLen::Dynamic))
    }

    /// Returns an iterator over all type definitions in the `Prelude`.
    pub fn iter<Prim: WithBoolean>() -> impl Iterator<Item = (&'static str, Type<Prim>)> {
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

impl<Prim: WithBoolean> From<Assertions> for Type<Prim> {
    fn from(value: Assertions) -> Self {
        match value {
            Assertions::Assert => Function::builder()
                .with_arg(Type::BOOL)
                .returning(Type::void())
                .into(),
            Assertions::AssertEq => Function::builder()
                .with_arg(Type::param(0))
                .with_arg(Type::param(0))
                .returning(Type::void())
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
    pub fn iter<Prim: WithBoolean>() -> impl Iterator<Item = (&'static str, Type<Prim>)> {
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
        ("if", "(Bool, 'T, 'T) -> 'T"),
        ("while", "('T, ('T) -> Bool, ('T) -> 'T) -> 'T"),
        ("map", "(['T; N], ('T) -> 'U) -> ['U; N]"),
        ("filter", "(['T], ('T) -> Bool) -> ['T]"),
        ("fold", "(['T], 'U, ('U, 'T) -> 'U) -> 'U"),
        ("push", "(['T; N], 'T) -> ['T; N + 1]"),
        ("merge", "(['T], ['T]) -> ['T]"),
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

    #[test]
    fn string_presentation_of_array_type() {
        let array_fn = Prelude::array(Num::Num);
        assert_eq!(array_fn.to_string(), "(Num, (Num) -> 'T) -> ['T]");
    }
}
