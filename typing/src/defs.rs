//! Type definitions for the standard types from the [`arithmetic-eval`] crate.
//!
//! [`arithmetic-eval`]: https://docs.rs/arithmetic-eval/

use std::iter;

use crate::{arith::WithBoolean, Function, Object, PrimitiveType, Type, UnknownLen};

/// Map containing type definitions for all variables from `Prelude` in the eval crate.
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
/// use arithmetic_typing::{defs::Prelude, Annotated, TypeEnvironment, Type};
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
/// # use arithmetic_typing::{defs::Prelude, error::ErrorKind, Annotated, TypeEnvironment, Type};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let code = r#"
///     len = |xs| xs.fold(0, |acc, _| acc + 1);
///     slice = (1, 2).merge((3, 4));
///     slice.len(); // methods working on slices are applicable
///     (_, _, _, z) = slice; // but destructuring is not
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
    /// Type of the `defer` function.
    Defer,
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
    /// Type of the `all` function.
    All,
    /// Type of the `any` function.
    Any,
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

            Prelude::Defer => {
                let fn_arg = Function::builder()
                    .with_arg(Type::param(0))
                    .returning(Type::param(0));
                Function::builder()
                    .with_arg(fn_arg)
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

            Prelude::All | Prelude::Any => {
                let predicate_arg = Function::builder()
                    .with_arg(Type::param(0))
                    .returning(Type::BOOL);

                Function::builder()
                    .with_arg(Type::param(0).repeat(UnknownLen::Dynamic))
                    .with_arg(predicate_arg)
                    .returning(Type::BOOL)
                    .into()
            }
        }
    }
}

impl Prelude {
    const VALUES: &'static [Self] = &[Self::True, Self::False, Self::If, Self::While, Self::Defer];

    const ARRAY_FUNCTIONS: &'static [Self] = &[
        Self::Map,
        Self::Filter,
        Self::Fold,
        Self::Push,
        Self::Merge,
        Self::All,
        Self::Any,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::True => "true",
            Self::False => "false",
            Self::If => "if",
            Self::While => "while",
            Self::Defer => "defer",
            Self::Map => "map",
            Self::Filter => "filter",
            Self::Fold => "fold",
            Self::Push => "push",
            Self::Merge => "merge",
            Self::All => "all",
            Self::Any => "any",
        }
    }

    /// Returns the type of the `array` generation function from the eval crate.
    ///
    /// The `array` function is not included into [`Self::iter()`] because in the general case
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

    fn array_namespace<Prim: WithBoolean>() -> Object<Prim> {
        Self::ARRAY_FUNCTIONS
            .iter()
            .map(|&value| (value.as_str(), Type::from(value)))
            .collect()
    }

    /// Returns an iterator over all type definitions in the `Prelude`.
    pub fn iter<Prim: WithBoolean>() -> impl Iterator<Item = (&'static str, Type<Prim>)> {
        Self::VALUES
            .iter()
            .chain(Self::ARRAY_FUNCTIONS)
            .map(|&value| (value.as_str(), value.into()))
            .chain(iter::once(("Array", Self::array_namespace().into())))
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
    /// Type of the `assert_fails` function.
    AssertFails,
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
            Assertions::AssertFails => {
                let checked_fn = Function::builder().returning(Type::param(0));
                Function::builder()
                    .with_arg(checked_fn)
                    .returning(Type::void())
                    .into()
            }
        }
    }
}

impl Assertions {
    const VALUES: &'static [Self] = &[Self::Assert, Self::AssertEq, Self::AssertFails];

    fn as_str(self) -> &'static str {
        match self {
            Self::Assert => "assert",
            Self::AssertEq => "assert_eq",
            Self::AssertFails => "assert_fails",
        }
    }

    /// Returns an iterator over all type definitions in `Assertions`.
    pub fn iter<Prim: WithBoolean>() -> impl Iterator<Item = (&'static str, Type<Prim>)> {
        Self::VALUES.iter().map(|&val| (val.as_str(), val.into()))
    }

    /// Returns the type of the `assert_close` function from the eval crate.
    ///
    /// This function is not included into [`Self::iter()`] because in the general case
    /// we don't know the type of arguments it accepts.
    pub fn assert_close<T: PrimitiveType>(value: T) -> Function<T> {
        Function::builder()
            .with_arg(Type::Prim(value.clone()))
            .with_arg(Type::Prim(value))
            .returning(Type::void())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::Num;

    use std::collections::{HashMap, HashSet};

    const EXPECTED_PRELUDE_TYPES: &[(&str, &str)] = &[
        ("false", "Bool"),
        ("true", "Bool"),
        ("if", "(Bool, 'T, 'T) -> 'T"),
        ("while", "('T, ('T) -> Bool, ('T) -> 'T) -> 'T"),
        ("defer", "(('T) -> 'T) -> 'T"),
        ("map", "(['T; N], ('T) -> 'U) -> ['U; N]"),
        ("filter", "(['T], ('T) -> Bool) -> ['T]"),
        ("fold", "(['T], 'U, ('U, 'T) -> 'U) -> 'U"),
        ("push", "(['T; N], 'T) -> ['T; N + 1]"),
        ("merge", "(['T], ['T]) -> ['T]"),
        ("all", "(['T], ('T) -> Bool) -> Bool"),
        ("any", "(['T], ('T) -> Bool) -> Bool"),
    ];

    #[test]
    fn string_presentations_of_prelude_types() {
        let expected_types: HashMap<_, _> = EXPECTED_PRELUDE_TYPES.iter().copied().collect();

        for (name, ty) in Prelude::iter::<Num>() {
            if name != "Array" {
                assert_eq!(ty.to_string(), expected_types[name]);
            }
        }

        assert_eq!(
            Prelude::iter::<Num>()
                .filter_map(|(name, _)| (name != "Array").then_some(name))
                .collect::<HashSet<_>>(),
            expected_types.keys().copied().collect::<HashSet<_>>()
        );
    }

    #[test]
    fn string_presentation_of_array_type() {
        let array_fn = Prelude::array(Num::Num);
        assert_eq!(array_fn.to_string(), "(Num, (Num) -> 'T) -> ['T]");
    }

    const EXPECTED_ASSERT_TYPES: &[(&str, &str)] = &[
        ("assert", "(Bool) -> ()"),
        ("assert_eq", "('T, 'T) -> ()"),
        ("assert_fails", "(() -> 'T) -> ()"),
    ];

    #[test]
    fn string_representation_of_assert_types() {
        let expected_types: HashMap<_, _> = EXPECTED_ASSERT_TYPES.iter().copied().collect();

        for (name, ty) in Assertions::iter::<Num>() {
            assert_eq!(ty.to_string(), expected_types[name]);
        }

        assert_eq!(
            Assertions::iter::<Num>()
                .map(|(name, _)| name)
                .collect::<HashSet<_>>(),
            expected_types.keys().copied().collect::<HashSet<_>>()
        );
    }

    #[test]
    fn string_representation_of_assert_close() {
        let assert_close_fn = Assertions::assert_close(Num::Num);
        assert_eq!(assert_close_fn.to_string(), "(Num, Num) -> ()");
    }
}
