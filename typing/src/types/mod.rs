//! Base types, such as `ValueType` and `FnType`.

use std::{borrow::Cow, fmt};

use crate::{LiteralType, Num};

mod fn_type;

pub(crate) use self::fn_type::{ConstParamDescription, TypeParamDescription};
pub use self::fn_type::{FnArgs, FnType, FnTypeBuilder};

/// Length of a tuple.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TupleLength {
    /// Wildcard length.
    Some {
        /// Is this length dynamic (can vary at runtime)?
        is_dynamic: bool,
    },
    /// Exact known length.
    Exact(usize),
    /// Length parameter in a function definition.
    Param(usize),

    /// Length variable. In contrast to `Param`s, `Var`s are used exclusively during
    /// inference and cannot occur in standalone function signatures.
    Var(usize),
}

impl fmt::Display for TupleLength {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Some { is_dynamic: false } => formatter.write_str("_"),
            Self::Some { is_dynamic: true } => formatter.write_str("*"),
            Self::Exact(len) => fmt::Display::fmt(len, formatter),
            Self::Var(idx) | Self::Param(idx) => {
                formatter.write_str(Self::const_param(*idx).as_ref())
            }
        }
    }
}

impl TupleLength {
    fn const_param(index: usize) -> Cow<'static, str> {
        const PARAM_NAMES: &str = "NMLKJI";
        PARAM_NAMES.get(index..=index).map_or_else(
            || Cow::from(format!("N{}", index - PARAM_NAMES.len())),
            Cow::from,
        )
    }
}

/// Possible value type.
#[derive(Debug, Clone)]
pub enum ValueType<Lit: LiteralType = Num> {
    /// Any type.
    // TODO: rename to `Some`
    Any,
    /// Boolean.
    // TODO: consider uniting literals and `Bool` as primitive types
    Bool,
    /// Literal.
    Lit(Lit),
    /// Function.
    Function(Box<FnType<Lit>>),
    /// Tuple.
    // TODO: support start / middle / end structuring
    Tuple(Vec<ValueType<Lit>>),
    /// Slice.
    Slice {
        /// Type of slice elements.
        element: Box<ValueType<Lit>>,
        /// Slice length.
        length: TupleLength,
    },
    /// Type parameter in a function definition.
    Param(usize),

    /// Type variable. In contrast to `Param`s, `Var`s are used exclusively during
    /// inference and cannot occur in standalone function signatures.
    Var(usize),
}

impl<Lit: LiteralType> PartialEq for ValueType<Lit> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Any, _) | (_, Self::Any) | (Self::Bool, Self::Bool) => true,

            (Self::Lit(x), Self::Lit(y)) => x == y,

            (Self::Var(x), Self::Var(y)) | (Self::Param(x), Self::Param(y)) => x == y,

            (Self::Tuple(xs), Self::Tuple(ys)) => xs == ys,

            (
                Self::Slice { element, length },
                Self::Slice {
                    element: other_element,
                    length: other_length,
                },
            ) => length == other_length && element == other_element,

            (Self::Tuple(xs), Self::Slice { element, length })
            | (Self::Slice { element, length }, Self::Tuple(xs)) => {
                *length == TupleLength::Exact(xs.len()) && xs.iter().all(|x| x == element.as_ref())
            }

            // TODO: function equality?
            _ => false,
        }
    }
}

impl<Lit: LiteralType> fmt::Display for ValueType<Lit> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Any => formatter.write_str("_"),
            Self::Var(idx) | Self::Param(idx) => formatter.write_str(type_param(*idx).as_ref()),

            Self::Bool => formatter.write_str("Bool"),
            Self::Lit(num) => fmt::Display::fmt(num, formatter),
            Self::Function(fn_type) => fmt::Display::fmt(fn_type, formatter),

            Self::Tuple(fragments) => {
                formatter.write_str("(")?;
                for (i, frag) in fragments.iter().enumerate() {
                    fmt::Display::fmt(frag, formatter)?;
                    if i + 1 < fragments.len() {
                        formatter.write_str(", ")?;
                    }
                }
                formatter.write_str(")")
            }

            Self::Slice {
                element,
                length: TupleLength::Exact(len),
            } => {
                // Format slice as a tuple since its size is statically known.
                formatter.write_str("(")?;
                for i in 0..*len {
                    fmt::Display::fmt(element, formatter)?;
                    if i + 1 < *len {
                        formatter.write_str(", ")?;
                    }
                }
                formatter.write_str(")")
            }
            Self::Slice { element, length } => {
                write!(formatter, "[{}; {}]", element, length)
            }
        }
    }
}

impl<Lit: LiteralType> From<FnType<Lit>> for ValueType<Lit> {
    fn from(fn_type: FnType<Lit>) -> Self {
        Self::Function(Box::new(fn_type))
    }
}

macro_rules! impl_from_tuple_for_value_type {
    ($($var:tt : $ty:ident),*) => {
        impl<Lit, $($ty : Into<ValueType<Lit>>,)*> From<($($ty,)*)> for ValueType<Lit>
        where
            Lit: LiteralType,
        {
            #[allow(unused_variables)] // `tuple` is unused for empty tuple
            fn from(tuple: ($($ty,)*)) -> Self {
                Self::Tuple(vec![$(tuple.$var.into(),)*])
            }
        }
    };
}

impl_from_tuple_for_value_type!();
impl_from_tuple_for_value_type!(0: T);
impl_from_tuple_for_value_type!(0: T, 1: U);
impl_from_tuple_for_value_type!(0: T, 1: U, 2: V);
impl_from_tuple_for_value_type!(0: T, 1: U, 2: V, 3: W);
impl_from_tuple_for_value_type!(0: T, 1: U, 2: V, 3: W, 4: X);
impl_from_tuple_for_value_type!(0: T, 1: U, 2: V, 3: W, 4: X, 5: Y);
impl_from_tuple_for_value_type!(0: T, 1: U, 2: V, 3: W, 4: X, 5: Y, 6: Z);
impl_from_tuple_for_value_type!(0: T, 1: U, 2: V, 3: W, 4: X, 5: Y, 6: Z, 7: A);
impl_from_tuple_for_value_type!(0: T, 1: U, 2: V, 3: W, 4: X, 5: Y, 6: Z, 7: A, 8: B);
impl_from_tuple_for_value_type!(0: T, 1: U, 2: V, 3: W, 4: X, 5: Y, 6: Z, 7: A, 8: B, 9: C);

fn type_param(index: usize) -> Cow<'static, str> {
    const PARAM_NAMES: &str = "TUVXYZ";
    PARAM_NAMES.get(index..=index).map_or_else(
        || Cow::from(format!("T{}", index - PARAM_NAMES.len())),
        Cow::from,
    )
}

impl ValueType {
    /// Numeric literal type.
    pub const NUM: Self = ValueType::Lit(Num);
}

impl<Lit: LiteralType> ValueType<Lit> {
    /// Returns a void type (an empty tuple).
    pub fn void() -> Self {
        Self::Tuple(Vec::new())
    }

    /// Creates a slice type.
    pub fn slice(element: impl Into<ValueType<Lit>>, length: TupleLength) -> Self {
        Self::Slice {
            element: Box::new(element.into()),
            length,
        }
    }

    /// Creates a slice type by repeating this type.
    pub fn repeat(self, length: TupleLength) -> Self {
        Self::Slice {
            element: Box::new(self),
            length,
        }
    }

    /// Checks if this type is void (i.e., an empty tuple).
    pub fn is_void(&self) -> bool {
        matches!(self, Self::Tuple(elements) if elements.is_empty())
            || matches!(self, Self::Slice { length, .. } if *length == TupleLength::Exact(0))
    }

    /// Returns `Some(true)` if this type is known to be a literal, `Some(false)` if it's known
    /// not to be a literal, and `None` if either case is possible.
    pub(crate) fn is_literal(&self) -> Option<bool> {
        match self {
            Self::Lit(_) => Some(true),
            Self::Tuple(_) | Self::Slice { .. } | Self::Bool | Self::Function(_) => Some(false),
            _ => None,
        }
    }
}
