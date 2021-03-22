//! Base types, such as `ValueType` and `FnType`.

use std::{borrow::Cow, fmt};

use crate::{Num, PrimitiveType};

mod fn_type;

pub use self::fn_type::{FnArgs, FnType, FnTypeBuilder};
pub(crate) use self::fn_type::{LenParamDescription, TypeParamDescription};

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
            Self::Some { is_dynamic: false } | Self::Var(_) => formatter.write_str("_"),
            Self::Some { is_dynamic: true } => formatter.write_str("*"),
            Self::Exact(len) => fmt::Display::fmt(len, formatter),
            Self::Param(idx) => formatter.write_str(Self::const_param(*idx).as_ref()),
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

    fn is_concrete(&self) -> bool {
        matches!(self, Self::Param(_) | Self::Exact(_))
    }
}

/// Kind of a length parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum LengthKind {
    /// Parameter is static (can be found during type inference / "in compile time").
    Static,
    /// Parameter is dynamic (can vary at runtime).
    Dynamic,
}

/// Enumeration encompassing all types supported by the type system.
///
/// Parametric by the [`PrimitiveType`] (i.e., primitive types besides Booleans).
///
/// # Examples
///
/// There are conversions to construct `ValueType`s eloquently:
///
/// ```
/// # use arithmetic_typing::{FnType, TupleLength, ValueType};
/// let tuple: ValueType = (ValueType::Bool, ValueType::NUM).into();
/// assert_eq!(tuple.to_string(), "(Bool, Num)");
/// let slice: ValueType = tuple.repeat(TupleLength::Param(0));
/// assert_eq!(slice.to_string(), "[(Bool, Num); N]");
/// let fn_type: ValueType = FnType::builder()
///     .with_len_params(0..1)
///     .with_arg(slice)
///     .returning(ValueType::NUM)
///     .into();
/// assert_eq!(fn_type.to_string(), "fn<len N>([(Bool, Num); N]) -> Num");
/// ```
///
/// A `ValueType` can also be parsed from a string:
///
/// ```
/// # use arithmetic_typing::ValueType;
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let slice: ValueType = "[(Bool, Num); _]".parse()?;
/// assert_matches!(slice, ValueType::Slice { .. });
/// let fn_type: ValueType = "fn<len N>([(Bool, Num); N]) -> Num".parse()?;
/// assert_matches!(fn_type, ValueType::Function(_));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub enum ValueType<Prim: PrimitiveType = Num> {
    /// Wildcard type, i.e. some type that is not specified. Similar to `_` in type annotations
    /// in Rust.
    Some,
    /// Boolean type.
    // TODO: consider uniting literals and `Bool` as primitive types
    Bool,
    /// Primitive type excluding Booleans.
    Prim(Prim),
    /// Functional type.
    Function(Box<FnType<Prim>>),
    /// Tuple type.
    // TODO: support start / middle / end structuring
    Tuple(Vec<ValueType<Prim>>),
    /// Slice type. Unlike in Rust, slices are a subset of tuples. If `length` is
    /// [`Exact`](TupleLength::Exact), the slice is completely equivalent
    /// to the corresponding tuple.
    Slice {
        /// Type of slice elements.
        element: Box<ValueType<Prim>>,
        /// Slice length.
        length: TupleLength,
    },
    /// Type parameter in a function definition.
    Param(usize),

    /// Type variable. In contrast to `Param`s, `Var`s are used exclusively during
    /// inference and cannot occur in standalone function signatures.
    Var(usize),
}

impl<Prim: PrimitiveType> PartialEq for ValueType<Prim> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Some, _) | (_, Self::Some) | (Self::Bool, Self::Bool) => true,

            (Self::Prim(x), Self::Prim(y)) => x == y,

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

impl<Prim: PrimitiveType> fmt::Display for ValueType<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Some | Self::Var(_) => formatter.write_str("_"),
            Self::Param(idx) => formatter.write_str(type_param(*idx).as_ref()),

            Self::Bool => formatter.write_str("Bool"),
            Self::Prim(num) => fmt::Display::fmt(num, formatter),
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

impl<Prim: PrimitiveType> From<FnType<Prim>> for ValueType<Prim> {
    fn from(fn_type: FnType<Prim>) -> Self {
        Self::Function(Box::new(fn_type))
    }
}

macro_rules! impl_from_tuple_for_value_type {
    ($($var:tt : $ty:ident),*) => {
        impl<Prim, $($ty : Into<ValueType<Prim>>,)*> From<($($ty,)*)> for ValueType<Prim>
        where
            Prim: PrimitiveType,
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
    /// Numeric primitive type.
    pub const NUM: Self = ValueType::Prim(Num);
}

impl<Prim: PrimitiveType> ValueType<Prim> {
    /// Returns a void type (an empty tuple).
    pub fn void() -> Self {
        Self::Tuple(Vec::new())
    }

    /// Creates a slice type.
    pub fn slice(element: impl Into<ValueType<Prim>>, length: TupleLength) -> Self {
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
            Self::Prim(_) => Some(true),
            Self::Tuple(_) | Self::Slice { .. } | Self::Bool | Self::Function(_) => Some(false),
            _ => None,
        }
    }

    /// Returns `true` iff this type does not contain type / length variables.
    ///
    /// See [`TypeEnvironment`](crate::TypeEnvironment) for caveats of dealing with
    /// non-concrete types.
    pub fn is_concrete(&self) -> bool {
        match self {
            Self::Var(_) | Self::Some => false,
            Self::Param(_) | Self::Bool | Self::Prim(_) => true,

            Self::Function(fn_type) => fn_type.is_concrete(),
            Self::Tuple(elements) => elements.iter().all(Self::is_concrete),
            Self::Slice { element, length } => length.is_concrete() && element.is_concrete(),
        }
    }
}