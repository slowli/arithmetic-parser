//! Base types, such as `ValueType` and `FnType`.

use std::{borrow::Cow, fmt};

use crate::{arith::WithBoolean, Num, PrimitiveType};

mod fn_type;
mod quantifier;
mod tuple;

pub(crate) use self::{
    fn_type::{FnParams, ParamConstraints},
    quantifier::ParamQuantifier,
};
pub use self::{
    fn_type::{FnType, FnTypeBuilder, FnWithConstraints},
    tuple::{LengthKind, LengthVar, Slice, Tuple, TupleLen, UnknownLen},
};

/// Type variable.
///
/// A variable represents a certain unknown type. Variables can be either *free*
/// or *bound* to a [function](FnType) (these are known as type params in Rust).
/// Types input to a [`TypeEnvironment`] can only have bounded variables (this is
/// verified in runtime), but types output by the inference process can contain both.
///
/// # Notation
///
/// - Bounded type variables are represented as `'T`, `'U`, `'V`, etc.
///   The tick is inspired by lifetimes in Rust and implicit type params in [F*]. It allows
///   to easily distinguish between vars and primitive types.
/// - Free variables are represented as `_`.
///
/// [`TypeEnvironment`]: crate::TypeEnvironment
/// [F*]: http://www.fstar-lang.org/tutorial/
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeVar {
    index: usize,
    is_free: bool,
}

impl fmt::Display for TypeVar {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_free {
            formatter.write_str("_")
        } else {
            write!(formatter, "'{}", Self::param_str(self.index))
        }
    }
}

impl TypeVar {
    fn param_str(index: usize) -> Cow<'static, str> {
        const PARAM_NAMES: &str = "TUVXYZ";
        PARAM_NAMES.get(index..=index).map_or_else(
            || Cow::from(format!("T{}", index - PARAM_NAMES.len())),
            Cow::from,
        )
    }

    /// Creates a bounded type variable that can be used to [build functions](FnTypeBuilder).
    pub const fn param(index: usize) -> Self {
        Self {
            index,
            is_free: false,
        }
    }

    /// Returns the 0-based index of this variable.
    pub fn index(self) -> usize {
        self.index
    }

    /// Is this variable free (not bounded in a function declaration)?
    pub fn is_free(self) -> bool {
        self.is_free
    }
}

/// Enumeration encompassing all types supported by the type system.
///
/// Parametric by the [`PrimitiveType`].
///
/// # Notation
///
/// - [`Self::Some`] is represented as `_`.
/// - [`Prim`](Self::Prim)itive types are represented using the [`Display`](fmt::Display)
///   implementation of the corresponding [`PrimitiveType`].
/// - [`Var`](Self::Var)s are represented as documented in [`TypeVar`].
/// - Notation for [functional](FnType) and [tuple](Tuple) types is documented separately.
///
/// # Examples
///
/// There are conversions to construct `ValueType`s eloquently:
///
/// ```
/// # use arithmetic_typing::{FnType, UnknownLen, ValueType};
/// let tuple: ValueType = (ValueType::BOOL, ValueType::NUM).into();
/// assert_eq!(tuple.to_string(), "(Bool, Num)");
/// let slice = tuple.repeat(UnknownLen::Some);
/// assert_eq!(slice.to_string(), "[(Bool, Num); _]");
/// let fn_type: ValueType = FnType::builder()
///     .with_arg(slice)
///     .returning(ValueType::NUM)
///     .into();
/// assert_eq!(fn_type.to_string(), "([(Bool, Num); _]) -> Num");
/// ```
///
/// A `ValueType` can also be parsed from a string:
///
/// ```
/// # use arithmetic_typing::ValueType;
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let slice: ValueType = "[(Bool, Num); _]".parse()?;
/// assert_matches!(slice, ValueType::Tuple(t) if t.as_slice().is_some());
/// let fn_type: ValueType = "([(Bool, Num); N]) -> Num".parse()?;
/// assert_matches!(fn_type, ValueType::Function(_));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ValueType<Prim: PrimitiveType = Num> {
    /// Wildcard type, i.e. some type that is not specified. Similar to `_` in type annotations
    /// in Rust.
    Some,
    /// Any type aka "I'll think about typing later". Similar to `any` type in TypeScript.
    /// `any` type can be used in any context (destructured, called with any params, etc.).
    Any,
    /// Primitive type.
    Prim(Prim),
    /// Functional type.
    Function(Box<FnType<Prim>>),
    /// Tuple type.
    Tuple(Tuple<Prim>),
    /// Type variable.
    Var(TypeVar),
}

impl<Prim: PrimitiveType> PartialEq for ValueType<Prim> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Some, Self::Some) | (Self::Any, _) | (_, Self::Any) => true,
            (Self::Prim(x), Self::Prim(y)) => x == y,
            (Self::Var(x), Self::Var(y)) => x == y,
            (Self::Tuple(xs), Self::Tuple(ys)) => xs == ys,
            (Self::Function(x), Self::Function(y)) => x == y,
            _ => false,
        }
    }
}

impl<Prim: PrimitiveType> fmt::Display for ValueType<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Some => formatter.write_str("_"),
            Self::Any => formatter.write_str("any"),
            Self::Var(var) => fmt::Display::fmt(var, formatter),
            Self::Prim(num) => fmt::Display::fmt(num, formatter),
            Self::Function(fn_type) => fmt::Display::fmt(fn_type, formatter),
            Self::Tuple(tuple) => fmt::Display::fmt(tuple, formatter),
        }
    }
}

impl<Prim: PrimitiveType> From<FnType<Prim>> for ValueType<Prim> {
    fn from(fn_type: FnType<Prim>) -> Self {
        Self::Function(Box::new(fn_type))
    }
}

impl<Prim: PrimitiveType> From<Tuple<Prim>> for ValueType<Prim> {
    fn from(tuple: Tuple<Prim>) -> ValueType<Prim> {
        Self::Tuple(tuple)
    }
}

impl<Prim: PrimitiveType> From<Slice<Prim>> for ValueType<Prim> {
    fn from(slice: Slice<Prim>) -> ValueType<Prim> {
        Self::Tuple(slice.into())
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
                Self::Tuple(Tuple::from(vec![$(tuple.$var.into(),)*]))
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

impl ValueType {
    /// Numeric primitive type.
    pub const NUM: Self = ValueType::Prim(Num::Num);
}

impl<Prim: WithBoolean> ValueType<Prim> {
    /// Boolean primitive type.
    pub const BOOL: Self = ValueType::Prim(Prim::BOOL);
}

impl<Prim: PrimitiveType> ValueType<Prim> {
    /// Returns a void type (an empty tuple).
    pub fn void() -> Self {
        Self::Tuple(Tuple::empty())
    }

    /// Creates a bounded type variable with the specified `index`.
    pub fn param(index: usize) -> Self {
        Self::Var(TypeVar::param(index))
    }

    pub(crate) fn free_var(index: usize) -> Self {
        Self::Var(TypeVar {
            index,
            is_free: true,
        })
    }

    /// Creates a slice type.
    pub fn slice(element: impl Into<ValueType<Prim>>, length: impl Into<TupleLen>) -> Self {
        Self::Tuple(Slice::new(element.into(), length).into())
    }

    /// Creates a slice type by repeating this type.
    pub fn repeat(self, length: impl Into<TupleLen>) -> Slice<Prim> {
        Slice::new(self, length)
    }

    /// Checks if this type is void (i.e., an empty tuple).
    pub fn is_void(&self) -> bool {
        matches!(self, Self::Tuple(tuple) if tuple.is_empty())
    }

    /// Returns `Some(true)` if this type is known to be primitive,
    /// `Some(false)` if it's known not to be primitive, and `None` if either case is possible.
    pub(crate) fn is_primitive(&self) -> Option<bool> {
        match self {
            Self::Prim(_) => Some(true),
            Self::Tuple(_) | Self::Function(_) => Some(false),
            _ => None,
        }
    }

    /// Returns `true` iff this type does not contain type / length variables.
    ///
    /// See [`TypeEnvironment`](crate::TypeEnvironment) for caveats of dealing with
    /// non-concrete types.
    pub fn is_concrete(&self) -> bool {
        match self {
            Self::Var(var) => !var.is_free,
            Self::Some => false,
            Self::Any | Self::Prim(_) => true,
            Self::Function(fn_type) => fn_type.is_concrete(),
            Self::Tuple(tuple) => tuple.is_concrete(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_types_are_equal_to_self() {
        const SAMPLE_TYPES: &[&str] = &[
            "Num",
            "(Num, Bool)",
            "[Num; _]",
            "(Num, ...[Bool; _])",
            "(Num) -> Num",
            "for<'T: Lin> (['T; _]) -> 'T",
        ];

        for &sample_type in SAMPLE_TYPES {
            let ty: ValueType = sample_type.parse().unwrap();
            assert!(ty.eq(&ty), "Type is not equal to self: {}", ty);
        }
    }

    #[test]
    fn equality_is_preserved_on_renaming_params() {
        const EQUAL_FNS: &[&str] = &[
            "for<'T: Lin> (['T; N]) -> 'T",
            "for<'T: Lin> (['T; L]) -> 'T",
            "for<'Ty: Lin> (['Ty; N]) -> 'Ty",
            "for<'N: Lin> (['N; T]) -> 'N",
        ];

        let functions: Vec<ValueType> = EQUAL_FNS.iter().map(|s| s.parse().unwrap()).collect();
        for (i, function) in functions.iter().enumerate() {
            for other_function in &functions[(i + 1)..] {
                assert_eq!(function, other_function);
            }
        }
    }

    #[test]
    fn unequal_functions() {
        const FUNCTIONS: &[&str] = &[
            "for<'T: Lin> (['T; N]) -> 'T",
            "for<len N*; 'T: Lin> (['T; N]) -> 'T",
            "(['T; N]) -> 'T",
            "for<'T: Lin> (['T; N], 'T) -> 'T",
            "for<'T: Lin> (['T; N]) -> ('T)",
        ];

        let functions: Vec<ValueType> = FUNCTIONS.iter().map(|s| s.parse().unwrap()).collect();
        for (i, function) in functions.iter().enumerate() {
            for other_function in &functions[(i + 1)..] {
                assert_ne!(function, other_function);
            }
        }
    }

    #[test]
    fn concrete_types() {
        let sample_types = &[
            ValueType::NUM,
            ValueType::BOOL,
            ValueType::Any,
            (ValueType::BOOL, ValueType::NUM).into(),
            "for<'T: Lin> (['T; N]) -> 'T".parse().unwrap(),
        ];

        for ty in sample_types {
            assert!(ty.is_concrete(), "{:?}", ty);
        }
    }

    #[test]
    fn non_concrete_types() {
        let sample_types = &[
            ValueType::Some,
            ValueType::free_var(2),
            (ValueType::NUM, ValueType::Some).into(),
            FnType::builder()
                .with_arg(ValueType::Some)
                .returning(ValueType::void())
                .into(),
        ];

        for ty in sample_types {
            assert!(!ty.is_concrete(), "{:?}", ty);
        }
    }
}
