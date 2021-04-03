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
    fn_type::{FnType, FnTypeBuilder},
    tuple::{LengthKind, Slice, Tuple, TupleLen, UnknownLen},
};

/// Enumeration encompassing all types supported by the type system.
///
/// Parametric by the [`PrimitiveType`].
///
/// # Notation
///
/// - [`Self::Some`] and [`Self::Var`] are represented as `_`.
/// - [`Prim`](Self::Prim)itive types are represented using the [`Display`](fmt::Display)
///   implementation of the corresponding [`PrimitiveType`].
/// - [`Param`](Self::Param)s are represented as `T`, `U`, `V` etc.
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
/// assert_eq!(fn_type.to_string(), "fn([(Bool, Num); _]) -> Num");
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
/// let fn_type: ValueType = "fn<len N>([(Bool, Num); N]) -> Num".parse()?;
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
    /// Primitive type.
    Prim(Prim),
    /// Functional type.
    Function(Box<FnType<Prim>>),
    /// Tuple type.
    Tuple(Tuple<Prim>),
    /// Type parameter in a function definition.
    Param(usize),

    /// Type variable. In contrast to `Param`s, `Var`s are used exclusively during
    /// inference and cannot occur in standalone function signatures.
    Var(usize),
}

impl<Prim: PrimitiveType> PartialEq for ValueType<Prim> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Some, _) | (_, Self::Some) => true,
            (Self::Prim(x), Self::Prim(y)) => x == y,
            (Self::Var(x), Self::Var(y)) | (Self::Param(x), Self::Param(y)) => x == y,
            (Self::Tuple(xs), Self::Tuple(ys)) => xs == ys,
            (Self::Function(x), Self::Function(y)) => x == y,
            _ => false,
        }
    }
}

impl<Prim: PrimitiveType> fmt::Display for ValueType<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Some | Self::Var(_) => formatter.write_str("_"),
            Self::Param(idx) => formatter.write_str(type_param(*idx).as_ref()),
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

fn type_param(index: usize) -> Cow<'static, str> {
    const PARAM_NAMES: &str = "TUVXYZ";
    PARAM_NAMES.get(index..=index).map_or_else(
        || Cow::from(format!("T{}", index - PARAM_NAMES.len())),
        Cow::from,
    )
}

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
            Self::Var(_) => false,
            Self::Some | Self::Param(_) | Self::Prim(_) => true,

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
            "fn(Num) -> Num",
            "fn<len N; T: Lin>([T; N]) -> T",
        ];

        for &sample_type in SAMPLE_TYPES {
            let ty: ValueType = sample_type.parse().unwrap();
            assert!(ty.eq(&ty), "Type is not equal to self: {}", ty);
        }
    }

    #[test]
    fn equality_is_preserved_on_renaming_params() {
        const EQUAL_FNS: &[&str] = &[
            "fn<len N; T: Lin>([T; N]) -> T",
            "fn<len L; T: Lin>([T; L]) -> T",
            "fn<len N; Ty: Lin>([Ty; N]) -> Ty",
            "fn<len T; N: Lin>([N; T]) -> N",
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
            "fn<len N; T: Lin>([T; N]) -> T",
            "fn<len N*; T: Lin>([T; N]) -> T",
            "fn<len N; T>([T; N]) -> T",
            "fn<len N; T: Lin>([T; N], T) -> T",
            "fn<len N; T: Lin>([T; N]) -> (T)",
        ];

        let functions: Vec<ValueType> = FUNCTIONS.iter().map(|s| s.parse().unwrap()).collect();
        for (i, function) in functions.iter().enumerate() {
            for other_function in &functions[(i + 1)..] {
                assert_ne!(function, other_function);
            }
        }
    }
}
