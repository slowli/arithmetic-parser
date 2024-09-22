//! Base types, such as `Type` and `DynConstraints`.

use core::fmt;

pub(crate) use self::{
    fn_type::{FnParams, ParamConstraints},
    quantifier::ParamQuantifier,
    tuple::IndexError,
};
pub use self::{
    fn_type::{FnWithConstraints, Function, FunctionBuilder},
    object::Object,
    tuple::{LengthVar, Slice, Tuple, TupleIndex, TupleLen, UnknownLen},
};
use crate::{
    alloc::{format, vec, Box, Cow},
    arith::{CompleteConstraints, ConstraintSet, Num, ObjectSafeConstraint, WithBoolean},
    PrimitiveType,
};

mod fn_type;
mod object;
mod quantifier;
mod tuple;

/// Type variable.
///
/// A variable represents a certain unknown type. Variables can be either *free*
/// or *bound* to a [`Function`] (these are known as type params in Rust).
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

    /// Creates a bounded type variable that can be used to [build functions](FunctionBuilder).
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
/// - [`Self::Any`] is represented as `any`.
/// - [`Self::Dyn`] types are represented as documented in [`DynConstraints`].
/// - [`Prim`](Self::Prim)itive types are represented using the [`Display`](fmt::Display)
///   implementation of the corresponding [`PrimitiveType`].
/// - [`Var`](Self::Var)s are represented as documented in [`TypeVar`].
/// - Notation for [functional](Function) and [tuple](Tuple) types is documented separately.
///
/// [`ConstraintSet`]: crate::arith::ConstraintSet
///
/// # Examples
///
/// There are conversions to construct `Type`s eloquently:
///
/// ```
/// # use arithmetic_typing::{Function, UnknownLen, Type};
/// let tuple: Type = (Type::BOOL, Type::NUM).into();
/// assert_eq!(tuple.to_string(), "(Bool, Num)");
/// let slice = tuple.repeat(UnknownLen::param(0));
/// assert_eq!(slice.to_string(), "[(Bool, Num); N]");
/// let fn_type: Type = Function::builder()
///     .with_arg(slice)
///     .returning(Type::NUM)
///     .into();
/// assert_eq!(fn_type.to_string(), "([(Bool, Num); N]) -> Num");
/// ```
///
/// A `Type` can also be parsed from a string:
///
/// ```
/// # use arithmetic_typing::{ast::TypeAst, Type};
/// # use std::convert::TryFrom;
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let slice = <Type>::try_from(&TypeAst::try_from("[(Bool, Num)]")?)?;
/// assert_matches!(slice, Type::Tuple(t) if t.as_slice().is_some());
/// let fn_type = <Type>::try_from(&TypeAst::try_from("([(Bool, Num); N]) -> Num")?)?;
/// assert_matches!(fn_type, Type::Function(_));
/// # Ok(())
/// # }
/// ```
///
/// # `Any` type
///
/// [`Self::Any`], denoted as `any`, is a catch-all type similar to `any` in TypeScript.
/// It allows to circumvent type system limitations at the cost of being extremely imprecise.
/// `any` type can be used in any context (destructured, called with args of any quantity
/// and type and so on), with each application of the type evaluated independently.
/// Thus, the same `any` variable can be treated as a function, a tuple, a primitive type, etc.
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse};
/// # use arithmetic_typing::{Annotated, TypeEnvironment, Type};
/// # use assert_matches::assert_matches;
///
/// # fn main() -> anyhow::Result<()> {
/// let code = "
///     wildcard: any = 1; // `any` can be assigned from anything
///     wildcard == 1 && wildcard == (2, 3);
///     (x, y, ...) = wildcard; // destructuring `any` always succeeds
///     wildcard(1, |x| x + 1); // calling `any` as a function works as well
/// ";
/// let ast = Annotated::<F32Grammar>::parse_statements(code)?;
/// let mut env = TypeEnvironment::new();
/// env.process_statements(&ast)?;
///
/// // Destructure outputs are certain types that can be inferred
/// // from their usage, rather than `any`!
/// assert_matches!(env["x"], Type::Var(_));
/// let bogus_code = "x + 1 == 2; x(1)";
/// let ast = Annotated::<F32Grammar>::parse_statements(bogus_code)?;
/// let errors = env.process_statements(&ast).unwrap_err();
/// # assert_eq!(errors.len(), 1);
/// let err = errors.iter().next().unwrap();
/// assert_eq!(err.main_location().span(bogus_code), "x(1)");
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Type<Prim: PrimitiveType = Num> {
    /// Any type aka "I'll think about typing later". Similar to `any` type in TypeScript.
    /// See [the dedicated section](#any-type) for more details.
    Any,
    /// Arbitrary type implementing certain constraints. Similar to `dyn _` types in Rust or use of
    /// interfaces in type position in TypeScript.
    ///
    /// See [`DynConstraints`] for details.
    Dyn(DynConstraints<Prim>),
    /// Primitive type.
    Prim(Prim),
    /// Functional type.
    Function(Box<Function<Prim>>),
    /// Tuple type.
    Tuple(Tuple<Prim>),
    /// Object type.
    Object(Object<Prim>),
    /// Type variable.
    Var(TypeVar),
}

impl<Prim: PrimitiveType> PartialEq for Type<Prim> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Any, _) | (_, Self::Any) => true,
            (Self::Dyn(x), Self::Dyn(y)) => x == y,
            (Self::Prim(x), Self::Prim(y)) => x == y,
            (Self::Var(x), Self::Var(y)) => x == y,
            (Self::Tuple(xs), Self::Tuple(ys)) => xs == ys,
            (Self::Object(x), Self::Object(y)) => x == y,
            (Self::Function(x), Self::Function(y)) => x == y,
            _ => false,
        }
    }
}

impl<Prim: PrimitiveType> fmt::Display for Type<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Any => formatter.write_str("any"),
            Self::Dyn(constraints) => {
                if constraints.inner.is_empty() {
                    formatter.write_str("dyn")
                } else {
                    write!(formatter, "dyn {constraints}")
                }
            }
            Self::Var(var) => fmt::Display::fmt(var, formatter),
            Self::Prim(num) => fmt::Display::fmt(num, formatter),
            Self::Function(fn_type) => fmt::Display::fmt(fn_type, formatter),
            Self::Tuple(tuple) => fmt::Display::fmt(tuple, formatter),
            Self::Object(obj) => fmt::Display::fmt(obj, formatter),
        }
    }
}

impl<Prim: PrimitiveType> From<Function<Prim>> for Type<Prim> {
    fn from(fn_type: Function<Prim>) -> Self {
        Self::Function(Box::new(fn_type))
    }
}

impl<Prim: PrimitiveType> From<Tuple<Prim>> for Type<Prim> {
    fn from(tuple: Tuple<Prim>) -> Self {
        Self::Tuple(tuple)
    }
}

impl<Prim: PrimitiveType> From<Slice<Prim>> for Type<Prim> {
    fn from(slice: Slice<Prim>) -> Self {
        Self::Tuple(slice.into())
    }
}

impl<Prim: PrimitiveType> From<Object<Prim>> for Type<Prim> {
    fn from(object: Object<Prim>) -> Self {
        Self::Object(object)
    }
}

impl<Prim: PrimitiveType> From<DynConstraints<Prim>> for Type<Prim> {
    fn from(constraints: DynConstraints<Prim>) -> Self {
        Self::Dyn(constraints)
    }
}

macro_rules! impl_from_tuple_for_type {
    ($($var:tt : $ty:ident),*) => {
        impl<Prim, $($ty : Into<Type<Prim>>,)*> From<($($ty,)*)> for Type<Prim>
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

impl_from_tuple_for_type!();
impl_from_tuple_for_type!(0: T);
impl_from_tuple_for_type!(0: T, 1: U);
impl_from_tuple_for_type!(0: T, 1: U, 2: V);
impl_from_tuple_for_type!(0: T, 1: U, 2: V, 3: W);
impl_from_tuple_for_type!(0: T, 1: U, 2: V, 3: W, 4: X);
impl_from_tuple_for_type!(0: T, 1: U, 2: V, 3: W, 4: X, 5: Y);
impl_from_tuple_for_type!(0: T, 1: U, 2: V, 3: W, 4: X, 5: Y, 6: Z);
impl_from_tuple_for_type!(0: T, 1: U, 2: V, 3: W, 4: X, 5: Y, 6: Z, 7: A);
impl_from_tuple_for_type!(0: T, 1: U, 2: V, 3: W, 4: X, 5: Y, 6: Z, 7: A, 8: B);
impl_from_tuple_for_type!(0: T, 1: U, 2: V, 3: W, 4: X, 5: Y, 6: Z, 7: A, 8: B, 9: C);

impl Type {
    /// Numeric primitive type.
    pub const NUM: Self = Type::Prim(Num::Num);
}

impl<Prim: WithBoolean> Type<Prim> {
    /// Boolean primitive type.
    pub const BOOL: Self = Type::Prim(Prim::BOOL);
}

impl<Prim: PrimitiveType> Type<Prim> {
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
    pub fn slice(element: impl Into<Type<Prim>>, length: impl Into<TupleLen>) -> Self {
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
            Self::Tuple(_) | Self::Object(_) | Self::Function(_) => Some(false),
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
            Self::Any | Self::Prim(_) => true,
            Self::Dyn(constraints) => constraints.is_concrete(),
            Self::Function(fn_type) => fn_type.is_concrete(),
            Self::Tuple(tuple) => tuple.is_concrete(),
            Self::Object(obj) => obj.is_concrete(),
        }
    }
}

/// Arbitrary type implementing certain constraints. Similar to `dyn _` types in Rust or use of
/// interfaces in type position in TypeScript.
///
/// [`Constraint`]s in this type must be [object-safe](crate::arith::ObjectSafeConstraint).
/// `DynConstraints` can also specify an [`Object`] constraint, which can be converted to it
/// using the [`From`] trait.
///
/// [`Constraint`]: crate::arith::Constraint
///
/// # Notation
///
/// - If the constraints do not include an object constraint, they are [`Display`](fmt::Display)ed
///   like a [`ConstraintSet`] with `dyn` prefix; e.g, `dyn Lin + Hash`.
/// - If the constraints include an object constraint, it is specified before all other constraints,
///   but after the `dyn` prefix; e.g., `dyn { x: Num } + Lin`.
///
/// # Examples
///
/// `dyn _` types can be used to express that any types satisfying certain constraints
/// should be accepted.
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse};
/// # use arithmetic_typing::{defs::Prelude, Annotated, TypeEnvironment, Type, Function};
/// #
/// # fn main() -> anyhow::Result<()> {
/// let code = "
///     sum_lengths = |...pts: dyn { x: _, y: _ }| {
///         pts.fold(0, |acc, { x, y }| acc + sqrt(x * x + y * y))
///     };
///     sum_lengths(#{ x: 1, y: 2 }, #{ x: 3, y: 4, z: 5 })
/// ";
/// let ast = Annotated::<F32Grammar>::parse_statements(code)?;
///
/// let mut env = TypeEnvironment::new();
/// let sqrt = Function::builder().with_arg(Type::NUM).returning(Type::NUM);
/// env.insert("fold", Prelude::Fold).insert("sqrt", sqrt);
/// env.process_statements(&ast)?;
///
/// assert_eq!(
///     env["sum_lengths"].to_string(),
///     "(...[dyn { x: Num, y: Num }; N]) -> Num"
/// );
/// # Ok(())
/// # }
/// ```
///
/// One of primary use cases of `dyn _` is restricting varargs of a function:
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse};
/// # use arithmetic_typing::{
/// #     ast::TypeAst, defs::Prelude, error::ErrorKind, Annotated, TypeEnvironment, Type,
/// # };
/// # use std::convert::TryFrom;
/// # use assert_matches::assert_matches;
/// #
/// # fn main() -> anyhow::Result<()> {
/// // Function that accepts any amount of linear args (not necessarily
/// // of the same type) and returns a number.
/// let digest_fn = Type::try_from(&TypeAst::try_from("(...[dyn Lin; N]) -> Num")?)?;
/// let mut env = TypeEnvironment::new();
/// env.insert("true", Prelude::True).insert("digest", digest_fn);
///
/// let code = "
///     digest(1, 2, (3, 4), #{ x: 5, y: (6,) }) == 1;
///     digest(3, true) == 0; // fails: `true` is not linear
/// ";
/// let ast = Annotated::<F32Grammar>::parse_statements(code)?;
/// let errors = env.process_statements(&ast).unwrap_err();
///
/// let err = errors.iter().next().unwrap();
/// assert_eq!(err.main_location().span(code), "true");
/// assert_matches!(err.kind(), ErrorKind::FailedConstraint { .. });
/// # Ok(())
/// # }
/// ```
#[derive(Clone, PartialEq)]
pub struct DynConstraints<Prim: PrimitiveType> {
    pub(crate) inner: CompleteConstraints<Prim>,
}

impl<Prim: PrimitiveType> fmt::Debug for DynConstraints<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, formatter)
    }
}

impl<Prim: PrimitiveType> fmt::Display for DynConstraints<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, formatter)
    }
}

impl<Prim: PrimitiveType> From<Object<Prim>> for DynConstraints<Prim> {
    fn from(object: Object<Prim>) -> Self {
        Self {
            inner: object.into(),
        }
    }
}

impl<Prim: PrimitiveType> DynConstraints<Prim> {
    /// Creates constraints based on a single constraint.
    pub fn just(constraint: impl ObjectSafeConstraint<Prim>) -> Self {
        Self {
            inner: CompleteConstraints::from(ConstraintSet::just(constraint)),
        }
    }

    /// Checks if this constraint set is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the enclosed object constraint, if any.
    pub fn object(&self) -> Option<&Object<Prim>> {
        self.inner.object.as_ref()
    }

    fn is_concrete(&self) -> bool {
        self.inner.object.as_ref().map_or(true, Object::is_concrete)
    }

    /// Adds the specified `constraint` to these constraints.
    pub fn insert(&mut self, constraint: impl ObjectSafeConstraint<Prim>) {
        self.inner.simple.insert(constraint);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{alloc::Vec, ast::TypeAst};

    #[test]
    fn types_are_equal_to_self() -> anyhow::Result<()> {
        const SAMPLE_TYPES: &[&str] = &[
            "Num",
            "(Num, Bool)",
            "(Num, ...[Bool; N]) -> ()",
            "(Num) -> Num",
            "for<'T: Lin> (['T; N]) -> 'T",
        ];

        for &sample_type in SAMPLE_TYPES {
            let ty = <Type>::try_from(&TypeAst::try_from(sample_type)?)?;
            assert!(ty.eq(&ty), "Type is not equal to self: {ty}");
        }
        Ok(())
    }

    #[test]
    fn equality_is_preserved_on_renaming_params() {
        const EQUAL_FNS: &[&str] = &[
            "for<'T: Lin> (['T; N]) -> 'T",
            "for<'T: Lin> (['T; L]) -> 'T",
            "for<'Ty: Lin> (['Ty; N]) -> 'Ty",
            "for<'N: Lin> (['N; T]) -> 'N",
        ];

        let functions: Vec<Type> = EQUAL_FNS
            .iter()
            .map(|&s| Type::try_from(&TypeAst::try_from(s).unwrap()).unwrap())
            .collect();
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
            "for<len! N; 'T: Lin> (['T; N]) -> 'T",
            "(['T; N]) -> 'T",
            "for<'T: Lin> (['T; N], 'T) -> 'T",
            "for<'T: Lin> (['T; N]) -> ('T)",
        ];

        let functions: Vec<Type> = FUNCTIONS
            .iter()
            .map(|&s| Type::try_from(&TypeAst::try_from(s).unwrap()).unwrap())
            .collect();
        for (i, function) in functions.iter().enumerate() {
            for other_function in &functions[(i + 1)..] {
                assert_ne!(function, other_function);
            }
        }
    }

    #[test]
    fn concrete_types() {
        let sample_types = &[
            Type::NUM,
            Type::BOOL,
            Type::Any,
            (Type::BOOL, Type::NUM).into(),
            Type::try_from(&TypeAst::try_from("for<'T: Lin> (['T; N]) -> 'T").unwrap()).unwrap(),
        ];

        for ty in sample_types {
            assert!(ty.is_concrete(), "{ty:?}");
        }
    }

    #[test]
    fn non_concrete_types() {
        let sample_types = &[
            Type::free_var(2),
            (Type::NUM, Type::free_var(0)).into(),
            Function::builder()
                .with_arg(Type::free_var(0))
                .returning(Type::void())
                .into(),
        ];

        for ty in sample_types {
            assert!(!ty.is_concrete(), "{ty:?}");
        }
    }
}
