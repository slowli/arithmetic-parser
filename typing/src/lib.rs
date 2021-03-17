//! Hindley-Milner type inference for arithmetic expressions parsed
//! by the [`arithmetic-parser`] crate.
//!
//! This crate allows to parse type annotations as a part of a [`Grammar`], and to infer
//! and check types for expressions / statements produced by `arithmetic-parser`.
//! Type inference is *partially* compatible with the interpreter from [`arithmetic-eval`];
//! if the inference algorithm succeeds on a certain expression / statement / block,
//! it will execute successfully, but not all successfully executing items pass type inference.
//!
//! # Type system
//!
//! The type system corresponds to types of `Value`s in `arithmetic-eval`:
//!
//! - There are 2 primitive types: Boolean (`Bool`) and number (`Num`).
//! - There is only one container type - a tuple. It can be represented either
//!   in the tuple form, such as `(Num, Bool)`, or as a slice, such as `[Num]` or `[Num; 3]`.
//!   As in Rust, all slice elements must have the same type. Unlike Rust, tuple and slice
//!   forms are equivalent; e.g., `[Num; 3]` and `(Num, Num, Num)` are the same type.
//! - Functions are first-class types. Functions can have type and/or const params.
//! - Type params can be constrained. The only currently supported constraint is type *linearity*.
//!   Linear types are types that support binary arithmetic ops (e.g., `+`): numbers and
//!   tuples consisting of linear elements.
//! - Const params always specify tuple length.
//!
//! # Inference rules
//!
//! FIXME
//!
//! # Examples
//!
//! ```
//! use arithmetic_parser::grammars::{Parse, Typed};
//! use arithmetic_typing::{NumGrammar, Prelude, TypeEnvironment, ValueType};
//!
//! # fn main() -> anyhow::Result<()> {
//! let code = "sum = |xs| xs.fold(0, |acc, x| acc + x);";
//! let ast = Typed::<NumGrammar<f32>>::parse_statements(code)?;
//!
//! let mut env = TypeEnvironment::new();
//! env.insert_type("fold", Prelude::fold_type().into());
//!
//! // Evaluate `code` to get the inferred `sum` function signature.
//! let output_type = env.process_statements(&ast)?;
//! assert!(output_type.is_void());
//! assert_eq!(env["sum"].to_string(), "fn<const N>([Num; N]) -> Num");
//! # Ok(())
//! # }
//! ```
//!
//! [`arithmetic-parser`]: https://crates.io/crates/arithmetic-parser
//! [`Grammar`]: arithmetic_parser::grammars::Grammar
//! [`arithmetic-eval`]: https://crates.io/crates/arithmetic-eval

#![warn(missing_docs, missing_debug_implementations)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::module_name_repetitions,
    clippy::similar_names // too many false positives because of lhs / rhs
)]

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fmt,
    marker::PhantomData,
    str::FromStr,
};

use arithmetic_parser::{
    grammars::{Grammar, NumLiteral, ParseLiteral},
    InputSpan, NomResult,
};

pub mod arith;
pub mod ast;
mod env;
mod error;
mod substitutions;
mod type_map;

pub use self::{
    env::TypeEnvironment,
    error::{TypeError, TypeErrorKind, TypeResult},
    substitutions::Substitutions,
    type_map::{Assertions, Prelude},
};

// Reexports for the macros.
#[doc(hidden)]
pub mod _reexports {
    pub use anyhow::{anyhow, Error};
}

/// Description of a type parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TypeParamDescription {
    /// Can this type param be non-linear?
    maybe_non_linear: bool,
}

/// Types of literals in a certain grammar.
///
/// More complex types, like [`ValueType`] and [`FnType`], are defined with a type param
/// which determines the literal type. This type param must implement [`LiteralType`].
///
/// [`TypeArithmetic`] has a `LiteralType` impl as an associated type, and one of the required
/// operations of this trait is to be able to infer type for literal values from a [`Grammar`].
///
/// # Implementation Requirements
///
/// - [`Display`](fmt::Display) and [`FromStr`] implementations must be consistent; i.e.,
///   `Display` should produce output parseable by `FromStr`. `Display` will be used in
///   `Display` impls for `ValueType` etc. `FromStr` will be used to read type annotations.
/// - `Display` presentations must be identifiers, such as `Num`.
///
/// [`Grammar`]: arithmetic_parser::grammars::Grammar
///
/// # Examples
///
/// ```
/// # use std::{fmt, str::FromStr};
/// use arithmetic_typing::LiteralType;
///
/// #[derive(Debug, Clone, Copy, PartialEq)]
/// enum NumOrBytes {
///     /// Numeric value, such as 1.
///     Num,
///     /// Bytes value, such as 0x1234 or "hello".
///     Bytes,
/// }
///
/// // `NumOrBytes` should correspond to a "value" type in the `Grammar`,
/// // for example:
/// enum NumOrBytesValue {
///     Num(f64),
///     Bytes(Vec<u8>),
/// }
///
/// impl fmt::Display for NumOrBytes {
///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
///         match self {
///             Self::Num => formatter.write_str("Num"),
///             Self::Bytes => formatter.write_str("Bytes"),
///         }
///     }
/// }
///
/// impl FromStr for NumOrBytes {
///     type Err = anyhow::Error;
///
///     fn from_str(s: &str) -> Result<Self, Self::Err> {
///         match s {
///             "Num" => Ok(Self::Num),
///             "Bytes" => Ok(Self::Bytes),
///             _ => Err(anyhow::anyhow!("expected `Num` or `Bytes`")),
///         }
///     }
/// }
///
/// impl LiteralType for NumOrBytes {}
/// ```
pub trait LiteralType:
    Clone + PartialEq + fmt::Debug + fmt::Display + FromStr + Send + Sync + 'static
{
}

/// Implements [`Display`](fmt::Display), [`FromStr`] and [`LiteralType`] for the provided type,
/// which must be a no-field struct.
///
/// # Examples
///
/// ```
/// use arithmetic_typing::impl_singleton_literal_type;
///
/// #[derive(Debug, Clone, Copy, PartialEq)]
/// pub struct SomeType;
///
/// impl_singleton_literal_type!(SomeType, "Some");
/// ```
#[macro_export]
macro_rules! impl_singleton_literal_type {
    ($ty:ident, $name:tt) => {
        impl core::fmt::Display for $ty {
            fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str($name)
            }
        }

        impl core::str::FromStr for $ty {
            type Err = $crate::_reexports::Error;

            fn from_str(s: &str) -> core::result::Result<Self, Self::Err> {
                if s == $name {
                    core::result::Result::Ok($ty)
                } else {
                    core::result::Result::Err($crate::_reexports::anyhow!(concat!(
                        "Expected `",
                        $name,
                        "`"
                    )))
                }
            }
        }

        impl $crate::LiteralType for $ty {}
    };
}

/// Generic numeric type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Num;

impl_singleton_literal_type!(Num, "Num");

/// Maps a literal value from a certain [`Grammar`] to its type.
///
/// [`Grammar`]: arithmetic_parser::grammars::Grammar
pub trait MapLiteralType<Val> {
    /// Types of literals output by this mapper.
    type Lit: LiteralType;

    /// Gets the type of the provided literal value.
    fn type_of_literal(&self, lit: &Val) -> Self::Lit;
}

/// Functional type.
#[derive(Debug, Clone, PartialEq)]
pub struct FnType<Lit = Num> {
    /// Type of function arguments.
    args: FnArgs<Lit>,
    /// Type of the value returned by the function.
    return_type: ValueType<Lit>,
    /// Type params associated with this function. The indexes of params should
    /// monotonically increase (necessary for correct display) and must be distinct.
    type_params: Vec<(usize, TypeParamDescription)>,
    /// Indexes of const params associated with this function. The indexes should
    /// monotonically increase (necessary for correct display) and must be distinct.
    const_params: Vec<usize>,
}

impl<Lit: fmt::Display> fmt::Display for FnType<Lit> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("fn")?;

        if self.const_params.len() + self.type_params.len() > 0 {
            formatter.write_str("<")?;

            if !self.const_params.is_empty() {
                formatter.write_str("const ")?;
                for (i, &var_idx) in self.const_params.iter().enumerate() {
                    formatter.write_str(TupleLength::const_param(var_idx).as_ref())?;
                    if i + 1 < self.const_params.len() {
                        formatter.write_str(", ")?;
                    }
                }

                if !self.type_params.is_empty() {
                    formatter.write_str("; ")?;
                }
            }

            for (i, (var_idx, description)) in self.type_params.iter().enumerate() {
                formatter.write_str(type_param(*var_idx).as_ref())?;
                if description.maybe_non_linear {
                    formatter.write_str(": ?Lin")?;
                }
                if i + 1 < self.type_params.len() {
                    formatter.write_str(", ")?;
                }
            }

            formatter.write_str(">")?;
        }

        write!(formatter, "({})", self.args)?;
        if !self.return_type.is_void() {
            write!(formatter, " -> {}", self.return_type)?;
        }
        Ok(())
    }
}

impl<Lit: LiteralType> FnType<Lit> {
    pub(crate) fn new(args: FnArgs<Lit>, return_type: ValueType<Lit>) -> Self {
        Self {
            args,
            return_type,
            type_params: Vec::new(),  // filled in later
            const_params: Vec::new(), // filled in later
        }
    }

    pub(crate) fn with_const_params(mut self, mut params: Vec<usize>) -> Self {
        params.sort_unstable();
        self.const_params = params;
        self
    }

    pub(crate) fn with_type_params(
        mut self,
        mut params: Vec<(usize, TypeParamDescription)>,
    ) -> Self {
        params.sort_unstable_by_key(|(idx, _)| *idx);
        self.type_params = params;
        self
    }

    /// Returns a builder for `FnType`s.
    pub fn builder() -> FnTypeBuilder<Lit> {
        FnTypeBuilder::default()
    }

    /// Returns `true` iff the function has at least one const or type param.
    pub fn is_parametric(&self) -> bool {
        !self.const_params.is_empty() || !self.type_params.is_empty()
    }

    /// Checks if a type variable with the specified index is linear.
    pub(crate) fn linear_type_params(&self) -> impl Iterator<Item = usize> + '_ {
        self.type_params.iter().filter_map(|(idx, description)| {
            if description.maybe_non_linear {
                None
            } else {
                Some(*idx)
            }
        })
    }

    pub(crate) fn arg_and_return_types(&self) -> impl Iterator<Item = &ValueType<Lit>> + '_ {
        let args_slice = match &self.args {
            FnArgs::List(args) => args.as_slice(),
            FnArgs::Any => &[],
        };
        args_slice.iter().chain(Some(&self.return_type))
    }

    fn arg_and_return_types_mut(&mut self) -> impl Iterator<Item = &mut ValueType<Lit>> + '_ {
        let args_slice = match &mut self.args {
            FnArgs::List(args) => args.as_mut_slice(),
            FnArgs::Any => &mut [],
        };
        args_slice.iter_mut().chain(Some(&mut self.return_type))
    }

    /// Maps argument and return types. The mapping function must not touch type params
    /// of the function.
    pub(crate) fn map_types<F>(&self, mut map_fn: F) -> Self
    where
        F: FnMut(&ValueType<Lit>) -> ValueType<Lit>,
    {
        Self {
            args: match &self.args {
                FnArgs::List(args) => FnArgs::List(args.iter().map(&mut map_fn).collect()),
                FnArgs::Any => FnArgs::Any,
            },
            return_type: map_fn(&self.return_type),
            type_params: self.type_params.clone(),
            const_params: self.const_params.clone(),
        }
    }
}

/// Builder for functional types.
///
/// The builder does not check well-formedness of the function, such as that all referenced
/// const / type params are declared.
///
/// **Tip.** You may also use [`FromStr`](core::str::FromStr) implementation to parse
/// functional types.
///
/// # Examples
///
/// Signature for a function summing a slice of numbers:
///
/// ```
/// # use arithmetic_typing::{FnType, TupleLength, ValueType};
/// # use std::iter;
/// let sum_fn_type = FnType::builder()
///     .with_const_params(iter::once(0))
///     .with_arg(ValueType::NUM.repeat(TupleLength::Param(0)))
///     .returning(ValueType::NUM);
/// assert_eq!(
///     sum_fn_type.to_string(),
///     "fn<const N>([Num; N]) -> Num"
/// );
/// ```
///
/// Signature for a slice mapping function:
///
/// ```
/// # use arithmetic_typing::{FnType, TupleLength, ValueType};
/// # use std::iter;
/// // Definition of the mapping arg. Note that the definition uses type params,
/// // but does not declare them (they are bound to the parent function).
/// let map_fn_arg = <FnType>::builder()
///     .with_arg(ValueType::Param(0))
///     .returning(ValueType::Param(1));
///
/// let map_fn_type = <FnType>::builder()
///     .with_const_params(iter::once(0))
///     .with_type_params(0..=1, false)
///     .with_arg(ValueType::Param(0).repeat(TupleLength::Param(0)))
///     .with_arg(map_fn_arg)
///     .returning(ValueType::Param(1).repeat(TupleLength::Param(0)));
/// assert_eq!(
///     map_fn_type.to_string(),
///     "fn<const N; T: ?Lin, U: ?Lin>([T; N], fn(T) -> U) -> [U; N]"
/// );
/// ```
#[derive(Debug)]
pub struct FnTypeBuilder<Lit = Num> {
    args: FnArgs<Lit>,
    type_params: HashMap<usize, TypeParamDescription>,
    const_params: HashSet<usize>,
}

impl<Lit> Default for FnTypeBuilder<Lit> {
    fn default() -> Self {
        Self {
            args: FnArgs::List(Vec::new()),
            type_params: HashMap::new(),
            const_params: HashSet::new(),
        }
    }
}

// TODO: support validation similarly to AST conversions.
impl<Lit: LiteralType> FnTypeBuilder<Lit> {
    /// Adds the const params with the specified `indexes` to the function definition.
    pub fn with_const_params(mut self, indexes: impl Iterator<Item = usize>) -> Self {
        self.const_params.extend(indexes);
        self
    }

    /// Adds the type params with the specified `indexes` to the function definition.
    /// `linear` determines if the type params are linear (i.e., can be used as arguments
    /// in binary ops).
    pub fn with_type_params(mut self, indexes: impl Iterator<Item = usize>, linear: bool) -> Self {
        let description = TypeParamDescription {
            maybe_non_linear: !linear,
        };
        self.type_params.extend(indexes.map(|i| (i, description)));
        self
    }

    /// Adds a new argument to the function definition.
    pub fn with_arg(mut self, arg: impl Into<ValueType<Lit>>) -> Self {
        match &mut self.args {
            FnArgs::List(args) => {
                args.push(arg.into());
            }
            FnArgs::Any => unreachable!(),
        }
        self
    }

    /// Declares the return type of the function and builds it.
    pub fn returning(self, return_type: ValueType<Lit>) -> FnType<Lit> {
        FnType::new(self.args, return_type)
            .with_const_params(self.const_params.into_iter().collect())
            .with_type_params(self.type_params.into_iter().collect())
    }
}

/// Type of function arguments.
#[derive(Debug, Clone, PartialEq)]
pub enum FnArgs<Lit> {
    /// Any arguments are accepted.
    // TODO: allow to parse any args
    Any,
    /// Lists accepted arguments.
    List(Vec<ValueType<Lit>>),
}

impl<Lit: fmt::Display> fmt::Display for FnArgs<Lit> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FnArgs::Any => formatter.write_str("..."),
            FnArgs::List(args) => {
                for (i, arg) in args.iter().enumerate() {
                    fmt::Display::fmt(arg, formatter)?;
                    if i + 1 < args.len() {
                        formatter.write_str(", ")?;
                    }
                }
                Ok(())
            }
        }
    }
}

/// Length of a tuple.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TupleLength {
    /// Dynamic length that can vary at runtime.
    Dynamic,
    /// Wildcard length.
    Any,
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
            Self::Dynamic => formatter.write_str("*"),
            Self::Any => formatter.write_str("_"),
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
pub enum ValueType<Lit = Num> {
    /// Any type.
    Any,
    /// Boolean.
    Bool,
    /// Literal.
    Lit(Lit),
    /// Function.
    Function(Box<FnType<Lit>>),
    /// Tuple.
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

impl<Num: PartialEq> PartialEq for ValueType<Num> {
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

            // FIXME: function equality?
            _ => false,
        }
    }
}

impl<Num: fmt::Display> fmt::Display for ValueType<Num> {
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
                length: TupleLength::Dynamic,
            } => {
                write!(formatter, "[{}]", element)
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

impl<Num> From<FnType<Num>> for ValueType<Num> {
    fn from(fn_type: FnType<Num>) -> Self {
        Self::Function(Box::new(fn_type))
    }
}

macro_rules! impl_from_tuple_for_value_type {
    ($($var:tt : $ty:ident),*) => {
        impl<Num, $($ty : Into<ValueType<Num>>,)*> From<($($ty,)*)> for ValueType<Num> {
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

impl<Num> ValueType<Num> {
    /// Returns a void type (an empty tuple).
    pub const fn void() -> Self {
        Self::Tuple(Vec::new())
    }

    /// Creates a slice type.
    pub fn slice(element: impl Into<ValueType<Num>>, length: TupleLength) -> Self {
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

/// Grammar with support of type annotations.
#[derive(Debug)]
pub struct NumGrammar<T>(PhantomData<T>);

impl<T> Default for NumGrammar<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T> Clone for NumGrammar<T> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T> Copy for NumGrammar<T> {}

impl<T: NumLiteral> ParseLiteral for NumGrammar<T> {
    type Lit = T;

    fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
        <T as NumLiteral>::parse(input)
    }
}

impl<T: NumLiteral> Grammar for NumGrammar<T> {
    type Type = ValueType<Num>;

    fn parse_type(input: InputSpan<'_>) -> NomResult<'_, Self::Type> {
        ValueType::parse(input)
    }
}
