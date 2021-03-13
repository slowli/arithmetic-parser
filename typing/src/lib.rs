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

use std::{borrow::Cow, collections::BTreeMap, fmt, marker::PhantomData, str::FromStr};

use arithmetic_parser::{
    grammars::{Grammar, NumLiteral, ParseLiteral},
    InputSpan, NomResult,
};

pub mod ast;
mod env;
mod error;
mod substitutions;
mod type_map;

pub use self::{
    env::TypeEnvironment,
    error::{TypeError, TypeErrorKind},
    type_map::{Assertions, Prelude},
};

/// Description of a type parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TypeParamDescription {
    /// Can this type param be non-linear?
    maybe_non_linear: bool,
}

/// Description of a constant parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ConstParamDescription;

/// FIXME
pub trait LiteralType:
    Clone + PartialEq + fmt::Debug + fmt::Display + FromStr + Send + Sync + 'static
{
}

/// FIXME
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Num;

impl fmt::Display for Num {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Num")
    }
}

impl FromStr for Num {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "Num" {
            Ok(Num)
        } else {
            Err(())
        }
    }
}

impl LiteralType for Num {}

/// Functional type.
#[derive(Debug, Clone, PartialEq)]
pub struct FnType<Lit = Num> {
    /// Type of function arguments.
    args: FnArgs<Lit>,
    /// Type of the value returned by the function.
    return_type: ValueType<Lit>,
    /// Indexes of type params associated with this function.
    type_params: BTreeMap<usize, TypeParamDescription>,
    /// Indexes of const params associated with this function.
    const_params: BTreeMap<usize, ConstParamDescription>,
}

impl<Lit: fmt::Display> fmt::Display for FnType<Lit> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("fn")?;

        if self.const_params.len() + self.type_params.len() > 0 {
            formatter.write_str("<")?;

            if !self.const_params.is_empty() {
                formatter.write_str("const ")?;
                for (i, (&var_idx, _)) in self.const_params.iter().enumerate() {
                    formatter.write_str(TupleLength::const_param(var_idx).as_ref())?;
                    if i + 1 < self.const_params.len() {
                        formatter.write_str(", ")?;
                    }
                }

                if !self.type_params.is_empty() {
                    formatter.write_str("; ")?;
                }
            }

            for (i, (&var_idx, description)) in self.type_params.iter().enumerate() {
                formatter.write_str(type_param(var_idx).as_ref())?;
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
    pub(crate) fn new(args: Vec<ValueType<Lit>>, return_type: ValueType<Lit>) -> Self {
        Self {
            args: FnArgs::List(args),
            return_type,
            type_params: BTreeMap::new(),  // filled in later
            const_params: BTreeMap::new(), // filled in later
        }
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
    pub(crate) fn is_linear(&self, var_idx: usize) -> bool {
        !self.type_params[&var_idx].maybe_non_linear
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
///     .with_arg(ValueType::Lit(Num).repeat(TupleLength::Param(0)))
///     .returning(ValueType::Lit(Num));
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
/// let map_fn_arg = FnType::builder()
///     .with_arg(ValueType::Param(0))
///     .returning(ValueType::Param(1));
///
/// let map_fn_type = FnType::builder()
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
    inner: FnType<Lit>,
}

impl<Lit> Default for FnTypeBuilder<Lit> {
    fn default() -> Self {
        Self {
            inner: FnType {
                args: FnArgs::List(vec![]),
                return_type: ValueType::void(),
                type_params: BTreeMap::default(),
                const_params: BTreeMap::default(),
            },
        }
    }
}

// TODO: support validation similarly to AST conversions.
impl<Lit: LiteralType> FnTypeBuilder<Lit> {
    /// Adds the const params with the specified `indexes` to the function definition.
    pub fn with_const_params(mut self, indexes: impl Iterator<Item = usize>) -> Self {
        self.inner
            .const_params
            .extend(indexes.map(|i| (i, ConstParamDescription)));
        self
    }

    /// Adds the type params with the specified `indexes` to the function definition.
    /// `linear` determines if the type params are linear (i.e., can be used as arguments
    /// in binary ops).
    pub fn with_type_params(mut self, indexes: impl Iterator<Item = usize>, linear: bool) -> Self {
        let description = TypeParamDescription {
            maybe_non_linear: !linear,
        };
        self.inner
            .type_params
            .extend(indexes.map(|i| (i, description)));
        self
    }

    /// Adds a new argument to the function definition.
    pub fn with_arg(mut self, arg: impl Into<ValueType<Lit>>) -> Self {
        match &mut self.inner.args {
            FnArgs::List(args) => {
                args.push(arg.into());
            }
            FnArgs::Any => unreachable!(),
        }
        self
    }

    /// Declares the return type of the function and builds it.
    pub fn returning(mut self, return_type: ValueType<Lit>) -> FnType<Lit> {
        self.inner.return_type = return_type;
        self.inner
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

    /// Returns `Some(true)` if this type is known to be a number, `Some(false)` if it's known
    /// not to be a number, and `None` if either case is possible.
    pub(crate) fn is_number(&self) -> Option<bool> {
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
