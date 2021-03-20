//! Hindley–Milner type inference for arithmetic expressions parsed
//! by the [`arithmetic-parser`] crate.
//!
//! This crate allows parsing type annotations as a part of a [`Grammar`], and to infer
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
//! use arithmetic_parser::grammars::{NumGrammar, Parse, Typed};
//! use arithmetic_typing::{Annotated, Prelude, TypeEnvironment, ValueType};
//!
//! type Parser = Typed<Annotated<NumGrammar<f32>>>;
//! # fn main() -> anyhow::Result<()> {
//! let code = "sum = |xs| xs.fold(0, |acc, x| acc + x);";
//! let ast = Parser::parse_statements(code)?;
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

use std::{fmt, marker::PhantomData, ops, str::FromStr};

use arithmetic_parser::{
    grammars::{Grammar, ParseLiteral},
    InputSpan, NomResult,
};

pub mod arith;
pub mod ast;
mod env;
mod error;
mod substitutions;
mod type_map;
mod types;

pub use self::{
    env::TypeEnvironment,
    error::{TypeError, TypeErrorKind, TypeResult},
    substitutions::Substitutions,
    type_map::{Assertions, Prelude},
    types::{FnArgs, FnType, FnTypeBuilder, TupleLength, ValueType},
};

// Reexports for the macros.
#[doc(hidden)]
pub mod _reexports {
    pub use anyhow::{anyhow, Error};
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
/// [`TypeArithmetic`]: crate::arith::TypeArithmetic
///
/// # Examples
///
/// ```
/// # use std::{fmt, str::FromStr};
/// use arithmetic_typing::{LiteralType, NumConstraints};
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
/// impl LiteralType for NumOrBytes {
///     type Constraints = NumConstraints;
/// }
/// ```
pub trait LiteralType:
    Clone + PartialEq + fmt::Debug + fmt::Display + FromStr + Send + Sync + 'static
{
    /// Constraints that can be placed on type parameters.
    type Constraints: TypeConstraints;
}

/// Container for constraints that can be placed on type parameters / variables.
///
/// Constraints can be placed on [function](FnType) type params, and can be applied to types
/// in [`TypeArithmetic`] impls. For example, [`NumArithmetic`] places
/// a [linearity constraint](NumConstraints::LIN) on types involved in arithmetic ops.
///
/// # Implementation rules
///
/// Usually, this trait should be implemented with something akin to [`bitflags`].
///
/// [`bitflags`]: https://docs.rs/bitflags/
///
/// - [`Default`] must return a container with no restrictions.
/// - [`BitAndAssign`](ops::BitAndAssign) must perform the union of the provided constraints.
/// - [`Display`](fmt::Display) must display constraints in the form `Foo + Bar + Quux`,
///   where `Foo`, `Bar` and `Quux` are *primitive* constraints (i.e., ones not reduced
///   to a combination of other constraints). The primitive constraints must be represented
///   as identifiers (i.e., consist of alphanumeric chars and start with an alphabetic char
///   or `_`).
/// - [`FromStr`] must parse primitive constraints.
///
/// [`TypeArithmetic`]: crate::arith::TypeArithmetic
/// [`NumArithmetic`]: crate::arith::NumArithmetic
pub trait TypeConstraints:
    Clone
    + Default
    + PartialEq
    + fmt::Debug
    + fmt::Display
    + FromStr
    + for<'op> ops::BitAndAssign<&'op Self>
    + Send
    + Sync
    + 'static
{
    /// Applies these constraints to the provided `ty`pe. Returns an error if the type
    /// contradicts the constraints.
    ///
    /// A typical implementation will use `substitutions` to
    /// [place constraints on type vars](Substitutions::insert_constraint()), e.g.,
    /// by recursively traversing and resolving the provided type.
    fn apply<Lit>(
        &self,
        ty: &ValueType<Lit>,
        substitutions: &mut Substitutions<Lit>,
    ) -> Result<(), TypeErrorKind<Lit>>
    where
        Lit: LiteralType<Constraints = Self>;
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
    ($ty:ident, $name:tt, $constraints:ty) => {
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

        impl $crate::LiteralType for $ty {
            type Constraints = $constraints;
        }
    };
}

/// Generic numeric type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Num;

/// Constraints associated with the [`Num`] literal.
///
/// There is only one supported constraint: [linearity](Self::LIN). Linear types are types
/// that can be used as arguments of arithmetic ops. They are recursively defined as [`Num`]
/// and tuples in which all elements are linear.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NumConstraints {
    is_linear: bool,
}

impl fmt::Display for NumConstraints {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_linear {
            formatter.write_str("Lin")
        } else {
            Ok(())
        }
    }
}

impl FromStr for NumConstraints {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "" => Ok(Self { is_linear: false }),
            "Lin" => Ok(Self { is_linear: true }),
            _ => Err(anyhow::anyhow!("Expected empty string or `Lin`")),
        }
    }
}

impl ops::BitAndAssign<&Self> for NumConstraints {
    #[allow(clippy::suspicious_op_assign_impl)] // "logical or" is intentional
    fn bitand_assign(&mut self, rhs: &Self) {
        self.is_linear = self.is_linear || rhs.is_linear;
    }
}

impl NumConstraints {
    /// Linearity constraint.
    pub const LIN: Self = Self { is_linear: true };
}

impl TypeConstraints for NumConstraints {
    fn apply<Lit>(
        &self,
        ty: &ValueType<Lit>,
        substitutions: &mut Substitutions<Lit>,
    ) -> Result<(), TypeErrorKind<Lit>>
    where
        Lit: LiteralType<Constraints = Self>,
    {
        if !self.is_linear {
            // The default constraint: does nothing.
            return Ok(());
        }

        let resolved_ty = if let ValueType::Var(idx) = ty {
            substitutions.insert_constraint(*idx, self);
            substitutions.fast_resolve(ty)
        } else {
            ty
        };

        match resolved_ty {
            // `Var`s are taken care of previously. `Lit`s are always linear.
            ValueType::Var(_) | ValueType::Lit(_) => Ok(()),

            ValueType::Any | ValueType::Param(_) => unreachable!(),

            ValueType::Bool | ValueType::Function(_) => Err(TypeErrorKind::failed_constraint(
                ty.to_owned(),
                self.to_owned(),
            )),

            ValueType::Tuple(elements) => {
                for element in elements.to_owned() {
                    self.apply(&element, substitutions)?;
                }
                Ok(())
            }
            ValueType::Slice { element, .. } => self.apply(&element.to_owned(), substitutions),
        }
    }
}

impl_singleton_literal_type!(Num, "Num", NumConstraints);

/// Maps a literal value from a certain [`Grammar`] to its type.
///
/// [`Grammar`]: arithmetic_parser::grammars::Grammar
pub trait MapLiteralType<Val> {
    /// Types of literals output by this mapper.
    type Lit: LiteralType;

    /// Gets the type of the provided literal value.
    fn type_of_literal(&self, lit: &Val) -> Self::Lit;
}

/// Grammar with support of type annotations. Works as a decorator.
///
/// # Examples
///
/// ```
/// use arithmetic_parser::grammars::{NumGrammar, Parse, Typed};
/// use arithmetic_typing::Annotated;
///
/// type F32Grammar = Annotated<NumGrammar<f32>>;
///
/// # fn main() -> anyhow::Result<()> {
/// let code = "x: [Num] = (1, 2, 3);";
/// let ast = Typed::<F32Grammar>::parse_statements(code)?;
/// # assert_eq!(ast.statements.len(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Annotated<T, Lit = Num>(PhantomData<(T, Lit)>);

impl<T: ParseLiteral, Lit: LiteralType> ParseLiteral for Annotated<T, Lit> {
    type Lit = T::Lit;

    fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
        <T as ParseLiteral>::parse_literal(input)
    }
}

impl<T: ParseLiteral, Lit: LiteralType> Grammar for Annotated<T, Lit> {
    type Type = ValueType<Lit>;

    fn parse_type(input: InputSpan<'_>) -> NomResult<'_, Self::Type> {
        ValueType::<Lit>::parse(input)
    }
}
