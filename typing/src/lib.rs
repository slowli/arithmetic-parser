//! Hindley–Milner type inference for arithmetic expressions parsed
//! by the [`arithmetic-parser`] crate.
//!
//! This crate allows parsing type annotations as a part of a [`Grammar`], and to infer
//! and check types for expressions / statements produced by `arithmetic-parser`.
//! Type inference is *partially* compatible with the interpreter from [`arithmetic-eval`];
//! if the inference algorithm succeeds on a certain expression / statement / block,
//! it will execute successfully, but not all successfully executing items pass type inference.
//! (An exception here is [`Type::Any`], which is specifically designed to circumvent
//! the type system limitations. If `Any` is used too liberally, it can result in code passing
//! type checks, but failing during execution.)
//!
//! # Type system
//!
//! The type system corresponds to types of `Value`s in `arithmetic-eval`:
//!
//! - Primitive types are customizeable via [`PrimitiveType`] impl. In the simplest case,
//!   there can be 2 primitive types: Booleans (`Bool`) and numbers (`Num`),
//!   as ecapsulated in [`Num`].
//! - There is only one container type - a tuple. It can be represented either
//!   in the tuple form, such as `(Num, Bool)`, or as a slice, such as `[Num; 3]`.
//!   As in Rust, all slice elements must have the same type. Unlike Rust, tuple and slice
//!   forms are equivalent; e.g., `[Num; 3]` and `(Num, Num, Num)` are the same type.
//! - Functions are first-class types. Functions can have type and/or const params.
//! - Type params can be constrained. Constraints are expressed via [`TypeConstraints`].
//!   As an example, [`Num`] has an only supported constraint – type *linearity*
//!   (via [`NumConstraints`]).
//! - Const params always specify tuple length.
//!
//! # Inference rules
//!
//! Inference mostly corresponds to [Hindley–Milner typing rules]. It does not require
//! type annotations, but utilizes them if present. Type unification (encapsulated in
//! [`Substitutions`]) is performed at each variable use or assignment. Variable uses include
//! function calls and unary and binary ops; the op behavior is customizable
//! via [`TypeArithmetic`].
//!
//! Whenever possible, the most generic type satisfying the constraints is used. In particular,
//! this means that all type / length variables not resolved at the function definition site become
//! parameters of the function. Symmetrically, each function call instantiates a separate instance
//! of a generic function; type / length params for each call are assigned independently.
//! See the example below for more details.
//!
//! [Hindley–Milner typing rules]: https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system#Typing_rules
//! [`TypeArithmetic`]: crate::arith::TypeArithmetic
//!
//! # Examples
//!
//! ```
//! use arithmetic_parser::grammars::{NumGrammar, Parse, Typed};
//! use arithmetic_typing::{Annotated, Prelude, TypeEnvironment, Type};
//!
//! type Parser = Typed<Annotated<NumGrammar<f32>>>;
//! # fn main() -> anyhow::Result<()> {
//! let code = "sum = |xs| xs.fold(0, |acc, x| acc + x);";
//! let ast = Parser::parse_statements(code)?;
//!
//! let mut env = TypeEnvironment::new();
//! env.insert("fold", Prelude::Fold);
//!
//! // Evaluate `code` to get the inferred `sum` function signature.
//! let output_type = env.process_statements(&ast)?;
//! assert!(output_type.is_void());
//! assert_eq!(env["sum"].to_string(), "([Num; N]) -> Num");
//! # Ok(())
//! # }
//! ```
//!
//! Defining and using generic functions:
//!
//! ```
//! # use arithmetic_parser::grammars::{NumGrammar, Parse, Typed};
//! # use arithmetic_typing::{Annotated, Prelude, TypeEnvironment, Type};
//! # type Parser = Typed<Annotated<NumGrammar<f32>>>;
//! # fn main() -> anyhow::Result<()> {
//! let code = "sum_with = |xs, init| xs.fold(init, |acc, x| acc + x);";
//! let ast = Parser::parse_statements(code)?;
//!
//! let mut env = TypeEnvironment::new();
//! env.insert("fold", Prelude::Fold);
//!
//! let output_type = env.process_statements(&ast)?;
//! assert!(output_type.is_void());
//! assert_eq!(
//!     env["sum_with"].to_string(),
//!     "for<'T: Ops> (['T; N], 'T) -> 'T"
//! );
//! // Note that `sum_with` is parametric by the element of the slice
//! // (for which the linearity constraint is applied based on the arg usage)
//! // *and* by its length.
//!
//! let usage_code = r#"
//!     num_sum: Num = (1, 2, 3).sum_with(0);
//!     tuple_sum: (Num, Num) = ((1, 2), (3, 4)).sum_with((0, 0));
//! "#;
//! let ast = Parser::parse_statements(usage_code)?;
//! // Both lengths and element types differ in these invocations,
//! // but it works fine since they are treated independently.
//! env.process_statements(&ast)?;
//! # Ok(())
//! # }
//! ```
//!
//! [`arithmetic-parser`]: https://crates.io/crates/arithmetic-parser
//! [`Grammar`]: arithmetic_parser::grammars::Grammar
//! [`arithmetic-eval`]: https://crates.io/crates/arithmetic-eval

#![doc(html_root_url = "https://docs.rs/arithmetic-typing/0.2.0")]
#![warn(missing_docs, missing_debug_implementations)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::module_name_repetitions,
    clippy::similar_names // too many false positives because of lhs / rhs
)]

use std::{fmt, marker::PhantomData, str::FromStr};

use arithmetic_parser::{
    grammars::{Grammar, ParseLiteral},
    InputSpan, NomResult,
};

pub mod arith;
pub mod ast;
mod env;
pub mod error;
mod substitutions;
mod type_map;
mod types;
pub mod visit;

pub use self::{
    env::TypeEnvironment,
    error::{Error, ErrorKind},
    substitutions::Substitutions,
    type_map::{Assertions, Prelude},
    types::{
        FnType, FnTypeBuilder, FnWithConstraints, LengthVar, Slice, Tuple, TupleIndex, TupleLen,
        Type, TypeVar, UnknownLen,
    },
};

use self::{
    arith::{ConstraintSet, LinearType, NumConstraints, WithBoolean},
    ast::TypeAst,
};

/// Primitive types in a certain type system.
///
/// More complex types, like [`Type`] and [`FnType`], are defined with a type param
/// which determines the primitive type(s). This type param must implement [`PrimitiveType`].
///
/// [`TypeArithmetic`] has a `PrimitiveType` impl as an associated type, and one of the required
/// operations of this trait is to be able to infer type for literal values from a [`Grammar`].
///
/// # Implementation Requirements
///
/// - [`Display`](fmt::Display) and [`FromStr`] implementations must be consistent; i.e.,
///   `Display` should produce output parseable by `FromStr`. `Display` will be used in
///   `Display` impls for `Type` etc. `FromStr` will be used to read type annotations.
/// - `Display` presentations must be identifiers, such as `Num`.
/// - While not required, a `PrimitiveType` should usually contain a Boolean type and
///   implement [`WithBoolean`]. This allows to reuse [`BoolArithmetic`] and/or [`NumArithmetic`]
///   as building blocks for your [`TypeArithmetic`].
///
/// [`Grammar`]: arithmetic_parser::grammars::Grammar
/// [`TypeArithmetic`]: crate::arith::TypeArithmetic
/// [`BoolArithmetic`]: crate::arith::BoolArithmetic
/// [`NumArithmetic`]: crate::arith::NumArithmetic
///
/// # Examples
///
/// ```
/// # use std::{fmt, str::FromStr};
/// use arithmetic_typing::{arith::NoConstraints, PrimitiveType};
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
/// impl PrimitiveType for NumOrBytes {}
/// ```
pub trait PrimitiveType:
    Clone + PartialEq + fmt::Debug + fmt::Display + FromStr + Send + Sync + 'static
{
    /// Returns well-known constraints for this type. These constraints are used
    /// in standalone parsing of type signatures.
    ///
    /// The default implementation returns an empty set.
    fn well_known_constraints() -> ConstraintSet<Self> {
        ConstraintSet::default()
    }
}

/// Primitive types for numeric arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Num {
    /// Numeric type (e.g., 1).
    Num,
    /// Boolean value (true or false).
    Bool,
}

impl fmt::Display for Num {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Num => "Num",
            Self::Bool => "Bool",
        })
    }
}

impl FromStr for Num {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Num" => Ok(Self::Num),
            "Bool" => Ok(Self::Bool),
            _ => Err(anyhow::anyhow!("Expected `Num` or `Bool`")),
        }
    }
}

impl PrimitiveType for Num {
    fn well_known_constraints() -> ConstraintSet<Self> {
        let mut constraints = ConstraintSet::default();
        constraints.insert(NumConstraints::Lin);
        constraints.insert(NumConstraints::Ops);
        constraints
    }
}

impl WithBoolean for Num {
    const BOOL: Self = Self::Bool;
}

impl LinearType for Num {
    fn is_linear(&self) -> bool {
        matches!(self, Self::Num) // numbers are linear, booleans are not
    }
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
pub struct Annotated<T>(PhantomData<T>);

impl<T: ParseLiteral> ParseLiteral for Annotated<T> {
    type Lit = T::Lit;

    fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
        <T as ParseLiteral>::parse_literal(input)
    }
}

impl<'a, T: ParseLiteral> Grammar<'a> for Annotated<T> {
    type Type = TypeAst<'a>;

    fn parse_type(input: InputSpan<'a>) -> NomResult<'a, Self::Type> {
        use nom::combinator::map;
        map(TypeAst::parse, |ast| ast.extra)(input)
    }
}
