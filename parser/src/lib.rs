//! Parser for arithmetic expressions with flexible definition of literals and support
//! of type annotations.
//!
//! Overall, parsed grammars are similar to Rust syntax,
//! [with a few notable differences](#differences-with-rust).
//!
//! # Supported syntax features
//!
//! - **Variables.** A variable name is defined similar to Rust and other programming languages,
//!   as a sequence of alphanumeric chars and underscores that does not start with a digit.
//! - **Literals.** The parser for literals is user-provided, thus allowing to apply the library
//!   to different domains (e.g., finite group arithmetic).
//! - `//` and `/* .. */` **comments**.
//! - Basic **arithmetic operations**: `+`, `-` (binary and unary), `*`, `/`, `^` (power).
//!   The parser outputs AST with nodes organized according to the operation priority.
//! - **Function calls**: `foo(1.0, x)`.
//! - **Parentheses** which predictably influence operation priority.
//!
//! The parser supports both complete and streaming (incomplete) modes; the latter is useful
//! for REPLs and similar applications.
//!
//! ## Optional syntax features
//!
//! These features can be switched on or off when defining a [`Parse`](grammars::Parse) impl
//! by declaring the corresponding [`Features`](grammars::Features).
//!
//! - **Tuples.** A tuple is two or more elements separated by commas, such as `(x, y)`
//!   or `(1, 2 * x)`. Tuples are parsed both as lvalues and rvalues.
//! - **Tuple destructuring.** Using a tuple as an lvalue, for example, `(x, y, z) = foo`.
//!   The "rest" syntax is also supported, either named or unnamed: `(head, ...tail) = foo`,
//!   `(a, ..., b, c) = foo`.
//! - **Function definitions.** A definition looks like a closure definition in Rust, e.g.,
//!   `|x| x - 10` or `|x, y| { z = max(x, y); (z - x, z - y) }`. A definition may be
//!   assigned to a variable (which is the way to define named functions).
//! - **Destructuring for function args.** Similar to tuple destructuring, it is possible to
//!   destructure and group args in function definitions, for example, `|(x, y), ...zs| { }`.
//! - **Blocks.** A block is several `;`-delimited statements enclosed in `{}` braces,
//!   e.g, `{ z = max(x, y); (z - x, z - y) }`. The blocks can be used in all contexts
//!   instead of a simple expression; for example, `min({ z = 5; z - 1 }, 3)`.
//! - **Objects.** Object is a mapping of string fields to values. Objects are defined via
//!   *object expressions*, which look similar to struct initialization in Rust or object
//!   initialization in JavaScript; for example, `#{ x: 1, y }`. (Note the `#` char at the start
//!   of the block; it is used to distinguish object expressions from blocks.)
//! - **Methods.** Method call is a function call separated from the receiver with a `.` char;
//!   for example, `foo.bar(2, x)`.
//! - **Type annotations.** A type annotation in the form `var: Type` can be present
//!   in the lvalues or in the function argument definitions. The parser for type annotations
//!   is user-defined.
//! - **Boolean operations**: `==`, `!=`, `&&`, `||`, `!`.
//! - **Order comparisons,** that is, `>`, `<`, `>=`, and `<=` boolean ops.
//!
//! ## Differences with Rust
//!
//! *(within shared syntax constructs; of course, Rust is much more expressive)*
//!
//! - No keyword for assigning a variable (i.e., no `let` / `let mut`). There are no
//!   keywords in general.
//! - Functions are only defined via the closure syntax.
//! - There is "rest" destructuting for tuples and function arguments.
//! - Type annotations are placed within tuple elements, for example, `(x: Num, _) = y`.
//! - Object expressions are enclosed in `#{ ... }`, similarly to [Rhai](https://rhai.rs/).
//!
//! # Crate features
//!
//! - `std`. Enables support of types from `std`, such as the `Error` trait, and propagates
//!   to dependencies.
//! - `num-complex`. Implements [`NumLiteral`](crate::grammars::NumLiteral) for floating-point
//!   complex numbers (`Complex32` and `Complex64`).
//! - `num-bigint`. Implements [`NumLiteral`](crate::grammars::NumLiteral) for `BigInt` and
//!   `BigUint` from the `num-bigint` crate.
//!
//! # Examples
//!
//! Using a grammar for arithmetic on real values.
//!
//! ```
//! # use assert_matches::assert_matches;
//! use arithmetic_parser::{
//!     grammars::{F32Grammar, Parse, Untyped},
//!     NomResult, Statement, Expr, FnDefinition, LvalueLen,
//! };
//!
//! const PROGRAM: &str = r#"
//!     // This is a comment.
//!     x = 1 + 2.5 * 3 + sin(a^3 / b^2 /* another comment */);
//!     // Function declarations have syntax similar to Rust closures.
//!     some_function = |a, b| (a + b, a - b);
//!     other_function = |x| {
//!         r = min(rand(), 0.5);
//!         r * x
//!     };
//!     // Tuples and blocks are supported and have a similar syntax to Rust.
//!     (y, z) = some_function({ x = x - 0.5; x }, x);
//!     other_function(y - z)
//! "#;
//!
//! # fn main() -> anyhow::Result<()> {
//! let block = Untyped::<F32Grammar>::parse_statements(PROGRAM)?;
//! // First statement is an assignment.
//! assert_matches!(
//!     block.statements[0].extra,
//!     Statement::Assignment { ref lhs, .. } if *lhs.fragment() == "x"
//! );
//! // The RHS of the second statement is a function.
//! let some_function = match &block.statements[1].extra {
//!     Statement::Assignment { rhs, .. } => &rhs.extra,
//!     _ => panic!("Unexpected parsing result"),
//! };
//! // This function has a single argument and a single statement in the body.
//! assert_matches!(
//!     some_function,
//!     Expr::FnDefinition(FnDefinition { ref args, ref body, .. })
//!         if args.extra.len() == LvalueLen::Exact(2)
//!             && body.statements.is_empty()
//!             && body.return_value.is_some()
//! );
//! # Ok(())
//! # }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![doc(html_root_url = "https://docs.rs/arithmetic-parser/0.3.0")]
#![warn(missing_docs, missing_debug_implementations)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::module_name_repetitions
)]

// Polyfill for `alloc` types.
mod alloc {
    #[cfg(not(feature = "std"))]
    extern crate alloc;

    #[cfg(not(feature = "std"))]
    pub use alloc::{borrow::ToOwned, boxed::Box, format, string::String, vec, vec::Vec};
    #[cfg(feature = "std")]
    pub use std::{borrow::ToOwned, boxed::Box, format, string::String, vec, vec::Vec};
}

pub use crate::{
    error::{Context, Error, ErrorKind, SpannedError, UnsupportedType},
    ops::{BinaryOp, Op, OpPriority, UnaryOp},
    parser::is_valid_variable_name,
    spans::{
        with_span, CodeFragment, InputSpan, LocatedSpan, MaybeSpanned, NomResult, Spanned,
        StripCode, StripResultExt,
    },
};

use core::fmt;

use crate::{
    alloc::{vec, Box, Vec},
    grammars::Grammar,
};

mod error;
pub mod grammars;
mod ops;
mod parser;
mod spans;

/// Object expression, such as `#{ x, y: x + 2 }`.
#[derive(Debug)]
#[non_exhaustive]
pub struct ObjectExpr<'a, T: Grammar<'a>> {
    /// Fields. Each field is the field name and an optional expression (that is, parts
    /// before and after the colon char `:`, respectively).
    pub fields: Vec<(Spanned<'a>, Option<SpannedExpr<'a, T>>)>,
}

impl<'a, T: Grammar<'a>> Clone for ObjectExpr<'a, T> {
    fn clone(&self) -> Self {
        Self {
            fields: self.fields.clone(),
        }
    }
}

impl<'a, T: Grammar<'a>> PartialEq for ObjectExpr<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}

/// Separators between the method call receiver and the method name.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MethodCallSeparator {
    /// Dot separator, e.g., in `foo.bar(1, 3)`.
    Dot,
    /// Double colon separator, e.g., in `foo::bar(4)`.
    Colon2,
}

/// Arithmetic expression with an abstract types for type annotations and literals.
#[derive(Debug)]
#[non_exhaustive]
pub enum Expr<'a, T: Grammar<'a>> {
    /// Variable use, e.g., `x`.
    Variable,

    /// Literal (semantic depends on `T`).
    Literal(T::Lit),

    /// Function definition, e.g., `|x, y| { x + y }`.
    FnDefinition(FnDefinition<'a, T>),

    /// Type cast, e.g., `x as Bool`.
    TypeCast {
        /// Value being cast, e.g., `x` in `x as Bool`.
        value: Box<SpannedExpr<'a, T>>,
        /// Type annotation for the case, e.g., `Bool` in `x as Bool`.
        ty: Spanned<'a, T::Type>,
    },

    /// Function call, e.g., `foo(x, y)` or `|x| { x + 5 }(3)`.
    Function {
        /// Function value. In the simplest case, this is a variable, but may also be another
        /// kind of expression, such as `|x| { x + 5 }` in `|x| { x + 5 }(3)`.
        name: Box<SpannedExpr<'a, T>>,
        /// Function arguments.
        args: Vec<SpannedExpr<'a, T>>,
    },

    /// Field access, e.g., `foo.bar`.
    FieldAccess {
        /// Name of the called method, e.g. `bar` in `foo.bar`.
        name: Spanned<'a>,
        /// Receiver of the call, e.g., `foo` in `foo.bar(x, 5)`.
        receiver: Box<SpannedExpr<'a, T>>,
    },

    /// Method call, e.g., `foo.bar(x, 5)` or `foo::bar(x, 5)`.
    Method {
        /// Name of the called method, e.g. `bar` in `foo.bar(x, 5)`.
        name: Spanned<'a>,
        /// Receiver of the call, e.g., `foo` in `foo.bar(x, 5)`.
        receiver: Box<SpannedExpr<'a, T>>,
        /// Separator between the receiver and the called method, e.g., `.` in `foo.bar(x, 5)`.
        separator: Spanned<'a, MethodCallSeparator>,
        /// Arguments; e.g., `x, 5` in `foo.bar(x, 5)`.
        args: Vec<SpannedExpr<'a, T>>,
    },

    /// Unary operation, e.g., `-x`.
    Unary {
        /// Operator.
        op: Spanned<'a, UnaryOp>,
        /// Inner expression.
        inner: Box<SpannedExpr<'a, T>>,
    },

    /// Binary operation, e.g., `x + 1`.
    Binary {
        /// LHS of the operation.
        lhs: Box<SpannedExpr<'a, T>>,
        /// Operator.
        op: Spanned<'a, BinaryOp>,
        /// RHS of the operation.
        rhs: Box<SpannedExpr<'a, T>>,
    },

    /// Tuple expression, e.g., `(x, y + z)`.
    Tuple(Vec<SpannedExpr<'a, T>>),

    /// Block expression, e.g., `{ x = 3; x + y }`.
    Block(Block<'a, T>),

    /// Object expression, e.g., `#{ x, y: x + 2 }`.
    Object(ObjectExpr<'a, T>),
}

impl<'a, T: Grammar<'a>> Expr<'a, T> {
    /// Returns LHS of the binary expression. If this is not a binary expression, returns `None`.
    pub fn binary_lhs(&self) -> Option<&SpannedExpr<'a, T>> {
        match self {
            Expr::Binary { ref lhs, .. } => Some(lhs),
            _ => None,
        }
    }

    /// Returns RHS of the binary expression. If this is not a binary expression, returns `None`.
    pub fn binary_rhs(&self) -> Option<&SpannedExpr<'a, T>> {
        match self {
            Expr::Binary { ref rhs, .. } => Some(rhs),
            _ => None,
        }
    }

    /// Returns the type of this expression.
    pub fn ty(&self) -> ExprType {
        match self {
            Self::Variable => ExprType::Variable,
            Self::Literal(_) => ExprType::Literal,
            Self::FnDefinition(_) => ExprType::FnDefinition,
            Self::TypeCast { .. } => ExprType::Cast,
            Self::Tuple(_) => ExprType::Tuple,
            Self::Object(_) => ExprType::Object,
            Self::Block(_) => ExprType::Block,
            Self::Function { .. } => ExprType::Function,
            Self::FieldAccess { .. } => ExprType::FieldAccess,
            Self::Method { .. } => ExprType::Method,
            Self::Unary { .. } => ExprType::Unary,
            Self::Binary { .. } => ExprType::Binary,
        }
    }
}

impl<'a, T: Grammar<'a>> Clone for Expr<'a, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Variable => Self::Variable,
            Self::Literal(lit) => Self::Literal(lit.clone()),
            Self::FnDefinition(function) => Self::FnDefinition(function.clone()),
            Self::TypeCast { value, ty } => Self::TypeCast {
                value: value.clone(),
                ty: ty.clone(),
            },
            Self::Tuple(tuple) => Self::Tuple(tuple.clone()),
            Self::Object(statements) => Self::Object(statements.clone()),
            Self::Block(block) => Self::Block(block.clone()),
            Self::Function { name, args } => Self::Function {
                name: name.clone(),
                args: args.clone(),
            },
            Self::FieldAccess { name, receiver } => Self::FieldAccess {
                name: *name,
                receiver: receiver.clone(),
            },
            Self::Method {
                name,
                receiver,
                separator,
                args,
            } => Self::Method {
                name: *name,
                receiver: receiver.clone(),
                separator: *separator,
                args: args.clone(),
            },
            Self::Unary { op, inner } => Self::Unary {
                op: *op,
                inner: inner.clone(),
            },
            Self::Binary { op, lhs, rhs } => Self::Binary {
                op: *op,
                lhs: lhs.clone(),
                rhs: rhs.clone(),
            },
        }
    }
}

impl<'a, T> PartialEq for Expr<'a, T>
where
    T: Grammar<'a>,
    T::Lit: PartialEq,
    T::Type: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Variable, Self::Variable) => true,
            (Self::Literal(this), Self::Literal(that)) => this == that,
            (Self::FnDefinition(this), Self::FnDefinition(that)) => this == that,

            (
                Self::TypeCast { value, ty },
                Self::TypeCast {
                    value: other_value,
                    ty: other_ty,
                },
            ) => value == other_value && ty == other_ty,

            (Self::Tuple(this), Self::Tuple(that)) => this == that,
            (Self::Object(this), Self::Object(that)) => this == that,
            (Self::Block(this), Self::Block(that)) => this == that,

            (
                Self::Function { name, args },
                Self::Function {
                    name: that_name,
                    args: that_args,
                },
            ) => name == that_name && args == that_args,

            (
                Self::FieldAccess { name, receiver },
                Self::FieldAccess {
                    name: that_name,
                    receiver: that_receiver,
                },
            ) => name == that_name && receiver == that_receiver,

            (
                Self::Method {
                    name,
                    receiver,
                    separator,
                    args,
                },
                Self::Method {
                    name: that_name,
                    receiver: that_receiver,
                    separator: that_separator,
                    args: that_args,
                },
            ) => {
                name == that_name
                    && receiver == that_receiver
                    && args == that_args
                    && separator == that_separator
            }

            (
                Self::Unary { op, inner },
                Self::Unary {
                    op: that_op,
                    inner: that_inner,
                },
            ) => op == that_op && inner == that_inner,

            (
                Self::Binary { lhs, op, rhs },
                Self::Binary {
                    lhs: that_lhs,
                    op: that_op,
                    rhs: that_rhs,
                },
            ) => op == that_op && lhs == that_lhs && rhs == that_rhs,

            _ => false,
        }
    }
}

/// `Expr` with the associated type and code span.
pub type SpannedExpr<'a, T> = Spanned<'a, Expr<'a, T>>;

/// Type of an `Expr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ExprType {
    /// Variable use, e.g., `x`.
    Variable,
    /// Literal (semantic depends on the grammar).
    Literal,
    /// Function definition, e.g., `|x, y| { x + y }`.
    FnDefinition,
    /// Cast, e.g., `x as Bool`.
    Cast,
    /// Function call, e.g., `foo(x, y)` or `|x| { x + 5 }(3)`.
    Function,
    /// Field access, e.g., `foo.bar`.
    FieldAccess,
    /// Method call, e.g., `foo.bar(x, 5)`.
    Method,
    /// Unary operation, e.g., `-x`.
    Unary,
    /// Binary operation, e.g., `x + 1`.
    Binary,
    /// Tuple expression, e.g., `(x, y + z)`.
    Tuple,
    /// Object expression, e.g., `#{ x = 1; y = x + 2; }`.
    Object,
    /// Block expression, e.g., `{ x = 3; x + y }`.
    Block,
}

impl fmt::Display for ExprType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Variable => "variable",
            Self::Literal => "literal",
            Self::FnDefinition => "function definition",
            Self::Cast => "type cast",
            Self::Function => "function call",
            Self::FieldAccess => "field access",
            Self::Method => "method call",
            Self::Unary => "unary operation",
            Self::Binary => "binary operation",
            Self::Tuple => "tuple",
            Self::Object => "object",
            Self::Block => "block",
        })
    }
}

/// Length of an assigned lvalue.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum LvalueLen {
    /// Exact length.
    Exact(usize),
    /// Minimum length.
    AtLeast(usize),
}

impl LvalueLen {
    /// Checks if this length matches the provided length of the rvalue.
    pub fn matches(self, value: usize) -> bool {
        match self {
            Self::Exact(len) => value == len,
            Self::AtLeast(len) => value >= len,
        }
    }
}

impl fmt::Display for LvalueLen {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(len) => write!(formatter, "{}", len),
            Self::AtLeast(len) => write!(formatter, "at least {}", len),
        }
    }
}

impl From<usize> for LvalueLen {
    fn from(value: usize) -> Self {
        Self::Exact(value)
    }
}

/// Tuple destructuring, such as `(a, b, ..., c)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Destructure<'a, T> {
    /// Start part of the destructuring, e.g, `a` and `b` in `(a, b, ..., c)`.
    pub start: Vec<SpannedLvalue<'a, T>>,
    /// Middle part of the destructuring, e.g., `rest` in `(a, b, ...rest, _)`.
    pub middle: Option<Spanned<'a, DestructureRest<'a, T>>>,
    /// End part of the destructuring, e.g., `c` in `(a, b, ..., c)`.
    pub end: Vec<SpannedLvalue<'a, T>>,
}

impl<T> Destructure<'_, T> {
    /// Returns the length of destructured elements.
    pub fn len(&self) -> LvalueLen {
        if self.middle.is_some() {
            LvalueLen::AtLeast(self.start.len() + self.end.len())
        } else {
            LvalueLen::Exact(self.start.len())
        }
    }

    /// Checks if the destructuring is empty.
    pub fn is_empty(&self) -> bool {
        self.start.is_empty()
    }
}

/// Rest syntax, such as `...rest` in `(a, ...rest, b)`.
#[derive(Debug, Clone, PartialEq)]
pub enum DestructureRest<'a, T> {
    /// Unnamed rest syntax, i.e., `...`.
    Unnamed,
    /// Named rest syntax, e.g., `...rest`.
    Named {
        /// Variable span, e.g., `rest`.
        variable: Spanned<'a>,
        /// Type annotation of the value.
        ty: Option<Spanned<'a, T>>,
    },
}

impl<'a, T> DestructureRest<'a, T> {
    /// Tries to convert this rest declaration into an lvalue. Return `None` if the rest declaration
    /// is unnamed.
    pub fn to_lvalue(&self) -> Option<SpannedLvalue<'a, T>> {
        match self {
            Self::Named { variable, .. } => {
                Some(variable.copy_with_extra(Lvalue::Variable { ty: None }))
            }
            Self::Unnamed => None,
        }
    }
}

/// Object destructuring, such as `{ x, y: new_y }`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ObjectDestructure<'a, T> {
    /// Fields mentioned in the destructuring.
    pub fields: Vec<ObjectDestructureField<'a, T>>,
}

/// Single field in [`ObjectDestructure`], such as `x` and `y: new_y` in `{ x, y: new_y }`.
///
/// In addition to the "ordinary" `field: lvalue` syntax for a field with binding,
/// an alternative one is supported: `field -> lvalue`. This makes the case
/// of a field with type annotation easier to recognize (for humans); `field -> lvalue: Type` is
/// arguably more readable than `field: lvalue: Type` (although the latter is still valid syntax).
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectDestructureField<'a, T> {
    /// Field name, such as `xs` in `xs: (x, ...tail)`.
    pub field_name: Spanned<'a>,
    /// Binding for the field, such as `(x, ...tail)` in `xs: (x, ...tail)`.
    pub binding: Option<SpannedLvalue<'a, T>>,
}

/// Assignable value.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Lvalue<'a, T> {
    /// Simple variable, e.g., `x`.
    Variable {
        /// Type annotation of the value.
        ty: Option<Spanned<'a, T>>,
    },
    /// Tuple destructuring, e.g., `(x, y)`.
    Tuple(Destructure<'a, T>),
    /// Object destructuring, e.g., `{ x, y }`.
    Object(ObjectDestructure<'a, T>),
}

impl<T> Lvalue<'_, T> {
    /// Returns type of this lvalue.
    pub fn ty(&self) -> LvalueType {
        match self {
            Self::Variable { .. } => LvalueType::Variable,
            Self::Tuple(_) => LvalueType::Tuple,
            Self::Object(_) => LvalueType::Object,
        }
    }
}

/// [`Lvalue`] with the associated code span.
pub type SpannedLvalue<'a, T> = Spanned<'a, Lvalue<'a, T>>;

/// Type of an [`Lvalue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LvalueType {
    /// Simple variable, e.g., `x`.
    Variable,
    /// Tuple destructuring, e.g., `(x, y)`.
    Tuple,
    /// Object destructuring, e.g., `{ x, y }`.
    Object,
}

impl fmt::Display for LvalueType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Variable => "simple variable",
            Self::Tuple => "tuple destructuring",
            Self::Object => "object destructuring",
        })
    }
}

/// Statement: an expression or a variable assignment.
#[derive(Debug)]
#[non_exhaustive]
pub enum Statement<'a, T: Grammar<'a>> {
    /// Expression, e.g., `x + (1, 2)`.
    Expr(SpannedExpr<'a, T>),

    /// Assigment, e.g., `(x, y) = (5, 8)`.
    Assignment {
        /// LHS of the assignment.
        lhs: SpannedLvalue<'a, T::Type>,
        /// RHS of the assignment.
        rhs: Box<SpannedExpr<'a, T>>,
    },
}

impl<'a, T: Grammar<'a>> Statement<'a, T> {
    /// Returns the type of this statement.
    pub fn ty(&self) -> StatementType {
        match self {
            Self::Expr(_) => StatementType::Expr,
            Self::Assignment { .. } => StatementType::Assignment,
        }
    }
}

impl<'a, T: Grammar<'a>> Clone for Statement<'a, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Expr(expr) => Self::Expr(expr.clone()),
            Self::Assignment { lhs, rhs } => Self::Assignment {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
            },
        }
    }
}

impl<'a, T> PartialEq for Statement<'a, T>
where
    T: Grammar<'a>,
    T::Lit: PartialEq,
    T::Type: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Expr(this), Self::Expr(that)) => this == that,

            (
                Self::Assignment { lhs, rhs },
                Self::Assignment {
                    lhs: that_lhs,
                    rhs: that_rhs,
                },
            ) => lhs == that_lhs && rhs == that_rhs,

            _ => false,
        }
    }
}

/// Statement with the associated code span.
pub type SpannedStatement<'a, T> = Spanned<'a, Statement<'a, T>>;

/// Type of a [`Statement`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StatementType {
    /// Expression, e.g., `x + (1, 2)`.
    Expr,
    /// Assigment, e.g., `(x, y) = (5, 8)`.
    Assignment,
}

impl fmt::Display for StatementType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Expr => "expression",
            Self::Assignment => "variable assignment",
        })
    }
}

/// Block of statements.
///
/// A block may end with a return expression, e.g., `{ x = 1; x }`.
#[derive(Debug)]
#[non_exhaustive]
pub struct Block<'a, T: Grammar<'a>> {
    /// Statements in the block.
    pub statements: Vec<SpannedStatement<'a, T>>,
    /// The last statement in the block which is returned from the block.
    pub return_value: Option<Box<SpannedExpr<'a, T>>>,
}

impl<'a, T: Grammar<'a>> Clone for Block<'a, T> {
    fn clone(&self) -> Self {
        Self {
            statements: self.statements.clone(),
            return_value: self.return_value.clone(),
        }
    }
}

impl<'a, T> PartialEq for Block<'a, T>
where
    T: Grammar<'a>,
    T::Lit: PartialEq,
    T::Type: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.return_value == other.return_value && self.statements == other.statements
    }
}

impl<'a, T: Grammar<'a>> Block<'a, T> {
    /// Creates an empty block.
    pub fn empty() -> Self {
        Self {
            statements: vec![],
            return_value: None,
        }
    }
}

/// Function definition, e.g., `|x, y| x + y`.
///
/// A function definition consists of a list of arguments and the function body.
#[derive(Debug)]
#[non_exhaustive]
pub struct FnDefinition<'a, T: Grammar<'a>> {
    /// Function arguments, e.g., `x, y`.
    pub args: Spanned<'a, Destructure<'a, T::Type>>,
    /// Function body, e.g., `x + y`.
    pub body: Block<'a, T>,
}

impl<'a, T: Grammar<'a>> Clone for FnDefinition<'a, T> {
    fn clone(&self) -> Self {
        Self {
            args: self.args.clone(),
            body: self.body.clone(),
        }
    }
}

impl<'a, T> PartialEq for FnDefinition<'a, T>
where
    T: Grammar<'a>,
    T::Lit: PartialEq,
    T::Type: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.args == other.args && self.body == other.body
    }
}
