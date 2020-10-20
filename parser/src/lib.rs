//! Parser for arithmetic expressions with flexible definition of literals and support
//! of type annotations.
//!
//! # Examples
//!
//! Using a grammar for arithmetic on real values.
//!
//! ```
//! # use assert_matches::assert_matches;
//! use arithmetic_parser::{
//!     grammars::F32Grammar,
//!     GrammarExt, NomResult, Statement, Expr, FnDefinition, LvalueLen,
//! };
//! use nom::number::complete::float;
//!
//! const PROGRAM: &str = r#"
//!     ## This is a comment.
//!     x = 1 + 2.5 * 3 + sin(a^3 / b^2);
//!     ## Function declarations have syntax similar to Rust closures.
//!     some_function = |a, b| (a + b, a - b);
//!     other_function = |x| {
//!         r = min(rand(), 0.5);
//!         r * x
//!     };
//!     ## Tuples and blocks are supported and have a similar syntax to Rust.
//!     (y, z) = some_function({ x = x - 0.5; x }, x);
//!     other_function(y - z)
//! "#;
//!
//! let block = F32Grammar::parse_statements(PROGRAM).unwrap();
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
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
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
    error::{Error, ErrorKind, SpannedError},
    parser::is_valid_variable_name,
    spans::{
        CodeFragment, InputSpan, LocatedSpan, MaybeSpanned, NomResult, Spanned, StripCode,
        StripResultExt,
    },
    traits::{Features, Grammar, GrammarExt},
};

use core::fmt;

use crate::alloc::{vec, Box, Vec};

mod error;
pub mod grammars;
mod parser;
mod spans;
mod traits;

/// Parsing context.
// TODO: Add more fine-grained contexts.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Context {
    /// Variable name.
    Var,
    /// Function invocation.
    Fun,
    /// Arithmetic expression.
    Expr,
}

impl fmt::Display for Context {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Context::Var => formatter.write_str("variable"),
            Context::Fun => formatter.write_str("function call"),
            Context::Expr => formatter.write_str("arithmetic expression"),
        }
    }
}

impl Context {
    fn new(s: &str) -> Self {
        match s {
            "var" => Self::Var,
            "fn" => Self::Fun,
            "expr" => Self::Expr,
            _ => unreachable!(),
        }
    }

    fn to_str(self) -> &'static str {
        match self {
            Self::Var => "var",
            Self::Fun => "fn",
            Self::Expr => "expr",
        }
    }
}

/// Arithmetic expression with an abstract types for type hints and literals.
#[derive(Debug)]
#[non_exhaustive]
pub enum Expr<'a, T>
where
    T: Grammar,
{
    /// Variable use, e.g., `x`.
    Variable,

    /// Literal (semantic depends on `T`).
    Literal(T::Lit),

    /// Function definition, e.g., `|x, y| { x + y }`.
    FnDefinition(FnDefinition<'a, T>),

    /// Function call, e.g., `foo(x, y)` or `|x| { x + 5 }(3)`.
    Function {
        /// Function value. In the simplest case, this is a variable, but may also be another
        /// kind of expression, such as `|x| { x + 5 }` in `|x| { x + 5 }(3)`.
        name: Box<SpannedExpr<'a, T>>,
        /// Function arguments.
        args: Vec<SpannedExpr<'a, T>>,
    },

    /// Method call, e.g., `foo.bar(x, 5)`.
    Method {
        /// Name of the called method, e.g. `bar` in `foo.bar(x, 5)`.
        name: Spanned<'a>,
        /// Receiver of the call, e.g., `foo` in `foo.bar(x, 5)`.
        receiver: Box<SpannedExpr<'a, T>>,
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
}

impl<T: Grammar> Expr<'_, T> {
    /// Returns LHS of the binary expression. If this is not a binary expression, returns `None`.
    pub fn binary_lhs(&self) -> Option<&SpannedExpr<'_, T>> {
        match self {
            Expr::Binary { ref lhs, .. } => Some(lhs),
            _ => None,
        }
    }

    /// Returns RHS of the binary expression. If this is not a binary expression, returns `None`.
    pub fn binary_rhs(&self) -> Option<&SpannedExpr<'_, T>> {
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
            Self::Tuple(_) => ExprType::Tuple,
            Self::Block(_) => ExprType::Block,
            Self::Function { .. } => ExprType::Function,
            Self::Method { .. } => ExprType::Method,
            Self::Unary { .. } => ExprType::Unary,
            Self::Binary { .. } => ExprType::Binary,
        }
    }
}

impl<T: Grammar> Clone for Expr<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Variable => Self::Variable,
            Self::Literal(lit) => Self::Literal(lit.clone()),
            Self::FnDefinition(function) => Self::FnDefinition(function.clone()),
            Self::Tuple(tuple) => Self::Tuple(tuple.clone()),
            Self::Block(block) => Self::Block(block.clone()),
            Self::Function { name, args } => Self::Function {
                name: name.clone(),
                args: args.clone(),
            },
            Self::Method {
                name,
                receiver,
                args,
            } => Self::Method {
                name: *name,
                receiver: receiver.clone(),
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

impl<T> PartialEq for Expr<'_, T>
where
    T: Grammar,
    T::Lit: PartialEq,
    T::Type: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Variable, Self::Variable) => true,
            (Self::Literal(this), Self::Literal(that)) => this == that,
            (Self::FnDefinition(this), Self::FnDefinition(that)) => this == that,
            (Self::Tuple(this), Self::Tuple(that)) => this == that,
            (Self::Block(this), Self::Block(that)) => this == that,

            (
                Self::Function { name, args },
                Self::Function {
                    name: that_name,
                    args: that_args,
                },
            ) => name == that_name && args == that_args,

            (
                Self::Method {
                    name,
                    receiver,
                    args,
                },
                Self::Method {
                    name: that_name,
                    receiver: that_receiver,
                    args: that_args,
                },
            ) => name == that_name && receiver == that_receiver && args == that_args,

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
    /// Function call, e.g., `foo(x, y)` or `|x| { x + 5 }(3)`.
    Function,
    /// Method call, e.g., `foo.bar(x, 5)`.
    Method,
    /// Unary operation, e.g., `-x`.
    Unary,
    /// Binary operation, e.g., `x + 1`.
    Binary,
    /// Tuple expression, e.g., `(x, y + z)`.
    Tuple,
    /// Block expression, e.g., `{ x = 3; x + y }`.
    Block,
}

impl fmt::Display for ExprType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Variable => "variable",
            Self::Literal => "literal",
            Self::FnDefinition => "function definition",
            Self::Function => "function call",
            Self::Method => "method call",
            Self::Unary => "unary operation",
            Self::Binary => "binary operation",
            Self::Tuple => "tuple",
            Self::Block => "block",
        })
    }
}

/// Unary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnaryOp {
    /// Negation (`-`).
    Neg,
    /// Boolean negation (`!`).
    Not,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            UnaryOp::Neg => formatter.write_str("negation"),
            UnaryOp::Not => formatter.write_str("logical negation"),
        }
    }
}

impl UnaryOp {
    /// Priority of unary operations.
    // TODO: replace with enum?
    pub const PRIORITY: usize = 5;
}

/// Binary arithmetic operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BinaryOp {
    /// Addition (`+`).
    Add,
    /// Subtraction (`-`).
    Sub,
    /// Multiplication (`*`).
    Mul,
    /// Division (`/`).
    Div,
    /// Power (`^`).
    Power,
    /// Equality (`==`).
    Eq,
    /// Non-equality (`!=`).
    NotEq,
    /// Boolean AND (`&&`).
    And,
    /// Boolean OR (`||`).
    Or,
    /// "Greater than" comparison.
    Gt,
    /// "Lesser than" comparison.
    Lt,
    /// "Greater or equal" comparison.
    Ge,
    /// "Lesser or equal" comparison.
    Le,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(match self {
            Self::Add => "addition",
            Self::Sub => "subtraction",
            Self::Mul => "multiplication",
            Self::Div => "division",
            Self::Power => "exponentiation",
            Self::Eq => "equality comparison",
            Self::NotEq => "non-equality comparison",
            Self::And => "AND",
            Self::Or => "OR",
            Self::Gt => "greater comparison",
            Self::Lt => "lesser comparison",
            Self::Ge => "greater-or-equal comparison",
            Self::Le => "lesser-or-equal comparison",
        })
    }
}

impl BinaryOp {
    /// Returns the string representation of this operation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Power => "^",
            Self::Eq => "==",
            Self::NotEq => "!=",
            Self::And => "&&",
            Self::Or => "||",
            Self::Gt => ">",
            Self::Lt => "<",
            Self::Ge => ">=",
            Self::Le => "<=",
        }
    }

    /// Returns the priority of this operation.
    // TODO: replace with enum?
    pub fn priority(self) -> usize {
        match self {
            Self::And | Self::Or => 0,
            Self::Eq | Self::NotEq | Self::Gt | Self::Lt | Self::Le | Self::Ge => 1,
            Self::Add | Self::Sub => 2,
            Self::Mul | Self::Div => 3,
            Self::Power => 4,
        }
    }

    /// Checks if this operation is arithmetic.
    pub fn is_arithmetic(self) -> bool {
        matches!(
            self,
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Power
        )
    }

    /// Checks if this operation is a comparison.
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            Self::Eq | Self::NotEq | Self::Gt | Self::Lt | Self::Le | Self::Ge
        )
    }

    /// Checks if this operation is an order comparison.
    pub fn is_order_comparison(self) -> bool {
        matches!(self, Self::Gt | Self::Lt | Self::Le | Self::Ge)
    }
}

/// Generic operation, either unary or binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Op {
    /// Unary operation.
    Unary(UnaryOp),
    /// Binary operation.
    Binary(BinaryOp),
}

impl fmt::Display for Op {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::Unary(inner) => fmt::Display::fmt(inner, formatter),
            Op::Binary(inner) => fmt::Display::fmt(inner, formatter),
        }
    }
}

impl From<UnaryOp> for Op {
    fn from(value: UnaryOp) -> Self {
        Op::Unary(value)
    }
}

impl From<BinaryOp> for Op {
    fn from(value: BinaryOp) -> Self {
        Op::Binary(value)
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
            _ => None,
        }
    }
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
}

impl<T> Lvalue<'_, T> {
    /// Returns type of this lvalue.
    pub fn ty(&self) -> LvalueType {
        match self {
            Self::Variable { .. } => LvalueType::Variable,
            Self::Tuple(_) => LvalueType::Tuple,
        }
    }
}

/// `Lvalue` with the associated code span.
pub type SpannedLvalue<'a, T> = Spanned<'a, Lvalue<'a, T>>;

/// Type of an `Lvalue`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LvalueType {
    /// Simple variable, e.g., `x`.
    Variable,
    /// Tuple destructuring, e.g., `(x, y)`.
    Tuple,
}

impl fmt::Display for LvalueType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Variable => "simple variable",
            Self::Tuple => "tuple destructuring",
        })
    }
}

/// Statement: an expression or a variable assignment.
#[derive(Debug)]
#[non_exhaustive]
pub enum Statement<'a, T>
where
    T: Grammar,
{
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

impl<T: Grammar> Statement<'_, T> {
    /// Returns the type of this statement.
    pub fn ty(&self) -> StatementType {
        match self {
            Self::Expr(_) => StatementType::Expr,
            Self::Assignment { .. } => StatementType::Assignment,
        }
    }
}

impl<T: Grammar> Clone for Statement<'_, T> {
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

impl<T> PartialEq for Statement<'_, T>
where
    T: Grammar,
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

/// Type of a `Statement`.
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
pub struct Block<'a, T: Grammar> {
    /// Statements in the block.
    pub statements: Vec<SpannedStatement<'a, T>>,
    /// The last statement in the block which is returned from the block.
    pub return_value: Option<Box<SpannedExpr<'a, T>>>,
}

impl<T: Grammar> Clone for Block<'_, T> {
    fn clone(&self) -> Self {
        Self {
            statements: self.statements.clone(),
            return_value: self.return_value.clone(),
        }
    }
}

impl<T> PartialEq for Block<'_, T>
where
    T: Grammar,
    T::Lit: PartialEq,
    T::Type: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.return_value == other.return_value && self.statements == other.statements
    }
}

impl<T: Grammar> Block<'_, T> {
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
pub struct FnDefinition<'a, T: Grammar> {
    /// Function arguments, e.g., `x, y`.
    pub args: Spanned<'a, Destructure<'a, T::Type>>,
    /// Function body, e.g., `x + y`.
    pub body: Block<'a, T>,
}

impl<T: Grammar> Clone for FnDefinition<'_, T> {
    fn clone(&self) -> Self {
        Self {
            args: self.args.clone(),
            body: self.body.clone(),
        }
    }
}

impl<T> PartialEq for FnDefinition<'_, T>
where
    T: Grammar,
    T::Lit: PartialEq,
    T::Type: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.args == other.args && self.body == other.body
    }
}
