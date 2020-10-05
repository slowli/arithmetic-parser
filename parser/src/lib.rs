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
//!     GrammarExt, NomResult, InputSpan, Statement, Expr, FnDefinition, LvalueLen,
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
//! let block = F32Grammar::parse_statements(InputSpan::new(PROGRAM)).unwrap();
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
//!     Expr::FnDefinition(FnDefinition { ref args, ref body })
//!         if args.extra.len() == LvalueLen::Exact(2)
//!             && body.statements.is_empty()
//!             && body.return_value.is_some()
//! );
//! ```

#![no_std]
#![warn(missing_docs, missing_debug_implementations)]

extern crate alloc;

pub use crate::{
    parser::{Error, SpannedError},
    traits::{Features, Grammar, GrammarExt},
};

use alloc::{borrow::ToOwned, boxed::Box, format, string::String, vec, vec::Vec};
use core::fmt;

pub mod grammars;
mod helpers;
mod parser;
mod traits;

/// Code span.
pub type InputSpan<'a> = nom_locate::LocatedSpan<&'a str, ()>;
/// Value with an associated code span.
pub type Spanned<'a, T = ()> = LocatedSpan<&'a str, T>;
/// Parsing outcome generalized by the type returned on success.
pub type NomResult<'a, T> = nom::IResult<InputSpan<'a>, T, SpannedError<'a>>;

/// FIXME
#[derive(Debug, Clone, Copy)]
pub struct LocatedSpan<Span, T = ()> {
    offset: usize,
    line: u32,
    column: usize,
    fragment: Span,

    /// Extra information that can be embedded by the user.
    pub extra: T,
}

impl<Span: PartialEq, T> PartialEq for LocatedSpan<Span, T> {
    fn eq(&self, other: &Self) -> bool {
        self.line == other.line && self.offset == other.offset && self.fragment == other.fragment
    }
}

impl<Span, T> LocatedSpan<Span, T> {
    /// The offset represents the position of the fragment relatively to
    /// the input of the parser. It starts at offset 0.
    pub fn location_offset(&self) -> usize {
        self.offset
    }

    /// The line number of the fragment relatively to the input of the parser. It starts at line 1.
    pub fn location_line(&self) -> u32 {
        self.line
    }

    /// The column of the fragment start.
    pub fn get_column(&self) -> usize {
        self.column
    }

    /// The fragment that is spanned. The fragment represents a part of the input of the parser.
    pub fn fragment(&self) -> &Span {
        &self.fragment
    }

    pub(crate) fn map_extra<U>(self, map_fn: impl FnOnce(T) -> U) -> LocatedSpan<Span, U> {
        LocatedSpan {
            offset: self.offset,
            line: self.line,
            column: self.column,
            fragment: self.fragment,
            extra: map_fn(self.extra),
        }
    }

    /// FIXME
    pub fn map_span<U>(self, map_fn: impl FnOnce(Span) -> U) -> LocatedSpan<U, T> {
        LocatedSpan {
            offset: self.offset,
            line: self.line,
            column: self.column,
            fragment: map_fn(self.fragment),
            extra: self.extra,
        }
    }
}

impl<'a, T> LocatedSpan<&'a str, T> {
    pub(crate) fn new(span: InputSpan<'a>, extra: T) -> Self {
        Self {
            offset: span.location_offset(),
            line: span.location_line(),
            column: span.get_column(),
            fragment: *span.fragment(),
            extra,
        }
    }
}

impl<Span: Copy, T> LocatedSpan<Span, T> {
    /// FIXME
    pub fn copy_with_extra<U>(&self, value: U) -> LocatedSpan<Span, U> {
        LocatedSpan {
            offset: self.offset,
            line: self.line,
            column: self.column,
            fragment: self.fragment,
            extra: value,
        }
    }

    /// FIXME
    pub fn with_no_extra(&self) -> LocatedSpan<Span> {
        self.copy_with_extra(())
    }
}

impl<Span: Copy, T: Clone> LocatedSpan<Span, T> {
    pub(crate) fn with_mapped_span<U>(&self, map_fn: impl FnOnce(Span) -> U) -> LocatedSpan<U, T> {
        LocatedSpan {
            offset: self.offset,
            line: self.line,
            column: self.column,
            fragment: map_fn(self.fragment),
            extra: self.extra.clone(),
        }
    }
}

impl<'a, T> From<nom_locate::LocatedSpan<&'a str, T>> for LocatedSpan<&'a str, T> {
    fn from(value: nom_locate::LocatedSpan<&'a str, T>) -> Self {
        Self {
            offset: value.location_offset(),
            line: value.location_line(),
            column: value.get_column(),
            fragment: *value.fragment(),
            extra: value.extra,
        }
    }
}

/// FIXME
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Code<'a> {
    /// FIXME
    Str(&'a str),
    /// FIXME
    Stripped(usize),
}

impl Code<'_> {
    /// FIXME
    pub fn strip(self) -> Code<'static> {
        match self {
            Self::Str(string) => Code::Stripped(string.len()),
            Self::Stripped(len) => Code::Stripped(len),
        }
    }
}

impl<'a> From<&'a str> for Code<'a> {
    fn from(value: &'a str) -> Self {
        Code::Str(value)
    }
}

/// Value with an optional associated code span.
pub type MaybeSpanned<'a, T = ()> = LocatedSpan<Code<'a>, T>;

impl<T> MaybeSpanned<'_, T> {
    /// FIXME
    pub fn code_or_location(&self, default_name: &str) -> String {
        match self.fragment {
            Code::Str(code) => code.to_owned(),
            Code::Stripped(_) => format!("{} at {}:{}", default_name, self.line, self.column),
        }
    }
}

impl<'a, T> From<Spanned<'a, T>> for MaybeSpanned<'a, T> {
    fn from(value: Spanned<'a, T>) -> Self {
        value.map_span(Code::from)
    }
}

/// FIXME
pub trait StripCode {
    /// FIXME
    type Stripped: 'static;

    /// FIXME
    fn strip_code(&self) -> Self::Stripped;
}

impl<T: Clone + 'static> StripCode for MaybeSpanned<'_, T> {
    type Stripped = MaybeSpanned<'static, T>;

    fn strip_code(&self) -> Self::Stripped {
        self.with_mapped_span(|code| code.strip())
    }
}

/// Parsing context.
// TODO: Add more fine-grained contexts.
#[derive(Debug, Clone, Copy, PartialEq)]
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
        use self::Context::*;
        match s {
            "var" => Var,
            "fn" => Fun,
            "expr" => Expr,
            _ => unreachable!(),
        }
    }

    fn to_str(self) -> &'static str {
        use self::Context::*;
        match self {
            Var => "var",
            Fun => "fn",
            Expr => "expr",
        }
    }
}

/// Arithmetic expression with an abstract types for type hints and literals.
#[derive(Debug)]
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
        use self::Expr::*;

        match (self, other) {
            (Variable, Variable) => true,
            (Literal(this), Literal(that)) => this == that,
            (FnDefinition(this), FnDefinition(that)) => this == that,
            (Tuple(this), Tuple(that)) => this == that,
            (Block(this), Block(that)) => this == that,

            (
                Function { name, args },
                Function {
                    name: that_name,
                    args: that_args,
                },
            ) => name == that_name && args == that_args,

            (
                Method {
                    name,
                    receiver,
                    args,
                },
                Method {
                    name: that_name,
                    receiver: that_receiver,
                    args: that_args,
                },
            ) => name == that_name && receiver == that_receiver && args == that_args,

            (
                Unary { op, inner },
                Unary {
                    op: that_op,
                    inner: that_inner,
                },
            ) => op == that_op && inner == that_inner,

            (
                Binary { lhs, op, rhs },
                Binary {
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

/// Unary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Power => true,
            _ => false,
        }
    }

    /// Checks if this operation is a comparison.
    pub fn is_comparison(self) -> bool {
        match self {
            Self::Eq | Self::NotEq | Self::Gt | Self::Lt | Self::Le | Self::Ge => true,
            _ => false,
        }
    }

    /// Checks if this operation is an order comparison.
    pub fn is_order_comparison(self) -> bool {
        match self {
            Self::Gt | Self::Lt | Self::Le | Self::Ge => true,
            _ => false,
        }
    }
}

/// Generic operation, either unary or binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
pub enum Lvalue<'a, T> {
    /// Simple variable, e.g., `x`.
    Variable {
        /// Type annotation of the value.
        ty: Option<Spanned<'a, T>>,
    },
    /// Tuple destructuring, e.g., `(x, y)`.
    Tuple(Destructure<'a, T>),
}

/// `Lvalue` with the associated code span.
pub type SpannedLvalue<'a, T> = Spanned<'a, Lvalue<'a, T>>;

/// Statement: an expression or a variable assignment.
#[derive(Debug)]
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
        use self::Statement::*;

        match (self, other) {
            (Expr(this), Expr(that)) => this == that,

            (
                Assignment { lhs, rhs },
                Assignment {
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

/// Block of statements.
///
/// A block may end with a return expression, e.g., `{ x = 1; x }`.
#[derive(Debug)]
pub struct Block<'a, T>
where
    T: Grammar,
{
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
pub struct FnDefinition<'a, T>
where
    T: Grammar,
{
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
