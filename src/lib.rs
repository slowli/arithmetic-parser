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
//!     Features, Grammar, GrammarExt, NomResult, Span, Statement, Expr,
//!     FnDefinition,
//! };
//! use nom::number::complete::float;
//!
//! #[derive(Debug)]
//! struct SimpleGrammar;
//!
//! impl Grammar for SimpleGrammar {
//!     type Lit = f32;
//!     // We switch off type hint parsing in `FEATURES`, so can specify
//!     // a void type for type annotations.
//!     type Type = ();
//!
//!     // Support all features, except for type annotations.
//!     const FEATURES: Features = Features {
//!         type_annotations: false,
//!         ..Features::all()
//!     };
//!
//!     /// Parses a literal using a standard `nom` parser.
//!     fn parse_literal(input: Span<'_>) -> NomResult<'_, Self::Lit> {
//!         float(input)
//!     }
//!
//!     /// Since we've disabled type annotations, we may leave type parsing
//!     /// unimplemented.
//!     fn parse_type(input: Span<'_>) -> NomResult<'_, Self::Type> {
//!         unreachable!()
//!     }
//! }
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
//! let statements = SimpleGrammar::parse_statements(Span::new(PROGRAM)).unwrap();
//! // First statement is an assignment.
//! assert_matches!(
//!     statements[0].extra,
//!     Statement::Assignment { ref lhs, .. } if lhs.fragment == "x"
//! );
//! // The RHS of the second statement is a function.
//! let some_function = match &statements[1].extra {
//!     Statement::Assignment { rhs, .. } => &rhs.extra,
//!     _ => panic!("Unexpected parsing result"),
//! };
//! // This function has a single argument and a single statement in the body.
//! assert_matches!(
//!     some_function,
//!     Expr::FnDefinition(FnDefinition { ref args, ref body })
//!         if args.len() == 2 && body.len() == 1
//! );
//! ```

#![warn(missing_docs, missing_debug_implementations)]

pub use crate::{
    parser::{Error, SpannedError},
    traits::{Features, Grammar, GrammarExt},
};

use nom_locate::{LocatedSpan, LocatedSpanEx};

use std::fmt;

mod parser;
mod traits;

/// Code span.
pub type Span<'a> = LocatedSpan<&'a str>;
/// Value with an associated code span.
pub type Spanned<'a, T> = LocatedSpanEx<&'a str, T>;
/// Parsing outcome generalized by the type returned on success.
pub type NomResult<'a, T> = nom::IResult<Span<'a>, T, SpannedError<'a>>;

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
#[derive(Debug, Clone)]
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

    /// Function call, e.g., `foo(x, y)`.
    Function {
        /// Function name.
        name: Span<'a>,
        /// Function arguments.
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
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BinaryOp::Add => formatter.write_str("addition"),
            BinaryOp::Sub => formatter.write_str("subtraction"),
            BinaryOp::Mul => formatter.write_str("multiplication"),
            BinaryOp::Div => formatter.write_str("division"),
            BinaryOp::Power => formatter.write_str("exponentiation"),
            BinaryOp::Eq => formatter.write_str("comparison"),
            BinaryOp::NotEq => formatter.write_str("non-equality comparison"),
            BinaryOp::And => formatter.write_str("AND"),
            BinaryOp::Or => formatter.write_str("OR"),
        }
    }
}

impl BinaryOp {
    /// Returns the string representation of this operation.
    pub fn as_str(self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Power => "^",
            BinaryOp::Eq => "==",
            BinaryOp::NotEq => "!=",
            BinaryOp::And => "&&",
            BinaryOp::Or => "||",
        }
    }

    /// Returns the priority of this operation.
    // TODO: replace with enum?
    pub fn priority(self) -> usize {
        match self {
            BinaryOp::And | BinaryOp::Or => 0,
            BinaryOp::Eq | BinaryOp::NotEq => 1,
            BinaryOp::Add | BinaryOp::Sub => 2,
            BinaryOp::Mul | BinaryOp::Div => 3,
            BinaryOp::Power => 4,
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

/// Assignable value.
#[derive(Debug, Clone, PartialEq)]
pub enum Lvalue<'a, T> {
    /// Simple variable, e.g., `x`.
    Variable {
        /// Type annotation of the value.
        ty: Option<Spanned<'a, T>>,
    },
    /// Tuple destructuring, e.g., `(x, y)`.
    Tuple(Vec<SpannedLvalue<'a, T>>),
}

/// `Lvalue` with the associated code span.
pub type SpannedLvalue<'a, T> = Spanned<'a, Lvalue<'a, T>>;

/// Statement: an expression or a variable assignment.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct Block<'a, T>
where
    T: Grammar,
{
    /// Statements in the block.
    pub statements: Vec<SpannedStatement<'a, T>>,
    /// The last statement in the block which is returned from the block.
    pub return_value: Option<Box<SpannedExpr<'a, T>>>,
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

/// Function definition, e.g., `|x, y| x + y`.
///
/// A function definition consists of a list of arguments and the function body.
#[derive(Debug, Clone)]
pub struct FnDefinition<'a, T>
where
    T: Grammar,
{
    /// Function arguments, e.g., `x, y`.
    pub args: Vec<SpannedLvalue<'a, T::Type>>,
    /// Function body, e.g., `x + y`.
    pub body: Block<'a, T>,
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
