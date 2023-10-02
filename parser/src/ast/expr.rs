//! `Expr` and tightly related types.

use core::fmt;

use super::{Block, FnDefinition, ObjectExpr};
use crate::{
    alloc::{Box, Vec},
    grammars::Grammar,
    ops::{BinaryOp, UnaryOp},
    spans::Spanned,
};

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
        name: Box<SpannedExpr<'a, T>>,
        /// Receiver of the call, e.g., `foo` in `foo.bar(x, 5)`.
        receiver: Box<SpannedExpr<'a, T>>,
    },
    /// Method call, e.g., `foo.bar(x, 5)`.
    Method {
        /// Name of the called method, e.g. `bar` in `foo.bar(x, 5)`.
        name: Box<SpannedExpr<'a, T>>,
        /// Receiver of the call, e.g., `foo` in `foo.bar(x, 5)`.
        receiver: Box<SpannedExpr<'a, T>>,
        /// Separator between the receiver and the called method, e.g., `.` in `foo.bar(x, 5)`.
        separator: Spanned<'a>,
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
                name: name.clone(),
                receiver: receiver.clone(),
            },
            Self::Method {
                name,
                receiver,
                separator,
                args,
            } => Self::Method {
                name: name.clone(),
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
