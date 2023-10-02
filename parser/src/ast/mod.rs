//! ASTs for arithmetic expressions and statements.

use core::fmt;

mod expr;
mod lvalue;

pub use self::{
    expr::{Expr, ExprType, SpannedExpr},
    lvalue::{
        Destructure, DestructureRest, Lvalue, LvalueLen, LvalueType, ObjectDestructure,
        ObjectDestructureField, SpannedLvalue,
    },
};
use crate::{
    alloc::{vec, Box, Vec},
    grammars::Grammar,
    spans::Spanned,
};

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
