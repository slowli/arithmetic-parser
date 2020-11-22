//! Operation-related types.

use core::fmt;

/// Priority of an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum OpPriority {
    /// Boolean OR (`||`).
    Or,
    /// Boolean AND (`&&`).
    And,
    /// Equality and order comparisons: `==`, `!=`, `>`, `<`, `>=`, `<=`.
    Comparison,
    /// Addition or subtraction: `+` or `-`.
    AddOrSub,
    /// Multiplication or division: `*` or `/`.
    MulOrDiv,
    /// Power (`^`).
    Power,
    /// Numeric or Boolean negation: `!` or unary `-`.
    Negation,
    /// Function or method call.
    Call,
}

impl OpPriority {
    /// Returns the maximum priority.
    pub const fn max_priority() -> Self {
        Self::Call
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
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => formatter.write_str("negation"),
            UnaryOp::Not => formatter.write_str("logical negation"),
        }
    }
}

impl UnaryOp {
    /// Returns a relative priority of this operation.
    pub fn priority(self) -> OpPriority {
        match self {
            Self::Neg | Self::Not => OpPriority::Negation,
        }
    }
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
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    pub fn priority(self) -> OpPriority {
        match self {
            Self::Or => OpPriority::Or,
            Self::And => OpPriority::And,
            Self::Eq | Self::NotEq | Self::Gt | Self::Lt | Self::Le | Self::Ge => {
                OpPriority::Comparison
            }
            Self::Add | Self::Sub => OpPriority::AddOrSub,
            Self::Mul | Self::Div => OpPriority::MulOrDiv,
            Self::Power => OpPriority::Power,
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
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unary(inner) => fmt::Display::fmt(inner, formatter),
            Self::Binary(inner) => fmt::Display::fmt(inner, formatter),
        }
    }
}

impl From<UnaryOp> for Op {
    fn from(value: UnaryOp) -> Self {
        Self::Unary(value)
    }
}

impl From<BinaryOp> for Op {
    fn from(value: BinaryOp) -> Self {
        Self::Binary(value)
    }
}
