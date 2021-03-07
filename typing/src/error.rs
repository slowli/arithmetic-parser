//! Errors related to type inference.

use std::fmt;

use crate::{substitutions::Substitutions, TupleLength, ValueType};
use arithmetic_parser::{BinaryOp, UnsupportedType};

/// Errors that can occur during type inference.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TypeError {
    /// Trying to unify tuples of different sizes.
    TupleLenMismatch(usize, usize),

    /// Error trying to unify operands of a binary operation.
    OperandMismatch {
        /// LHS type.
        lhs_ty: ValueType,
        /// RHS type.
        rhs_ty: ValueType,
        /// Operator.
        op: BinaryOp,
    },

    /// Trying to unify incompatible types.
    IncompatibleTypes(ValueType, ValueType),
    /// Incompatible tuple lengths.
    IncompatibleLengths(TupleLength, TupleLength),

    /// Trying to call a non-function type.
    NotCallable(ValueType),

    /// Undefined variable occurrence.
    UndefinedVar(String),

    /// Mismatch between the number of args in the function definition and call.
    ArgLenMismatch {
        /// Number of args in function definition.
        expected: usize,
        /// Number of args supplied in the call.
        actual: usize,
    },

    /// Trying to unify a type with a type containing it.
    RecursiveType(ValueType),

    /// Non-linear type.
    NonLinearType(ValueType),

    /// Language construct not supported by the type inference.
    Unsupported(UnsupportedType),
    /// Unsupported use of destructuring in an lvalue or function arguments.
    UnsupportedDestructure,
}

impl fmt::Display for TypeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TupleLenMismatch(first, second) => write!(
                formatter,
                "Tuple with {} elements cannot be unified with a tuple with {} elements",
                first, second
            ),

            Self::OperandMismatch { op, .. } => write!(
                formatter,
                "Operands of {} operation must have the same type",
                op
            ),

            Self::IncompatibleTypes(first, second) => write!(
                formatter,
                "Trying to unify incompatible types `{}` and `{}`",
                first, second
            ),
            Self::IncompatibleLengths(first, second) => write!(
                formatter,
                "Trying to unify incompatible lengths {} and {}",
                first, second
            ),

            Self::NotCallable(ty) => write!(formatter, "Trying to call non-function type: {}", ty),

            Self::UndefinedVar(name) => write!(formatter, "Variable `{}` is not defined", name),

            Self::ArgLenMismatch { expected, actual } => write!(
                formatter,
                "Function expects {} args, but is called with {} args",
                expected, actual
            ),

            Self::RecursiveType(ty) => write!(
                formatter,
                "Trying to unify a type `T` with a type containing it: {}",
                ty
            ),

            Self::NonLinearType(ty) => write!(formatter, "Non-linear type: {}", ty),

            Self::Unsupported(ty) => write!(formatter, "Unsupported {}", ty),
            Self::UnsupportedDestructure => {
                formatter.write_str("Destructuring is not supported yet")
            }
        }
    }
}

impl TypeError {
    /// Creates an error for an lvalue type not supported by the interpreter.
    pub fn unsupported<T: Into<UnsupportedType>>(ty: T) -> Self {
        Self::Unsupported(ty.into())
    }

    pub(crate) fn into_op_mismatch(
        self,
        substitutions: &Substitutions,
        lhs_ty: &ValueType,
        rhs_ty: &ValueType,
        op: BinaryOp,
    ) -> Self {
        match self {
            TypeError::TupleLenMismatch(..) | TypeError::IncompatibleTypes(..) => {
                TypeError::OperandMismatch {
                    lhs_ty: substitutions.sanitize_type(None, lhs_ty),
                    rhs_ty: substitutions.sanitize_type(None, rhs_ty),
                    op,
                }
            }
            err => err,
        }
    }
}
