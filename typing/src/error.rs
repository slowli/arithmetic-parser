//! Errors related to type inference.

use std::fmt;

use crate::{PrimitiveType, TupleLength, ValueType};
use arithmetic_parser::{BinaryOp, Spanned, UnsupportedType};

/// Errors that can occur during type inference.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TypeErrorKind<Prim: PrimitiveType> {
    /// Error trying to unify operands of a binary operation.
    OperandMismatch {
        /// LHS type.
        lhs_ty: ValueType<Prim>,
        /// RHS type.
        rhs_ty: ValueType<Prim>,
        /// Operator.
        op: BinaryOp,
    },

    /// Trying to unify incompatible types. The first type is LHS, the second one is RHS.
    IncompatibleTypes(ValueType<Prim>, ValueType<Prim>),
    /// Incompatible tuple lengths. The first length is LHS, the second one is RHS.
    IncompatibleLengths(TupleLength, TupleLength),

    /// Trying to call a non-function type.
    NotCallable(ValueType<Prim>),

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
    RecursiveType(ValueType<Prim>),

    /// Failure when applying constraint to a type.
    FailedConstraint {
        /// Type that fails constraint requirement.
        ty: ValueType<Prim>,
        /// Failing constraint(s).
        constraint: Prim::Constraints,
    },

    /// Language construct not supported by the type inference.
    Unsupported(UnsupportedType),
    /// Unsupported use of destructuring in an lvalue or function arguments.
    ///
    /// Destructuring with a specified middle, such as `(x, ...ys)` are not supported yet.
    UnsupportedDestructure,
    /// Unsupported use of type or const params in function declaration.
    ///
    /// Type or const params are currently not supported in type annotations, such as
    ///
    /// ```text
    /// identity: fn<T>(T) -> T = |x| x;
    /// ```
    UnsupportedParam,
}

impl<Prim: PrimitiveType> fmt::Display for TypeErrorKind<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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

            Self::FailedConstraint { ty, constraint } => {
                write!(formatter, "Type `{}` fails constraint {}", ty, constraint)
            }

            Self::Unsupported(ty) => write!(formatter, "Unsupported {}", ty),
            Self::UnsupportedDestructure => {
                formatter.write_str("Destructuring with middle is not supported yet")
            }
            Self::UnsupportedParam => {
                formatter.write_str("Type params in declared function types is not supported yet")
            }
        }
    }
}

impl<Prim: PrimitiveType> std::error::Error for TypeErrorKind<Prim> {}

impl<Prim: PrimitiveType> TypeErrorKind<Prim> {
    /// Creates an error for an lvalue type not supported by the interpreter.
    pub fn unsupported<T: Into<UnsupportedType>>(ty: T) -> Self {
        Self::Unsupported(ty.into())
    }

    /// Creates a "failed constraint" error.
    pub fn failed_constraint(ty: ValueType<Prim>, constraint: Prim::Constraints) -> Self {
        Self::FailedConstraint { ty, constraint }
    }

    /// Creates an error from this error kind and the specified span.
    pub fn with_span<'a, T>(self, span: &Spanned<'a, T>) -> TypeError<'a, Prim> {
        TypeError {
            inner: span.copy_with_extra(self),
        }
    }

    pub(crate) fn into_op_mismatch(
        self,
        lhs_ty: ValueType<Prim>,
        rhs_ty: ValueType<Prim>,
        op: BinaryOp,
    ) -> Self {
        match self {
            TypeErrorKind::IncompatibleLengths(..) | TypeErrorKind::IncompatibleTypes(..) => {
                TypeErrorKind::OperandMismatch { lhs_ty, rhs_ty, op }
            }
            err => err,
        }
    }
}

/// Type error together with the corresponding code span.
#[derive(Debug, Clone)]
pub struct TypeError<'a, Prim: PrimitiveType> {
    inner: Spanned<'a, TypeErrorKind<Prim>>,
}

impl<Prim: PrimitiveType> fmt::Display for TypeError<'_, Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}:{}: {}",
            self.span().location_line(),
            self.span().location_offset(),
            self.kind()
        )
    }
}

impl<Prim: PrimitiveType> std::error::Error for TypeError<'_, Prim> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.kind())
    }
}

impl<'a, Prim: PrimitiveType> TypeError<'a, Prim> {
    /// Gets the kind of this error.
    pub fn kind(&self) -> &TypeErrorKind<Prim> {
        &self.inner.extra
    }

    /// Gets the code span of this error.
    pub fn span(&self) -> Spanned<'a> {
        self.inner.with_no_extra()
    }
}

/// Result of inferring type for a certain expression.
pub type TypeResult<'a, Prim> = Result<ValueType<Prim>, TypeError<'a, Prim>>;
