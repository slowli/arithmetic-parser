//! Errors related to type inference.

use std::fmt;

use crate::{PrimitiveType, TupleLength, ValueType};
use arithmetic_parser::{BinaryOp, Spanned, UnsupportedType};

/// Context for [`TypeErrorKind::TupleLenMismatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TupleLenMismatchContext {
    /// An error has occurred during assignment.
    Assignment,
    /// An error has occurred when calling a function.
    FnArgs,
}

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
    // FIXME: Rename to `TypeMismatch`?
    IncompatibleTypes(ValueType<Prim>, ValueType<Prim>),
    /// Incompatible tuple lengths.
    // FIXME: Inconsistency: `TupleLenMismatch` vs `TupleLength`
    TupleLenMismatch {
        /// Length of the LHS. This is the length determined by type annotations
        /// for assignments and the number of actually supplied args in function calls.
        lhs: TupleLength,
        /// Length of the RHS. This is usually the actual tuple length in assignments
        /// and the number of expected args in function calls.
        rhs: TupleLength,
        /// Context in which the error has occurred.
        context: TupleLenMismatchContext,
    },
    /// Undefined variable occurrence.
    UndefinedVar(String),
    /// Trying to unify a type with a type containing it.
    RecursiveType(ValueType<Prim>),

    /// Mention of [`ValueType::Param`] or [`TupleLength::Param`] in a type.
    ///
    /// `Param`s are instantiated into `Var`s automatically, so this error
    /// can only occur with types manually supplied to [`Substitutions::unify()`].
    ///
    /// [`Substitutions::unify()`]: crate::Substitutions::unify()
    UnresolvedParam,

    /// Failure when applying constraint to a type.
    FailedConstraint {
        /// Type that fails constraint requirement.
        ty: ValueType<Prim>,
        /// Failing constraint(s).
        constraint: Prim::Constraints,
    },

    /// Language construct not supported by the type inference.
    Unsupported(UnsupportedType),
    /// Unsupported use of type or length params in a function declaration.
    ///
    /// Type or length params are currently not supported in type annotations. Here's an example
    /// of code that triggers this error:
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
            Self::TupleLenMismatch {
                lhs,
                rhs,
                context: TupleLenMismatchContext::FnArgs,
            } => write!(
                formatter,
                "Function expects {} args, but is called with {} args",
                rhs, lhs
            ),
            Self::TupleLenMismatch { lhs, rhs, .. } => write!(
                formatter,
                "Trying to unify incompatible lengths {} and {}",
                lhs, rhs
            ),

            Self::UndefinedVar(name) => write!(formatter, "Variable `{}` is not defined", name),

            Self::RecursiveType(ty) => write!(
                formatter,
                "Trying to unify a type `T` with a type containing it: {}",
                ty
            ),

            Self::UnresolvedParam => {
                formatter.write_str("Params not instantiated into variables cannot be unified")
            }

            Self::FailedConstraint { ty, constraint } => {
                write!(formatter, "Type `{}` fails constraint {}", ty, constraint)
            }

            Self::Unsupported(ty) => write!(formatter, "Unsupported {}", ty),
            Self::UnsupportedParam => {
                formatter.write_str("Params in declared function types are not supported yet")
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
            TypeErrorKind::TupleLenMismatch { .. } | TypeErrorKind::IncompatibleTypes(..) => {
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
