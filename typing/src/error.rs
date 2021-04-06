//! Errors related to type inference.

use std::fmt;

use crate::{PrimitiveType, TupleLen, ValueType};
use arithmetic_parser::{Spanned, UnsupportedType};

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
    /// Trying to unify incompatible types. The first type is LHS, the second one is RHS.
    TypeMismatch(ValueType<Prim>, ValueType<Prim>),
    /// Incompatible tuple lengths.
    TupleLenMismatch {
        /// Length of the LHS. This is the length determined by type annotations
        /// for assignments and the number of actually supplied args in function calls.
        lhs: TupleLen,
        /// Length of the RHS. This is usually the actual tuple length in assignments
        /// and the number of expected args in function calls.
        rhs: TupleLen,
        /// Context in which the error has occurred.
        context: TupleLenMismatchContext,
    },
    /// Undefined variable occurrence.
    UndefinedVar(String),
    /// Trying to unify a type with a type containing it.
    RecursiveType(ValueType<Prim>),

    /// Mention of a bounded type or length variable in a type supplied
    /// to [`Substitutions::unify()`].
    ///
    /// Bounded variables are instantiated into free vars automatically during
    /// type inference, so this error
    /// can only occur with types manually supplied to `Substitutions::unify()`.
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

    /// Language construct not supported by type inference logic.
    Unsupported(UnsupportedType),

    /// Type not supported by type inference logic. For example,
    /// a [`TypeArithmetic`] or [`TypeConstraints`] implementations may return this error
    /// if they encounter an unknown [`ValueType`] variant.
    ///
    /// [`TypeArithmetic`]: crate::arith::TypeArithmetic
    /// [`TypeConstraints`]: crate::arith::TypeConstraints
    UnsupportedType(ValueType<Prim>),

    /// Unsupported use of type or length params in a function declaration.
    ///
    /// Type or length params are currently not supported in type annotations. Here's an example
    /// of code that triggers this error:
    ///
    /// ```text
    /// identity: ('T) -> 'T = |x| x;
    /// ```
    UnsupportedParam,
}

impl<Prim: PrimitiveType> fmt::Display for TypeErrorKind<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeMismatch(first, second) => write!(
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
            Self::UnsupportedType(ty) => write!(formatter, "Unsupported type: {}", ty),
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
