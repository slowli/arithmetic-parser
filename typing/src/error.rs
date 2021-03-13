//! Errors related to type inference.

use std::fmt;

use crate::{substitutions::Substitutions, LiteralType, TupleLength, ValueType};
use arithmetic_parser::{BinaryOp, Spanned, UnsupportedType};

/// Errors that can occur during type inference.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TypeErrorKind<Lit> {
    /// Error trying to unify operands of a binary operation.
    OperandMismatch {
        /// LHS type.
        lhs_ty: ValueType<Lit>,
        /// RHS type.
        rhs_ty: ValueType<Lit>,
        /// Operator.
        op: BinaryOp,
    },

    /// Trying to unify incompatible types. The first type is LHS, the second one is RHS.
    IncompatibleTypes(ValueType<Lit>, ValueType<Lit>),
    /// Incompatible tuple lengths. The first length is LHS, the second one is RHS.
    IncompatibleLengths(TupleLength, TupleLength),

    /// Trying to call a non-function type.
    NotCallable(ValueType<Lit>),

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
    RecursiveType(ValueType<Lit>),

    /// Non-linear type encountered where linearity is required.
    NonLinearType(ValueType<Lit>),

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

impl<Lit: fmt::Display> fmt::Display for TypeErrorKind<Lit> {
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

            Self::NonLinearType(ty) => write!(formatter, "Non-linear type: {}", ty),

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

impl<Lit: LiteralType> std::error::Error for TypeErrorKind<Lit> {}

impl<Lit: LiteralType> TypeErrorKind<Lit> {
    /// Creates an error for an lvalue type not supported by the interpreter.
    pub fn unsupported<T: Into<UnsupportedType>>(ty: T) -> Self {
        Self::Unsupported(ty.into())
    }

    pub(crate) fn into_op_mismatch(
        self,
        substitutions: &Substitutions<Lit>,
        lhs_ty: &ValueType<Lit>,
        rhs_ty: &ValueType<Lit>,
        op: BinaryOp,
    ) -> Self {
        match self {
            TypeErrorKind::IncompatibleLengths(..) | TypeErrorKind::IncompatibleTypes(..) => {
                TypeErrorKind::OperandMismatch {
                    lhs_ty: substitutions.sanitize_type(None, lhs_ty),
                    rhs_ty: substitutions.sanitize_type(None, rhs_ty),
                    op,
                }
            }
            err => err,
        }
    }
}

/// Type error together with the corresponding code span.
#[derive(Debug, Clone)]
pub struct TypeError<'a, Lit> {
    inner: Spanned<'a, TypeErrorKind<Lit>>,
}

impl<Lit: fmt::Display> fmt::Display for TypeError<'_, Lit> {
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

impl<Lit: LiteralType> std::error::Error for TypeError<'_, Lit> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.kind())
    }
}

impl<'a, Lit> TypeError<'a, Lit> {
    pub(crate) fn new(inner: Spanned<'a, TypeErrorKind<Lit>>) -> Self {
        Self { inner }
    }

    /// Gets the kind of this error.
    pub fn kind(&self) -> &TypeErrorKind<Lit> {
        &self.inner.extra
    }

    /// Gets the code span of this error.
    pub fn span(&self) -> Spanned<'a> {
        self.inner.with_no_extra()
    }
}
