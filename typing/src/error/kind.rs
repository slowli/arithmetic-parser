//! `ErrorKind` and tightly related types.

use std::{collections::HashSet, fmt};

use crate::{
    arith::Constraint, ast::AstConversionError, error::ErrorLocation, PrimitiveType, TupleIndex,
    TupleLen, Type,
};
use arithmetic_parser::UnsupportedType;

/// Context in which a tuple is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TupleContext {
    /// Generic tuple use: assignment, destructuring, or creating a tuple from elements.
    Generic,
    /// The tuple represents function arguments.
    FnArgs,
}

impl TupleContext {
    pub(crate) fn element(self, index: usize) -> ErrorLocation {
        let index = TupleIndex::Start(index);
        match self {
            Self::Generic => ErrorLocation::TupleElement(Some(index)),
            Self::FnArgs => ErrorLocation::FnArg(Some(index)),
        }
    }

    pub(crate) fn end_element(self, index: usize) -> ErrorLocation {
        let index = TupleIndex::End(index);
        match self {
            Self::Generic => ErrorLocation::TupleElement(Some(index)),
            Self::FnArgs => ErrorLocation::FnArg(Some(index)),
        }
    }
}

/// Kinds of errors that can occur during type inference.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ErrorKind<Prim: PrimitiveType> {
    /// Trying to unify incompatible types. The first type is LHS, the second one is RHS.
    TypeMismatch(Type<Prim>, Type<Prim>),
    /// Incompatible tuple lengths.
    TupleLenMismatch {
        /// Length of the LHS. This is the length determined by type annotations
        /// for assignments and the number of actually supplied args in function calls.
        lhs: TupleLen,
        /// Length of the RHS. This is usually the actual tuple length in assignments
        /// and the number of expected args in function calls.
        rhs: TupleLen,
        /// Context in which the error has occurred.
        context: TupleContext,
    },
    /// Undefined variable occurrence.
    UndefinedVar(String),
    /// Trying to unify a type with a type containing it.
    RecursiveType(Type<Prim>),

    /// Field name is invalid.
    InvalidFieldName(String),
    /// Value cannot be indexed (i.e., not a tuple).
    CannotIndex,
    /// Unsupported indexing operation. For example, the receiver type is not known,
    /// or it is a tuple with an unknown length, and the type of the element cannot be decided.
    UnsupportedIndex,
    /// Index is out of bounds for the indexed tuple.
    IndexOutOfBounds {
        /// Index.
        index: usize,
        /// Actual tuple length.
        len: TupleLen,
    },

    /// Field set differs between LHS and RHS, which are both concrete objects.
    FieldsMismatch {
        /// Fields in LHS.
        lhs_fields: HashSet<String>,
        /// Fields in RHS.
        rhs_fields: HashSet<String>,
    },

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
        ty: Type<Prim>,
        /// Failing constraint.
        constraint: Box<dyn Constraint<Prim>>,
    },
    /// Length with the static constraint is actually dynamic (contains [`UnknownLen::Dynamic`]).
    ///
    /// [`UnknownLen::Dynamic`]: crate::UnknownLen::Dynamic
    DynamicLen(TupleLen),

    /// Language feature not supported by type inference logic.
    UnsupportedFeature(UnsupportedType),

    /// Type not supported by type inference logic. For example,
    /// a [`TypeArithmetic`] or [`Constraint`] implementations may return this error
    /// if they encounter an unknown [`Type`] variant.
    ///
    /// [`TypeArithmetic`]: crate::arith::TypeArithmetic
    /// [`Constraint`]: crate::arith::Constraint
    UnsupportedType(Type<Prim>),

    /// Unsupported use of type or length params in a function declaration.
    ///
    /// Type or length params are currently not supported in type annotations. Here's an example
    /// of code that triggers this error:
    ///
    /// ```text
    /// identity: (('T,)) -> ('T,) = |x| x;
    /// ```
    UnsupportedParam,

    /// Error while instantiating a type from AST.
    AstConversion(AstConversionError),
}

impl<Prim: PrimitiveType> fmt::Display for ErrorKind<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeMismatch(lhs, rhs) => write!(
                formatter,
                "Type `{}` is not assignable to type `{}`",
                rhs, lhs
            ),
            Self::TupleLenMismatch {
                lhs,
                rhs,
                context: TupleContext::FnArgs,
            } => write!(
                formatter,
                "Function expects {} args, but is called with {} args",
                lhs, rhs
            ),
            Self::TupleLenMismatch { lhs, rhs, .. } => write!(
                formatter,
                "Expected a tuple with {} elements, got one with {} elements",
                lhs, rhs
            ),

            Self::UndefinedVar(name) => write!(formatter, "Variable `{}` is not defined", name),

            Self::RecursiveType(ty) => write!(
                formatter,
                "Cannot unify type 'T with a type containing it: {}",
                ty
            ),

            Self::InvalidFieldName(name) => {
                write!(formatter, "`{}` is not a valid field name", name)
            }
            Self::CannotIndex => formatter.write_str("Value cannot be indexed"),
            Self::UnsupportedIndex => formatter.write_str("Unsupported indexing operation"),
            Self::IndexOutOfBounds { index, len } => write!(
                formatter,
                "Attempting to get element {} from tuple with length {}",
                index, len
            ),

            Self::FieldsMismatch {
                lhs_fields,
                rhs_fields,
            } => write!(
                formatter,
                "Cannot assign object with fields {rhs:?} to object with fields {lhs:?}",
                lhs = lhs_fields,
                rhs = rhs_fields
            ),

            Self::UnresolvedParam => {
                formatter.write_str("Params not instantiated into variables cannot be unified")
            }

            Self::FailedConstraint { ty, constraint } => {
                write!(formatter, "Type `{}` fails constraint `{}`", ty, constraint)
            }
            Self::DynamicLen(len) => {
                write!(formatter, "Length `{}` is required to be static", len)
            }

            Self::UnsupportedFeature(ty) => write!(formatter, "Unsupported {}", ty),
            Self::UnsupportedType(ty) => write!(formatter, "Unsupported type: {}", ty),
            Self::UnsupportedParam => {
                formatter.write_str("Params in declared function types are not supported yet")
            }

            Self::AstConversion(err) => write!(
                formatter,
                "Error instantiating type from annotation: {}",
                err
            ),
        }
    }
}

impl<Prim: PrimitiveType> std::error::Error for ErrorKind<Prim> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AstConversion(err) => Some(err),
            _ => None,
        }
    }
}

impl<Prim: PrimitiveType> ErrorKind<Prim> {
    /// Creates an error for an lvalue type not supported by the interpreter.
    pub fn unsupported<T: Into<UnsupportedType>>(ty: T) -> Self {
        Self::UnsupportedFeature(ty.into())
    }

    /// Creates a "failed constraint" error.
    pub fn failed_constraint(ty: Type<Prim>, constraint: impl Constraint<Prim> + Clone) -> Self {
        Self::FailedConstraint {
            ty,
            constraint: Box::new(constraint),
        }
    }
}
