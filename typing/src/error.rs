//! Errors related to type inference.

use std::{fmt, ops};

use crate::{
    arith::{BinaryOpContext, UnaryOpContext},
    PrimitiveType, TupleLen, ValueType,
};
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
    /// Length with the static constraint is actually dynamic (contains [`UnknownLen::Dynamic`]).
    ///
    /// [`UnknownLen::Dynamic`]: crate::UnknownLen::Dynamic
    DynamicLen(TupleLen),

    /// Language construct not supported by type inference logic.
    // TODO: rename to `UnsupportedFeature`?
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
            Self::TypeMismatch(lhs, rhs) => write!(
                formatter,
                "Type `{}` is not assignable to type `{}`",
                rhs, lhs
            ),
            Self::TupleLenMismatch {
                lhs,
                rhs,
                context: TupleLenMismatchContext::FnArgs,
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
                "Trying to unify a type `T` with a type containing it: {}",
                ty
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
}

/// Type error together with the corresponding code span.
#[derive(Debug, Clone)]
pub struct TypeError<'a, Prim: PrimitiveType> {
    inner: Spanned<'a, TypeErrorKind<Prim>>,
    context: ErrorContext<Prim>,
    location: Vec<ErrorLocation>,
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
    pub(crate) fn lvalue(kind: TypeErrorKind<Prim>, span: &Spanned<'a, ValueType<Prim>>) -> Self {
        Self {
            inner: span.copy_with_extra(kind),
            context: ErrorContext::Lvalue(span.extra.clone()),
            location: vec![],
        }
    }

    pub(crate) fn unsupported<T>(
        unsupported: impl Into<UnsupportedType>,
        span: &Spanned<'a, T>,
    ) -> Self {
        let kind = TypeErrorKind::unsupported(unsupported);
        Self {
            inner: span.copy_with_extra(kind),
            context: ErrorContext::None,
            location: vec![],
        }
    }

    pub(crate) fn undefined_var<T>(span: &Spanned<'a, T>) -> Self {
        let ident = (*span.fragment()).to_owned();
        Self {
            inner: span.copy_with_extra(TypeErrorKind::UndefinedVar(ident)),
            context: ErrorContext::None,
            location: vec![],
        }
    }

    /// Gets the kind of this error.
    pub fn kind(&self) -> &TypeErrorKind<Prim> {
        &self.inner.extra
    }

    /// Gets the code span of this error.
    pub fn span(&self) -> Spanned<'a> {
        self.inner.with_no_extra()
    }

    /// Gets the location of this error relative to the spanned type.
    pub fn location(&self) -> &[ErrorLocation] {
        &self.location
    }
}

/// List of [`TypeError`]s.
#[derive(Debug, Clone)]
pub struct TypeErrors<'a, Prim: PrimitiveType> {
    inner: Vec<TypeError<'a, Prim>>,
}

impl<'a, Prim: PrimitiveType> TypeErrors<'a, Prim> {
    pub(crate) fn new() -> Self {
        Self { inner: vec![] }
    }

    pub(crate) fn push(&mut self, err: TypeError<'a, Prim>) {
        self.inner.push(err);
    }

    pub(crate) fn extend(&mut self, errors: impl Iterator<Item = TypeError<'a, Prim>>) {
        self.inner.extend(errors);
    }

    /// Returns the number of errors in this list.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Checks if this list is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterates over errors contained in this list.
    pub fn iter(&self) -> impl Iterator<Item = &TypeError<'a, Prim>> + '_ {
        self.inner.iter()
    }

    #[cfg(test)]
    pub(crate) fn single(mut self) -> TypeError<'a, Prim> {
        if self.len() == 1 {
            self.inner.pop().unwrap()
        } else {
            panic!("Expected 1 error, got {:?}", self)
        }
    }
}

impl<Prim: PrimitiveType> fmt::Display for TypeErrors<'_, Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, error) in self.inner.iter().enumerate() {
            write!(formatter, "{}", error)?;
            if i + 1 < self.inner.len() {
                formatter.write_str("\n")?;
            }
        }
        Ok(())
    }
}

impl<Prim: PrimitiveType> std::error::Error for TypeErrors<'_, Prim> {}

impl<'a, Prim: PrimitiveType> IntoIterator for TypeErrors<'a, Prim> {
    type Item = TypeError<'a, Prim>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

/// Mutable borrow of [`TypeErrors`] together with a code span.
#[derive(Debug)]
pub struct SpannedTypeErrors<'a, Prim: PrimitiveType> {
    errors: Goat<'a, Vec<TypeErrorPrecursor<Prim>>>,
    current_location: Vec<ErrorLocation>,
}

impl<Prim: PrimitiveType> SpannedTypeErrors<'_, Prim> {
    /// Adds a new `error` into this the error list.
    pub fn push(&mut self, kind: TypeErrorKind<Prim>) {
        self.errors.push(TypeErrorPrecursor {
            kind,
            location: self.current_location.clone(),
        });
    }

    /// Invokes the provided closure and returns `false` if new errors were
    /// added during the closure execution.
    pub fn check(&mut self, check: impl FnOnce(SpannedTypeErrors<'_, Prim>)) -> bool {
        let error_count = self.errors.len();
        check(self.by_ref());
        self.errors.len() == error_count
    }

    /// FIXME
    pub fn by_ref(&mut self) -> SpannedTypeErrors<'_, Prim> {
        SpannedTypeErrors {
            errors: Goat::Borrowed(&mut *self.errors),
            current_location: self.current_location.clone(),
        }
    }

    /// Narrows down the location of the error.
    pub fn with_location(&mut self, location: ErrorLocation) -> SpannedTypeErrors<'_, Prim> {
        let mut current_location = self.current_location.clone();
        current_location.push(location);
        SpannedTypeErrors {
            errors: Goat::Borrowed(&mut *self.errors),
            current_location,
        }
    }

    #[cfg(test)]
    pub(crate) fn into_vec(self) -> Vec<TypeErrorKind<Prim>> {
        let errors = match self.errors {
            Goat::Owned(errors) => errors,
            Goat::Borrowed(_) => panic!("Attempt to call `into_vec` for borrowed errors"),
        };
        errors.into_iter().map(|err| err.kind).collect()
    }
}

impl<Prim: PrimitiveType> SpannedTypeErrors<'static, Prim> {
    pub(crate) fn new() -> Self {
        Self {
            errors: Goat::Owned(vec![]),
            current_location: vec![],
        }
    }

    pub(crate) fn contextualize<'a, T>(
        self,
        context: impl Into<ErrorContext<Prim>>,
        span: &Spanned<'a, T>,
    ) -> impl Iterator<Item = TypeError<'a, Prim>> {
        let errors = match self.errors {
            Goat::Owned(errors) => errors,
            Goat::Borrowed(_) => unreachable!(),
        };

        let context = context.into();
        let span = span.with_no_extra();
        errors
            .into_iter()
            .map(move |item| item.into_error(context.clone(), span))
    }
}

#[derive(Debug)]
enum Goat<'a, T> {
    Owned(T),
    Borrowed(&'a mut T),
}

impl<T> ops::Deref for Goat<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(value) => value,
            Self::Borrowed(mut_ref) => mut_ref,
        }
    }
}

impl<T> ops::DerefMut for Goat<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Owned(value) => value,
            Self::Borrowed(mut_ref) => *mut_ref,
        }
    }
}

#[derive(Debug)]
struct TypeErrorPrecursor<Prim: PrimitiveType> {
    kind: TypeErrorKind<Prim>,
    location: Vec<ErrorLocation>,
}

impl<Prim: PrimitiveType> TypeErrorPrecursor<Prim> {
    // TODO: narrow down span based on `self.location` and `root_span`.
    fn into_error(
        self,
        context: ErrorContext<Prim>,
        root_span: Spanned<'_>,
    ) -> TypeError<'_, Prim> {
        TypeError {
            inner: root_span.copy_with_extra(self.kind),
            context,
            location: self.location,
        }
    }
}

/// Result of inferring type for a certain expression.
pub type TypeResult<'a, Prim> = Result<ValueType<Prim>, TypeError<'a, Prim>>;

/// Fragment of an error location.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ErrorLocation {
    /// Function argument (0-based).
    FnArg(usize),
    /// Function return type.
    FnReturnType,
    /// Tuple element (0-based).
    TupleElement(usize),
    /// Left-hand side of a binary operation.
    Lhs,
    /// Right-hand side of a binary operation.
    Rhs,
}

/// Context of a [`TypeError`] corresponding to the top-level operation that has errored.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ErrorContext<Prim: PrimitiveType> {
    /// No context.
    None,
    /// Processing lvalue (before assignment).
    Lvalue(ValueType<Prim>),
    /// Function call.
    FnCall {
        /// Function definition. Note that this is not necessarily a [`FnType`].
        definition: ValueType<Prim>,
        /// Signature of the call.
        call_signature: ValueType<Prim>,
    },
    /// Assignment.
    Assignment {
        /// Left-hand side of the assignment.
        lhs: ValueType<Prim>,
        /// Right-hand side of the assignment.
        rhs: ValueType<Prim>,
    },
    /// Unary operation.
    UnaryOp(UnaryOpContext<Prim>),
    /// Binary operation.
    BinaryOp(BinaryOpContext<Prim>),
}

impl<Prim: PrimitiveType> From<UnaryOpContext<Prim>> for ErrorContext<Prim> {
    fn from(value: UnaryOpContext<Prim>) -> Self {
        Self::UnaryOp(value)
    }
}

impl<Prim: PrimitiveType> From<BinaryOpContext<Prim>> for ErrorContext<Prim> {
    fn from(value: BinaryOpContext<Prim>) -> Self {
        Self::BinaryOp(value)
    }
}
