//! Errors related to type inference.

use std::{fmt, ops};

use crate::{
    arith::{BinaryOpContext, UnaryOpContext},
    ast::{ConversionError, ValueTypeAst},
    visit::VisitMut,
    PrimitiveType, Tuple, TupleLen, ValueType,
};
use arithmetic_parser::{
    grammars::Grammar, Destructure, DestructureRest, Expr, InputSpan, Lvalue, Spanned, SpannedExpr,
    SpannedLvalue, UnsupportedType,
};

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

    /// Language feature not supported by type inference logic.
    UnsupportedFeature(UnsupportedType),

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

    /// Error while instantiating a type from AST.
    Conversion(ConversionError),
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
                "Cannot unify type 'T with a type containing it: {}",
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

            Self::UnsupportedFeature(ty) => write!(formatter, "Unsupported {}", ty),
            Self::UnsupportedType(ty) => write!(formatter, "Unsupported type: {}", ty),
            Self::UnsupportedParam => {
                formatter.write_str("Params in declared function types are not supported yet")
            }

            Self::Conversion(err) => write!(
                formatter,
                "Error instantiating type from annotation: {}",
                err
            ),
        }
    }
}

impl<Prim: PrimitiveType> std::error::Error for TypeErrorKind<Prim> {}

impl<Prim: PrimitiveType> TypeErrorKind<Prim> {
    /// Creates an error for an lvalue type not supported by the interpreter.
    pub fn unsupported<T: Into<UnsupportedType>>(ty: T) -> Self {
        Self::UnsupportedFeature(ty.into())
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
            self.span().get_column(),
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

    pub(crate) fn conversion(kind: ConversionError, span: InputSpan<'a>) -> Self {
        let kind = TypeErrorKind::Conversion(kind);
        Self {
            inner: Spanned::from(span).copy_with_extra(kind),
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

    /// Top-level operation that has errored.
    pub fn context(&self) -> &ErrorContext<Prim> {
        &self.context
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

    pub(crate) fn extend(&mut self, errors: Vec<TypeError<'a, Prim>>) {
        self.inner.extend(errors.into_iter());
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

    /// Post-processes these errors, resolving the contained `ValueType`s using
    /// the provided `type_resolver`.
    pub(crate) fn post_process(&mut self, type_resolver: &mut impl VisitMut<Prim>) {
        for error in &mut self.inner {
            error.context.map_types(type_resolver);
        }
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

/// Error container tied to a particular top-level operation.
#[derive(Debug)]
pub struct OpTypeErrors<'a, Prim: PrimitiveType> {
    errors: Goat<'a, Vec<TypeErrorPrecursor<Prim>>>,
    current_location: Vec<ErrorLocation>,
}

impl<Prim: PrimitiveType> OpTypeErrors<'_, Prim> {
    /// Adds a new `error` into this the error list.
    pub fn push(&mut self, kind: TypeErrorKind<Prim>) {
        self.errors.push(TypeErrorPrecursor {
            kind,
            location: self.current_location.clone(),
        });
    }

    /// Invokes the provided closure and returns `false` if new errors were
    /// added during the closure execution.
    pub fn check(&mut self, check: impl FnOnce(OpTypeErrors<'_, Prim>)) -> bool {
        let error_count = self.errors.len();
        check(self.by_ref());
        self.errors.len() == error_count
    }

    /// Mutably borrows this container allowing to use it multiple times.
    pub fn by_ref(&mut self) -> OpTypeErrors<'_, Prim> {
        OpTypeErrors {
            errors: Goat::Borrowed(&mut *self.errors),
            current_location: self.current_location.clone(),
        }
    }

    /// Narrows down the location of the error.
    pub fn with_location(&mut self, location: ErrorLocation) -> OpTypeErrors<'_, Prim> {
        let mut current_location = self.current_location.clone();
        current_location.push(location);
        OpTypeErrors {
            errors: Goat::Borrowed(&mut *self.errors),
            current_location,
        }
    }

    // Sort of unsafe (the user shouldn't forget to call `pop_location`, hence not public).
    pub(crate) fn push_location(&mut self, location: ErrorLocation) {
        self.current_location.push(location);
    }

    pub(crate) fn pop_location(&mut self) {
        self.current_location.pop();
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

impl<Prim: PrimitiveType> OpTypeErrors<'static, Prim> {
    pub(crate) fn new() -> Self {
        Self {
            errors: Goat::Owned(vec![]),
            current_location: vec![],
        }
    }

    pub(crate) fn contextualize<'a, T: Grammar<'a>>(
        self,
        span: &SpannedExpr<'a, T>,
        context: impl Into<ErrorContext<Prim>>,
    ) -> Vec<TypeError<'a, Prim>> {
        let context = context.into();
        self.do_contextualize(|item| item.into_expr_error(context.clone(), span))
    }

    fn do_contextualize<'a>(
        self,
        map_fn: impl Fn(TypeErrorPrecursor<Prim>) -> TypeError<'a, Prim>,
    ) -> Vec<TypeError<'a, Prim>> {
        let errors = match self.errors {
            Goat::Owned(errors) => errors,
            Goat::Borrowed(_) => unreachable!(),
        };
        errors.into_iter().map(map_fn).collect()
    }

    pub(crate) fn contextualize_assignment<'a>(
        self,
        span: &SpannedLvalue<'a, ValueTypeAst<'a>>,
        context: &ErrorContext<Prim>,
    ) -> Vec<TypeError<'a, Prim>> {
        if self.errors.is_empty() {
            vec![]
        } else {
            self.do_contextualize(|item| item.into_assignment_error(context.clone(), span))
        }
    }

    pub(crate) fn contextualize_destructure<'a>(
        self,
        span: &Spanned<'a, Destructure<'a, ValueTypeAst<'a>>>,
        create_context: impl FnOnce() -> ErrorContext<Prim>,
    ) -> Vec<TypeError<'a, Prim>> {
        if self.errors.is_empty() {
            vec![]
        } else {
            let context = create_context();
            self.do_contextualize(|item| item.into_destructure_error(context.clone(), span))
        }
    }
}

/// Analogue of `Cow` with a mutable ref.
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
    fn into_expr_error<'a, T: Grammar<'a>>(
        self,
        context: ErrorContext<Prim>,
        root_expr: &SpannedExpr<'a, T>,
    ) -> TypeError<'a, Prim> {
        TypeError {
            inner: ErrorLocation::walk_expr(&self.location, root_expr).copy_with_extra(self.kind),
            context,
            location: self.location,
        }
    }

    fn into_assignment_error<'a>(
        self,
        context: ErrorContext<Prim>,
        root_lvalue: &SpannedLvalue<'a, ValueTypeAst<'a>>,
    ) -> TypeError<'a, Prim> {
        TypeError {
            inner: ErrorLocation::walk_lvalue(&self.location, root_lvalue)
                .copy_with_extra(self.kind),
            context,
            location: self.location,
        }
    }

    fn into_destructure_error<'a>(
        self,
        context: ErrorContext<Prim>,
        root_destructure: &Spanned<'a, Destructure<'a, ValueTypeAst<'a>>>,
    ) -> TypeError<'a, Prim> {
        TypeError {
            inner: ErrorLocation::walk_destructure(&self.location, root_destructure)
                .copy_with_extra(self.kind),
            context,
            location: self.location,
        }
    }
}

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
    /// Tuple middle.
    TupleMiddle,
    /// Tuple end element (0-based).
    TupleEnd(usize),
    /// Left-hand side of a binary operation.
    Lhs,
    /// Right-hand side of a binary operation.
    Rhs,
}

impl ErrorLocation {
    /// Walks the provided `expr` and returns the most exact span found in it.
    fn walk_expr<'a, T: Grammar<'a>>(location: &[Self], expr: &SpannedExpr<'a, T>) -> Spanned<'a> {
        Self::walk(location, expr, Self::step_into_expr).with_no_extra()
    }

    fn walk<T: Copy>(mut location: &[Self], init: T, refine: impl Fn(Self, T) -> Option<T>) -> T {
        let mut refined = init;
        while !location.is_empty() {
            if let Some(refinement) = refine(location[0], refined) {
                refined = refinement;
                location = &location[1..];
            } else {
                break;
            }
        }
        refined
    }

    fn step_into_expr<'r, 'a, T: Grammar<'a>>(
        self,
        expr: &'r SpannedExpr<'a, T>,
    ) -> Option<&'r SpannedExpr<'a, T>> {
        match self {
            Self::FnArg(index) => match &expr.extra {
                Expr::Function { args, .. } => Some(&args[index]),
                Expr::Method { receiver, args, .. } => Some(if index == 0 {
                    receiver.as_ref()
                } else {
                    &args[index - 1]
                }),
                _ => None,
            },

            Self::Lhs => {
                if let Expr::Binary { lhs, .. } = &expr.extra {
                    Some(lhs.as_ref())
                } else {
                    None
                }
            }
            Self::Rhs => {
                if let Expr::Binary { rhs, .. } = &expr.extra {
                    Some(rhs.as_ref())
                } else {
                    None
                }
            }

            Self::TupleElement(index) => {
                if let Expr::Tuple(elements) = &expr.extra {
                    Some(&elements[index])
                } else {
                    None
                }
            }

            _ => None,
        }
    }

    fn walk_lvalue<'a>(
        location: &[Self],
        lvalue: &SpannedLvalue<'a, ValueTypeAst<'a>>,
    ) -> Spanned<'a> {
        Self::walk(location, LvalueTree::Lvalue(lvalue), Self::step_into_lvalue)
            .refine_lvalue()
            .with_no_extra()
    }

    fn walk_destructure<'a>(
        location: &[Self],
        destructure: &Spanned<'a, Destructure<'a, ValueTypeAst<'a>>>,
    ) -> Spanned<'a> {
        let destructure = LvalueTree::Destructure(destructure);
        Self::walk(location, destructure, Self::step_into_lvalue)
            .refine_lvalue()
            .with_no_extra()
    }

    fn step_into_lvalue<'r, 'a, T>(
        self,
        lvalue: LvalueTree<'r, 'a, T>,
    ) -> Option<LvalueTree<'r, 'a, T>> {
        match lvalue {
            LvalueTree::Type(ty) => self.step_into_type(&ty.extra),
            LvalueTree::Destructure(destructure) => self.step_into_destructure(&destructure.extra),
            LvalueTree::Lvalue(lvalue) => match &lvalue.extra {
                Lvalue::Tuple(destructure) => self.step_into_destructure(destructure),
                Lvalue::Variable { ty: Some(ty) } => self.step_into_type(&ty.extra),
                _ => None,
            },
        }
    }

    #[allow(clippy::unused_self)] // FIXME
    fn step_into_type<'r, 'a, T>(self, _ty: &'r T) -> Option<LvalueTree<'r, 'a, T>> {
        None
    }

    fn step_into_destructure<'r, 'a, T>(
        self,
        destructure: &'r Destructure<'a, T>,
    ) -> Option<LvalueTree<'r, 'a, T>> {
        match self {
            Self::TupleElement(index) => destructure.start.get(index).map(LvalueTree::Lvalue),
            Self::TupleEnd(index) => destructure.end.get(index).map(LvalueTree::Lvalue),
            Self::TupleMiddle => {
                let middle = &destructure.middle.as_ref()?.extra;
                if let DestructureRest::Named { ty, .. } = middle {
                    ty.as_ref().map(LvalueTree::Type)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Enumeration of all types encountered on the lvalue side of assignments.
#[derive(Debug)]
enum LvalueTree<'r, 'a, T> {
    Lvalue(&'r SpannedLvalue<'a, T>),
    Destructure(&'r Spanned<'a, Destructure<'a, T>>),
    Type(&'r Spanned<'a, T>),
}

impl<T> Clone for LvalueTree<'_, '_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Lvalue(lvalue) => Self::Lvalue(*lvalue),
            Self::Destructure(destructure) => Self::Destructure(*destructure),
            Self::Type(ty) => Self::Type(*ty),
        }
    }
}

impl<T> Copy for LvalueTree<'_, '_, T> {}

impl<'a, T> LvalueTree<'_, 'a, T> {
    fn with_no_extra(self) -> Spanned<'a> {
        match self {
            Self::Lvalue(lvalue) => lvalue.with_no_extra(),
            Self::Destructure(destructure) => destructure.with_no_extra(),
            Self::Type(ty) => ty.with_no_extra(),
        }
    }

    /// Refines the `Lvalue` variant if it is a variable with an annotation.
    fn refine_lvalue(mut self) -> Self {
        if let Self::Lvalue(lvalue) = self {
            if let Lvalue::Variable { ty: Some(ty) } = &lvalue.extra {
                self = Self::Type(ty);
            }
        }
        self
    }
}

/// Context of a [`TypeError`] corresponding to the top-level operation that has errored.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ErrorContext<Prim: PrimitiveType> {
    /// No context.
    None,
    /// Processing lvalue (before assignment).
    Lvalue(ValueType<Prim>),
    /// Function definition.
    FnDefinition {
        /// Types of function arguments.
        args: Tuple<Prim>,
    },
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

impl<Prim: PrimitiveType> ErrorContext<Prim> {
    fn map_types(&mut self, mapper: &mut impl VisitMut<Prim>) {
        match self {
            Self::None => { /* Do nothing. */ }
            Self::Lvalue(lvalue) => mapper.visit_type_mut(lvalue),
            Self::FnDefinition { args } => mapper.visit_tuple_mut(args),
            Self::FnCall {
                definition,
                call_signature,
            } => {
                mapper.visit_type_mut(definition);
                mapper.visit_type_mut(call_signature);
            }
            Self::Assignment { lhs, rhs } | Self::BinaryOp(BinaryOpContext { lhs, rhs, .. }) => {
                mapper.visit_type_mut(lhs);
                mapper.visit_type_mut(rhs);
            }
            Self::UnaryOp(UnaryOpContext { arg, .. }) => {
                mapper.visit_type_mut(arg);
            }
        }
    }
}
