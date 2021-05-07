//! Errors related to type inference.

// TODO: split into a couple of modules

use std::{fmt, ops};

use crate::{
    arith::{BinaryOpContext, UnaryOpContext},
    ast::{AstConversionError, SpannedTypeAst, TupleAst, TypeAst},
    visit::VisitMut,
    PrimitiveType, Tuple, TupleIndex, TupleLen, Type,
};
use arithmetic_parser::{
    grammars::Grammar, Destructure, DestructureRest, Expr, Lvalue, Spanned, SpannedExpr,
    SpannedLvalue, UnsupportedType,
};

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
    /// if they encounter an unknown [`Type`] variant.
    ///
    /// [`TypeArithmetic`]: crate::arith::TypeArithmetic
    /// [`TypeConstraints`]: crate::arith::TypeConstraints
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
    pub fn failed_constraint(ty: Type<Prim>, constraint: Prim::Constraints) -> Self {
        Self::FailedConstraint { ty, constraint }
    }
}

/// Type error together with the corresponding code span.
// TODO: implement `StripCode`?
#[derive(Debug, Clone)]
pub struct Error<'a, Prim: PrimitiveType> {
    inner: Spanned<'a, ErrorKind<Prim>>,
    root_span: Spanned<'a>,
    context: ErrorContext<Prim>,
    location: Vec<ErrorLocation>,
}

impl<Prim: PrimitiveType> fmt::Display for Error<'_, Prim> {
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

impl<Prim: PrimitiveType> std::error::Error for Error<'_, Prim> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.kind())
    }
}

impl<'a, Prim: PrimitiveType> Error<'a, Prim> {
    pub(crate) fn unsupported<T>(
        unsupported: impl Into<UnsupportedType>,
        span: &Spanned<'a, T>,
    ) -> Self {
        let kind = ErrorKind::unsupported(unsupported);
        Self {
            inner: span.copy_with_extra(kind),
            root_span: span.with_no_extra(),
            context: ErrorContext::None,
            location: vec![],
        }
    }

    pub(crate) fn undefined_var<T>(span: &Spanned<'a, T>) -> Self {
        let ident = (*span.fragment()).to_owned();
        Self {
            inner: span.copy_with_extra(ErrorKind::UndefinedVar(ident)),
            root_span: span.with_no_extra(),
            context: ErrorContext::None,
            location: vec![],
        }
    }

    pub(crate) fn conversion<T>(kind: AstConversionError, span: &Spanned<'a, T>) -> Self {
        let kind = ErrorKind::AstConversion(kind);
        Self {
            inner: span.copy_with_extra(kind),
            root_span: span.with_no_extra(),
            context: ErrorContext::None,
            location: vec![],
        }
    }

    /// Gets the kind of this error.
    pub fn kind(&self) -> &ErrorKind<Prim> {
        &self.inner.extra
    }

    /// Gets the most specific code span of this error.
    pub fn span(&self) -> Spanned<'a> {
        self.inner.with_no_extra()
    }

    /// Gets the root code span of the failed operation. May coincide with [`Self::span()`].
    pub fn root_span(&self) -> Spanned<'a> {
        self.root_span
    }

    /// Gets the context for an operation that has failed.
    pub fn context(&self) -> &ErrorContext<Prim> {
        &self.context
    }

    /// Gets the location of this error relative to the failed top-level operation.
    /// This can be used for highlighting relevant parts of types in [`Self::context()`].
    pub fn location(&self) -> &[ErrorLocation] {
        &self.location
    }
}

/// List of [`Error`]s.
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{NumGrammar, Parse, Typed};
/// # use arithmetic_typing::{error::Errors, Annotated, Prelude, TypeEnvironment};
/// # use std::collections::HashSet;
/// # type Parser = Typed<Annotated<NumGrammar<f32>>>;
/// # fn main() -> anyhow::Result<()> {
/// let buggy_code = Parser::parse_statements(r#"
///     numbers: ['T; _] = (1, 2, 3);
///     numbers.filter(|x| x, 1)
/// "#)?;
/// let mut env: TypeEnvironment = Prelude::iter().collect();
/// let errors: Errors<_> = env.process_statements(&buggy_code).unwrap_err();
/// assert_eq!(errors.len(), 3);
///
/// let messages: HashSet<_> = errors.iter().map(ToString::to_string).collect();
/// assert!(messages
///     .iter()
///     .any(|msg| msg.contains("Type param `T` is not scoped by function definition")));
/// assert!(messages
///     .contains("3:20: Type `Num` is not assignable to type `Bool`"));
/// assert!(messages
///     .contains("3:5: Function expects 2 args, but is called with 3 args"));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Errors<'a, Prim: PrimitiveType> {
    inner: Vec<Error<'a, Prim>>,
}

impl<'a, Prim: PrimitiveType> Errors<'a, Prim> {
    pub(crate) fn new() -> Self {
        Self { inner: vec![] }
    }

    pub(crate) fn push(&mut self, err: Error<'a, Prim>) {
        self.inner.push(err);
    }

    pub(crate) fn extend(&mut self, errors: Vec<Error<'a, Prim>>) {
        self.inner.extend(errors.into_iter());
    }

    /// Returns the number of errors in this list.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Checks if this list is empty (there are no errors).
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterates over errors contained in this list.
    pub fn iter(&self) -> impl Iterator<Item = &Error<'a, Prim>> + '_ {
        self.inner.iter()
    }

    /// Post-processes these errors, resolving the contained `Type`s using
    /// the provided `type_resolver`.
    pub(crate) fn post_process(&mut self, type_resolver: &mut impl VisitMut<Prim>) {
        for error in &mut self.inner {
            error.context.map_types(type_resolver);
        }
    }

    #[cfg(test)]
    pub(crate) fn single(mut self) -> Error<'a, Prim> {
        if self.len() == 1 {
            self.inner.pop().unwrap()
        } else {
            panic!("Expected 1 error, got {:?}", self);
        }
    }
}

impl<Prim: PrimitiveType> fmt::Display for Errors<'_, Prim> {
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

impl<Prim: PrimitiveType> std::error::Error for Errors<'_, Prim> {}

impl<'a, Prim: PrimitiveType> IntoIterator for Errors<'a, Prim> {
    type Item = Error<'a, Prim>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

/// Error container tied to a particular top-level operation that has a certain span
/// and [context](ErrorContext).
///
/// Supplied as an argument to [`TypeArithmetic`] methods and [`Substitutions::unify()`].
///
/// [`TypeArithmetic`]: crate::arith::TypeArithmetic
/// [`Substitutions::unify()`]: crate::Substitutions::unify()
#[derive(Debug)]
pub struct OpErrors<'a, Prim: PrimitiveType> {
    errors: Goat<'a, Vec<ErrorPrecursor<Prim>>>,
    current_location: Vec<ErrorLocation>,
}

impl<Prim: PrimitiveType> OpErrors<'_, Prim> {
    /// Adds a new `error` into this the error list.
    pub fn push(&mut self, kind: ErrorKind<Prim>) {
        self.errors.push(ErrorPrecursor {
            kind,
            location: self.current_location.clone(),
        });
    }

    /// Invokes the provided closure and returns `false` if new errors were
    /// added during the closure execution.
    pub fn check(&mut self, check: impl FnOnce(OpErrors<'_, Prim>)) -> bool {
        let error_count = self.errors.len();
        check(self.by_ref());
        self.errors.len() == error_count
    }

    /// Mutably borrows this container allowing to use it multiple times.
    pub fn by_ref(&mut self) -> OpErrors<'_, Prim> {
        OpErrors {
            errors: Goat::Borrowed(&mut *self.errors),
            current_location: self.current_location.clone(),
        }
    }

    /// Narrows down the location of the error.
    pub fn with_location(&mut self, location: impl Into<ErrorLocation>) -> OpErrors<'_, Prim> {
        let mut current_location = self.current_location.clone();
        current_location.push(location.into());
        OpErrors {
            errors: Goat::Borrowed(&mut *self.errors),
            current_location,
        }
    }

    #[cfg(test)]
    pub(crate) fn into_vec(self) -> Vec<ErrorKind<Prim>> {
        let errors = match self.errors {
            Goat::Owned(errors) => errors,
            Goat::Borrowed(_) => panic!("Attempt to call `into_vec` for borrowed errors"),
        };
        errors.into_iter().map(|err| err.kind).collect()
    }
}

impl<Prim: PrimitiveType> OpErrors<'static, Prim> {
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
    ) -> Vec<Error<'a, Prim>> {
        let context = context.into();
        self.do_contextualize(|item| item.into_expr_error(context.clone(), span))
    }

    fn do_contextualize<'a>(
        self,
        map_fn: impl Fn(ErrorPrecursor<Prim>) -> Error<'a, Prim>,
    ) -> Vec<Error<'a, Prim>> {
        let errors = match self.errors {
            Goat::Owned(errors) => errors,
            Goat::Borrowed(_) => unreachable!(),
        };
        errors.into_iter().map(map_fn).collect()
    }

    pub(crate) fn contextualize_assignment<'a>(
        self,
        span: &SpannedLvalue<'a, TypeAst<'a>>,
        context: &ErrorContext<Prim>,
    ) -> Vec<Error<'a, Prim>> {
        if self.errors.is_empty() {
            vec![]
        } else {
            self.do_contextualize(|item| item.into_assignment_error(context.clone(), span))
        }
    }

    pub(crate) fn contextualize_destructure<'a>(
        self,
        span: &Spanned<'a, Destructure<'a, TypeAst<'a>>>,
        create_context: impl FnOnce() -> ErrorContext<Prim>,
    ) -> Vec<Error<'a, Prim>> {
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
struct ErrorPrecursor<Prim: PrimitiveType> {
    kind: ErrorKind<Prim>,
    location: Vec<ErrorLocation>,
}

impl<Prim: PrimitiveType> ErrorPrecursor<Prim> {
    fn into_expr_error<'a, T: Grammar<'a>>(
        self,
        context: ErrorContext<Prim>,
        root_expr: &SpannedExpr<'a, T>,
    ) -> Error<'a, Prim> {
        Error {
            inner: ErrorLocation::walk_expr(&self.location, root_expr).copy_with_extra(self.kind),
            root_span: root_expr.with_no_extra(),
            context,
            location: self.location,
        }
    }

    fn into_assignment_error<'a>(
        self,
        context: ErrorContext<Prim>,
        root_lvalue: &SpannedLvalue<'a, TypeAst<'a>>,
    ) -> Error<'a, Prim> {
        Error {
            inner: ErrorLocation::walk_lvalue(&self.location, root_lvalue)
                .copy_with_extra(self.kind),
            root_span: root_lvalue.with_no_extra(),
            context,
            location: self.location,
        }
    }

    fn into_destructure_error<'a>(
        self,
        context: ErrorContext<Prim>,
        root_destructure: &Spanned<'a, Destructure<'a, TypeAst<'a>>>,
    ) -> Error<'a, Prim> {
        Error {
            inner: ErrorLocation::walk_destructure(&self.location, root_destructure)
                .copy_with_extra(self.kind),
            root_span: root_destructure.with_no_extra(),
            context,
            location: self.location,
        }
    }
}

impl TupleIndex {
    fn get_from_tuple<'r, 'a>(self, tuple: &'r TupleAst<'a>) -> Option<&'r SpannedTypeAst<'a>> {
        match self {
            Self::Start(i) => tuple.start.get(i),
            Self::Middle => tuple
                .middle
                .as_ref()
                .map(|middle| middle.extra.element.as_ref()),
            Self::End(i) => tuple.end.get(i),
        }
    }

    fn get_from_destructure<'r, 'a>(
        self,
        destructure: &'r Destructure<'a, TypeAst<'a>>,
    ) -> Option<LvalueTree<'r, 'a>> {
        match self {
            Self::Start(i) => destructure.start.get(i).map(LvalueTree::Lvalue),
            Self::Middle => destructure
                .middle
                .as_ref()
                .and_then(|middle| match &middle.extra {
                    DestructureRest::Named { ty, .. } => ty.as_ref(),
                    _ => None,
                })
                .map(LvalueTree::Type),
            Self::End(i) => destructure.end.get(i).map(LvalueTree::Lvalue),
        }
    }
}

/// Fragment of an error location.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ErrorLocation {
    /// Function argument with the specified index (0-based; can be `None` if the error cannot
    /// be attributed to a specific index).
    FnArg(Option<TupleIndex>),
    /// Function return type.
    FnReturnType,
    /// Tuple element with the specified index (0-based; can be `None` if the error cannot
    /// be attributed to a specific index).
    TupleElement(Option<TupleIndex>),
    /// Left-hand side of a binary operation.
    Lhs,
    /// Right-hand side of a binary operation.
    Rhs,
}

impl From<TupleIndex> for ErrorLocation {
    fn from(index: TupleIndex) -> Self {
        Self::TupleElement(Some(index))
    }
}

// TODO: test walking
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
            // `TupleIndex::FromEnd` should not occur in this context.
            Self::FnArg(Some(TupleIndex::Start(index))) => match &expr.extra {
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

            Self::TupleElement(Some(TupleIndex::Start(index))) => {
                if let Expr::Tuple(elements) = &expr.extra {
                    Some(&elements[index])
                } else {
                    None
                }
            }

            _ => None,
        }
    }

    fn walk_lvalue<'a>(location: &[Self], lvalue: &SpannedLvalue<'a, TypeAst<'a>>) -> Spanned<'a> {
        Self::walk(location, LvalueTree::Lvalue(lvalue), Self::step_into_lvalue)
            .refine_lvalue()
            .with_no_extra()
    }

    fn walk_destructure<'a>(
        location: &[Self],
        destructure: &Spanned<'a, Destructure<'a, TypeAst<'a>>>,
    ) -> Spanned<'a> {
        let destructure = LvalueTree::Destructure(destructure);
        Self::walk(location, destructure, Self::step_into_lvalue)
            .refine_lvalue()
            .with_no_extra()
    }

    fn step_into_lvalue<'r, 'a>(self, lvalue: LvalueTree<'r, 'a>) -> Option<LvalueTree<'r, 'a>> {
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

    fn step_into_type<'r, 'a>(self, ty: &'r TypeAst<'a>) -> Option<LvalueTree<'r, 'a>> {
        let tuple = match ty {
            TypeAst::Tuple(tuple) => tuple,
            _ => return None,
        };
        match self {
            Self::TupleElement(Some(i)) => i.get_from_tuple(tuple).map(LvalueTree::Type),
            _ => None,
        }
    }

    fn step_into_destructure<'r, 'a>(
        self,
        destructure: &'r Destructure<'a, TypeAst<'a>>,
    ) -> Option<LvalueTree<'r, 'a>> {
        match self {
            Self::TupleElement(Some(i)) => i.get_from_destructure(destructure),
            _ => None,
        }
    }
}

/// Enumeration of all types encountered on the lvalue side of assignments.
#[derive(Debug, Clone, Copy)]
enum LvalueTree<'r, 'a> {
    Lvalue(&'r SpannedLvalue<'a, TypeAst<'a>>),
    Destructure(&'r Spanned<'a, Destructure<'a, TypeAst<'a>>>),
    Type(&'r Spanned<'a, TypeAst<'a>>),
}

impl<'a> LvalueTree<'_, 'a> {
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

/// Context of a [`Error`] corresponding to a top-level operation that has errored.
/// Generally, contains resolved types concerning the operation, such as operands of
/// a binary arithmetic op.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ErrorContext<Prim: PrimitiveType> {
    /// No context.
    None,
    /// Processing lvalue (before assignment).
    Lvalue(Type<Prim>),
    /// Function definition.
    FnDefinition {
        /// Types of function arguments.
        args: Tuple<Prim>,
    },
    /// Function call.
    FnCall {
        /// Function definition. Note that this is not necessarily a [`FnType`](crate::FnType).
        definition: Type<Prim>,
        /// Signature of the call.
        call_signature: Type<Prim>,
    },
    /// Assignment.
    Assignment {
        /// Left-hand side of the assignment.
        lhs: Type<Prim>,
        /// Right-hand side of the assignment.
        rhs: Type<Prim>,
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
