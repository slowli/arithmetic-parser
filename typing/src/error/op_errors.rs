//! `OpErrors` type.

use core::ops;

use arithmetic_parser::{grammars::Grammar, Destructure, Spanned, SpannedExpr, SpannedLvalue};

use crate::{
    alloc::Vec,
    ast::TypeAst,
    error::{Error, ErrorContext, ErrorKind, ErrorPathFragment},
    PrimitiveType,
};

/// Error container tied to a particular top-level operation that has a certain span
/// and [context](ErrorContext).
///
/// Supplied as an argument to [`TypeArithmetic`] methods and [`Substitutions::unify()`].
///
/// [`TypeArithmetic`]: crate::arith::TypeArithmetic
/// [`Substitutions::unify()`]: crate::arith::Substitutions::unify()
#[derive(Debug)]
pub struct OpErrors<'a, Prim: PrimitiveType> {
    errors: Goat<'a, Vec<ErrorPrecursor<Prim>>>,
    current_path: Vec<ErrorPathFragment>,
}

impl<Prim: PrimitiveType> OpErrors<'_, Prim> {
    /// Adds a new `error` into this the error list.
    pub fn push(&mut self, kind: ErrorKind<Prim>) {
        self.errors.push(ErrorPrecursor {
            kind,
            path: self.current_path.clone(),
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
            current_path: self.current_path.clone(),
        }
    }

    /// Narrows down the path to the error.
    pub fn join_path(&mut self, path: impl Into<ErrorPathFragment>) -> OpErrors<'_, Prim> {
        let mut current_path = self.current_path.clone();
        current_path.push(path.into());
        OpErrors {
            errors: Goat::Borrowed(&mut *self.errors),
            current_path,
        }
    }

    pub(crate) fn push_path_fragment(&mut self, path: impl Into<ErrorPathFragment>) {
        self.current_path.push(path.into());
    }

    pub(crate) fn pop_path_fragment(&mut self) {
        self.current_path.pop().expect("Location is empty");
    }

    #[cfg(test)]
    pub(crate) fn into_vec(self) -> Vec<ErrorKind<Prim>> {
        let Goat::Owned(errors) = self.errors else {
            panic!("Attempt to call `into_vec` for borrowed errors");
        };
        errors.into_iter().map(|err| err.kind).collect()
    }
}

impl<Prim: PrimitiveType> OpErrors<'static, Prim> {
    pub(crate) fn new() -> Self {
        Self {
            errors: Goat::Owned(Vec::new()),
            current_path: Vec::new(),
        }
    }

    pub(crate) fn contextualize<T: Grammar>(
        self,
        span: &SpannedExpr<'_, T>,
        context: impl Into<ErrorContext<Prim>>,
    ) -> Vec<Error<Prim>> {
        let context = context.into();
        self.do_contextualize(|item| item.into_expr_error(context.clone(), span))
    }

    fn do_contextualize(
        self,
        map_fn: impl Fn(ErrorPrecursor<Prim>) -> Error<Prim>,
    ) -> Vec<Error<Prim>> {
        let Goat::Owned(errors) = self.errors else {
            unreachable!()
        };
        errors.into_iter().map(map_fn).collect()
    }

    pub(crate) fn contextualize_assignment<'a>(
        self,
        span: &SpannedLvalue<'a, TypeAst<'a>>,
        context: &ErrorContext<Prim>,
    ) -> Vec<Error<Prim>> {
        if self.errors.is_empty() {
            Vec::new()
        } else {
            self.do_contextualize(|item| item.into_assignment_error(context.clone(), span))
        }
    }

    pub(crate) fn contextualize_destructure<'a>(
        self,
        span: &Spanned<'a, Destructure<'a, TypeAst<'a>>>,
        create_context: impl FnOnce() -> ErrorContext<Prim>,
    ) -> Vec<Error<Prim>> {
        if self.errors.is_empty() {
            Vec::new()
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
            Self::Borrowed(mut_ref) => mut_ref,
        }
    }
}

#[derive(Debug)]
struct ErrorPrecursor<Prim: PrimitiveType> {
    kind: ErrorKind<Prim>,
    path: Vec<ErrorPathFragment>,
}

impl<Prim: PrimitiveType> ErrorPrecursor<Prim> {
    fn into_expr_error<T: Grammar>(
        self,
        context: ErrorContext<Prim>,
        root_expr: &SpannedExpr<'_, T>,
    ) -> Error<Prim> {
        Error {
            inner: ErrorPathFragment::walk_expr(&self.path, root_expr)
                .copy_with_extra(self.kind)
                .into(),
            root_location: root_expr.with_no_extra().into(),
            context,
            path: self.path,
        }
    }

    fn into_assignment_error<'a>(
        self,
        context: ErrorContext<Prim>,
        root_lvalue: &SpannedLvalue<'a, TypeAst<'a>>,
    ) -> Error<Prim> {
        Error {
            inner: ErrorPathFragment::walk_lvalue(&self.path, root_lvalue)
                .copy_with_extra(self.kind)
                .into(),
            root_location: root_lvalue.with_no_extra().into(),
            context,
            path: self.path,
        }
    }

    fn into_destructure_error<'a>(
        self,
        context: ErrorContext<Prim>,
        root_destructure: &Spanned<'a, Destructure<'a, TypeAst<'a>>>,
    ) -> Error<Prim> {
        Error {
            inner: ErrorPathFragment::walk_destructure(&self.path, root_destructure)
                .copy_with_extra(self.kind)
                .into(),
            root_location: root_destructure.with_no_extra().into(),
            context,
            path: self.path,
        }
    }
}
