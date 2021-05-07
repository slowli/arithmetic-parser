//! Errors related to type inference.

use std::fmt;

use crate::{
    arith::{BinaryOpContext, UnaryOpContext},
    ast::AstConversionError,
    visit::VisitMut,
    PrimitiveType, Tuple, Type,
};
use arithmetic_parser::{Spanned, UnsupportedType};

mod kind;
mod location;
mod op_errors;

pub use self::{
    kind::{ErrorKind, TupleContext},
    location::ErrorLocation,
    op_errors::OpErrors,
};

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
