//! Errors related to type inference.

use std::fmt;

use arithmetic_parser::{Location, Spanned, UnsupportedType};

pub use self::{
    kind::{ErrorKind, TupleContext},
    op_errors::OpErrors,
    path::ErrorPathFragment,
};
use crate::{
    arith::{BinaryOpContext, UnaryOpContext},
    ast::AstConversionError,
    visit::VisitMut,
    PrimitiveType, Tuple, Type,
};

mod kind;
mod op_errors;
mod path;

/// Type error together with the corresponding code span.
#[derive(Debug, Clone)]
pub struct Error<Prim: PrimitiveType> {
    inner: Location<ErrorKind<Prim>>,
    root_location: Location,
    context: ErrorContext<Prim>,
    path: Vec<ErrorPathFragment>,
}

impl<Prim: PrimitiveType> fmt::Display for Error<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}:{}: {}",
            self.main_location().location_line(),
            self.main_location().get_column(),
            self.kind()
        )
    }
}

impl<Prim: PrimitiveType> std::error::Error for Error<Prim> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.kind())
    }
}

impl<Prim: PrimitiveType> Error<Prim> {
    pub(crate) fn unsupported<T>(
        unsupported: impl Into<UnsupportedType>,
        span: &Spanned<'_, T>,
    ) -> Self {
        let kind = ErrorKind::unsupported(unsupported);
        Self {
            inner: span.copy_with_extra(kind).into(),
            root_location: span.with_no_extra().into(),
            context: ErrorContext::None,
            path: vec![],
        }
    }

    pub(crate) fn undefined_var<T>(span: &Spanned<'_, T>) -> Self {
        let ident = (*span.fragment()).to_owned();
        Self {
            inner: span.copy_with_extra(ErrorKind::UndefinedVar(ident)).into(),
            root_location: span.with_no_extra().into(),
            context: ErrorContext::None,
            path: vec![],
        }
    }

    pub(crate) fn repeated_assignment(span: Spanned<'_>) -> Self {
        let ident = (*span.fragment()).to_owned();
        Self {
            inner: span
                .copy_with_extra(ErrorKind::RepeatedAssignment(ident))
                .into(),
            root_location: span.with_no_extra().into(),
            context: ErrorContext::None,
            path: vec![],
        }
    }

    pub(crate) fn repeated_field(span: Spanned<'_>) -> Self {
        let ident = (*span.fragment()).to_owned();
        Self {
            inner: span.copy_with_extra(ErrorKind::RepeatedField(ident)).into(),
            root_location: span.with_no_extra().into(),
            context: ErrorContext::None,
            path: vec![],
        }
    }

    pub(crate) fn conversion<T>(kind: AstConversionError, span: &Spanned<'_, T>) -> Self {
        let kind = ErrorKind::AstConversion(kind);
        Self {
            inner: span.copy_with_extra(kind).into(),
            root_location: span.with_no_extra().into(),
            context: ErrorContext::None,
            path: vec![],
        }
    }

    pub(crate) fn invalid_field_name(span: Spanned<'_>) -> Self {
        let ident = (*span.fragment()).to_owned();
        Self {
            inner: span
                .copy_with_extra(ErrorKind::InvalidFieldName(ident))
                .into(),
            root_location: span.into(),
            context: ErrorContext::None,
            path: vec![],
        }
    }

    pub(crate) fn index_out_of_bounds<T>(
        receiver: Tuple<Prim>,
        span: &Spanned<'_, T>,
        index: usize,
    ) -> Self {
        Self {
            inner: span
                .copy_with_extra(ErrorKind::IndexOutOfBounds {
                    index,
                    len: receiver.len(),
                })
                .into(),
            root_location: span.with_no_extra().into(),
            context: ErrorContext::TupleIndex {
                ty: Type::Tuple(receiver),
            },
            path: vec![],
        }
    }

    pub(crate) fn cannot_index<T>(receiver: Type<Prim>, span: &Spanned<'_, T>) -> Self {
        Self {
            inner: span.copy_with_extra(ErrorKind::CannotIndex).into(),
            root_location: span.with_no_extra().into(),
            context: ErrorContext::TupleIndex { ty: receiver },
            path: vec![],
        }
    }

    pub(crate) fn unsupported_index<T>(receiver: Type<Prim>, span: &Spanned<'_, T>) -> Self {
        Self {
            inner: span.copy_with_extra(ErrorKind::UnsupportedIndex).into(),
            root_location: span.with_no_extra().into(),
            context: ErrorContext::TupleIndex { ty: receiver },
            path: vec![],
        }
    }

    /// Gets the kind of this error.
    pub fn kind(&self) -> &ErrorKind<Prim> {
        &self.inner.extra
    }

    /// Gets the most specific code span of this error.
    pub fn main_location(&self) -> Location {
        self.inner.with_no_extra()
    }

    /// Gets the root code location of the failed operation. May coincide with [`Self::main_location()`].
    pub fn root_location(&self) -> Location {
        self.root_location
    }

    /// Gets the context for an operation that has failed.
    pub fn context(&self) -> &ErrorContext<Prim> {
        &self.context
    }

    /// Gets the path of this error relative to the failed top-level operation.
    /// This can be used for highlighting relevant parts of types in [`Self::context()`].
    pub fn path(&self) -> &[ErrorPathFragment] {
        &self.path
    }
}

/// List of [`Error`]s.
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse};
/// # use arithmetic_typing::{defs::Prelude, error::Errors, Annotated, TypeEnvironment};
/// # use std::collections::HashSet;
/// # fn main() -> anyhow::Result<()> {
/// let buggy_code = Annotated::<F32Grammar>::parse_statements(r#"
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
pub struct Errors<Prim: PrimitiveType> {
    inner: Vec<Error<Prim>>,
    first_failing_statement: usize,
}

impl<Prim: PrimitiveType> Errors<Prim> {
    pub(crate) fn new() -> Self {
        Self {
            inner: vec![],
            first_failing_statement: 0,
        }
    }

    pub(crate) fn push(&mut self, err: Error<Prim>) {
        self.inner.push(err);
    }

    pub(crate) fn extend(&mut self, errors: Vec<Error<Prim>>) {
        self.inner.extend(errors);
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
    pub fn iter(&self) -> impl Iterator<Item = &Error<Prim>> + '_ {
        self.inner.iter()
    }

    /// Returns the index of the first failing statement within a `Block` that has errored.
    /// If the error is in the return value, this index will be equal to the number of statements
    /// in the block.
    pub fn first_failing_statement(&self) -> usize {
        self.first_failing_statement
    }

    pub(crate) fn set_first_failing_statement(&mut self, index: usize) {
        self.first_failing_statement = index;
    }

    /// Post-processes these errors, resolving the contained `Type`s using
    /// the provided `type_resolver`.
    pub(crate) fn post_process(&mut self, type_resolver: &mut impl VisitMut<Prim>) {
        for error in &mut self.inner {
            error.context.map_types(type_resolver);
        }
    }
}

impl<Prim: PrimitiveType> fmt::Display for Errors<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, error) in self.inner.iter().enumerate() {
            write!(formatter, "{error}")?;
            if i + 1 < self.inner.len() {
                formatter.write_str("\n")?;
            }
        }
        Ok(())
    }
}

impl<Prim: PrimitiveType> std::error::Error for Errors<Prim> {}

impl<Prim: PrimitiveType> IntoIterator for Errors<Prim> {
    type Item = Error<Prim>;
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
        /// Function definition. Note that this is not necessarily a [`Function`](crate::Function).
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
    /// Type cast.
    TypeCast {
        /// Source type of the casted value.
        source: Type<Prim>,
        /// Target type of the cast.
        target: Type<Prim>,
    },
    /// Unary operation.
    UnaryOp(UnaryOpContext<Prim>),
    /// Binary operation.
    BinaryOp(BinaryOpContext<Prim>),
    /// Tuple indexing operation.
    TupleIndex {
        /// Type being indexed.
        ty: Type<Prim>,
    },
    /// Field access for an object.
    ObjectFieldAccess {
        /// Type being accessed.
        ty: Type<Prim>,
    },
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
            Self::TypeCast { source, target } => {
                mapper.visit_type_mut(source);
                mapper.visit_type_mut(target);
            }
            Self::UnaryOp(UnaryOpContext { arg, .. }) => {
                mapper.visit_type_mut(arg);
            }
            Self::TupleIndex { ty } | Self::ObjectFieldAccess { ty } => {
                mapper.visit_type_mut(ty);
            }
        }
    }
}
