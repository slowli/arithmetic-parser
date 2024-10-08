//! Error handling.

use core::fmt;

use nom::{
    error::{ContextError, ErrorKind as NomErrorKind, FromExternalError, ParseError},
    Slice,
};

use crate::{
    BinaryOp, ExprType, InputSpan, LocatedSpan, Location, LvalueType, Op, Spanned, StatementType,
    UnaryOp,
};

/// Parsing context.
// TODO: Add more fine-grained contexts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Context {
    /// Variable name.
    Var,
    /// Function invocation.
    Fun,
    /// Arithmetic expression.
    Expr,
    /// Comment.
    Comment,
}

impl fmt::Display for Context {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Var => "variable",
            Self::Fun => "function call",
            Self::Expr => "arithmetic expression",
            Self::Comment => "comment",
        })
    }
}

impl Context {
    pub(crate) fn new(s: &str) -> Self {
        match s {
            "var" => Self::Var,
            "fn" => Self::Fun,
            "expr" => Self::Expr,
            "comment" => Self::Comment,
            _ => unreachable!(),
        }
    }

    pub(crate) fn to_str(self) -> &'static str {
        match self {
            Self::Var => "var",
            Self::Fun => "fn",
            Self::Expr => "expr",
            Self::Comment => "comment",
        }
    }
}

/// Parsing error kind.
#[derive(Debug)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Input is not in ASCII.
    NonAsciiInput,
    /// Error parsing literal.
    Literal(anyhow::Error),
    /// Literal is used where a name is expected, e.g., as a function identifier.
    ///
    /// An example of input triggering this error is `1(2, x)`; `1` is used as the function
    /// identifier.
    LiteralName,
    /// Error parsing type annotation.
    Type(anyhow::Error),
    /// Unary or binary operation switched off in the parser features.
    UnsupportedOp(Op),
    /// No rules where expecting this character.
    UnexpectedChar {
        /// Parsing context.
        context: Option<Context>,
    },
    /// Unexpected expression end.
    UnexpectedTerm {
        /// Parsing context.
        context: Option<Context>,
    },
    /// Leftover symbols after parsing.
    Leftovers,
    /// Input is incomplete.
    Incomplete,
    /// Unfinished comment.
    UnfinishedComment,
    /// Chained comparison, such as `1 < 2 < 3`.
    ChainedComparison,
    /// Other parsing error.
    Other {
        /// `nom`-defined error kind.
        kind: NomErrorKind,
        /// Parsing context.
        context: Option<Context>,
    },
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonAsciiInput => formatter.write_str("Non-ASCII inputs are not supported"),
            Self::Literal(err) => write!(formatter, "Invalid literal: {err}"),
            Self::LiteralName => formatter.write_str("Literal used in place of an identifier"),

            Self::Type(err) => write!(formatter, "Invalid type annotation: {err}"),

            Self::UnsupportedOp(op) => write!(
                formatter,
                "Encountered operation switched off in the parser features: {op}"
            ),

            Self::UnexpectedChar { context: Some(ctx) } => {
                write!(formatter, "Unexpected character in {ctx}")
            }
            Self::UnexpectedChar { .. } => formatter.write_str("Unexpected character"),

            Self::UnexpectedTerm { context: Some(ctx) } => write!(formatter, "Unfinished {ctx}"),
            Self::UnexpectedTerm { .. } => formatter.write_str("Unfinished expression"),

            Self::Leftovers => formatter.write_str("Uninterpreted characters after parsing"),
            Self::Incomplete => formatter.write_str("Incomplete input"),
            Self::UnfinishedComment => formatter.write_str("Unfinished comment"),
            Self::ChainedComparison => formatter.write_str("Chained comparisons"),
            Self::Other { .. } => write!(formatter, "Cannot parse sequence"),
        }
    }
}

impl ErrorKind {
    /// Creates a `Literal` variant with the specified error.
    pub fn literal<T: Into<anyhow::Error>>(error: T) -> Self {
        Self::Literal(error.into())
    }

    fn context_mut(&mut self) -> Option<&mut Option<Context>> {
        match self {
            Self::UnexpectedChar { context }
            | Self::UnexpectedTerm { context }
            | Self::Other { context, .. } => Some(context),
            _ => None,
        }
    }

    /// Returns optional error context.
    pub fn context(&self) -> Option<Context> {
        match self {
            Self::UnexpectedChar { context }
            | Self::UnexpectedTerm { context }
            | Self::Other { context, .. } => *context,
            _ => None,
        }
    }

    /// Returns `true` if this is `Incomplete`.
    pub fn is_incomplete(&self) -> bool {
        matches!(self, Self::Incomplete)
    }

    #[doc(hidden)]
    pub fn with_span<T>(self, span: &Spanned<'_, T>) -> Error {
        Error {
            inner: span.copy_with_extra(self).map_fragment(str::len),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ErrorKind {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Literal(err) | Self::Type(err) => Some(err.as_ref()),
            _ => None,
        }
    }
}

/// Parsing error with a generic code span.
///
/// Two primary cases of the `Span` type param are `&str` (for original errors produced by
/// the parser) and `usize` (for *stripped* errors that have a static lifetime).
#[derive(Debug)]
pub struct Error {
    inner: Location<ErrorKind>,
}

impl Error {
    pub(crate) fn new(span: InputSpan<'_>, kind: ErrorKind) -> Self {
        Self {
            inner: LocatedSpan::from(span)
                .map_fragment(str::len)
                .copy_with_extra(kind),
        }
    }

    /// Returns the kind of this error.
    pub fn kind(&self) -> &ErrorKind {
        &self.inner.extra
    }

    /// Returns the span of this error.
    pub fn location(&self) -> Location {
        self.inner.with_no_extra()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}:{}: {}",
            self.inner.location_line(),
            self.inner.get_column(),
            self.inner.extra
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        std::error::Error::source(&self.inner.extra)
    }
}

impl<'a> ParseError<InputSpan<'a>> for Error {
    fn from_error_kind(mut input: InputSpan<'a>, kind: NomErrorKind) -> Self {
        if kind == NomErrorKind::Char && !input.fragment().is_empty() {
            // Truncate the error span to the first ineligible char.
            input = input.slice(..1);
        }

        let error_kind = if kind == NomErrorKind::Char {
            if input.fragment().is_empty() {
                ErrorKind::UnexpectedTerm { context: None }
            } else {
                ErrorKind::UnexpectedChar { context: None }
            }
        } else {
            ErrorKind::Other {
                kind,
                context: None,
            }
        };

        Error::new(input, error_kind)
    }

    fn append(_: InputSpan<'a>, _: NomErrorKind, other: Self) -> Self {
        other
    }
}

impl<'a> ContextError<InputSpan<'a>> for Error {
    fn add_context(input: InputSpan<'a>, ctx: &'static str, mut target: Self) -> Self {
        let ctx = Context::new(ctx);
        if ctx == Context::Comment {
            target.inner.extra = ErrorKind::UnfinishedComment;
        }

        if input.location_offset() < target.inner.location_offset() {
            if let Some(context) = target.inner.extra.context_mut() {
                *context = Some(ctx);
            }
        }
        target
    }
}

impl<'a> FromExternalError<InputSpan<'a>, ErrorKind> for Error {
    fn from_external_error(input: InputSpan<'a>, _: NomErrorKind, err: ErrorKind) -> Self {
        Self::new(input, err)
    }
}

/// Description of a construct not supported by a certain module (e.g., interpreter
/// or type inference).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnsupportedType {
    /// Unary operation.
    UnaryOp(UnaryOp),
    /// Binary operation.
    BinaryOp(BinaryOp),
    /// Expression.
    Expr(ExprType),
    /// Statement.
    Statement(StatementType),
    /// Lvalue.
    Lvalue(LvalueType),
}

impl fmt::Display for UnsupportedType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnaryOp(op) => write!(formatter, "unary op: {op}"),
            Self::BinaryOp(op) => write!(formatter, "binary op: {op}"),
            Self::Expr(expr) => write!(formatter, "expression: {expr}"),
            Self::Statement(statement) => write!(formatter, "statement: {statement}"),
            Self::Lvalue(lvalue) => write!(formatter, "lvalue: {lvalue}"),
        }
    }
}

impl From<UnaryOp> for UnsupportedType {
    fn from(value: UnaryOp) -> Self {
        Self::UnaryOp(value)
    }
}

impl From<BinaryOp> for UnsupportedType {
    fn from(value: BinaryOp) -> Self {
        Self::BinaryOp(value)
    }
}

impl From<ExprType> for UnsupportedType {
    fn from(value: ExprType) -> Self {
        Self::Expr(value)
    }
}

impl From<StatementType> for UnsupportedType {
    fn from(value: StatementType) -> Self {
        Self::Statement(value)
    }
}

impl From<LvalueType> for UnsupportedType {
    fn from(value: LvalueType) -> Self {
        Self::Lvalue(value)
    }
}
