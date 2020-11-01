//! Error handling.

use nom::{
    error::{ErrorKind as NomErrorKind, ParseError},
    Slice,
};

use core::fmt;

use crate::{alloc::ToOwned, InputSpan, LocatedSpan, Op, Spanned, StripCode};

/// Parsing context.
// TODO: Add more fine-grained contexts.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
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
    /// Error parsing type hint.
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
            Self::Literal(e) => write!(formatter, "Invalid literal: {}", e),
            Self::LiteralName => formatter.write_str("Literal used in place of an identifier"),

            Self::Type(e) => write!(formatter, "Invalid type hint: {}", e),

            Self::UnsupportedOp(op) => write!(
                formatter,
                "Encountered operation switched off in the parser features: {}",
                op
            ),

            Self::UnexpectedChar { context: Some(ctx) } => {
                write!(formatter, "Unexpected character in {}", ctx)
            }
            Self::UnexpectedChar { .. } => formatter.write_str("Unexpected character"),

            Self::UnexpectedTerm { context: Some(ctx) } => write!(formatter, "Unfinished {}", ctx),
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
            | Self::Other { context, .. } => context.to_owned(),
            _ => None,
        }
    }

    /// Returns `true` if this is `Incomplete`.
    pub fn is_incomplete(&self) -> bool {
        matches!(self, Self::Incomplete)
    }

    pub(crate) fn with_span<'a, T>(self, span: &Spanned<'a, T>) -> Error<'a> {
        Error {
            inner: span.copy_with_extra(self),
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
pub struct SpannedError<Span> {
    inner: LocatedSpan<Span, ErrorKind>,
}

/// Error with code span available as a string reference.
pub type Error<'a> = SpannedError<&'a str>;

impl<'a> Error<'a> {
    pub(crate) fn new(span: InputSpan<'a>, kind: ErrorKind) -> Self {
        Self {
            inner: Spanned::new(span, kind),
        }
    }
}

impl<Span> SpannedError<Span> {
    /// Returns the kind of this error.
    pub fn kind(&self) -> &ErrorKind {
        &self.inner.extra
    }
}

impl<Span: Copy> SpannedError<Span> {
    /// Returns the span of this error.
    pub fn span(&self) -> LocatedSpan<Span> {
        self.inner.with_no_extra()
    }
}

impl<Span> fmt::Display for SpannedError<Span> {
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
impl<Span: fmt::Debug> std::error::Error for SpannedError<Span> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        std::error::Error::source(&self.inner.extra)
    }
}

impl StripCode for Error<'_> {
    type Stripped = SpannedError<usize>;

    fn strip_code(self) -> Self::Stripped {
        SpannedError {
            inner: self.inner.map_fragment(str::len),
        }
    }
}

impl<'a> ParseError<InputSpan<'a>> for Error<'a> {
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
