//! Error handling.

use nom::{
    error::{ErrorKind as NomErrorKind, ParseError},
    Slice,
};

use alloc::borrow::ToOwned;
use core::fmt;

use crate::{Context, InputSpan, Op, Spanned};

/// Parsing error kind.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Input is not in ASCII.
    NonAsciiInput,

    /// Error parsing literal.
    Literal(anyhow::Error),

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

    /// Other parsing error.
    Other {
        /// `nom`-defined error kind.
        kind: NomErrorKind,
        /// Parsing context.
        context: Option<Context>,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonAsciiInput => formatter.write_str("Non-ASCII inputs are not supported"),
            Self::Literal(e) => write!(formatter, "Invalid literal: {}", e),
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
            Self::Other { .. } => write!(formatter, "Cannot parse sequence"),
        }
    }
}

impl Error {
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

    pub(crate) fn with_span<'a, T>(self, span: &Spanned<'a, T>) -> SpannedError<'a> {
        SpannedError {
            inner: span.copy_with_extra(self),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Literal(err) | Self::Type(err) => Some(err),
            _ => None,
        }
    }
}

/// Parsing error with the associated code span.
#[derive(Debug)]
pub struct SpannedError<'a> {
    inner: Spanned<'a, Error>,
}

impl<'a> SpannedError<'a> {
    pub(crate) fn new(span: InputSpan<'a>, kind: Error) -> Self {
        Self {
            inner: Spanned::new(span, kind),
        }
    }

    pub(crate) fn from_parts(span: Spanned<'a>, kind: Error) -> Self {
        Self {
            inner: span.copy_with_extra(kind),
        }
    }

    /// Returns the kind of this error.
    pub fn kind(&self) -> &Error {
        &self.inner.extra
    }

    /// Returns the span of this error.
    pub fn span(&self) -> Spanned<'a> {
        self.inner.with_no_extra()
    }
}

impl fmt::Display for SpannedError<'_> {
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
impl std::error::Error for SpannedError<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        std::error::Error::source(&self.inner)
    }
}

impl<'a> ParseError<InputSpan<'a>> for SpannedError<'a> {
    fn from_error_kind(mut input: InputSpan<'a>, kind: NomErrorKind) -> Self {
        if kind == NomErrorKind::Char && !input.fragment().is_empty() {
            // Truncate the error span to the first ineligible char.
            input = input.slice(..1);
        }

        let error_kind = if kind == NomErrorKind::Char {
            if input.fragment().is_empty() {
                Error::UnexpectedTerm { context: None }
            } else {
                Error::UnexpectedChar { context: None }
            }
        } else {
            Error::Other {
                kind,
                context: None,
            }
        };

        SpannedError::new(input, error_kind)
    }

    fn append(_: InputSpan<'a>, _: NomErrorKind, other: Self) -> Self {
        other
    }

    fn add_context(input: InputSpan<'a>, ctx: &'static str, mut target: Self) -> Self {
        if input.location_offset() < target.inner.location_offset() {
            if let Some(context) = target.inner.extra.context_mut() {
                *context = Some(Context::new(ctx));
            }
        }
        target
    }
}
