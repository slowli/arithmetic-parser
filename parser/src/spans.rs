//! Types related to spanning parsed code.

use nom::Slice;

use alloc::{borrow::ToOwned, format, string::String};

use crate::SpannedError;

/// Code span.
pub type InputSpan<'a> = nom_locate::LocatedSpan<&'a str, ()>;
/// Parsing outcome generalized by the type returned on success.
pub type NomResult<'a, T> = nom::IResult<InputSpan<'a>, T, SpannedError<'a>>;

/// Code span together with information related to where it is located in the code.
///
/// This type is similar to one from the [`nom_locate`] crate, but it has slightly different
/// functionality. In particular, this type provides no method to access other parts of the code
/// (which is performed in `nom_locate`'s `LocatedSpan::get_column()` among other methods).
/// As such, this allows to safely replace [span info](#method.fragment) without worrying
/// about undefined behavior.
///
/// [`nom_locate`]: https://crates.io/crates/nom_locate
#[derive(Debug, Clone, Copy)]
pub struct LocatedSpan<Span, T = ()> {
    offset: usize,
    line: u32,
    column: usize,
    fragment: Span,

    /// Extra information that can be embedded by the user.
    pub extra: T,
}

impl<Span: PartialEq, T> PartialEq for LocatedSpan<Span, T> {
    fn eq(&self, other: &Self) -> bool {
        self.line == other.line && self.offset == other.offset && self.fragment == other.fragment
    }
}

impl<Span, T> LocatedSpan<Span, T> {
    /// The offset represents the position of the fragment relatively to the input of the parser.
    /// It starts at offset 0.
    pub fn location_offset(&self) -> usize {
        self.offset
    }

    /// The line number of the fragment relatively to the input of the parser. It starts at line 1.
    pub fn location_line(&self) -> u32 {
        self.line
    }

    /// The column of the fragment start.
    pub fn get_column(&self) -> usize {
        self.column
    }

    /// The fragment that is spanned. The fragment represents a part of the input of the parser.
    pub fn fragment(&self) -> &Span {
        &self.fragment
    }

    /// Maps the `extra` field of this span using the provided closure.
    pub fn map_extra<U>(self, map_fn: impl FnOnce(T) -> U) -> LocatedSpan<Span, U> {
        LocatedSpan {
            offset: self.offset,
            line: self.line,
            column: self.column,
            fragment: self.fragment,
            extra: map_fn(self.extra),
        }
    }

    /// Maps the fragment field of this span using the provided closure.
    pub fn map_fragment<U>(self, map_fn: impl FnOnce(Span) -> U) -> LocatedSpan<U, T> {
        LocatedSpan {
            offset: self.offset,
            line: self.line,
            column: self.column,
            fragment: map_fn(self.fragment),
            extra: self.extra,
        }
    }
}

impl<Span: Copy, T> LocatedSpan<Span, T> {
    /// Copies this span with the provided `extra` field.
    pub fn copy_with_extra<U>(&self, value: U) -> LocatedSpan<Span, U> {
        LocatedSpan {
            offset: self.offset,
            line: self.line,
            column: self.column,
            fragment: self.fragment,
            extra: value,
        }
    }

    /// Removes `extra` field from this span.
    pub fn with_no_extra(&self) -> LocatedSpan<Span> {
        self.copy_with_extra(())
    }
}

impl<Span: Copy, T: Clone> LocatedSpan<Span, T> {
    pub(crate) fn with_mapped_span<U>(&self, map_fn: impl FnOnce(Span) -> U) -> LocatedSpan<U, T> {
        LocatedSpan {
            offset: self.offset,
            line: self.line,
            column: self.column,
            fragment: map_fn(self.fragment),
            extra: self.extra.clone(),
        }
    }
}

impl<'a, T> From<nom_locate::LocatedSpan<&'a str, T>> for LocatedSpan<&'a str, T> {
    fn from(value: nom_locate::LocatedSpan<&'a str, T>) -> Self {
        Self {
            offset: value.location_offset(),
            line: value.location_line(),
            column: value.get_column(),
            fragment: *value.fragment(),
            extra: value.extra,
        }
    }
}

/// Value with an associated code span.
pub type Spanned<'a, T = ()> = LocatedSpan<&'a str, T>;

impl<'a, T> Spanned<'a, T> {
    pub(crate) fn new(span: InputSpan<'a>, extra: T) -> Self {
        Self {
            offset: span.location_offset(),
            line: span.location_line(),
            column: span.get_column(),
            fragment: *span.fragment(),
            extra,
        }
    }
}

impl<'a> Spanned<'a> {
    /// Creates a span from a `range` in the provided `code`. This is mostly useful for testing.
    pub fn from_str<R>(code: &'a str, range: R) -> Self
    where
        InputSpan<'a>: Slice<R>,
    {
        let input = InputSpan::new(code);
        Self::new(input.slice(range), ())
    }
}

/// Container for a code fragment that can be in one of the two states: either the code string
/// is retained, or it is stripped away.
///
/// The stripped version allows to retain information about code location within [`LocatedSpan`]
/// without a restriction by the code lifetime.
///
/// [`LocatedSpan`]: struct.LocatedSpan.html
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CodeFragment<'a> {
    /// Original code fragment: a string reference.
    Str(&'a str),
    /// Stripped code fragment: just the string length.
    Stripped(usize),
}

impl PartialEq<&str> for CodeFragment<'_> {
    fn eq(&self, &other: &&str) -> bool {
        match self {
            Self::Str(string) => *string == other,
            Self::Stripped(_) => false,
        }
    }
}

impl CodeFragment<'_> {
    /// Strips this code fragment, extending its lifetime beyond the lifetime of the code.
    pub fn strip(self) -> CodeFragment<'static> {
        match self {
            Self::Str(string) => CodeFragment::Stripped(string.len()),
            Self::Stripped(len) => CodeFragment::Stripped(len),
        }
    }

    /// Gets the length of this code fragment.
    pub fn len(self) -> usize {
        match self {
            Self::Str(string) => string.len(),
            Self::Stripped(len) => len,
        }
    }

    /// Checks if this code fragment is empty.
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

impl<'a> From<&'a str> for CodeFragment<'a> {
    fn from(value: &'a str) -> Self {
        CodeFragment::Str(value)
    }
}

/// Value with an optional associated code span.
pub type MaybeSpanned<'a, T = ()> = LocatedSpan<CodeFragment<'a>, T>;

impl<'a> MaybeSpanned<'a> {
    /// Creates a span from a `range` in the provided `code`. This is mostly useful for testing.
    pub fn from_str<R>(code: &'a str, range: R) -> Self
    where
        InputSpan<'a>: Slice<R>,
    {
        Spanned::from_str(code, range).into()
    }
}

impl<T> MaybeSpanned<'_, T> {
    /// Returns either the original code fragment (if it's retained), or a string in the form
    /// `{default_name} at {line}:{column}`.
    pub fn code_or_location(&self, default_name: &str) -> String {
        match self.fragment {
            CodeFragment::Str(code) => code.to_owned(),
            CodeFragment::Stripped(_) => {
                format!("{} at {}:{}", default_name, self.line, self.column)
            }
        }
    }
}

impl<'a, T> From<Spanned<'a, T>> for MaybeSpanned<'a, T> {
    fn from(value: Spanned<'a, T>) -> Self {
        value.map_fragment(CodeFragment::from)
    }
}

/// Encapsulates stripping references to code fragments. The result can outlive the code.
///
/// Implementors of this trait are usually generic by the code lifetime: `Foo<'_, ..>`,
/// with the result of stripping being `Foo<'static, ..>`.
pub trait StripCode {
    /// Resulting type after code stripping.
    type Stripped: 'static;

    /// Strips references to code fragments in this type.
    fn strip_code(&self) -> Self::Stripped;
}

impl<T: Clone + 'static> StripCode for MaybeSpanned<'_, T> {
    type Stripped = MaybeSpanned<'static, T>;

    fn strip_code(&self) -> Self::Stripped {
        self.with_mapped_span(|code| code.strip())
    }
}

/// Wrapper around parsers allowing to capture both their output and the relevant span.
pub(crate) fn with_span<'a, O>(
    parser: impl Fn(InputSpan<'a>) -> NomResult<'a, O>,
) -> impl Fn(InputSpan<'a>) -> NomResult<'a, Spanned<O>> {
    move |input: InputSpan| {
        parser(input).map(|(rest, output)| {
            let len = rest.location_offset() - input.location_offset();
            let spanned = Spanned {
                offset: input.location_offset(),
                line: input.location_line(),
                column: input.get_column(),
                fragment: &input.fragment()[..len],
                extra: output,
            };
            (rest, spanned)
        })
    }
}

pub(crate) fn unite_spans<'a, T, U>(
    input: InputSpan<'a>,
    start: &Spanned<'_, T>,
    end: &Spanned<'_, U>,
) -> InputSpan<'a> {
    debug_assert!(input.location_offset() <= start.location_offset());
    debug_assert!(start.location_offset() <= end.location_offset());
    debug_assert!(
        input.location_offset() + input.fragment().len()
            >= end.location_offset() + end.fragment().len()
    );

    let start_idx = start.location_offset() - input.location_offset();
    let end_idx = end.location_offset() + end.fragment().len() - input.location_offset();
    unsafe {
        // SAFETY: Safe since offset coincides with the input offset (which we consider
        // well-formed).
        InputSpan::new_from_raw_offset(
            start.location_offset(),
            start.location_line(),
            &input.fragment()[start_idx..end_idx],
            (),
        )
    }
}
