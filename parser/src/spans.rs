//! Types related to spanning parsed code.

use nom::Slice;

use crate::{
    alloc::{format, String},
    Error,
};

/// Code span.
pub type InputSpan<'a> = nom_locate::LocatedSpan<&'a str, ()>;
/// Parsing outcome generalized by the type returned on success.
pub type NomResult<'a, T> = nom::IResult<InputSpan<'a>, T, Error>;

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
    /// Returns a copy of this span with borrowed `extra` field.
    pub fn as_ref(&self) -> LocatedSpan<Span, &T> {
        LocatedSpan {
            offset: self.offset,
            line: self.line,
            column: self.column,
            fragment: self.fragment,
            extra: &self.extra,
        }
    }

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

#[allow(clippy::mismatching_type_param_order)] // weird false positive
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

/// Value with an associated code location. Unlike [`Spanned`], `Location` does not retain a reference
/// to the original code span, just its start position and length.
pub type Location<T = ()> = LocatedSpan<usize, T>;

impl Location {
    /// Creates a location from a `range` in the provided `code`. This is mostly useful for testing.
    pub fn from_str<'a, R>(code: &'a str, range: R) -> Self
    where
        InputSpan<'a>: Slice<R>,
    {
        Spanned::from_str(code, range).into()
    }
}

impl<T> Location<T> {
    /// Returns a string representation of this location in the form `{default_name} at {line}:{column}`.
    pub fn to_string(&self, default_name: &str) -> String {
        format!("{default_name} at {}:{}", self.line, self.column)
    }

    /// Returns this location in the provided `code`. It is caller's responsibility to ensure that this
    /// is called with the original `code` that produced this location.
    pub fn span<'a>(&self, code: &'a str) -> &'a str {
        &code[self.offset..(self.offset + self.fragment)]
    }
}

impl<T> From<Spanned<'_, T>> for Location<T> {
    fn from(value: Spanned<'_, T>) -> Self {
        value.map_fragment(str::len)
    }
}

/// Wrapper around parsers allowing to capture both their output and the relevant span.
pub fn with_span<'a, O>(
    mut parser: impl FnMut(InputSpan<'a>) -> NomResult<'a, O>,
) -> impl FnMut(InputSpan<'a>) -> NomResult<'a, Spanned<'a, O>> {
    move |input: InputSpan<'_>| {
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
) -> Spanned<'a> {
    debug_assert!(input.location_offset() <= start.location_offset());
    debug_assert!(start.location_offset() <= end.location_offset());
    debug_assert!(
        input.location_offset() + input.fragment().len()
            >= end.location_offset() + end.fragment().len()
    );

    let start_idx = start.location_offset() - input.location_offset();
    let end_idx = end.location_offset() + end.fragment().len() - input.location_offset();
    Spanned {
        offset: start.offset,
        line: start.line,
        column: start.column,
        fragment: &input.fragment()[start_idx..end_idx],
        extra: (),
    }
}
