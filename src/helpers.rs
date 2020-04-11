//! Misc helpers.

use crate::{NomResult, Span, Spanned};

pub fn create_span<T, U>(span: Spanned<T>, extra: U) -> Spanned<U> {
    Spanned {
        offset: span.offset,
        line: span.line,
        fragment: span.fragment,
        extra,
    }
}

pub fn create_span_ref<'a, T, U>(span: &Spanned<'a, T>, extra: U) -> Spanned<'a, U> {
    Spanned {
        offset: span.offset,
        line: span.line,
        fragment: span.fragment,
        extra,
    }
}

pub fn map_span<T, U>(span: Spanned<'_, T>, f: impl FnOnce(T) -> U) -> Spanned<'_, U> {
    Spanned {
        offset: span.offset,
        line: span.line,
        fragment: span.fragment,
        extra: f(span.extra),
    }
}

/// Wrapper around parsers allowing to capture both their output and the relevant span.
pub fn with_span<'a, O>(
    parser: impl Fn(Span<'a>) -> NomResult<'a, O>,
) -> impl Fn(Span<'a>) -> NomResult<'a, Spanned<O>> {
    move |input: Span| {
        parser(input).map(|(rest, output)| {
            let spanned = Spanned {
                offset: input.offset,
                line: input.line,
                extra: output,
                fragment: &input.fragment[..(rest.offset - input.offset)],
            };
            (rest, spanned)
        })
    }
}

pub fn unite_spans<'a, T, U>(
    input: Span<'a>,
    start: &Spanned<'_, T>,
    end: &Spanned<'_, U>,
) -> Span<'a> {
    debug_assert!(input.offset <= start.offset);
    debug_assert!(start.offset <= end.offset);
    debug_assert!(input.offset + input.fragment.len() >= end.offset + end.fragment.len());

    let start_idx = start.offset - input.offset;
    let end_idx = end.offset + end.fragment.len() - input.offset;
    Span {
        offset: start.offset,
        line: start.line,
        fragment: &input.fragment[start_idx..end_idx],
        extra: (),
    }
}

pub fn cover_spans<'a, T>(input: Span<'a>, items: &[Spanned<'a, T>]) -> Span<'a> {
    match items {
        [] => Span {
            offset: input.offset,
            line: input.line,
            fragment: "",
            extra: (),
        },
        [item] => create_span_ref(item, ()),
        items => unite_spans(input, items.first().unwrap(), items.last().unwrap()),
    }
}
