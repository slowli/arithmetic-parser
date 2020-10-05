//! Misc helpers.

use crate::{NomResult, Span, Spanned};

/// Wrapper around parsers allowing to capture both their output and the relevant span.
pub fn with_span<'a, O>(
    parser: impl Fn(Span<'a>) -> NomResult<'a, O>,
) -> impl Fn(Span<'a>) -> NomResult<'a, Spanned<O>> {
    move |input: Span| {
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

pub fn unite_spans<'a, T, U>(
    input: Span<'a>,
    start: &Spanned<'_, T>,
    end: &Spanned<'_, U>,
) -> Span<'a> {
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
        Span::new_from_raw_offset(
            start.location_offset(),
            start.location_line(),
            &input.fragment()[start_idx..end_idx],
            (),
        )
    }
}
