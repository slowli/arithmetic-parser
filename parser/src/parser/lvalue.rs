//! Lvalue-related parsing functions.

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::char as tag_char,
    combinator::{cut, map, not, opt, peek},
    multi::{separated_list0, separated_list1},
    sequence::{delimited, preceded, terminated, tuple},
    Err as NomErr,
};

use super::helpers::{comma_sep, var_name, ws, GrammarType};
use crate::{
    alloc::{vec, Vec},
    grammars::{Features, Grammar, Parse},
    spans::with_span,
    Destructure, DestructureRest, ErrorKind, InputSpan, Lvalue, NomResult, ObjectDestructure,
    ObjectDestructureField, Spanned, SpannedLvalue,
};

fn comma_separated_lvalues<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, Vec<GrammarLvalue<'_, T>>>
where
    T: Parse,
    Ty: GrammarType,
{
    separated_list0(comma_sep::<Ty>, lvalue::<T, Ty>)(input)
}

fn destructure_rest<T, Ty>(
    input: InputSpan<'_>,
) -> NomResult<'_, Spanned<'_, DestructureRest<'_, <T::Base as Grammar>::Type<'_>>>>
where
    T: Parse,
    Ty: GrammarType,
{
    map(
        with_span(preceded(
            terminated(tag("..."), ws::<Ty>),
            cut(opt(simple_lvalue_with_type::<T, Ty>)),
        )),
        |spanned| {
            spanned.map_extra(|maybe_lvalue| {
                maybe_lvalue.map_or(DestructureRest::Unnamed, |lvalue| DestructureRest::Named {
                    variable: lvalue.with_no_extra(),
                    ty: match lvalue.extra {
                        Lvalue::Variable { ty } => ty,
                        _ => None,
                    },
                })
            })
        },
    )(input)
}

type DestructureTail<'a, T> = (
    Spanned<'a, DestructureRest<'a, T>>,
    Option<Vec<SpannedLvalue<'a, T>>>,
);

fn destructure_tail<T, Ty>(
    input: InputSpan<'_>,
) -> NomResult<'_, DestructureTail<'_, <T::Base as Grammar>::Type<'_>>>
where
    T: Parse,
    Ty: GrammarType,
{
    tuple((
        destructure_rest::<T, Ty>,
        opt(preceded(comma_sep::<Ty>, comma_separated_lvalues::<T, Ty>)),
    ))(input)
}

/// Parse the destructuring *without* the surrounding delimiters.
pub(super) fn destructure<T, Ty>(
    input: InputSpan<'_>,
) -> NomResult<'_, Destructure<'_, <T::Base as Grammar>::Type<'_>>>
where
    T: Parse,
    Ty: GrammarType,
{
    let main_parser = alt((
        // `destructure_tail` has fast fail path: the input must start with `...`.
        map(destructure_tail::<T, Ty>, |rest| (vec![], Some(rest))),
        tuple((
            comma_separated_lvalues::<T, Ty>,
            opt(preceded(comma_sep::<Ty>, destructure_tail::<T, Ty>)),
        )),
    ));
    // Allow for `,`-terminated lists.
    let main_parser = terminated(main_parser, opt(comma_sep::<Ty>));

    map(main_parser, |(start, maybe_rest)| {
        if let Some((middle, end)) = maybe_rest {
            Destructure {
                start,
                middle: Some(middle),
                end: end.unwrap_or_default(),
            }
        } else {
            Destructure {
                start,
                middle: None,
                end: vec![],
            }
        }
    })(input)
}

type GrammarLvalue<'a, T> = SpannedLvalue<'a, <<T as Parse>::Base as Grammar>::Type<'a>>;

fn parenthesized_destructure<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, GrammarLvalue<'_, T>>
where
    T: Parse,
    Ty: GrammarType,
{
    with_span(map(
        delimited(
            terminated(tag_char('('), ws::<Ty>),
            destructure::<T, Ty>,
            preceded(ws::<Ty>, tag_char(')')),
        ),
        Lvalue::Tuple,
    ))(input)
}

/// Simple lvalue with an optional type annotation, e.g., `x` or `x: Num`.
fn simple_lvalue_with_type<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, GrammarLvalue<'_, T>>
where
    T: Parse,
    Ty: GrammarType,
{
    // Do not consider `::` as a type delimiter; otherwise, parsing will be inappropriately cut
    // when `$var_name::...` is encountered.
    let type_delimiter = terminated(tag_char(':'), peek(not(tag_char(':'))));
    map(
        tuple((
            var_name,
            opt(preceded(
                delimited(ws::<Ty>, type_delimiter, ws::<Ty>),
                cut(with_span(<T::Base>::parse_type)),
            )),
        )),
        |(name, ty)| Spanned::new(name, Lvalue::Variable { ty }),
    )(input)
}

fn simple_lvalue_without_type<T>(input: InputSpan<'_>) -> NomResult<'_, GrammarLvalue<'_, T>>
where
    T: Parse,
{
    map(var_name, |name| {
        Spanned::new(name, Lvalue::Variable { ty: None })
    })(input)
}

fn object_destructure_field<T, Ty>(
    input: InputSpan<'_>,
) -> NomResult<'_, ObjectDestructureField<'_, <T::Base as Grammar>::Type<'_>>>
where
    T: Parse,
    Ty: GrammarType,
{
    let field_sep = alt((tag(":"), tag("->")));
    let field_sep = tuple((ws::<Ty>, field_sep, ws::<Ty>));
    let field = tuple((var_name, opt(preceded(field_sep, lvalue::<T, Ty>))));
    map(field, |(name, maybe_binding)| ObjectDestructureField {
        field_name: Spanned::new(name, ()),
        binding: maybe_binding,
    })(input)
}

pub(super) fn object_destructure<T, Ty>(
    input: InputSpan<'_>,
) -> NomResult<'_, ObjectDestructure<'_, <T::Base as Grammar>::Type<'_>>>
where
    T: Parse,
    Ty: GrammarType,
{
    let inner = separated_list1(comma_sep::<Ty>, object_destructure_field::<T, Ty>);
    let inner = terminated(inner, opt(comma_sep::<Ty>));
    let inner = delimited(
        terminated(tag_char('{'), ws::<Ty>),
        inner,
        preceded(ws::<Ty>, tag_char('}')),
    );
    map(inner, |fields| ObjectDestructure { fields })(input)
}

fn mapped_object_destructure<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, GrammarLvalue<'_, T>>
where
    T: Parse,
    Ty: GrammarType,
{
    with_span(map(object_destructure::<T, Ty>, Lvalue::Object))(input)
}

/// Parses an `Lvalue`.
pub(super) fn lvalue<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, GrammarLvalue<'_, T>>
where
    T: Parse,
    Ty: GrammarType,
{
    fn error<T>(input: InputSpan<'_>) -> NomResult<'_, GrammarLvalue<'_, T>>
    where
        T: Parse,
    {
        let e = ErrorKind::Leftovers.with_span(&input.into());
        Err(NomErr::Error(e))
    }

    let simple_lvalue = if T::FEATURES.contains(Features::TYPE_ANNOTATIONS) {
        simple_lvalue_with_type::<T, Ty>
    } else {
        simple_lvalue_without_type::<T>
    };
    let destructure = if T::FEATURES.contains(Features::TUPLES) {
        parenthesized_destructure::<T, Ty>
    } else {
        error::<T>
    };
    let object_destructure = if T::FEATURES.contains(Features::OBJECTS) {
        mapped_object_destructure::<T, Ty>
    } else {
        error::<T>
    };
    alt((destructure, object_destructure, simple_lvalue))(input)
}
