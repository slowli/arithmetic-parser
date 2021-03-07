//! Parsing type annotations.

use nom::{
    branch::alt,
    bytes::complete::{tag, take_until, take_while, take_while1, take_while_m_n},
    character::complete::char as tag_char,
    combinator::{cut, map, opt, peek, recognize},
    multi::{many0, separated_list0, separated_list1},
    sequence::{delimited, preceded, terminated, tuple},
};

use arithmetic_parser::{InputSpan, NomResult};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq)]
pub enum RawValueType<'a> {
    Any,
    Bool,
    Number,
    Ident(InputSpan<'a>),
    Function(Box<RawFnType<'a>>),
    Tuple(Vec<RawValueType<'a>>),
    Slice {
        element: Box<RawValueType<'a>>,
        length: RawTupleLength<'a>,
    },
}

impl RawValueType<'_> {
    fn void() -> Self {
        Self::Tuple(vec![])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RawFnType<'a> {
    const_params: Vec<InputSpan<'a>>,
    type_params: Vec<InputSpan<'a>>,
    args: Vec<RawValueType<'a>>,
    return_type: RawValueType<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RawTupleLength<'a> {
    Dynamic,
    Ident(InputSpan<'a>),
}

/// Whitespace and comments.
fn ws(input: InputSpan<'_>) -> NomResult<'_, InputSpan<'_>> {
    fn narrow_ws(input: InputSpan<'_>) -> NomResult<'_, InputSpan<'_>> {
        take_while1(|c: char| c.is_ascii_whitespace())(input)
    }

    fn long_comment_body(input: InputSpan<'_>) -> NomResult<'_, InputSpan<'_>> {
        cut(take_until("*/"))(input)
    }

    let comment = preceded(tag("//"), take_while(|c: char| c != '\n'));
    let long_comment = delimited(tag("/*"), long_comment_body, tag("*/"));
    let ws_line = alt((narrow_ws, comment, long_comment));
    recognize(many0(ws_line))(input)
}

/// Comma separator.
fn comma_sep(input: InputSpan<'_>) -> NomResult<'_, char> {
    delimited(ws, tag_char(','), ws)(input)
}

/// Comma-separated list of types.
fn comma_separated_types(input: InputSpan<'_>) -> NomResult<'_, Vec<RawValueType<'_>>> {
    separated_list0(comma_sep, type_definition)(input)
}

fn ident(input: InputSpan<'_>) -> NomResult<'_, InputSpan<'_>> {
    preceded(
        peek(take_while_m_n(1, 1, |c: char| {
            c.is_ascii_alphabetic() || c == '_'
        })),
        take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_'),
    )(input)
}

fn tuple_definition(input: InputSpan<'_>) -> NomResult<'_, Vec<RawValueType<'_>>> {
    let maybe_comma = opt(preceded(ws, tag_char(',')));
    preceded(
        terminated(tag_char('('), ws),
        // Once we've encountered the opening `(`, the input *must* correspond to the parser.
        cut(terminated(
            separated_list0(delimited(ws, tag_char(','), ws), type_definition),
            tuple((maybe_comma, ws, tag_char(')'))),
        )),
    )(input)
}

fn slice_definition(input: InputSpan<'_>) -> NomResult<'_, (RawValueType<'_>, RawTupleLength<'_>)> {
    let semicolon = tuple((ws, tag_char(';'), ws));
    let tuple_len = map(opt(preceded(semicolon, ident)), |maybe_ident| {
        if let Some(ident) = maybe_ident {
            RawTupleLength::Ident(ident)
        } else {
            RawTupleLength::Dynamic
        }
    });

    preceded(
        terminated(tag_char('['), ws),
        // Once we've encountered the opening `[`, the input *must* correspond to the parser.
        cut(terminated(
            tuple((type_definition, tuple_len)),
            tuple((ws, tag_char(']'))),
        )),
    )(input)
}

/// Function params, including `<>` brackets.
fn fn_params(input: InputSpan<'_>) -> NomResult<'_, (Vec<InputSpan<'_>>, Vec<InputSpan<'_>>)> {
    let semicolon = tuple((ws, tag_char(';'), ws));
    let const_params = preceded(
        terminated(tag("const"), ws),
        separated_list1(comma_sep, ident),
    );
    let type_params = separated_list1(comma_sep, ident);

    let params_parser = alt((
        map(
            tuple((const_params, opt(preceded(semicolon, type_params)))),
            |(const_params, type_params)| (const_params, type_params.unwrap_or_default()),
        ),
        map(separated_list1(comma_sep, ident), |type_params| {
            (vec![], type_params)
        }),
    ));

    preceded(
        terminated(tag_char('<'), ws),
        cut(terminated(params_parser, tuple((ws, tag_char('>'))))),
    )(input)
}

fn fn_definition(input: InputSpan<'_>) -> NomResult<'_, RawFnType<'_>> {
    let return_type = preceded(tuple((ws, tag("->"), ws)), type_definition);
    let fn_parser = tuple((
        opt(fn_params),
        tuple_definition,
        map(opt(return_type), |ty| ty.unwrap_or_else(RawValueType::void)),
    ));

    preceded(
        terminated(tag("fn"), ws),
        map(fn_parser, |(params, args, return_type)| {
            let (const_params, type_params) = params.unwrap_or_default();
            RawFnType {
                const_params,
                type_params,
                args,
                return_type,
            }
        }),
    )(input)
}

pub fn type_definition(input: InputSpan<'_>) -> NomResult<'_, RawValueType<'_>> {
    alt((
        map(fn_definition, |fn_type| {
            RawValueType::Function(Box::new(fn_type))
        }),
        map(ident, |ident| match *ident.fragment() {
            "Num" => RawValueType::Number,
            "Bool" => RawValueType::Bool,
            "_" => RawValueType::Any,
            _ => RawValueType::Ident(ident),
        }),
        map(tuple_definition, RawValueType::Tuple),
        map(slice_definition, |(element, length)| RawValueType::Slice {
            element: Box::new(element),
            length,
        }),
    ))(input)
}
