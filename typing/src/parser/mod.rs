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

mod conversion;
#[cfg(test)]
mod tests;

/// Type annotation after parsing.
///
/// Compared to [`ValueType`], this enum is structured correspondingly to the AST.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ParsedValueType<'a> {
    /// Type placeholder (`_`).
    Any,
    /// Boolean type (`Bool`).
    Bool,
    /// Number type (`Num`).
    Number,
    /// Reference to a type param.
    Ident(InputSpan<'a>),
    /// Functional type.
    Function(Box<ParsedFnType<'a>>),
    /// Tuple type; for example, `(Num, Bool)`.
    Tuple(Vec<ParsedValueType<'a>>),
    /// Slice type; for example, `[Num]` or `[(Num, T); N]`.
    Slice {
        /// Element of this slice; for example, `Num` in `[Num; N]`.
        element: Box<ParsedValueType<'a>>,
        /// Length of this slice; for example, `N` in `[Num; N]`.
        length: ParsedTupleLength<'a>,
    },
}

impl<'a> ParsedValueType<'a> {
    fn void() -> Self {
        Self::Tuple(vec![])
    }

    /// Parses `input` as a type. This parser can be composed using `nom` infrastructure.
    pub fn parse(input: InputSpan<'a>) -> NomResult<'a, Self> {
        type_definition(input)
    }
}

/// Parsed functional type.
///
/// Compared to [`FnType`], this enum is structured correspondingly to the AST.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ParsedFnType<'a> {
    /// Constant params; e.g., `N` in `fn<const N>([Num; N]) -> Num`.
    pub const_params: Vec<InputSpan<'a>>,
    /// Type params together with their bounds. E.g., `T` in `fn<T>(T, T) -> T`.
    pub type_params: Vec<(InputSpan<'a>, ParsedTypeParamBounds)>,
    /// Function arguments.
    pub args: Vec<ParsedValueType<'a>>,
    /// Return type of the function. Will be set to void if not declared.
    pub return_type: ParsedValueType<'a>,
}

impl<'a> ParsedFnType<'a> {
    /// Parses `input` as a functional type. This parser can be composed using `nom` infrastructure.
    pub fn parse(input: InputSpan<'a>) -> NomResult<'a, Self> {
        fn_definition(input)
    }
}

/// Parsed tuple length.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ParsedTupleLength<'a> {
    /// Dynamic tuple length. This length is *implicit*, as in `[Num]`.
    Dynamic,
    /// Reference to a const; for example, `N` in `[Num; N]`.
    Ident(InputSpan<'a>),
}

/// Bounds that can be placed on a type param.
#[derive(Debug, Default, Clone, PartialEq)]
#[non_exhaustive]
pub struct ParsedTypeParamBounds {
    /// Can the type param be non-linear?
    pub maybe_non_linear: bool,
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

fn ident(input: InputSpan<'_>) -> NomResult<'_, InputSpan<'_>> {
    preceded(
        peek(take_while_m_n(1, 1, |c: char| {
            c.is_ascii_alphabetic() || c == '_'
        })),
        take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_'),
    )(input)
}

fn tuple_definition(input: InputSpan<'_>) -> NomResult<'_, Vec<ParsedValueType<'_>>> {
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

fn slice_definition(
    input: InputSpan<'_>,
) -> NomResult<'_, (ParsedValueType<'_>, ParsedTupleLength<'_>)> {
    let semicolon = tuple((ws, tag_char(';'), ws));
    let tuple_len = map(opt(preceded(semicolon, ident)), |maybe_ident| {
        if let Some(ident) = maybe_ident {
            ParsedTupleLength::Ident(ident)
        } else {
            ParsedTupleLength::Dynamic
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

fn type_bounds(input: InputSpan<'_>) -> NomResult<'_, ParsedTypeParamBounds> {
    map(terminated(tag("?Lin"), ws), |_| ParsedTypeParamBounds {
        maybe_non_linear: true,
    })(input)
}

fn type_params(input: InputSpan<'_>) -> NomResult<'_, Vec<(InputSpan<'_>, ParsedTypeParamBounds)>> {
    let maybe_type_bounds = opt(preceded(tuple((ws, tag_char(':'), ws)), type_bounds));
    let type_param = tuple((ident, map(maybe_type_bounds, Option::unwrap_or_default)));
    separated_list1(comma_sep, type_param)(input)
}

type FnParams<'a> = (
    Vec<InputSpan<'a>>,
    Vec<(InputSpan<'a>, ParsedTypeParamBounds)>,
);

/// Function params, including `<>` brackets.
fn fn_params(input: InputSpan<'_>) -> NomResult<'_, FnParams> {
    let semicolon = tuple((ws, tag_char(';'), ws));
    let const_params = preceded(
        terminated(tag("const"), ws),
        separated_list1(comma_sep, ident),
    );

    let params_parser = alt((
        map(
            tuple((const_params, opt(preceded(semicolon, type_params)))),
            |(const_params, type_params)| (const_params, type_params.unwrap_or_default()),
        ),
        map(type_params, |type_params| (vec![], type_params)),
    ));

    preceded(
        terminated(tag_char('<'), ws),
        cut(terminated(params_parser, tuple((ws, tag_char('>'))))),
    )(input)
}

fn fn_definition(input: InputSpan<'_>) -> NomResult<'_, ParsedFnType<'_>> {
    let return_type = preceded(tuple((ws, tag("->"), ws)), type_definition);
    let fn_parser = tuple((
        opt(fn_params),
        tuple_definition,
        map(opt(return_type), |ty| {
            ty.unwrap_or_else(ParsedValueType::void)
        }),
    ));

    preceded(
        terminated(tag("fn"), ws),
        map(fn_parser, |(params, args, return_type)| {
            let (const_params, type_params) = params.unwrap_or_default();
            ParsedFnType {
                const_params,
                type_params,
                args,
                return_type,
            }
        }),
    )(input)
}

fn type_definition(input: InputSpan<'_>) -> NomResult<'_, ParsedValueType<'_>> {
    alt((
        map(fn_definition, |fn_type| {
            ParsedValueType::Function(Box::new(fn_type))
        }),
        map(ident, |ident| match *ident.fragment() {
            "Num" => ParsedValueType::Number,
            "Bool" => ParsedValueType::Bool,
            "_" => ParsedValueType::Any,
            _ => ParsedValueType::Ident(ident),
        }),
        map(tuple_definition, ParsedValueType::Tuple),
        map(slice_definition, |(element, length)| {
            ParsedValueType::Slice {
                element: Box::new(element),
                length,
            }
        }),
    ))(input)
}
