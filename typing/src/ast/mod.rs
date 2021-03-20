//! ASTs for type annotations and their parsing logic.
//!
//! # Overview
//!
//! This module contains types representing AST for parsed type annotations; for example,
//! [`ValueTypeAst`] and [`FnTypeAst`]. These two types expose `parse` method which
//! allows to integrate them into `nom` parsing.

use nom::{
    branch::alt,
    bytes::complete::{tag, take_until, take_while, take_while1, take_while_m_n},
    character::complete::char as tag_char,
    combinator::{cut, map, opt, peek, recognize},
    multi::{many0, separated_list0, separated_list1},
    sequence::{delimited, preceded, terminated, tuple},
};

use crate::{LiteralType, Num};
use arithmetic_parser::{InputSpan, NomResult};

mod conversion;
#[cfg(test)]
mod tests;

pub use self::conversion::{ConversionError, ConversionErrorKind};

/// Type annotation after parsing.
///
/// Compared to [`ValueType`], this enum corresponds to AST, not to the logical presentation
/// of a type.
///
/// [`ValueType`]: crate::ValueType
///
/// # Examples
///
/// ```
/// use arithmetic_parser::InputSpan;
/// # use arithmetic_typing::{ast::ValueTypeAst, Num};
/// # use assert_matches::assert_matches;
///
/// # fn main() -> anyhow::Result<()> {
/// let input = InputSpan::new("(Num, fn<T>(T) -> (T, T))");
/// let elements = match ValueTypeAst::parse(input)?.1 {
///     ValueTypeAst::Tuple(elements) => elements,
///     _ => unreachable!(),
/// };
/// assert_eq!(elements[0], ValueTypeAst::Lit(Num));
/// assert_matches!(
///     &elements[1],
///     ValueTypeAst::Function(f) if f.type_params.len() == 1
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ValueTypeAst<'a, Lit = Num> {
    /// Type placeholder (`_`). Corresponds to any single type.
    Any,
    /// Boolean type (`Bool`).
    Bool,
    /// Literal types.
    Lit(Lit),
    /// Reference to a type param.
    Ident(InputSpan<'a>),
    /// Functional type.
    Function(Box<FnTypeAst<'a, Lit>>),
    /// Tuple type; for example, `(Num, Bool)`.
    Tuple(Vec<ValueTypeAst<'a, Lit>>),
    /// Slice type; for example, `[Num]` or `[(Num, T); N]`.
    Slice {
        /// Element of this slice; for example, `Num` in `[Num; N]`.
        element: Box<ValueTypeAst<'a, Lit>>,
        /// Length of this slice; for example, `N` in `[Num; N]`.
        length: TupleLengthAst<'a>,
    },
}

impl<'a, Lit: LiteralType> ValueTypeAst<'a, Lit> {
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
/// In contrast to [`FnType`], this struct corresponds to AST, not to the logical representation
/// of functional types.
///
/// [`FnType`]: crate::FnType
///
/// # Examples
///
/// ```
/// use arithmetic_parser::InputSpan;
/// # use assert_matches::assert_matches;
/// # use arithmetic_typing::{ast::{FnTypeAst, ValueTypeAst}, Num};
///
/// # fn main() -> anyhow::Result<()> {
/// let input = InputSpan::new("fn<const N>([Num; N]) -> Num");
/// let (rest, ty) = FnTypeAst::parse(input)?;
/// assert!(rest.fragment().is_empty());
/// assert_eq!(ty.const_params.len(), 1);
/// assert!(ty.type_params.is_empty());
/// assert_matches!(ty.args.as_slice(), [ValueTypeAst::Slice { .. }]);
/// assert_eq!(ty.return_type, ValueTypeAst::Lit(Num));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FnTypeAst<'a, Lit = Num> {
    /// Constant params; e.g., `N` in `fn<const N>([Num; N]) -> Num`.
    pub const_params: Vec<InputSpan<'a>>,
    /// Type params together with their bounds. E.g., `T` in `fn<T>(T, T) -> T`.
    pub type_params: Vec<(InputSpan<'a>, TypeConstraintsAst<'a>)>,
    /// Function arguments.
    pub args: Vec<ValueTypeAst<'a, Lit>>,
    /// Return type of the function. Will be set to void if not declared.
    pub return_type: ValueTypeAst<'a, Lit>,
}

impl<'a, Lit: LiteralType> FnTypeAst<'a, Lit> {
    /// Parses `input` as a functional type. This parser can be composed using `nom` infrastructure.
    pub fn parse(input: InputSpan<'a>) -> NomResult<'a, Self> {
        fn_definition(input)
    }
}

/// Parsed tuple length.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TupleLengthAst<'a> {
    /// Const placeholder (`_`). Corresponds to any single type.
    Any,
    /// Dynamic tuple length. This length is *implicit*, as in `[Num]`.
    Dynamic,
    /// Reference to a const; for example, `N` in `[Num; N]`.
    Ident(InputSpan<'a>),
}

/// Bounds that can be placed on a type param.
#[derive(Debug, Default, Clone, PartialEq)]
#[non_exhaustive]
pub struct TypeConstraintsAst<'a> {
    /// Spans corresponding to constraints, e.g. `Foo` and `Bar` in `Foo + Bar`.
    pub constraints: Vec<InputSpan<'a>>,
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

fn tuple_definition<Lit: LiteralType>(
    input: InputSpan<'_>,
) -> NomResult<'_, Vec<ValueTypeAst<'_, Lit>>> {
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

fn slice_definition<Lit: LiteralType>(
    input: InputSpan<'_>,
) -> NomResult<'_, (ValueTypeAst<'_, Lit>, TupleLengthAst<'_>)> {
    let semicolon = tuple((ws, tag_char(';'), ws));
    let tuple_len = map(
        opt(preceded(semicolon, ident)),
        |maybe_ident| match maybe_ident {
            Some(ident) if *ident.fragment() == "_" => TupleLengthAst::Any,
            Some(ident) => TupleLengthAst::Ident(ident),
            None => TupleLengthAst::Dynamic,
        },
    );

    preceded(
        terminated(tag_char('['), ws),
        // Once we've encountered the opening `[`, the input *must* correspond to the parser.
        cut(terminated(
            tuple((type_definition, tuple_len)),
            tuple((ws, tag_char(']'))),
        )),
    )(input)
}

fn type_bounds(input: InputSpan<'_>) -> NomResult<'_, TypeConstraintsAst<'_>> {
    let constraint_sep = tuple((ws, tag_char('+'), ws));
    map(separated_list1(constraint_sep, ident), |constraints| {
        TypeConstraintsAst { constraints }
    })(input)
}

fn type_params(input: InputSpan<'_>) -> NomResult<'_, Vec<(InputSpan<'_>, TypeConstraintsAst)>> {
    let maybe_type_bounds = opt(preceded(tuple((ws, tag_char(':'), ws)), type_bounds));
    let type_param = tuple((ident, map(maybe_type_bounds, Option::unwrap_or_default)));
    separated_list1(comma_sep, type_param)(input)
}

type FnParams<'a> = (
    Vec<InputSpan<'a>>,
    Vec<(InputSpan<'a>, TypeConstraintsAst<'a>)>,
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

fn fn_definition<Lit: LiteralType>(input: InputSpan<'_>) -> NomResult<'_, FnTypeAst<'_, Lit>> {
    let return_type = preceded(tuple((ws, tag("->"), ws)), type_definition);
    let fn_parser = tuple((
        opt(fn_params),
        tuple_definition,
        map(opt(return_type), |ty| ty.unwrap_or_else(ValueTypeAst::void)),
    ));

    preceded(
        terminated(tag("fn"), ws),
        map(fn_parser, |(params, args, return_type)| {
            let (const_params, type_params) = params.unwrap_or_default();
            FnTypeAst {
                const_params,
                type_params,
                args,
                return_type,
            }
        }),
    )(input)
}

fn type_definition<Lit: LiteralType>(input: InputSpan<'_>) -> NomResult<'_, ValueTypeAst<'_, Lit>> {
    alt((
        map(fn_definition, |fn_type| {
            ValueTypeAst::Function(Box::new(fn_type))
        }),
        map(ident, |ident| {
            if let Ok(res) = ident.fragment().parse::<Lit>() {
                return ValueTypeAst::Lit(res);
            }
            match *ident.fragment() {
                "Bool" => ValueTypeAst::Bool,
                "_" => ValueTypeAst::Any,
                _ => ValueTypeAst::Ident(ident),
            }
        }),
        map(tuple_definition, ValueTypeAst::Tuple),
        map(slice_definition, |(element, length)| ValueTypeAst::Slice {
            element: Box::new(element),
            length,
        }),
    ))(input)
}
