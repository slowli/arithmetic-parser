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
    Err as NomErr,
};

use std::str::FromStr;

use crate::{Num, PrimitiveType};
use arithmetic_parser::{ErrorKind as ParserErrorKind, InputSpan, NomResult};

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
/// let input = InputSpan::new("(Num, ('T) -> ('T, 'T))");
/// let elements = match ValueTypeAst::parse(input)?.1 {
///     ValueTypeAst::Tuple(elements) => elements,
///     _ => unreachable!(),
/// };
/// assert_eq!(elements.start[0], ValueTypeAst::Prim(Num::Num));
/// assert_matches!(
///     &elements.start[1],
///     ValueTypeAst::Function { .. }
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ValueTypeAst<'a, Prim: PrimitiveType = Num> {
    /// Type placeholder (`_`). Corresponds to any single type.
    Any,
    /// Primitive types.
    Prim(Prim),
    /// Ticked identifier, e.g., `'T`.
    Param(InputSpan<'a>),
    /// Functional type.
    Function {
        /// Constraints on function params.
        constraints: Option<ConstraintsAst<'a, Prim>>,
        /// Function body.
        function: Box<FnTypeAst<'a, Prim>>,
    },
    /// Tuple type; for example, `(Num, Bool)`.
    Tuple(TupleAst<'a, Prim>),
    /// Slice type; for example, `[Num]` or `[(Num, T); N]`.
    Slice(SliceAst<'a, Prim>),
}

impl<'a, Prim: PrimitiveType> ValueTypeAst<'a, Prim> {
    /// Parses `input` as a type. This parser can be composed using `nom` infrastructure.
    pub fn parse(input: InputSpan<'a>) -> NomResult<'a, Self> {
        type_definition(input)
    }
}

/// Parsed tuple type, such as `(Num, Bool)` or `(fn() -> Num, ...[Num; _])`.
#[derive(Debug, Clone, PartialEq)]
pub struct TupleAst<'a, Prim: PrimitiveType = Num> {
    /// Elements at the beginning of the tuple, e.g., `Num` and `Bool`
    /// in `(Num, Bool, ...[T; _])`.
    pub start: Vec<ValueTypeAst<'a, Prim>>,
    /// Middle of the tuple, e.g., `[T; _]` in `(Num, Bool, ...[T; _])`.
    pub middle: Option<SliceAst<'a, Prim>>,
    /// Elements at the end of the tuple, e.g., `Bool` in `(...[Num; _], Bool)`.
    /// Guaranteed to be empty if `middle` is not present.
    pub end: Vec<ValueTypeAst<'a, Prim>>,
}

/// Parsed slice type, such as `[Num; N]`.
#[derive(Debug, Clone, PartialEq)]
pub struct SliceAst<'a, Prim: PrimitiveType = Num> {
    /// Element of this slice; for example, `Num` in `[Num; N]`.
    pub element: Box<ValueTypeAst<'a, Prim>>,
    /// Length of this slice; for example, `N` in `[Num; N]`.
    pub length: TupleLenAst<'a>,
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
/// let input = InputSpan::new("([Num; N]) -> Num");
/// let (rest, ty) = FnTypeAst::parse(input)?;
/// assert!(rest.fragment().is_empty());
/// assert_matches!(ty.args.start.as_slice(), [ValueTypeAst::Slice(_)]);
/// assert_eq!(ty.return_type, ValueTypeAst::Prim(Num::Num));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FnTypeAst<'a, Prim: PrimitiveType = Num> {
    /// Function arguments.
    pub args: TupleAst<'a, Prim>,
    /// Return type of the function.
    pub return_type: ValueTypeAst<'a, Prim>,
}

impl<'a, Prim: PrimitiveType> FnTypeAst<'a, Prim> {
    /// Parses `input` as a functional type. This parser can be composed using `nom` infrastructure.
    pub fn parse(input: InputSpan<'a>) -> NomResult<'a, Self> {
        fn_definition(input)
    }
}

/// Parsed tuple length.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TupleLenAst<'a> {
    /// Length placeholder (`_`). Corresponds to any single length.
    Some,
    /// Dynamic tuple length. This length is *implicit*, as in `[Num]`.
    Dynamic,
    /// Reference to a length; for example, `N` in `[Num; N]`.
    Ident(InputSpan<'a>),
}

/// Parameter constraints, e.g. `for<len N*; T: Lin>`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ConstraintsAst<'a, Prim: PrimitiveType> {
    /// Span of the `for` keyword.
    pub for_keyword: InputSpan<'a>,
    /// Dynamic lengths, e.g., `N` in `for<len N*>`.
    pub dyn_lengths: Vec<InputSpan<'a>>,
    /// Type constraints.
    pub type_params: Vec<(InputSpan<'a>, TypeConstraintsAst<'a, Prim>)>,
}

/// Bounds that can be placed on a type variable.
#[derive(Debug, Default, Clone, PartialEq)]
#[non_exhaustive]
pub struct TypeConstraintsAst<'a, Prim: PrimitiveType> {
    /// Spans corresponding to constraints, e.g. `Foo` and `Bar` in `Foo + Bar`.
    pub terms: Vec<InputSpan<'a>>,
    /// Computed constraint.
    pub computed: Prim::Constraints,
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

fn type_param_ident(input: InputSpan<'_>) -> NomResult<'_, InputSpan<'_>> {
    preceded(tag_char('\''), ident)(input)
}

fn comma_separated_types<Prim: PrimitiveType>(
    input: InputSpan<'_>,
) -> NomResult<'_, Vec<ValueTypeAst<'_, Prim>>> {
    separated_list0(delimited(ws, tag_char(','), ws), type_definition)(input)
}

fn tuple_middle<Prim: PrimitiveType>(input: InputSpan<'_>) -> NomResult<'_, SliceAst<'_, Prim>> {
    preceded(terminated(tag("..."), ws), slice_definition)(input)
}

type TupleTailAst<'a, Prim> = (SliceAst<'a, Prim>, Vec<ValueTypeAst<'a, Prim>>);

fn tuple_tail<Prim: PrimitiveType>(input: InputSpan<'_>) -> NomResult<'_, TupleTailAst<'_, Prim>> {
    tuple((
        tuple_middle,
        map(
            opt(preceded(comma_sep, comma_separated_types)),
            Option::unwrap_or_default,
        ),
    ))(input)
}

fn tuple_definition<Prim: PrimitiveType>(
    input: InputSpan<'_>,
) -> NomResult<'_, TupleAst<'_, Prim>> {
    let maybe_comma = opt(comma_sep);

    let main_parser = alt((
        map(tuple_tail, |(middle, end)| TupleAst {
            start: Vec::new(),
            middle: Some(middle),
            end,
        }),
        map(
            tuple((comma_separated_types, opt(preceded(comma_sep, tuple_tail)))),
            |(start, maybe_tail)| {
                if let Some((middle, end)) = maybe_tail {
                    TupleAst {
                        start,
                        middle: Some(middle),
                        end,
                    }
                } else {
                    TupleAst {
                        start,
                        middle: None,
                        end: Vec::new(),
                    }
                }
            },
        ),
    ));

    preceded(
        terminated(tag_char('('), ws),
        // Once we've encountered the opening `(`, the input *must* correspond to the parser.
        cut(terminated(
            main_parser,
            tuple((maybe_comma, ws, tag_char(')'))),
        )),
    )(input)
}

fn slice_definition<Prim: PrimitiveType>(
    input: InputSpan<'_>,
) -> NomResult<'_, SliceAst<'_, Prim>> {
    let semicolon = tuple((ws, tag_char(';'), ws));
    let tuple_len = map(
        opt(preceded(semicolon, ident)),
        |maybe_ident| match maybe_ident {
            Some(ident) if *ident.fragment() == "_" => TupleLenAst::Some,
            Some(ident) => TupleLenAst::Ident(ident),
            None => TupleLenAst::Dynamic,
        },
    );

    preceded(
        terminated(tag_char('['), ws),
        // Once we've encountered the opening `[`, the input *must* correspond to the parser.
        cut(terminated(
            map(tuple((type_definition, tuple_len)), |(element, length)| {
                SliceAst {
                    element: Box::new(element),
                    length,
                }
            }),
            tuple((ws, tag_char(']'))),
        )),
    )(input)
}

fn type_bounds<Prim: PrimitiveType>(
    input: InputSpan<'_>,
) -> NomResult<'_, TypeConstraintsAst<'_, Prim>> {
    let constraint_sep = tuple((ws, tag_char('+'), ws));
    let (rest, terms) = separated_list1(constraint_sep, ident)(input)?;

    let computed = terms
        .iter()
        .try_fold(Prim::Constraints::default(), |mut acc, &input| {
            let input_str = *input.fragment();
            let partial = Prim::Constraints::from_str(input_str).map_err(|_| {
                let err = anyhow::anyhow!("Cannot parse type constraint");
                ParserErrorKind::Type(err).with_span(&input.into())
            })?;
            acc |= &partial;
            Ok(acc)
        })
        .map_err(NomErr::Failure)?;

    Ok((rest, TypeConstraintsAst { terms, computed }))
}

fn type_params<Prim: PrimitiveType>(
    input: InputSpan<'_>,
) -> NomResult<'_, Vec<(InputSpan<'_>, TypeConstraintsAst<'_, Prim>)>> {
    let type_bounds = preceded(tuple((ws, tag_char(':'), ws)), type_bounds);
    let type_param = tuple((type_param_ident, type_bounds));
    separated_list1(comma_sep, type_param)(input)
}

/// Function params, including the `for` keyword and `<>` brackets.
fn constraints<Prim: PrimitiveType>(
    input: InputSpan<'_>,
) -> NomResult<'_, ConstraintsAst<'_, Prim>> {
    let semicolon = tuple((ws, tag_char(';'), ws));

    let len_param = terminated(ident, tag_char('*'));
    let len_params = preceded(
        terminated(tag("len"), ws),
        separated_list1(comma_sep, len_param),
    );

    let params_parser = alt((
        map(
            tuple((len_params, opt(preceded(semicolon, type_params)))),
            |(dyn_lengths, type_params)| (dyn_lengths, type_params.unwrap_or_default()),
        ),
        map(type_params, |type_params| (vec![], type_params)),
    ));

    let constraints_parser = tuple((
        terminated(tag("for"), ws),
        terminated(tag_char('<'), ws),
        cut(terminated(params_parser, tuple((ws, tag_char('>'))))),
    ));

    map(
        constraints_parser,
        |(for_keyword, _, (dyn_lengths, type_params))| ConstraintsAst {
            for_keyword,
            dyn_lengths,
            type_params,
        },
    )(input)
}

fn return_type<Prim: PrimitiveType>(input: InputSpan<'_>) -> NomResult<'_, ValueTypeAst<'_, Prim>> {
    preceded(tuple((ws, tag("->"), ws)), cut(type_definition))(input)
}

#[allow(clippy::option_if_let_else)] // false positive; `args` is moved into both clauses
fn fn_or_tuple<Prim: PrimitiveType>(input: InputSpan<'_>) -> NomResult<'_, ValueTypeAst<'_, Prim>> {
    map(
        tuple((tuple_definition, opt(return_type))),
        |(args, return_type)| {
            if let Some(return_type) = return_type {
                ValueTypeAst::Function {
                    function: Box::new(FnTypeAst { args, return_type }),
                    constraints: None,
                }
            } else {
                ValueTypeAst::Tuple(args)
            }
        },
    )(input)
}

fn fn_definition<Prim: PrimitiveType>(input: InputSpan<'_>) -> NomResult<'_, FnTypeAst<'_, Prim>> {
    map(
        tuple((tuple_definition, return_type)),
        |(args, return_type)| FnTypeAst { args, return_type },
    )(input)
}

fn fn_definition_with_constraints<Prim: PrimitiveType>(
    input: InputSpan<'_>,
) -> NomResult<'_, (ConstraintsAst<'_, Prim>, FnTypeAst<'_, Prim>)> {
    map(
        tuple((constraints, ws, fn_definition)),
        |(constraints, _, fn_def)| (constraints, fn_def),
    )(input)
}

fn free_ident<Prim: PrimitiveType>(input: InputSpan<'_>) -> NomResult<'_, ValueTypeAst<'_, Prim>> {
    let (rest, ident) = ident(input)?;

    let output = if let Ok(res) = ident.fragment().parse::<Prim>() {
        ValueTypeAst::Prim(res)
    } else if *ident.fragment() == "_" {
        ValueTypeAst::Any
    } else {
        let err = anyhow::anyhow!("Unknown type name: {}", ident.fragment());
        let err = ParserErrorKind::Type(err).with_span(&ident.into());
        return Err(NomErr::Failure(err));
    };
    Ok((rest, output))
}

fn type_definition<Prim: PrimitiveType>(
    input: InputSpan<'_>,
) -> NomResult<'_, ValueTypeAst<'_, Prim>> {
    alt((
        fn_or_tuple,
        map(fn_definition_with_constraints, |(constraints, fn_type)| {
            ValueTypeAst::Function {
                constraints: Some(constraints),
                function: Box::new(fn_type),
            }
        }),
        map(type_param_ident, ValueTypeAst::Param),
        map(slice_definition, ValueTypeAst::Slice),
        free_ident,
    ))(input)
}
