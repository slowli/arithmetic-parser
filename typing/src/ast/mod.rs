//! ASTs for type annotations and their parsing logic.
//!
//! # Overview
//!
//! This module contains types representing AST for parsed type annotations; for example,
//! [`TypeAst`] and [`FnTypeAst`]. These two types expose `parse` method which
//! allows to integrate them into `nom` parsing.

use nom::{
    branch::alt,
    bytes::complete::{tag, take, take_until, take_while, take_while1, take_while_m_n},
    character::complete::char as tag_char,
    combinator::{cut, map, map_res, not, opt, peek, recognize},
    multi::{many0, separated_list0, separated_list1},
    sequence::{delimited, preceded, separated_pair, terminated, tuple},
};

use arithmetic_parser::{with_span, ErrorKind as ParseErrorKind, InputSpan, NomResult, Spanned};

mod conversion;
#[cfg(test)]
mod tests;

pub use self::conversion::AstConversionError;
pub(crate) use self::conversion::AstConversionState;

/// Type annotation after parsing.
///
/// Compared to [`Type`], this enum corresponds to AST, not to the logical presentation
/// of a type.
///
/// [`Type`]: crate::Type
///
/// # Examples
///
/// ```
/// use arithmetic_parser::InputSpan;
/// # use arithmetic_typing::{ast::TypeAst, Num};
/// # use assert_matches::assert_matches;
///
/// # fn main() -> anyhow::Result<()> {
/// let input = InputSpan::new("(Num, ('T) -> ('T, 'T))");
/// let (_, ty) = TypeAst::parse(input)?;
/// let elements = match ty.extra {
///     TypeAst::Tuple(elements) => elements,
///     _ => unreachable!(),
/// };
/// assert_eq!(elements.start[0].extra, TypeAst::Ident);
/// assert_matches!(
///     &elements.start[1].extra,
///     TypeAst::Function { .. }
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TypeAst<'a> {
    /// Type placeholder (`_`). Corresponds to a certain type that is not specified, like `_`
    /// in type annotations in Rust.
    Some,
    /// Any type (`any`).
    Any(TypeConstraintsAst<'a>),
    /// Non-ticked identifier, e.g., `Bool`.
    Ident,
    /// Ticked identifier, e.g., `'T`.
    Param,
    /// Functional type.
    Function(Box<FnTypeAst<'a>>),
    /// Functional type with constraints.
    FunctionWithConstraints {
        /// Constraints on function params.
        constraints: Spanned<'a, ConstraintsAst<'a>>,
        /// Function body.
        function: Box<Spanned<'a, FnTypeAst<'a>>>,
    },
    /// Tuple type; for example, `(Num, Bool)`.
    Tuple(TupleAst<'a>),
    /// Slice type; for example, `[Num]` or `[(Num, T); N]`.
    Slice(SliceAst<'a>),
    /// Object type; for example, `{ len: Num }`. Not to be confused with object constraints.
    Object(ObjectAst<'a>),
}

impl<'a> TypeAst<'a> {
    /// Parses `input` as a type. This parser can be composed using `nom` infrastructure.
    pub fn parse(input: InputSpan<'a>) -> NomResult<'a, Spanned<'a, Self>> {
        with_span(type_definition)(input)
    }
}

/// Spanned [`TypeAst`].
pub type SpannedTypeAst<'a> = Spanned<'a, TypeAst<'a>>;

/// Parsed tuple type, such as `(Num, Bool)` or `(fn() -> Num, ...[Num; _])`.
#[derive(Debug, Clone, PartialEq)]
pub struct TupleAst<'a> {
    /// Elements at the beginning of the tuple, e.g., `Num` and `Bool`
    /// in `(Num, Bool, ...[T; _])`.
    pub start: Vec<SpannedTypeAst<'a>>,
    /// Middle of the tuple, e.g., `[T; _]` in `(Num, Bool, ...[T; _])`.
    pub middle: Option<Spanned<'a, SliceAst<'a>>>,
    /// Elements at the end of the tuple, e.g., `Bool` in `(...[Num; _], Bool)`.
    /// Guaranteed to be empty if `middle` is not present.
    pub end: Vec<SpannedTypeAst<'a>>,
}

/// Parsed slice type, such as `[Num; N]`.
#[derive(Debug, Clone, PartialEq)]
pub struct SliceAst<'a> {
    /// Element of this slice; for example, `Num` in `[Num; N]`.
    pub element: Box<SpannedTypeAst<'a>>,
    /// Length of this slice; for example, `N` in `[Num; N]`.
    pub length: Spanned<'a, TupleLenAst>,
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
/// # use arithmetic_typing::{ast::{FnTypeAst, TypeAst}, Num};
///
/// # fn main() -> anyhow::Result<()> {
/// let input = InputSpan::new("([Num; N]) -> Num");
/// let (rest, ty) = FnTypeAst::parse(input)?;
/// assert!(rest.fragment().is_empty());
/// assert_matches!(ty.args.extra.start[0].extra, TypeAst::Slice(_));
/// assert_eq!(ty.return_type.extra, TypeAst::Ident);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FnTypeAst<'a> {
    /// Function arguments.
    pub args: Spanned<'a, TupleAst<'a>>,
    /// Return type of the function.
    pub return_type: SpannedTypeAst<'a>,
}

impl<'a> FnTypeAst<'a> {
    /// Parses `input` as a functional type. This parser can be composed using `nom` infrastructure.
    pub fn parse(input: InputSpan<'a>) -> NomResult<'a, Self> {
        fn_definition(input)
    }
}

/// Parsed tuple length.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TupleLenAst {
    /// Length placeholder (`_`). Corresponds to any single length.
    Some,
    /// Dynamic tuple length. This length is *implicit*, as in `[Num]`. As such, it has
    /// an empty span.
    Dynamic,
    /// Reference to a length; for example, `N` in `[Num; N]`.
    Ident,
}

/// Parameter constraints, e.g. `for<len! N; T: Lin>`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ConstraintsAst<'a> {
    /// Static lengths, e.g., `N` in `for<len! N>`.
    pub static_lengths: Vec<Spanned<'a>>,
    /// Type constraints.
    pub type_params: Vec<(Spanned<'a>, TypeConstraintsAst<'a>)>,
}

/// Bounds that can be placed on a type variable.
#[derive(Debug, Default, Clone, PartialEq)]
#[non_exhaustive]
pub struct TypeConstraintsAst<'a> {
    /// Object constraint, such as `{ x: 'T }`.
    pub object: Option<ObjectAst<'a>>,
    /// Spans corresponding to constraints, e.g. `Foo` and `Bar` in `Foo + Bar`.
    pub terms: Vec<Spanned<'a>>,
}

/// Object type or constraint, such as `{ x: Num, y: [(Num, Bool)] }`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ObjectAst<'a> {
    /// Fields of the object.
    pub fields: Vec<(Spanned<'a>, SpannedTypeAst<'a>)>,
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

fn ident(input: InputSpan<'_>) -> NomResult<'_, Spanned<'_>> {
    preceded(
        peek(take_while_m_n(1, 1, |c: char| {
            c.is_ascii_alphabetic() || c == '_'
        })),
        map(
            take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_'),
            Spanned::from,
        ),
    )(input)
}

fn not_keyword(input: InputSpan<'_>) -> NomResult<'_, Spanned<'_>> {
    map_res(ident, |ident| {
        if *ident.fragment() == "as" {
            Err(ParseErrorKind::Type(anyhow::anyhow!(
                "`as` is a reserved keyword"
            )))
        } else {
            Ok(ident)
        }
    })(input)
}

fn type_param_ident(input: InputSpan<'_>) -> NomResult<'_, Spanned<'_>> {
    preceded(tag_char('\''), ident)(input)
}

fn comma_separated_types(input: InputSpan<'_>) -> NomResult<'_, Vec<SpannedTypeAst<'_>>> {
    separated_list0(delimited(ws, tag_char(','), ws), with_span(type_definition))(input)
}

fn tuple_middle(input: InputSpan<'_>) -> NomResult<'_, Spanned<'_, SliceAst<'_>>> {
    preceded(terminated(tag("..."), ws), with_span(slice_definition))(input)
}

type TupleTailAst<'a> = (Spanned<'a, SliceAst<'a>>, Vec<SpannedTypeAst<'a>>);

fn tuple_tail(input: InputSpan<'_>) -> NomResult<'_, TupleTailAst<'_>> {
    tuple((
        tuple_middle,
        map(
            opt(preceded(comma_sep, comma_separated_types)),
            Option::unwrap_or_default,
        ),
    ))(input)
}

fn tuple_definition(input: InputSpan<'_>) -> NomResult<'_, TupleAst<'_>> {
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

fn tuple_len(input: InputSpan<'_>) -> NomResult<'_, Spanned<'_, TupleLenAst>> {
    let semicolon = tuple((ws, tag_char(';'), ws));
    let empty = map(take(0_usize), Spanned::from);
    map(alt((preceded(semicolon, not_keyword), empty)), |id| {
        id.map_extra(|()| match *id.fragment() {
            "_" => TupleLenAst::Some,
            "" => TupleLenAst::Dynamic,
            _ => TupleLenAst::Ident,
        })
    })(input)
}

fn slice_definition(input: InputSpan<'_>) -> NomResult<'_, SliceAst<'_>> {
    preceded(
        terminated(tag_char('['), ws),
        // Once we've encountered the opening `[`, the input *must* correspond to the parser.
        cut(terminated(
            map(
                tuple((with_span(type_definition), tuple_len)),
                |(element, length)| SliceAst {
                    element: Box::new(element),
                    length,
                },
            ),
            tuple((ws, tag_char(']'))),
        )),
    )(input)
}

fn object(input: InputSpan<'_>) -> NomResult<'_, ObjectAst<'_>> {
    let colon = tuple((ws, tag_char(':'), ws));
    let object_field = separated_pair(ident, colon, with_span(type_definition));
    let object_body = terminated(separated_list1(comma_sep, object_field), opt(comma_sep));
    let object = preceded(
        terminated(tag_char('{'), ws),
        cut(terminated(object_body, tuple((ws, tag_char('}'))))),
    );
    map(object, |fields| ObjectAst { fields })(input)
}

fn constraint_sep(input: InputSpan<'_>) -> NomResult<'_, ()> {
    map(tuple((ws, tag_char('+'), ws)), drop)(input)
}

fn simple_type_bounds(input: InputSpan<'_>) -> NomResult<'_, TypeConstraintsAst<'_>> {
    map(separated_list1(constraint_sep, not_keyword), |terms| {
        TypeConstraintsAst {
            object: None,
            terms,
        }
    })(input)
}

fn type_bounds(input: InputSpan<'_>) -> NomResult<'_, TypeConstraintsAst<'_>> {
    alt((
        map(
            tuple((
                object,
                opt(preceded(
                    constraint_sep,
                    separated_list1(constraint_sep, not_keyword),
                )),
            )),
            |(object, terms)| TypeConstraintsAst {
                object: Some(object),
                terms: terms.unwrap_or_default(),
            },
        ),
        simple_type_bounds,
    ))(input)
}

fn type_params(input: InputSpan<'_>) -> NomResult<'_, Vec<(Spanned<'_>, TypeConstraintsAst<'_>)>> {
    let type_bounds = preceded(tuple((ws, tag_char(':'), ws)), type_bounds);
    let type_param = tuple((type_param_ident, type_bounds));
    separated_list1(comma_sep, type_param)(input)
}

/// Function params, including the `for` keyword and `<>` brackets.
fn constraints(input: InputSpan<'_>) -> NomResult<'_, ConstraintsAst<'_>> {
    let semicolon = tuple((ws, tag_char(';'), ws));

    let len_params = preceded(
        terminated(tag("len!"), ws),
        separated_list1(comma_sep, not_keyword),
    );

    let params_parser = alt((
        map(
            tuple((len_params, opt(preceded(semicolon, type_params)))),
            |(static_lengths, type_params)| (static_lengths, type_params.unwrap_or_default()),
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
        |(_, _, (static_lengths, type_params))| ConstraintsAst {
            static_lengths,
            type_params,
        },
    )(input)
}

fn return_type(input: InputSpan<'_>) -> NomResult<'_, SpannedTypeAst<'_>> {
    preceded(tuple((ws, tag("->"), ws)), cut(with_span(type_definition)))(input)
}

fn fn_or_tuple(input: InputSpan<'_>) -> NomResult<'_, TypeAst<'_>> {
    map(
        tuple((with_span(tuple_definition), opt(return_type))),
        |(args, return_type)| {
            if let Some(return_type) = return_type {
                TypeAst::Function(Box::new(FnTypeAst { args, return_type }))
            } else {
                TypeAst::Tuple(args.extra)
            }
        },
    )(input)
}

fn fn_definition(input: InputSpan<'_>) -> NomResult<'_, FnTypeAst<'_>> {
    map(
        tuple((with_span(tuple_definition), return_type)),
        |(args, return_type)| FnTypeAst { args, return_type },
    )(input)
}

fn fn_definition_with_constraints(input: InputSpan<'_>) -> NomResult<'_, TypeAst> {
    map(
        tuple((with_span(constraints), ws, cut(with_span(fn_definition)))),
        |(constraints, _, function)| TypeAst::FunctionWithConstraints {
            constraints,
            function: Box::new(function),
        },
    )(input)
}

fn any_type(input: InputSpan<'_>) -> NomResult<'_, TypeConstraintsAst<'_>> {
    let not_ident_char = peek(not(take_while_m_n(1, 1, |c: char| {
        c.is_ascii_alphanumeric() || c == '_'
    })));
    map(
        preceded(
            terminated(tag("any"), not_ident_char),
            opt(preceded(ws, simple_type_bounds)),
        ),
        Option::unwrap_or_default,
    )(input)
}

fn free_ident(input: InputSpan<'_>) -> NomResult<'_, TypeAst<'_>> {
    map(not_keyword, |id| match *id.fragment() {
        "_" => TypeAst::Some,
        _ => TypeAst::Ident,
    })(input)
}

fn type_definition(input: InputSpan<'_>) -> NomResult<'_, TypeAst<'_>> {
    alt((
        fn_or_tuple,
        fn_definition_with_constraints,
        map(type_param_ident, |_| TypeAst::Param),
        map(slice_definition, TypeAst::Slice),
        map(object, TypeAst::Object),
        map(any_type, TypeAst::Any),
        free_ident,
    ))(input)
}
