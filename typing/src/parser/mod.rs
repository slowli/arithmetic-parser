//! Parsing type annotations.

use nom::{
    branch::alt,
    bytes::complete::{tag, take_until, take_while, take_while1, take_while_m_n},
    character::complete::char as tag_char,
    combinator::{cut, map, opt, peek, recognize},
    multi::{many0, separated_list0, separated_list1},
    sequence::{delimited, preceded, terminated, tuple},
};

use std::collections::HashMap;

use crate::{ConstParamDescription, FnArgs, FnType, TupleLength, TypeParamDescription, ValueType};
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

impl<'a> RawValueType<'a> {
    fn void() -> Self {
        Self::Tuple(vec![])
    }

    fn try_convert(&self, state: &ConversionState<'a>) -> Result<ValueType, ConversionError<'a>> {
        Ok(match self {
            Self::Any => ValueType::Any,
            Self::Bool => ValueType::Bool,
            Self::Number => ValueType::Number,

            Self::Ident(ident) => {
                let idx = state
                    .type_param_idx(ident.fragment())
                    .ok_or_else(|| ConversionError::UndefinedTypeParam(*ident))?;
                ValueType::TypeParam(idx)
            }

            Self::Function(fn_type) => {
                let converted_fn = fn_type.try_convert_inner(state.clone())?;
                ValueType::Function(Box::new(converted_fn))
            },

            Self::Tuple(elements) => {
                let converted_elements: Result<Vec<_>, _> =
                    elements.iter().map(|elt| elt.try_convert(state)).collect();
                ValueType::Tuple(converted_elements?)
            }

            Self::Slice { element, length } => {
                let converted_length = match length {
                    RawTupleLength::Ident(ident) => TupleLength::Param(
                        state
                            .const_param_idx(ident.fragment())
                            .ok_or_else(|| ConversionError::UndefinedConst(*ident))?,
                    ),
                    RawTupleLength::Dynamic => TupleLength::Dynamic,
                };
                ValueType::Slice {
                    element: Box::new(element.try_convert(state)?),
                    length: converted_length,
                }
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RawFnType<'a> {
    const_params: Vec<InputSpan<'a>>,
    type_params: Vec<(InputSpan<'a>, RawTypeParamBounds)>,
    args: Vec<RawValueType<'a>>,
    return_type: RawValueType<'a>,
}

#[derive(Debug)]
pub enum ConversionError<'a> {
    DuplicateConst {
        definition: InputSpan<'a>,
        previous: InputSpan<'a>,
    },
    DuplicateTypeParam {
        definition: InputSpan<'a>,
        previous: InputSpan<'a>,
    },
    UndefinedTypeParam(InputSpan<'a>),
    UndefinedConst(InputSpan<'a>),
}

/// Intermediate conversion state.
#[derive(Debug, Default, Clone)]
struct ConversionState<'a> {
    const_params: HashMap<&'a str, (InputSpan<'a>, usize)>,
    type_params: HashMap<&'a str, (InputSpan<'a>, usize)>,
}

impl<'a> ConversionState<'a> {
    fn insert_const_param(&mut self, param: InputSpan<'a>) -> Result<(), ConversionError<'a>> {
        let new_idx = self.const_params.len();
        if let Some((previous, _)) = self
            .const_params
            .insert(*param.fragment(), (param, new_idx))
        {
            Err(ConversionError::DuplicateConst {
                definition: param,
                previous,
            })
        } else {
            Ok(())
        }
    }

    fn insert_type_param(&mut self, param: InputSpan<'a>) -> Result<(), ConversionError<'a>> {
        let new_idx = self.type_params.len();
        if let Some((previous, _)) = self.type_params.insert(*param.fragment(), (param, new_idx)) {
            Err(ConversionError::DuplicateTypeParam {
                definition: param,
                previous,
            })
        } else {
            Ok(())
        }
    }

    fn type_param_idx(&self, param_name: &str) -> Option<usize> {
        self.type_params.get(param_name).map(|(_, idx)| *idx)
    }

    fn const_param_idx(&self, param_name: &str) -> Option<usize> {
        self.const_params.get(param_name).map(|(_, idx)| *idx)
    }
}

impl<'a> RawFnType<'a> {
    /// Tries to convert this into an `FnType`.
    pub fn try_convert(&self) -> Result<FnType, ConversionError<'a>> {
        self.try_convert_inner(ConversionState::default())
    }

    fn try_convert_inner(
        &self,
        mut state: ConversionState<'a>,
    ) -> Result<FnType, ConversionError<'a>> {
        // Check params for consistency.
        for param in &self.const_params {
            state.insert_const_param(*param)?;
        }
        for (param, _) in &self.type_params {
            state.insert_type_param(*param)?;
        }

        let args: Result<Vec<_>, _> = self
            .args
            .iter()
            .map(|arg| arg.try_convert(&state))
            .collect();
        Ok(FnType {
            args: FnArgs::List(args?),
            return_type: self.return_type.try_convert(&state)?,

            // FIXME: do we need to record external type params here?
            type_params: self
                .type_params
                .iter()
                .map(|(name, bounds)| {
                    (
                        state.type_param_idx(name.fragment()).unwrap(),
                        TypeParamDescription {
                            maybe_non_linear: bounds.maybe_non_linear,
                            is_external: false,
                        },
                    )
                })
                .collect(),

            const_params: self
                .const_params
                .iter()
                .map(|name| {
                    (
                        state.const_param_idx(name.fragment()).unwrap(),
                        ConstParamDescription { is_external: false },
                    )
                })
                .collect(),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RawTupleLength<'a> {
    Dynamic,
    Ident(InputSpan<'a>),
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct RawTypeParamBounds {
    maybe_non_linear: bool,
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

fn type_bounds(input: InputSpan<'_>) -> NomResult<'_, RawTypeParamBounds> {
    map(terminated(tag("?Lin"), ws), |_| RawTypeParamBounds {
        maybe_non_linear: true,
    })(input)
}

fn type_params(input: InputSpan<'_>) -> NomResult<'_, Vec<(InputSpan<'_>, RawTypeParamBounds)>> {
    let maybe_type_bounds = opt(preceded(tuple((ws, tag_char(':'), ws)), type_bounds));
    let type_param = tuple((ident, map(maybe_type_bounds, Option::unwrap_or_default)));
    separated_list1(comma_sep, type_param)(input)
}

type FnParams<'a> = (Vec<InputSpan<'a>>, Vec<(InputSpan<'a>, RawTypeParamBounds)>);

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
