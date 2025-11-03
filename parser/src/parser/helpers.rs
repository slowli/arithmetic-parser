//! Passing helpers.

use nom::{
    branch::alt,
    bytes::{
        complete::{tag, take_until, take_while, take_while1, take_while_m_n},
        streaming,
    },
    character::complete::char as tag_char,
    combinator::{cut, not, peek, recognize},
    error::context,
    multi::many0,
    sequence::{delimited, preceded},
    Parser as _,
};

use crate::{grammars::Features, BinaryOp, Context, InputSpan, NomResult, Spanned, UnaryOp};

pub(super) trait GrammarType {
    const COMPLETE: bool;
}

#[derive(Debug)]
pub(super) struct Complete(());

impl GrammarType for Complete {
    const COMPLETE: bool = true;
}

#[derive(Debug)]
pub(super) struct Streaming(());

impl GrammarType for Streaming {
    const COMPLETE: bool = false;
}

impl UnaryOp {
    pub(super) fn from_span(span: Spanned<'_, char>) -> Spanned<'_, Self> {
        match span.extra {
            '-' => span.copy_with_extra(UnaryOp::Neg),
            '!' => span.copy_with_extra(UnaryOp::Not),
            _ => unreachable!(),
        }
    }

    pub(super) fn try_from_byte(byte: u8) -> Option<Self> {
        match byte {
            b'-' => Some(Self::Neg),
            b'!' => Some(Self::Not),
            _ => None,
        }
    }
}

impl BinaryOp {
    pub(super) fn from_span(span: InputSpan<'_>) -> Spanned<'_, Self> {
        Spanned::new(
            span,
            match *span.fragment() {
                "+" => Self::Add,
                "-" => Self::Sub,
                "*" => Self::Mul,
                "/" => Self::Div,
                "^" => Self::Power,
                "==" => Self::Eq,
                "!=" => Self::NotEq,
                "&&" => Self::And,
                "||" => Self::Or,
                ">" => Self::Gt,
                "<" => Self::Lt,
                ">=" => Self::Ge,
                "<=" => Self::Le,
                _ => unreachable!(),
            },
        )
    }

    pub(super) fn is_supported(self, features: Features) -> bool {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Power => true,
            Self::Eq | Self::NotEq | Self::And | Self::Or => {
                features.contains(Features::BOOLEAN_OPS_BASIC)
            }
            Self::Gt | Self::Lt | Self::Ge | Self::Le => features.contains(Features::BOOLEAN_OPS),
        }
    }
}

/// Whitespace and comments.
pub(super) fn ws<Ty: GrammarType>(input: InputSpan<'_>) -> NomResult<'_, InputSpan<'_>> {
    fn narrow_ws<T: GrammarType>(input: InputSpan<'_>) -> NomResult<'_, InputSpan<'_>> {
        if T::COMPLETE {
            take_while1(|c: char| c.is_ascii_whitespace())(input)
        } else {
            streaming::take_while1(|c: char| c.is_ascii_whitespace())(input)
        }
    }

    fn long_comment_body<T: GrammarType>(input: InputSpan<'_>) -> NomResult<'_, InputSpan<'_>> {
        if T::COMPLETE {
            context(Context::Comment.to_str(), cut(take_until("*/"))).parse(input)
        } else {
            streaming::take_until("*/")(input)
        }
    }

    let comment = preceded(tag("//"), take_while(|c: char| c != '\n'));
    let long_comment = delimited(tag("/*"), long_comment_body::<Ty>, tag("*/"));
    let ws_line = alt((narrow_ws::<Ty>, comment, long_comment));
    recognize(many0(ws_line)).parse(input)
}

pub(super) fn mandatory_ws<Ty: GrammarType>(input: InputSpan<'_>) -> NomResult<'_, InputSpan<'_>> {
    let not_ident_char = peek(not(take_while_m_n(1, 1, |c: char| {
        c.is_ascii_alphanumeric() || c == '_'
    })));
    preceded(not_ident_char, ws::<Ty>).parse(input)
}

/// Variable name, like `a_foo` or `Bar`.
pub(super) fn var_name(input: InputSpan<'_>) -> NomResult<'_, InputSpan<'_>> {
    context(
        Context::Var.to_str(),
        preceded(
            peek(take_while_m_n(1, 1, |c: char| {
                c.is_ascii_alphabetic() || c == '_'
            })),
            take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_'),
        ),
    )
    .parse(input)
}

/// Checks if the provided string is a valid variable name.
pub fn is_valid_variable_name(name: &str) -> bool {
    if name.is_empty() || !name.is_ascii() {
        return false;
    }

    match var_name(InputSpan::new(name)) {
        Ok((rest, _)) => rest.fragment().is_empty(),
        Err(_) => false,
    }
}

pub(super) fn comma_sep<Ty: GrammarType>(input: InputSpan<'_>) -> NomResult<'_, char> {
    delimited(ws::<Ty>, tag_char(','), ws::<Ty>).parse(input)
}
