//! Parsers implemented with the help of `nom`.

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::char as tag_char,
    combinator::{cut, map, not, opt, peek},
    multi::many0,
    sequence::{delimited, preceded, terminated, tuple},
    Err as NomErr,
};

mod expr;
mod helpers;
mod lvalue;
#[cfg(test)]
mod tests;

pub use self::helpers::is_valid_variable_name;
use self::{
    expr::expr,
    helpers::{ws, Complete, GrammarType, Streaming},
    lvalue::{destructure, lvalue},
};
use crate::{
    alloc::{vec, Box},
    grammars::Parse,
    spans::with_span,
    Block, Error, ErrorKind, FnDefinition, InputSpan, NomResult, SpannedStatement, Statement,
};

#[allow(clippy::option_if_let_else)]
fn statement<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, SpannedStatement<'_, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    let assignment = tuple((tag("="), peek(not(tag_char('=')))));
    let assignment_parser = tuple((
        opt(terminated(
            lvalue::<T, Ty>,
            delimited(ws::<Ty>, assignment, ws::<Ty>),
        )),
        expr::<T, Ty>,
    ));

    with_span(map(assignment_parser, |(lvalue, rvalue)| {
        // Clippy lint is triggered here. `rvalue` cannot be moved into both branches, so it's a false positive.
        if let Some(lvalue) = lvalue {
            Statement::Assignment {
                lhs: lvalue,
                rhs: Box::new(rvalue),
            }
        } else {
            Statement::Expr(rvalue)
        }
    }))(input)
}

/// Parses a complete list of statements.
pub(crate) fn statements<T>(input_span: InputSpan<'_>) -> Result<Block<'_, T::Base>, Error<'_>>
where
    T: Parse,
{
    if !input_span.fragment().is_ascii() {
        return Err(Error::new(input_span, ErrorKind::NonAsciiInput));
    }
    statements_inner::<T, Complete>(input_span)
}

/// Parses a potentially incomplete list of statements.
pub(crate) fn streaming_statements<T>(
    input_span: InputSpan<'_>,
) -> Result<Block<'_, T::Base>, Error<'_>>
where
    T: Parse,
{
    if !input_span.fragment().is_ascii() {
        return Err(Error::new(input_span, ErrorKind::NonAsciiInput));
    }

    statements_inner::<T, Complete>(input_span)
        .or_else(|_| statements_inner::<T, Streaming>(input_span))
}

fn statements_inner<T, Ty>(input_span: InputSpan<'_>) -> Result<Block<'_, T::Base>, Error<'_>>
where
    T: Parse,
    Ty: GrammarType,
{
    delimited(ws::<Ty>, separated_statements::<T, Ty>, ws::<Ty>)(input_span)
        .map_err(|e| match e {
            NomErr::Failure(e) | NomErr::Error(e) => e,
            NomErr::Incomplete(_) => ErrorKind::Incomplete.with_span(&input_span.into()),
        })
        .and_then(|(remaining, statements)| {
            if remaining.fragment().is_empty() {
                Ok(statements)
            } else {
                Err(ErrorKind::Leftovers.with_span(&remaining.into()))
            }
        })
}

fn separated_statement<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, SpannedStatement<'_, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    terminated(statement::<T, Ty>, preceded(ws::<Ty>, tag_char(';')))(input)
}

/// List of statements separated by semicolons.
fn separated_statements<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, Block<'_, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    map(
        tuple((
            many0(terminated(separated_statement::<T, Ty>, ws::<Ty>)),
            opt(expr::<T, Ty>),
        )),
        |(statements, return_value)| Block {
            statements,
            return_value: return_value.map(Box::new),
        },
    )(input)
}

/// Block of statements, e.g., `{ x = 3; x + y }`.
fn block<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, Block<'_, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    preceded(
        terminated(tag_char('{'), ws::<Ty>),
        cut(terminated(
            separated_statements::<T, Ty>,
            preceded(ws::<Ty>, tag_char('}')),
        )),
    )(input)
}

/// Function definition, e.g., `|x, y: Sc| { x + y }`.
fn fn_def<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, FnDefinition<'_, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    let body_parser = alt((
        block::<T, Ty>,
        map(expr::<T, Ty>, |spanned| Block {
            statements: vec![],
            return_value: Some(Box::new(spanned)),
        }),
    ));

    let args_parser = preceded(
        terminated(tag_char('|'), ws::<Ty>),
        cut(terminated(
            destructure::<T, Ty>,
            preceded(ws::<Ty>, tag_char('|')),
        )),
    );

    let parser = tuple((with_span(args_parser), cut(preceded(ws::<Ty>, body_parser))));
    map(parser, |(args, body)| FnDefinition { args, body })(input)
}
