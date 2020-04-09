//! Parsers implemented with the help of `nom`.

use nom::{
    branch::alt,
    bytes::{
        complete::{tag, take_while, take_while1, take_while_m_n},
        streaming,
    },
    character::complete::{char as tag_char, one_of},
    combinator::{cut, map, not, opt, peek, recognize},
    error::{context, ErrorKind},
    multi::{many0, separated_list},
    sequence::{delimited, preceded, terminated, tuple},
    Err as NomErr,
};

use std::{fmt, mem};

use crate::{
    BinaryOp, Block, Context, Expr, FnDefinition, Grammar, Lvalue, NomResult, Span, Spanned,
    SpannedExpr, SpannedLvalue, SpannedStatement, Statement, UnaryOp,
};

#[cfg(test)]
mod tests;

trait GrammarType {
    const COMPLETE: bool;
}

#[derive(Debug)]
struct Complete(());

impl GrammarType for Complete {
    const COMPLETE: bool = true;
}

#[derive(Debug)]
struct Streaming(());

impl GrammarType for Streaming {
    const COMPLETE: bool = false;
}

fn create_span<T, U>(span: Spanned<T>, extra: U) -> Spanned<U> {
    Spanned {
        offset: span.offset,
        line: span.line,
        fragment: span.fragment,
        extra,
    }
}

fn create_span_ref<'a, T, U>(span: &Spanned<'a, T>, extra: U) -> Spanned<'a, U> {
    Spanned {
        offset: span.offset,
        line: span.line,
        fragment: span.fragment,
        extra,
    }
}

fn map_span<T, U>(span: Spanned<'_, T>, f: impl FnOnce(T) -> U) -> Spanned<'_, U> {
    Spanned {
        offset: span.offset,
        line: span.line,
        fragment: span.fragment,
        extra: f(span.extra),
    }
}

/// Wrapper around parsers allowing to capture both their output and the relevant span.
fn with_span<'a, O>(
    parser: impl Fn(Span<'a>) -> NomResult<'a, O>,
) -> impl Fn(Span<'a>) -> NomResult<'a, Spanned<O>> {
    move |input: Span| {
        parser(input).map(|(rest, output)| {
            let spanned = Spanned {
                offset: input.offset,
                line: input.line,
                extra: output,
                fragment: &input.fragment[..(rest.offset - input.offset)],
            };
            (rest, spanned)
        })
    }
}

fn unite_spans<'a, T, U>(
    input: Span<'a>,
    start: &Spanned<'_, T>,
    end: &Spanned<'_, U>,
) -> Span<'a> {
    debug_assert!(input.offset <= start.offset);
    debug_assert!(start.offset <= end.offset);
    debug_assert!(input.offset + input.fragment.len() >= end.offset + end.fragment.len());

    let start_idx = start.offset - input.offset;
    let end_idx = end.offset + end.fragment.len() - input.offset;
    Span {
        offset: start.offset,
        line: start.line,
        fragment: &input.fragment[start_idx..end_idx],
        extra: (),
    }
}

impl UnaryOp {
    fn from_span(span: Spanned<char>) -> Spanned<Self> {
        match span.extra {
            '-' => create_span(span, UnaryOp::Neg),
            '!' => create_span(span, UnaryOp::Not),
            _ => unreachable!(),
        }
    }
}

impl BinaryOp {
    fn from_span(span: Span) -> Spanned<Self> {
        Spanned {
            offset: span.offset,
            line: span.line,
            fragment: span.fragment,
            extra: match span.fragment {
                "+" => BinaryOp::Add,
                "-" => BinaryOp::Sub,
                "*" => BinaryOp::Mul,
                "/" => BinaryOp::Div,
                "^" => BinaryOp::Power,
                "==" => BinaryOp::Eq,
                "!=" => BinaryOp::NotEq,
                "&&" => BinaryOp::And,
                "||" => BinaryOp::Or,
                _ => unreachable!(),
            },
        }
    }
}

/// Parsing error.
#[derive(Debug)]
pub enum Error<'a> {
    /// Input is not in ASCII.
    NonAsciiInput,

    /// Error parsing literal.
    Literal(anyhow::Error),

    /// Error parsing type hint.
    Type(anyhow::Error),

    /// No rules where expecting this character.
    UnexpectedChar {
        /// Parsing context.
        context: Option<Spanned<'a, Context>>,
    },

    /// Unexpected expression end.
    UnexpectedTerm {
        /// Parsing context.
        context: Option<Spanned<'a, Context>>,
    },

    /// Leftover symbols after parsing.
    Leftovers,
    /// Input is incomplete.
    Incomplete,

    /// Other parsing error.
    Other {
        /// `nom`-defined error kind.
        kind: ErrorKind,
        /// Parsing context.
        context: Option<Spanned<'a, Context>>,
    },
}

impl Error<'_> {
    /// Removes the context from this error, thus extending its lifetime.
    pub fn without_context(self) -> Error<'static> {
        match self {
            Self::NonAsciiInput => Error::NonAsciiInput,
            Self::Leftovers => Error::Leftovers,
            Self::Incomplete => Error::Incomplete,
            Self::Literal(source) => Error::Literal(source),
            Self::Type(source) => Error::Type(source),
            Self::UnexpectedChar { .. } => Error::UnexpectedChar { context: None },
            Self::UnexpectedTerm { .. } => Error::UnexpectedTerm { context: None },
            Self::Other { kind, .. } => Error::Other {
                kind,
                context: None,
            },
        }
    }
}

impl fmt::Display for Error<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::NonAsciiInput => formatter.write_str("Non-ASCII inputs are not supported"),
            Error::Literal(e) => write!(formatter, "Invalid literal: {}", e),
            Error::Type(e) => write!(formatter, "Invalid type hint: {}", e),

            Error::UnexpectedChar { context: Some(ctx) } => {
                write!(formatter, "Unexpected character in {}", ctx.extra)
            }
            Error::UnexpectedChar { .. } => formatter.write_str("Unexpected character"),

            Error::UnexpectedTerm { context: Some(ctx) } => {
                write!(formatter, "Unfinished {}", ctx.extra)
            }
            Error::UnexpectedTerm { .. } => formatter.write_str("Unfinished expression"),
            Error::Leftovers => formatter.write_str("Uninterpreted characters after parsing"),
            Error::Incomplete => formatter.write_str("Incomplete input"),
            Error::Other { .. } => write!(formatter, "Cannot parse sequence"),
        }
    }
}

impl<'a> Error<'a> {
    fn accepts_context(&self) -> bool {
        match self {
            Error::UnexpectedChar { context }
            | Error::UnexpectedTerm { context }
            | Error::Other { context, .. } => context.is_none(),
            _ => false,
        }
    }

    /// Returns optional error context.
    pub fn context(&self) -> Option<Spanned<Context>> {
        match self {
            Error::UnexpectedChar { context }
            | Error::UnexpectedTerm { context }
            | Error::Other { context, .. } => context.to_owned(),
            _ => None,
        }
    }

    fn set_context(&mut self, ctx: Context, span: Span<'a>) {
        match self {
            Error::UnexpectedChar { context }
            | Error::UnexpectedTerm { context }
            | Error::Other { context, .. } => {
                *context = Some(create_span(span, ctx));
            }
            _ => { /* do nothing */ }
        }
    }

    fn with_span<T>(self, span: Spanned<'a, T>) -> SpannedError<'a> {
        SpannedError(create_span(span, self))
    }
}

/// Parsing error with the associated code span.
#[derive(Debug)]
pub struct SpannedError<'a>(Spanned<'a, Error<'a>>);

impl<'a> nom::error::ParseError<Span<'a>> for SpannedError<'a> {
    fn from_error_kind(mut input: Span<'a>, kind: ErrorKind) -> Self {
        if kind == ErrorKind::Char && !input.fragment.is_empty() {
            // Truncate the error span to the first ineligible char.
            input.fragment = &input.fragment[..1];
        }

        SpannedError(create_span(
            input,
            if kind == ErrorKind::Char {
                if input.fragment.is_empty() {
                    Error::UnexpectedTerm { context: None }
                } else {
                    Error::UnexpectedChar { context: None }
                }
            } else {
                Error::Other {
                    kind,
                    context: None,
                }
            },
        ))
    }

    fn append(_: Span<'a>, _: ErrorKind, other: Self) -> Self {
        other
    }

    fn add_context(input: Span<'a>, ctx: &'static str, mut other: Self) -> Self {
        if other.0.extra.accepts_context() && input.offset < other.0.offset {
            other.0.extra.set_context(Context::new(ctx), input);
        }
        other
    }
}

/// Whitespace and `#...` comments.
fn ws<Ty: GrammarType>(input: Span) -> NomResult<Span> {
    fn narrow_ws<T: GrammarType>(input: Span) -> NomResult<Span> {
        if T::COMPLETE {
            take_while1(|c: char| c.is_ascii_whitespace())(input)
        } else {
            streaming::take_while1(|c: char| c.is_ascii_whitespace())(input)
        }
    }

    let comment = preceded(tag_char('#'), take_while(|c: char| c != '\n'));
    let ws_line = alt((narrow_ws::<Ty>, comment));
    recognize(many0(ws_line))(input)
}

/// Variable name, like `a_foo` or `Bar`.
fn var_name(input: Span<'_>) -> NomResult<'_, Span<'_>> {
    context(
        Context::Var.to_str(),
        preceded(
            peek(take_while_m_n(1, 1, |c: char| {
                c.is_ascii_alphabetic() || c == '_'
            })),
            take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_'),
        ),
    )(input)
}

/// Function arguments, e.g., `(a, B + 1)`
fn fn_args<T, Ty>(input: Span<'_>) -> NomResult<'_, Vec<SpannedExpr<'_, T>>>
where
    T: Grammar,
    Ty: GrammarType,
{
    delimited(
        terminated(tag_char('('), ws::<Ty>),
        cut(separated_list(
            delimited(ws::<Ty>, tag_char(','), ws::<Ty>),
            expr::<T, Ty>,
        )),
        preceded(ws::<Ty>, tag_char(')')),
    )(input)
}

/// Expression enclosed in parentheses. This may be a simple value (e.g., `(1 + 2)`)
/// or a tuple (e.g., `(x, y)`), depending on the number of comma-separated terms.
fn paren_expr<T, Ty>(input: Span<'_>) -> NomResult<SpannedExpr<'_, T>>
where
    T: Grammar,
    Ty: GrammarType,
{
    with_span(fn_args::<T, Ty>)(input).and_then(|(rest, parsed)| match parsed.extra.len() {
        0 => Err(NomErr::Failure(
            Error::UnexpectedChar { context: None }.with_span(parsed),
        )),

        1 => Ok((
            rest,
            map_span(parsed, |mut terms| terms.pop().unwrap().extra),
        )),

        _ => {
            if T::FEATURES.tuples {
                Ok((rest, map_span(parsed, Expr::Tuple)))
            } else {
                Err(NomErr::Failure(
                    Error::UnexpectedTerm { context: None }.with_span(parsed),
                ))
            }
        }
    })
}

fn var_or_fn_call<T, Ty>(input: Span<'_>) -> NomResult<'_, SpannedExpr<'_, T>>
where
    T: Grammar,
    Ty: GrammarType,
{
    let parser = tuple((var_name, opt(preceded(ws::<Ty>, fn_args::<T, Ty>))));
    map(with_span(parser), |spanned| {
        map_span(spanned, |(name, maybe_args)| {
            if let Some(args) = maybe_args {
                Expr::Function { name, args }
            } else {
                Expr::Variable
            }
        })
    })(input)
}

/// Parses a simple expression, i.e., one not containing binary operations.
///
/// From the construction, the evaluation priorities within such an expression are always higher
/// than for possible binary ops surrounding it.
fn simple_expr<'a, T, Ty>(input: Span<'a>) -> NomResult<'a, SpannedExpr<'a, T>>
where
    T: Grammar,
    Ty: GrammarType,
{
    let block_parser: Box<dyn Fn(Span<'a>) -> NomResult<'a, SpannedExpr<'a, T>>> =
        if T::FEATURES.blocks {
            let parser = map(with_span(block::<T, Ty>), |spanned| {
                map_span(spanned, Expr::Block)
            });
            Box::new(parser)
        } else {
            // Always fail.
            Box::new(|input| {
                let e = Error::Leftovers.with_span(input);
                Err(NomErr::Error(e))
            })
        };

    let fn_def_parser: Box<dyn Fn(Span<'a>) -> NomResult<'a, SpannedExpr<'a, T>>> =
        if T::FEATURES.fn_definitions {
            let parser = map(with_span(fn_def::<T, Ty>), |span| Spanned {
                offset: span.offset,
                line: span.line,
                fragment: span.fragment,
                extra: Expr::FnDefinition(span.extra),
            });
            Box::new(parser)
        } else {
            // Always fail.
            Box::new(|input| {
                let e = Error::Leftovers.with_span(input);
                Err(NomErr::Error(e))
            })
        };

    alt((
        var_or_fn_call::<T, Ty>,
        map(with_span(T::parse_literal), |span| Spanned {
            offset: span.offset,
            line: span.line,
            fragment: span.fragment,
            extra: Expr::Literal(span.extra),
        }),
        fn_def_parser,
        map(
            with_span(tuple((
                terminated(with_span(one_of("-!")), ws::<Ty>),
                simple_expr::<T, Ty>,
            ))),
            |spanned| {
                let (op, inner) = spanned.extra;
                Spanned {
                    offset: spanned.offset,
                    line: spanned.line,
                    fragment: spanned.fragment,
                    extra: Expr::Unary {
                        op: UnaryOp::from_span(op),
                        inner: Box::new(inner),
                    },
                }
            },
        ),
        block_parser,
        paren_expr::<T, Ty>,
    ))(input)
}

/// Parses an expression with binary operations into a tree with the hierarchy reflecting
/// the evaluation order of the operations.
fn binary_expr<T, Ty>(input: Span<'_>) -> NomResult<'_, SpannedExpr<'_, T>>
where
    T: Grammar,
    Ty: GrammarType,
{
    // First, we parse the expression into a list with simple expressions interspersed
    // with binary operations. For example, `1 + 2 * foo(x, y)` is parsed into
    //
    //     [ 1, +, 2, *, foo(x, y) ]
    let binary_ops = alt((
        tag("+"),
        tag("-"),
        tag("*"),
        tag("/"),
        tag("^"),
        tag("=="),
        tag("!="),
        tag("&&"),
        tag("||"),
    ));
    let binary_ops = with_span(map(binary_ops, drop));
    let binary_parser = tuple((
        simple_expr::<T, Ty>,
        many0(tuple((
            delimited(ws::<Ty>, binary_ops, ws::<Ty>),
            cut(simple_expr::<T, Ty>),
        ))),
    ));

    // After obtaining the list, we fold it paying attention to the operation priorities.
    // We track the `right_contour` of the parsed tree and insert a new operation so that
    // operations in the contour with lower priority remain in place.
    //
    // As an example, consider expression `1 + 2 * foo(x, y) - 7` split into list
    // `[ 1, +, 2, *, foo(x, y), -, 7 ]`. First, we form a tree
    //
    //   +
    //  / \
    // 1   2
    //
    // Then, we find the place to insert the `*` op and `foo(x, y)` operand. Since `*` has
    // a higher priority than `+`, we insert it *below* the `+` op:
    //
    //   +
    //  / \
    // 1   *
    //    / \
    //   2  foo(x, y)
    //
    // The next op `-` has the same priority as the first op in the right contour (`+`), so
    // we insert it *above* it:
    //
    //    -
    //   / \
    //   +  7
    //  / \
    // 1   *
    //    / \
    //   2  foo(x, y)
    map(binary_parser, |(first, rest)| {
        let mut right_contour: Vec<BinaryOp> = vec![];
        rest.into_iter().fold(first, |mut acc, (op, expr)| {
            let new_op = BinaryOp::from_span(op);
            let united_span = unite_spans(input, &acc, &expr);

            let insert_pos = right_contour
                .iter()
                .position(|past_op| past_op.priority() >= new_op.extra.priority())
                .unwrap_or_else(|| right_contour.len());

            if insert_pos == 0 {
                right_contour.clear();
                right_contour.push(new_op.extra);

                create_span(
                    united_span,
                    Expr::Binary {
                        lhs: Box::new(acc),
                        op: new_op,
                        rhs: Box::new(expr),
                    },
                )
            } else {
                right_contour.truncate(insert_pos);
                right_contour.push(new_op.extra);

                let mut parent = &mut acc;
                for _ in 1..insert_pos {
                    parent = match parent.extra {
                        Expr::Binary { ref mut rhs, .. } => rhs,
                        _ => unreachable!(),
                    };
                }

                parent.fragment = unite_spans(input, &parent, &expr).fragment;
                if let Expr::Binary { ref mut rhs, .. } = parent.extra {
                    let rhs_span = unite_spans(input, rhs, &expr);
                    let dummy = Box::new(create_span_ref(rhs, Expr::Variable));
                    let old_rhs = mem::replace(rhs, dummy);
                    let new_expr = Expr::Binary {
                        lhs: old_rhs,
                        op: new_op,
                        rhs: Box::new(expr),
                    };
                    *rhs = Box::new(create_span(rhs_span, new_expr));
                }
                acc.fragment = united_span.fragment;
                acc
            }
        })
    })(input)
}

fn expr<T, Ty>(input: Span<'_>) -> NomResult<'_, SpannedExpr<'_, T>>
where
    T: Grammar,
    Ty: GrammarType,
{
    context(Context::Expr.to_str(), binary_expr::<T, Ty>)(input)
}

/// Parses an `Lvalue`.
fn lvalue<'a, T, Ty>(input: Span<'a>) -> NomResult<'a, SpannedLvalue<'a, T::Type>>
where
    T: Grammar,
    Ty: GrammarType,
{
    let simple_lvalue: Box<dyn Fn(Span<'a>) -> NomResult<'a, SpannedLvalue<'a, T::Type>>> =
        if T::FEATURES.type_annotations {
            let parser = map(
                tuple((
                    var_name,
                    opt(preceded(
                        delimited(ws::<Ty>, tag_char(':'), ws::<Ty>),
                        cut(with_span(T::parse_type)),
                    )),
                )),
                |(name, ty)| create_span(name, Lvalue::Variable { ty }),
            );
            Box::new(parser)
        } else {
            let parser = map(var_name, |name| {
                create_span(name, Lvalue::Variable { ty: None })
            });
            Box::new(parser)
        };

    if T::FEATURES.tuples {
        alt((
            with_span(map(
                delimited(
                    terminated(tag_char('('), ws::<Ty>),
                    separated_list(
                        delimited(ws::<Ty>, tag_char(','), ws::<Ty>),
                        lvalue::<T, Ty>,
                    ),
                    preceded(ws::<Ty>, tag_char(')')),
                ),
                Lvalue::Tuple,
            )),
            simple_lvalue,
        ))(input)
    } else {
        simple_lvalue(input)
    }
}

fn statement<T, Ty>(input: Span<'_>) -> NomResult<'_, SpannedStatement<'_, T>>
where
    T: Grammar,
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
pub(crate) fn statements<T>(input_span: Span<'_>) -> Result<Block<'_, T>, Spanned<'_, Error<'_>>>
where
    T: Grammar,
{
    if !input_span.fragment.is_ascii() {
        return Err(create_span(input_span, Error::NonAsciiInput));
    }

    statements_inner::<T, Complete>(input_span)
}

/// Parses a potentially incomplete list of statements.
pub(crate) fn streaming_statements<T>(
    input_span: Span<'_>,
) -> Result<Block<'_, T>, Spanned<'_, Error<'_>>>
where
    T: Grammar,
{
    if !input_span.fragment.is_ascii() {
        return Err(create_span(input_span, Error::NonAsciiInput));
    }

    statements_inner::<T, Complete>(input_span)
        .or_else(|_| statements_inner::<T, Streaming>(input_span))
}

fn statements_inner<T, Ty>(input_span: Span<'_>) -> Result<Block<'_, T>, Spanned<'_, Error<'_>>>
where
    T: Grammar,
    Ty: GrammarType,
{
    delimited(ws::<Ty>, separated_statements::<T, Ty>, ws::<Ty>)(input_span)
        .map_err(|e| match e {
            NomErr::Failure(e) | NomErr::Error(e) => e.0,
            NomErr::Incomplete(_) => Error::Incomplete.with_span(input_span).0,
        })
        .and_then(|(remaining, statements)| {
            if remaining.fragment.is_empty() {
                Ok(statements)
            } else {
                Err(Error::Leftovers.with_span(remaining).0)
            }
        })
}

fn separated_statement<T, Ty>(input: Span<'_>) -> NomResult<'_, SpannedStatement<'_, T>>
where
    T: Grammar,
    Ty: GrammarType,
{
    terminated(statement::<T, Ty>, preceded(ws::<Ty>, tag_char(';')))(input)
}

/// List of statements separated by semicolons.
fn separated_statements<T, Ty>(input: Span<'_>) -> NomResult<'_, Block<'_, T>>
where
    T: Grammar,
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
fn block<T, Ty>(input: Span<'_>) -> NomResult<'_, Block<'_, T>>
where
    T: Grammar,
    Ty: GrammarType,
{
    delimited(
        terminated(tag_char('{'), ws::<Ty>),
        separated_statements::<T, Ty>,
        preceded(ws::<Ty>, tag_char('}')),
    )(input)
}

/// Function definition, e.g., `|x, y: Sc| { x + y }`.
fn fn_def<T, Ty>(input: Span<'_>) -> NomResult<'_, FnDefinition<'_, T>>
where
    T: Grammar,
    Ty: GrammarType,
{
    let body_parser = alt((
        block::<T, Ty>,
        map(expr::<T, Ty>, |spanned| Block {
            statements: vec![],
            return_value: Some(Box::new(spanned)),
        }),
    ));
    let parser = tuple((
        delimited(
            terminated(tag_char('|'), ws::<Ty>),
            separated_list(
                delimited(ws::<Ty>, tag_char(','), ws::<Ty>),
                lvalue::<T, Ty>,
            ),
            preceded(ws::<Ty>, tag_char('|')),
        ),
        cut(preceded(ws::<Ty>, body_parser)),
    ));
    map(parser, |(args, body)| FnDefinition { args, body })(input)
}
