//! Parsers implemented with the help of `nom`.

use nom::{
    branch::alt,
    bytes::{
        complete::{tag, take_until, take_while, take_while1, take_while_m_n},
        streaming,
    },
    character::complete::{char as tag_char, one_of},
    combinator::{cut, map, not, opt, peek, recognize},
    error::context,
    multi::{many0, separated_list},
    sequence::{delimited, preceded, terminated, tuple},
    Err as NomErr, Slice,
};

use core::mem;

use crate::{
    alloc::{vec, Box, Vec},
    grammars::{BooleanOps, Features, Grammar, Parse, ParseLiteral},
    spans::{unite_spans, with_span},
    BinaryOp, Block, Context, Destructure, DestructureRest, Error, ErrorKind, Expr, FnDefinition,
    InputSpan, Lvalue, NomResult, Spanned, SpannedExpr, SpannedLvalue, SpannedStatement, Statement,
    UnaryOp,
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

impl UnaryOp {
    fn from_span(span: Spanned<char>) -> Spanned<Self> {
        match span.extra {
            '-' => span.copy_with_extra(UnaryOp::Neg),
            '!' => span.copy_with_extra(UnaryOp::Not),
            _ => unreachable!(),
        }
    }

    fn try_from_byte(byte: u8) -> Option<Self> {
        match byte {
            b'-' => Some(Self::Neg),
            b'!' => Some(Self::Not),
            _ => None,
        }
    }
}

impl BinaryOp {
    fn from_span(span: InputSpan<'_>) -> Spanned<'_, Self> {
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

    fn is_supported(self, features: &Features) -> bool {
        let boolean_ops = features.boolean_ops;

        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Power => true,
            Self::Eq | Self::NotEq | Self::And | Self::Or => boolean_ops >= BooleanOps::Basic,
            Self::Gt | Self::Lt | Self::Ge | Self::Le => {
                boolean_ops >= BooleanOps::OrderComparisons
            }
        }
    }
}

/// Whitespace and comments.
fn ws<Ty: GrammarType>(input: InputSpan<'_>) -> NomResult<'_, InputSpan<'_>> {
    fn narrow_ws<T: GrammarType>(input: InputSpan<'_>) -> NomResult<'_, InputSpan<'_>> {
        if T::COMPLETE {
            take_while1(|c: char| c.is_ascii_whitespace())(input)
        } else {
            streaming::take_while1(|c: char| c.is_ascii_whitespace())(input)
        }
    }

    fn long_comment_body<T: GrammarType>(input: InputSpan<'_>) -> NomResult<'_, InputSpan<'_>> {
        if T::COMPLETE {
            context(Context::Comment.to_str(), cut(take_until("*/")))(input)
        } else {
            streaming::take_until("*/")(input)
        }
    }

    let comment = preceded(tag("//"), take_while(|c: char| c != '\n'));
    let long_comment = delimited(tag("/*"), long_comment_body::<Ty>, tag("*/"));
    let ws_line = alt((narrow_ws::<Ty>, comment, long_comment));
    recognize(many0(ws_line))(input)
}

/// Variable name, like `a_foo` or `Bar`.
fn var_name(input: InputSpan<'_>) -> NomResult<'_, InputSpan<'_>> {
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

/// Function arguments in the call position; e.g., `(a, B + 1)`.
///
/// # Return value
///
/// The second component of the returned tuple is set to `true` if the list is `,`-terminated.
fn fn_args<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, (Vec<SpannedExpr<'_, T::Base>>, bool)>
where
    T: Parse,
    Ty: GrammarType,
{
    let maybe_comma = map(opt(preceded(ws::<Ty>, tag_char(','))), |c| c.is_some());

    preceded(
        terminated(tag_char('('), ws::<Ty>),
        // Once we've encountered the opening `(`, the input *must* correspond to the parser.
        cut(tuple((
            separated_list(delimited(ws::<Ty>, tag_char(','), ws::<Ty>), expr::<T, Ty>),
            terminated(maybe_comma, tuple((ws::<Ty>, tag_char(')')))),
        ))),
    )(input)
}

/// Expression enclosed in parentheses. This may be a simple value (e.g., `(1 + 2)`)
/// or a tuple (e.g., `(x, y)`), depending on the number of comma-separated terms.
fn paren_expr<T, Ty>(input: InputSpan<'_>) -> NomResult<SpannedExpr<'_, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    with_span(fn_args::<T, Ty>)(input).and_then(|(rest, parsed)| {
        let comma_terminated = parsed.extra.1;
        let terms = parsed.map_extra(|terms| terms.0);

        match (terms.extra.len(), comma_terminated) {
            (1, false) => Ok((
                rest,
                terms.map_extra(|mut terms| terms.pop().unwrap().extra),
            )),

            _ => {
                if T::FEATURES.tuples {
                    Ok((rest, terms.map_extra(Expr::Tuple)))
                } else {
                    Err(NomErr::Failure(
                        ErrorKind::UnexpectedTerm { context: None }.with_span(&terms),
                    ))
                }
            }
        }
    })
}

/// Parses a simple expression, i.e., one not containing binary operations or function calls.
///
/// From the construction, the evaluation priorities within such an expression are always higher
/// than for possible binary ops surrounding it.
fn simplest_expr<'a, T, Ty>(input: InputSpan<'a>) -> NomResult<'a, SpannedExpr<'a, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    let block_parser: Box<dyn Fn(InputSpan<'a>) -> NomResult<'a, SpannedExpr<'a, T::Base>>> =
        if T::FEATURES.blocks {
            let parser = map(with_span(block::<T, Ty>), |spanned| {
                spanned.map_extra(Expr::Block)
            });
            Box::new(parser)
        } else {
            // Always fail.
            Box::new(|input| {
                let e = ErrorKind::Leftovers.with_span(&input.into());
                Err(NomErr::Error(e))
            })
        };

    let fn_def_parser: Box<dyn Fn(InputSpan<'a>) -> NomResult<'a, SpannedExpr<'a, T::Base>>> =
        if T::FEATURES.fn_definitions {
            let parser = map(with_span(fn_def::<T, Ty>), |span| {
                span.map_extra(Expr::FnDefinition)
            });
            Box::new(parser)
        } else {
            // Always fail.
            Box::new(|input| {
                let e = ErrorKind::Leftovers.with_span(&input.into());
                Err(NomErr::Error(e))
            })
        };

    alt((
        map(with_span(<T::Base>::parse_literal), |span| {
            span.map_extra(Expr::Literal)
        }),
        map(with_span(var_name), |span| {
            span.copy_with_extra(Expr::Variable)
        }),
        fn_def_parser,
        map(
            with_span(tuple((
                terminated(with_span(one_of("-!")), ws::<Ty>),
                simple_expr::<T, Ty>,
            ))),
            |spanned| {
                spanned.map_extra(|(op, inner)| Expr::Unary {
                    op: UnaryOp::from_span(op),
                    inner: Box::new(inner),
                })
            },
        ),
        block_parser,
        paren_expr::<T, Ty>,
    ))(input)
}

#[derive(Debug)]
struct MethodOrFnCall<'a, T: Grammar> {
    fn_name: Option<InputSpan<'a>>,
    args: Vec<SpannedExpr<'a, T>>,
}

impl<T: Grammar> MethodOrFnCall<'_, T> {
    fn is_method(&self) -> bool {
        self.fn_name.is_some()
    }
}

/// Simple expression, which includes, besides `simplest_expr`s, function calls.
fn simple_expr<'a, T, Ty>(input: InputSpan<'a>) -> NomResult<'a, SpannedExpr<'a, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    let fn_call_parser = map(fn_args::<T, Ty>, |(args, _)| MethodOrFnCall {
        fn_name: None,
        args,
    });

    let method_or_fn_call: Box<
        dyn Fn(InputSpan<'a>) -> NomResult<'a, MethodOrFnCall<'a, T::Base>>,
    > = if T::FEATURES.methods {
        let method_parser = map(
            tuple((var_name, fn_args::<T, Ty>)),
            |(fn_name, (args, _))| MethodOrFnCall {
                fn_name: Some(fn_name),
                args,
            },
        );

        let parser = alt((
            preceded(tuple((tag_char('.'), ws::<Ty>)), cut(method_parser)),
            fn_call_parser,
        ));
        Box::new(parser)
    } else {
        Box::new(fn_call_parser)
    };

    let parser = tuple((
        simplest_expr::<T, Ty>,
        many0(with_span(preceded(ws::<Ty>, method_or_fn_call))),
    ));
    parser(input).and_then(|(rest, (base, calls))| {
        fold_args(input, base, calls).map(|folded| (rest, folded))
    })
}

#[allow(clippy::option_if_let_else, clippy::range_plus_one)]
// ^-- See explanations in the function code.
fn fold_args<'a, T: Grammar>(
    input: InputSpan<'a>,
    mut base: SpannedExpr<'a, T>,
    calls: Vec<Spanned<MethodOrFnCall<'a, T>>>,
) -> Result<SpannedExpr<'a, T>, NomErr<Error<'a>>> {
    // Do we need to reorder unary op application and method calls? This is only applicable if:
    //
    // - `base` is a literal (as a corollary, `-` / `!` may be a start of a literal)
    // - The next call is a method
    // - A literal is parsed without the unary op.
    //
    // If all these conditions hold, we reorder the unary op to execute *after* all calls.
    // This is necessary to correctly parse `-1.abs()` as `-(1.abs())`, not as `(-1).abs()`.
    let mut maybe_reordered_op = None;

    if matches!(base.extra, Expr::Literal(_)) {
        match calls.first() {
            Some(call) if !call.extra.is_method() => {
                // Bogus function call, such as `1(2, 3)`.
                let e = ErrorKind::LiteralName.with_span(&base);
                return Err(NomErr::Failure(e));
            }
            // Indexing should be safe: literals must have non-empty span.
            Some(_) => maybe_reordered_op = UnaryOp::try_from_byte(base.fragment().as_bytes()[0]),
            None => { /* Special processing is not required. */ }
        }
    }

    let reordered_op = if let Some(reordered_op) = maybe_reordered_op {
        let lit_start = base.location_offset() - input.location_offset();
        let unsigned_lit_input = input.slice((lit_start + 1)..(lit_start + base.fragment().len()));

        if let Ok((_, unsigned_lit)) = T::parse_literal(unsigned_lit_input) {
            base = SpannedExpr::new(unsigned_lit_input, Expr::Literal(unsigned_lit));

            // `nom::Slice` is not implemented for inclusive range types, so a Clippy warning
            // cannot be fixed.
            let op_span = input.slice(lit_start..(lit_start + 1));
            Some(Spanned::new(op_span, reordered_op))
        } else {
            None
        }
    } else {
        None
    };

    let folded = calls.into_iter().fold(base, |name, call| {
        let united_span = unite_spans(input, &name, &call);

        // Clippy lint is triggered here. `name` cannot be moved into both branches,
        // so it's a false positive.
        let expr = if let Some(fn_name) = call.extra.fn_name {
            Expr::Method {
                name: fn_name.into(),
                receiver: Box::new(name),
                args: call.extra.args,
            }
        } else {
            Expr::Function {
                name: Box::new(name),
                args: call.extra.args,
            }
        };
        united_span.copy_with_extra(expr)
    });

    Ok(if let Some(unary_op) = reordered_op {
        unite_spans(input, &unary_op, &folded).copy_with_extra(Expr::Unary {
            op: unary_op,
            inner: Box::new(folded),
        })
    } else {
        folded
    })
}

/// Parses an expression with binary operations into a tree with the hierarchy reflecting
/// the evaluation order of the operations.
fn binary_expr<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, SpannedExpr<'_, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    // First, we parse the expression into a list with simple expressions interspersed
    // with binary operations. For example, `1 + 2 * foo(x, y)` is parsed into
    //
    //     [ 1, +, 2, *, foo(x, y) ]
    #[rustfmt::skip]
    let binary_ops = alt((
        tag("+"), tag("-"), tag("*"), tag("/"), tag("^"), // simple ops
        tag("=="), tag("!="), // equality comparisons
        tag("&&"), tag("||"), // logical ops
        tag(">="), tag("<="), tag(">"), tag("<"), // order comparisons
    ));
    let binary_ops = map(binary_ops, BinaryOp::from_span);

    let full_binary_ops = move |input| {
        let (rest, spanned_op) = binary_ops(input)?;
        if spanned_op.extra.is_supported(&T::FEATURES) {
            Ok((rest, spanned_op))
        } else {
            // Immediately drop parsing on an unsupported op, since there are no alternatives.
            let err = ErrorKind::UnsupportedOp(spanned_op.extra.into());
            let spanned_err = err.with_span(&spanned_op);
            Err(NomErr::Failure(spanned_err))
        }
    };

    let binary_parser = tuple((
        simple_expr::<T, Ty>,
        many0(tuple((
            delimited(ws::<Ty>, full_binary_ops, ws::<Ty>),
            cut(simple_expr::<T, Ty>),
        ))),
    ));

    let (remaining_input, (first, rest)) = binary_parser(input)?;
    let folded = fold_binary_expr(input, first, rest).map_err(NomErr::Failure)?;
    Ok((remaining_input, folded))
}

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
fn fold_binary_expr<'a, T: Grammar>(
    input: InputSpan<'a>,
    first: SpannedExpr<'a, T>,
    rest: Vec<(Spanned<'a, BinaryOp>, SpannedExpr<'a, T>)>,
) -> Result<SpannedExpr<'a, T>, Error<'a>> {
    let mut right_contour: Vec<BinaryOp> = vec![];

    rest.into_iter().try_fold(first, |mut acc, (new_op, expr)| {
        let united_span = unite_spans(input, &acc, &expr);

        let insert_pos = right_contour
            .iter()
            .position(|past_op| past_op.priority() >= new_op.extra.priority())
            .unwrap_or_else(|| right_contour.len());

        // We determine the error span later.
        let chained_comparison = right_contour.get(insert_pos).map_or(false, |past_op| {
            past_op.is_comparison() && new_op.extra.is_comparison()
        });

        right_contour.truncate(insert_pos);
        right_contour.push(new_op.extra);

        if insert_pos == 0 {
            if chained_comparison {
                return Err(ErrorKind::ChainedComparison.with_span(&united_span));
            }

            Ok(united_span.copy_with_extra(Expr::Binary {
                lhs: Box::new(acc),
                op: new_op,
                rhs: Box::new(expr),
            }))
        } else {
            let mut parent = &mut acc;
            for _ in 1..insert_pos {
                parent = match parent.extra {
                    Expr::Binary { ref mut rhs, .. } => rhs,
                    _ => unreachable!(),
                };
            }

            *parent = unite_spans(input, &parent, &expr).copy_with_extra(parent.extra.clone());
            if let Expr::Binary { ref mut rhs, .. } = parent.extra {
                let rhs_span = unite_spans(input, rhs, &expr);
                if chained_comparison {
                    return Err(ErrorKind::ChainedComparison.with_span(&rhs_span));
                }

                let dummy = Box::new(rhs.copy_with_extra(Expr::Variable));
                let old_rhs = mem::replace(rhs, dummy);
                let new_expr = Expr::Binary {
                    lhs: old_rhs,
                    op: new_op,
                    rhs: Box::new(expr),
                };
                *rhs = Box::new(rhs_span.copy_with_extra(new_expr));
            }
            acc = united_span.copy_with_extra(acc.extra);
            Ok(acc)
        }
    })
}

fn expr<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, SpannedExpr<'_, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    context(Context::Expr.to_str(), binary_expr::<T, Ty>)(input)
}

fn comma_sep<Ty: GrammarType>(input: InputSpan<'_>) -> NomResult<'_, char> {
    delimited(ws::<Ty>, tag_char(','), ws::<Ty>)(input)
}

fn comma_separated_lvalues<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, Vec<GrammarLvalue<'_, T>>>
where
    T: Parse,
    Ty: GrammarType,
{
    separated_list(comma_sep::<Ty>, lvalue::<T, Ty>)(input)
}

fn destructure_rest<T, Ty>(
    input: InputSpan<'_>,
) -> NomResult<'_, Spanned<'_, DestructureRest<'_, <T::Base as Grammar>::Type>>>
where
    T: Parse,
    Ty: GrammarType,
{
    map(
        with_span(preceded(
            terminated(tag("..."), ws::<Ty>),
            cut(opt(var_name)),
        )),
        |spanned| {
            spanned.map_extra(|maybe_name| {
                maybe_name.map_or(DestructureRest::Unnamed, |name| DestructureRest::Named {
                    variable: name.into(),
                    ty: None,
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
) -> NomResult<'_, DestructureTail<'_, <T::Base as Grammar>::Type>>
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
fn destructure<T, Ty>(
    input: InputSpan<'_>,
) -> NomResult<'_, Destructure<'_, <T::Base as Grammar>::Type>>
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

type GrammarLvalue<'a, T> = SpannedLvalue<'a, <<T as Parse>::Base as Grammar>::Type>;

/// Parses an `Lvalue`.
fn lvalue<'a, T, Ty>(input: InputSpan<'a>) -> NomResult<'a, GrammarLvalue<'a, T>>
where
    T: Parse,
    Ty: GrammarType,
{
    let simple_lvalue: Box<dyn Fn(InputSpan<'a>) -> NomResult<'a, GrammarLvalue<'a, T>>> =
        if T::FEATURES.type_annotations {
            let parser = map(
                tuple((
                    var_name,
                    opt(preceded(
                        delimited(ws::<Ty>, tag_char(':'), ws::<Ty>),
                        cut(with_span(<T::Base>::parse_type)),
                    )),
                )),
                |(name, ty)| Spanned::new(name, Lvalue::Variable { ty }),
            );
            Box::new(parser)
        } else {
            let parser = map(var_name, |name| {
                Spanned::new(name, Lvalue::Variable { ty: None })
            });
            Box::new(parser)
        };

    if T::FEATURES.tuples {
        alt((
            with_span(map(
                delimited(
                    terminated(tag_char('('), ws::<Ty>),
                    destructure::<T, Ty>,
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
