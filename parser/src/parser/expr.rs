//! `Expr`-related parsing functions.

use core::mem;

use nom::{
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::complete::{char as tag_char, one_of},
    combinator::{cut, map, map_res, opt},
    error::context,
    multi::{many0, separated_list0},
    sequence::{delimited, preceded, terminated, tuple},
    Err as NomErr, Slice,
};

use super::{
    block, fn_def,
    helpers::{comma_sep, is_valid_variable_name, mandatory_ws, var_name, ws, GrammarType},
};
use crate::{
    alloc::{vec, Box, Vec},
    grammars::{Features, Grammar, Parse, ParseLiteral},
    spans::{unite_spans, with_span},
    BinaryOp, Context, Error, ErrorKind, Expr, InputSpan, NomResult, ObjectExpr, Spanned,
    SpannedExpr, UnaryOp,
};

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
            separated_list0(delimited(ws::<Ty>, tag_char(','), ws::<Ty>), expr::<T, Ty>),
            terminated(maybe_comma, tuple((ws::<Ty>, tag_char(')')))),
        ))),
    )(input)
}

/// Expression enclosed in parentheses. This may be a simple value (e.g., `(1 + 2)`)
/// or a tuple (e.g., `(x, y)`), depending on the number of comma-separated terms.
pub(super) fn paren_expr<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, SpannedExpr<'_, T::Base>>
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
                if T::FEATURES.contains(Features::TUPLES) {
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

/// Parses a block and wraps it into an `Expr`.
fn block_expr<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, SpannedExpr<'_, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    map(with_span(block::<T, Ty>), |spanned| {
        spanned.map_extra(Expr::Block)
    })(input)
}

fn object_expr_field<T, Ty>(
    input: InputSpan<'_>,
) -> NomResult<'_, (Spanned<'_>, Option<SpannedExpr<'_, T::Base>>)>
where
    T: Parse,
    Ty: GrammarType,
{
    let colon_sep = delimited(ws::<Ty>, tag_char(':'), ws::<Ty>);
    tuple((
        map(var_name, Spanned::from),
        opt(preceded(colon_sep, expr::<T, Ty>)),
    ))(input)
}

pub(super) fn object_expr<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, SpannedExpr<'_, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    let object = preceded(
        terminated(tag("#{"), ws::<Ty>),
        cut(terminated(
            terminated(
                separated_list0(comma_sep::<Ty>, object_expr_field::<T, Ty>),
                opt(comma_sep::<Ty>),
            ),
            preceded(ws::<Ty>, tag_char('}')),
        )),
    );

    map(with_span(object), |spanned| {
        spanned.map_extra(|fields| Expr::Object(ObjectExpr { fields }))
    })(input)
}

/// Parses a function definition and wraps it into an `Expr`.
fn fn_def_expr<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, SpannedExpr<'_, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    map(with_span(fn_def::<T, Ty>), |spanned| {
        spanned.map_extra(Expr::FnDefinition)
    })(input)
}

/// Parses a simple expression, i.e., one not containing binary operations or function calls.
///
/// From the construction, the evaluation priorities within such an expression are always higher
/// than for possible binary ops surrounding it.
fn simplest_expr<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, SpannedExpr<'_, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    fn error<T>(input: InputSpan<'_>) -> NomResult<'_, SpannedExpr<'_, T::Base>>
    where
        T: Parse,
    {
        let e = ErrorKind::Leftovers.with_span(&input.into());
        Err(NomErr::Error(e))
    }

    let block_parser = if T::FEATURES.contains(Features::BLOCKS) {
        block_expr::<T, Ty>
    } else {
        error::<T>
    };
    let object_parser = if T::FEATURES.contains(Features::OBJECTS) {
        object_expr::<T, Ty>
    } else {
        error::<T>
    };
    let fn_def_parser = if T::FEATURES.contains(Features::FN_DEFINITIONS) {
        fn_def_expr::<T, Ty>
    } else {
        error::<T>
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
                expr_with_calls::<T, Ty>,
            ))),
            |spanned| {
                spanned.map_extra(|(op, inner)| Expr::Unary {
                    op: UnaryOp::from_span(op),
                    inner: Box::new(inner),
                })
            },
        ),
        block_parser,
        object_parser,
        paren_expr::<T, Ty>,
    ))(input)
}

#[derive(Debug)]
struct MethodOrFnCall<'a, T: Grammar> {
    separator: Option<Spanned<'a>>,
    fn_name: Option<SpannedExpr<'a, T>>,
    args: Option<Vec<SpannedExpr<'a, T>>>,
}

impl<T: Grammar> MethodOrFnCall<'_, T> {
    fn is_method(&self) -> bool {
        self.fn_name.is_some() && self.args.is_some()
    }
}

type MethodParseResult<'a, T> = NomResult<'a, MethodOrFnCall<'a, <T as Parse>::Base>>;

fn fn_call<T, Ty>(input: InputSpan<'_>) -> MethodParseResult<'_, T>
where
    T: Parse,
    Ty: GrammarType,
{
    map(fn_args::<T, Ty>, |(args, _)| MethodOrFnCall {
        separator: None,
        fn_name: None,
        args: Some(args),
    })(input)
}

fn method_or_fn_call<T, Ty>(input: InputSpan<'_>) -> MethodParseResult<'_, T>
where
    T: Parse,
    Ty: GrammarType,
{
    let var_name_or_digits = alt((var_name, take_while1(|c: char| c.is_ascii_digit())));
    let method_or_field_expr = alt((
        map(with_span(var_name_or_digits), |span| {
            span.copy_with_extra(Expr::Variable)
        }),
        block_expr::<T, Ty>,
    ));
    let method_or_field_access_parser = map_res(
        tuple((
            method_or_field_expr,
            opt(preceded(ws::<Ty>, fn_args::<T, Ty>)),
        )),
        |(fn_name, maybe_args)| {
            if maybe_args.is_some() {
                let is_bogus_name = matches!(fn_name.extra, Expr::Variable)
                    && !is_valid_variable_name(fn_name.fragment());
                if is_bogus_name {
                    return Err(ErrorKind::LiteralName);
                }
            }
            Ok(MethodOrFnCall {
                separator: None,
                fn_name: Some(fn_name),
                args: maybe_args.map(|(args, _)| args),
            })
        },
    );
    let method_or_field_access_parser = map(
        tuple((
            terminated(with_span(tag_char('.')), ws::<Ty>),
            cut(method_or_field_access_parser),
        )),
        |(separator, mut call)| {
            call.separator = Some(separator.with_no_extra());
            call
        },
    );

    alt((method_or_field_access_parser, fn_call::<T, Ty>))(input)
}

/// Expression, which includes, besides `simplest_expr`s, function calls.
fn expr_with_calls<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, SpannedExpr<'_, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    let method_or_fn_call = if T::FEATURES.contains(Features::METHODS) {
        method_or_fn_call::<T, Ty>
    } else {
        fn_call::<T, Ty>
    };

    let mut parser = tuple((
        simplest_expr::<T, Ty>,
        many0(with_span(preceded(ws::<Ty>, method_or_fn_call))),
    ));
    parser(input).and_then(|(rest, (base, calls))| {
        fold_args(input, base, calls).map(|folded| (rest, folded))
    })
}

/// Simple expression, which includes `expr_with_calls` and type casts.
pub(super) fn simple_expr<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, SpannedExpr<'_, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    let as_keyword = delimited(ws::<Ty>, tag("as"), mandatory_ws::<Ty>);
    let parser = tuple((
        expr_with_calls::<T, Ty>,
        many0(preceded(as_keyword, cut(with_span(<T::Base>::parse_type)))),
    ));

    map(parser, |(base, casts)| {
        casts.into_iter().fold(base, |value, ty| {
            let united_span = unite_spans(input, &value, &ty);
            united_span.copy_with_extra(Expr::TypeCast {
                value: Box::new(value),
                ty,
            })
        })
    })(input)
}

#[allow(clippy::option_if_let_else, clippy::range_plus_one)]
// ^-- See explanations in the function code.
fn fold_args<'a, T: Grammar>(
    input: InputSpan<'a>,
    mut base: SpannedExpr<'a, T>,
    calls: Vec<Spanned<'a, MethodOrFnCall<'a, T>>>,
) -> Result<SpannedExpr<'a, T>, NomErr<Error>> {
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
                // Bogus function call, such as `1(2, 3)` or `1.x`.
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

            // `nom::Slice` is not implemented for inclusive range types, so the Clippy warning
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
            if let Some(args) = call.extra.args {
                Expr::Method {
                    name: fn_name.into(),
                    receiver: Box::new(name),
                    separator: call.extra.separator.unwrap(), // safe by construction
                    args,
                }
            } else {
                Expr::FieldAccess {
                    name: fn_name.into(),
                    receiver: Box::new(name),
                }
            }
        } else {
            Expr::Function {
                name: Box::new(name),
                args: call.extra.args.expect("Args must be present for functions"),
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
pub(super) fn binary_expr<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, SpannedExpr<'_, T::Base>>
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
    let mut binary_ops = map(binary_ops, BinaryOp::from_span);

    let full_binary_ops = move |input| {
        let (rest, spanned_op) = binary_ops(input)?;
        if spanned_op.extra.is_supported(T::FEATURES) {
            Ok((rest, spanned_op))
        } else {
            // Immediately drop parsing on an unsupported op, since there are no alternatives.
            let err = ErrorKind::UnsupportedOp(spanned_op.extra.into());
            let spanned_err = err.with_span(&spanned_op);
            Err(NomErr::Failure(spanned_err))
        }
    };

    let mut binary_parser = tuple((
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
) -> Result<SpannedExpr<'a, T>, Error> {
    let mut right_contour: Vec<BinaryOp> = vec![];

    rest.into_iter().try_fold(first, |mut acc, (new_op, expr)| {
        let united_span = unite_spans(input, &acc, &expr);

        let insert_pos = right_contour
            .iter()
            .position(|past_op| past_op.priority() >= new_op.extra.priority())
            .unwrap_or(right_contour.len());

        // We determine the error span later.
        let chained_comparison = right_contour
            .get(insert_pos)
            .is_some_and(|past_op| past_op.is_comparison() && new_op.extra.is_comparison());

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
                parent = match &mut parent.extra {
                    Expr::Binary { rhs, .. } => rhs,
                    _ => unreachable!(),
                };
            }

            *parent = unite_spans(input, parent, &expr).copy_with_extra(parent.extra.clone());
            if let Expr::Binary { rhs, .. } = &mut parent.extra {
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

pub(super) fn expr<T, Ty>(input: InputSpan<'_>) -> NomResult<'_, SpannedExpr<'_, T::Base>>
where
    T: Parse,
    Ty: GrammarType,
{
    context(Context::Expr.to_str(), binary_expr::<T, Ty>)(input)
}
