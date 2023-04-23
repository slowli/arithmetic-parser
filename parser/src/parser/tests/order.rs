//! Tests for order of operations.

use assert_matches::assert_matches;
use nom::Err as NomErr;

use super::{sp, span, FieldGrammar};
use crate::{
    grammars::{F32Grammar, Untyped},
    parser::{binary_expr, expr, Complete},
    BinaryOp, ErrorKind, Expr, InputSpan, UnaryOp,
};

#[test]
fn expr_evaluation_order() {
    let input = InputSpan::new("1 - 2 + 3 - 4;");
    assert_matches!(
        expr::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Expr::Binary { op, .. } if op == BinaryOp::from_span(span(10, "-"))
    );

    let input = InputSpan::new("1 / 2 * 3 / 4;");
    assert_matches!(
        expr::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Expr::Binary { op, .. } if op == BinaryOp::from_span(span(10, "/"))
    );

    let input = InputSpan::new("1 - 2 * 3 - 4;");
    assert_matches!(
        expr::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Expr::Binary { op, .. } if op == BinaryOp::from_span(span(10, "-"))
    );

    let input = InputSpan::new("X - G^2 + y * Z;");
    assert_matches!(
        expr::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Expr::Binary { op, .. } if op == BinaryOp::from_span(span(8, "+"))
    );
}

#[test]
fn evaluation_order_with_bool_expressions() {
    let input = InputSpan::new("x == 2 + 3 * 4 && y == G^x;");
    let output = expr::<FieldGrammar, Complete>(input).unwrap().1.extra;
    assert_matches!(
        output,
        Expr::Binary { op, ref lhs, ref rhs } if op == BinaryOp::from_span(span(15, "&&")) &&
            *lhs.fragment() == "x == 2 + 3 * 4" &&
            *rhs.fragment() == "y == G^x"
    );
    let scalar_expr = output.binary_lhs().unwrap().extra.binary_rhs().unwrap();
    assert_eq!(*scalar_expr.fragment(), "2 + 3 * 4");
    assert_matches!(
        scalar_expr.extra,
        Expr::Binary { op, .. } if op == BinaryOp::from_span(span(7, "+"))
    );
}

#[test]
fn evaluation_order_with_complex_bool_expressions() {
    let input = InputSpan::new("x == 2 * z + 3 * 4 && (y, z) == (G^x, 2);");
    let output = expr::<FieldGrammar, Complete>(input).unwrap().1.extra;
    assert_matches!(
        output,
        Expr::Binary { op, ref lhs, ref rhs } if op == BinaryOp::from_span(span(19, "&&")) &&
            *lhs.fragment() == "x == 2 * z + 3 * 4" &&
            *rhs.fragment() == "(y, z) == (G^x, 2)"
    );
    let scalar_expr = output.binary_lhs().unwrap().extra.binary_rhs().unwrap();
    assert_eq!(*scalar_expr.fragment(), "2 * z + 3 * 4");
    assert_matches!(
        scalar_expr.extra,
        Expr::Binary { op, .. } if op == BinaryOp::from_span(span(11, "+"))
    );
}

#[test]
fn methods_have_higher_priority_than_unary_ops() {
    const INPUTS: &[&str] = &["-5.abs();", "-5.5.sin();", "-5.25.foo(1, PI)(1, x).cos();"];

    for &input in INPUTS {
        let input = InputSpan::new(input);
        let expr = expr::<Untyped<F32Grammar>, Complete>(input)
            .unwrap()
            .1
            .extra;
        assert_matches!(
            expr,
            Expr::Unary { op, .. } if op.extra == UnaryOp::Neg,
            "Errored input: {}",
            input.fragment()
        );
    }
}

#[test]
fn methods_and_multiple_unary_ops() {
    let input = InputSpan::new("--1.abs();");
    let expr = expr::<Untyped<F32Grammar>, Complete>(input).unwrap().1;
    assert_eq!(*expr.fragment(), "--1.abs()");
    let Expr::Unary { inner, .. } = expr.extra else {
        panic!("Unexpected expr: {expr:?}");
    };
    assert_eq!(*inner.fragment(), "-1.abs()");
    assert_matches!(inner.extra, Expr::Unary { .. });
}

#[test]
fn and_has_higher_priority_than_or() {
    let input = InputSpan::new("x || y && z");
    let expr = expr::<FieldGrammar, Complete>(input).unwrap().1.extra;
    assert_eq!(
        expr,
        Expr::Binary {
            lhs: Box::new(sp(0, "x", Expr::Variable)),
            op: sp(2, "||", BinaryOp::Or),
            rhs: Box::new(sp(
                5,
                "y && z",
                Expr::Binary {
                    lhs: Box::new(sp(5, "y", Expr::Variable)),
                    op: sp(7, "&&", BinaryOp::And),
                    rhs: Box::new(sp(10, "z", Expr::Variable)),
                }
            ))
        }
    );
}

#[test]
fn chained_comparisons() {
    const INPUTS: &[&str] = &[
        "1 > 2 > 3",
        "1 == 2 == 3",
        "x < y > z",
        "x == 2 > y",
        "x <= 3.foo() > bar(1, 2)",
        "1 + 2 < 3 != (z + 2)",
    ];

    for &input in INPUTS {
        let err = binary_expr::<FieldGrammar, Complete>(InputSpan::new(input)).unwrap_err();
        let NomErr::Failure(err) = err else {
            panic!("Unexpected error: {err:?}");
        };
        assert_matches!(err.kind(), ErrorKind::ChainedComparison);
        assert_eq!(*err.span().fragment(), input);
    }
}

#[test]
fn chained_comparisons_with_larger_context() {
    let input = "x == 3 && 1 + 2 > x.abs() == 3 && T";
    let err = binary_expr::<FieldGrammar, Complete>(InputSpan::new(input)).unwrap_err();
    let NomErr::Failure(err) = err else {
        panic!("Unexpected error: {err:?}");
    };
    assert_matches!(err.kind(), ErrorKind::ChainedComparison);
    assert_eq!(*err.span().fragment(), "1 + 2 > x.abs() == 3");
}

#[test]
fn valid_sequential_comparisons() {
    const INPUTS: &[&str] = &["x == (y > z)", "(x < 3) < (y == 2)", "(1 == 2) != true"];

    for &input in INPUTS {
        let input = InputSpan::new(input);
        let expr = binary_expr::<FieldGrammar, Complete>(input)
            .unwrap()
            .1
            .extra;
        assert_matches!(expr, Expr::Binary { .. });
    }
}
