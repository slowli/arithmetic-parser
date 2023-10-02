//! Tests for tuples and destructuring.

use super::{lsp, lvalue_tuple, sp, span, FieldGrammar, Literal, ValueType};
use crate::{
    parser::{
        destructure,
        expr::{expr, paren_expr},
        lvalue, Complete,
    },
    BinaryOp, Destructure, DestructureRest, Expr, InputSpan, Lvalue, Spanned,
};

#[test]
fn tuples_are_parsed() {
    let input = InputSpan::new("(x, y)");
    assert_eq!(
        paren_expr::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Expr::Tuple(vec![sp(1, "x", Expr::Variable), sp(4, "y", Expr::Variable)])
    );

    let input = InputSpan::new("(x / 2, G^y, 1)");
    assert_eq!(
        paren_expr::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Expr::Tuple(vec![
            sp(
                1,
                "x / 2",
                Expr::Binary {
                    lhs: Box::new(sp(1, "x", Expr::Variable)),
                    op: BinaryOp::from_span(span(3, "/")),
                    rhs: Box::new(sp(5, "2", Expr::Literal(Literal::Number))),
                }
            ),
            sp(
                8,
                "G^y",
                Expr::Binary {
                    lhs: Box::new(sp(8, "G", Expr::Variable)),
                    op: BinaryOp::from_span(span(9, "^")),
                    rhs: Box::new(sp(10, "y", Expr::Variable)),
                }
            ),
            sp(13, "1", Expr::Literal(Literal::Number)),
        ]),
    );
}

#[test]
fn empty_tuple_is_parsed() {
    let input = InputSpan::new("();");
    let value = expr::<FieldGrammar, Complete>(input).unwrap().1;
    assert_eq!(value, sp(0, "()", Expr::Tuple(vec![])));
}

#[test]
fn single_value_tuple_is_parsed() {
    let input = InputSpan::new("(1,);");
    let value = expr::<FieldGrammar, Complete>(input).unwrap().1;
    assert_eq!(
        value,
        sp(
            0,
            "(1,)",
            Expr::Tuple(vec![sp(1, "1", Expr::Literal(Literal::Number))])
        )
    );
}

#[test]
fn destructuring_is_parsed() {
    let input = InputSpan::new("x, y)");
    let expected = Destructure {
        start: vec![
            lsp(0, "x", Lvalue::Variable { ty: None }),
            lsp(3, "y", Lvalue::Variable { ty: None }),
        ],
        middle: None,
        end: vec![],
    };
    assert_eq!(
        destructure::<FieldGrammar, Complete>(input).unwrap().1,
        expected
    );

    let input = InputSpan::new("x, y ,\n)");
    let (rest, parsed) = destructure::<FieldGrammar, Complete>(input).unwrap();
    assert_eq!(parsed, expected);
    assert_eq!(*rest.fragment(), ")");

    let input = InputSpan::new("x, ..., y)");
    assert_eq!(
        destructure::<FieldGrammar, Complete>(input).unwrap().1,
        Destructure {
            start: vec![lsp(0, "x", Lvalue::Variable { ty: None })],
            middle: Some(Spanned::new(span(3, "..."), DestructureRest::Unnamed)),
            end: vec![lsp(8, "y", Lvalue::Variable { ty: None })],
        }
    );

    let input = InputSpan::new("x, ...rest, y: Ge, z)");
    assert_eq!(
        destructure::<FieldGrammar, Complete>(input).unwrap().1,
        Destructure {
            start: vec![lsp(0, "x", Lvalue::Variable { ty: None })],
            middle: Some(Spanned::new(
                span(3, "...rest"),
                DestructureRest::Named {
                    variable: span(6, "rest").into(),
                    ty: None,
                }
            )),
            end: vec![
                lsp(
                    12,
                    "y",
                    Lvalue::Variable {
                        ty: Some(Spanned::new(span(15, "Ge"), ValueType::Element))
                    }
                ),
                lsp(19, "z", Lvalue::Variable { ty: None })
            ],
        }
    );

    let input = InputSpan::new("...xs: Ge, end)");
    assert_eq!(
        destructure::<FieldGrammar, Complete>(input).unwrap().1,
        Destructure {
            start: vec![],
            middle: Some(Spanned::new(
                span(0, "...xs: Ge"),
                DestructureRest::Named {
                    variable: span(3, "xs").into(),
                    ty: Some(Spanned::new(span(7, "Ge"), ValueType::Element))
                }
            )),
            end: vec![lsp(11, "end", Lvalue::Variable { ty: None })],
        }
    );
}

#[test]
fn nested_destructuring_is_parsed() {
    let input = InputSpan::new("(x, y), ..., (_, ...rest),\n|");
    let start = Destructure {
        start: vec![
            lsp(1, "x", Lvalue::Variable { ty: None }),
            lsp(4, "y", Lvalue::Variable { ty: None }),
        ],
        middle: None,
        end: vec![],
    };
    let end = Destructure {
        start: vec![lsp(14, "_", Lvalue::Variable { ty: None })],
        middle: Some(Spanned::new(
            span(17, "...rest"),
            DestructureRest::Named {
                variable: span(20, "rest").into(),
                ty: None,
            },
        )),
        end: vec![],
    };

    assert_eq!(
        destructure::<FieldGrammar, Complete>(input).unwrap().1,
        Destructure {
            start: vec![Spanned::new(span(0, "(x, y)"), Lvalue::Tuple(start))],
            middle: Some(Spanned::new(span(8, "..."), DestructureRest::Unnamed)),
            end: vec![Spanned::new(span(13, "(_, ...rest)"), Lvalue::Tuple(end))],
        }
    );
}

#[test]
fn lvalues_are_parsed() {
    let input = InputSpan::new("x =");
    assert_eq!(
        lvalue::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Lvalue::Variable { ty: None }
    );

    let input = InputSpan::new("(x, (y, z)) =");
    assert_eq!(
        lvalue::<FieldGrammar, Complete>(input).unwrap().1.extra,
        lvalue_tuple(vec![
            lsp(1, "x", Lvalue::Variable { ty: None }),
            lsp(
                4,
                "(y, z)",
                lvalue_tuple(vec![
                    lsp(5, "y", Lvalue::Variable { ty: None }),
                    lsp(8, "z", Lvalue::Variable { ty: None }),
                ])
            )
        ])
    );

    let input = InputSpan::new("(x: (Sc, _), (y, z: Ge)) =");
    assert_eq!(
        lvalue::<FieldGrammar, Complete>(input).unwrap().1.extra,
        lvalue_tuple(vec![
            lsp(
                1,
                "x",
                Lvalue::Variable {
                    ty: Some(Spanned::new(
                        span(4, "(Sc, _)"),
                        ValueType::Tuple(vec![ValueType::Scalar, ValueType::Any])
                    )),
                }
            ),
            lsp(
                13,
                "(y, z: Ge)",
                lvalue_tuple(vec![
                    lsp(14, "y", Lvalue::Variable { ty: None }),
                    lsp(
                        17,
                        "z",
                        Lvalue::Variable {
                            ty: Some(Spanned::new(span(20, "Ge"), ValueType::Element)),
                        }
                    ),
                ])
            )
        ])
    );
}
