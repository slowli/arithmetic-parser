//! Tests for functions / methods.

use assert_matches::assert_matches;
use nom::Err as NomErr;

use super::{args, lsp, lvalue_tuple, sp, span, FieldGrammar, Literal, LiteralType, ValueType};
use crate::{
    parser::{expr, fn_def, simple_expr, Complete},
    BinaryOp, Block, ErrorKind, Expr, FnDefinition, InputSpan, Lvalue, MethodCallSeparator,
    Spanned,
};

#[test]
fn fun_works() {
    let input = InputSpan::new("ge(0x123456) + 1");
    assert_eq!(
        simple_expr::<FieldGrammar, Complete>(input).unwrap().1,
        sp(
            0,
            "ge(0x123456)",
            Expr::Function {
                name: Box::new(sp(0, "ge", Expr::Variable)),
                args: vec![sp(
                    3,
                    "0x123456",
                    Expr::Literal(Literal::Bytes {
                        value: vec![0x12, 0x34, 0x56],
                        ty: LiteralType::Bytes,
                    })
                ),]
            }
        )
    );

    let input = InputSpan::new("ge (  0x123456\t) + A");
    assert_eq!(
        simple_expr::<FieldGrammar, Complete>(input).unwrap().1,
        sp(
            0,
            "ge (  0x123456\t)",
            Expr::Function {
                name: Box::new(sp(0, "ge", Expr::Variable)),
                args: vec![sp(
                    6,
                    "0x123456",
                    Expr::Literal(Literal::Bytes {
                        value: vec![0x12, 0x34, 0x56],
                        ty: LiteralType::Bytes,
                    })
                )]
            }
        )
    );
}

#[test]
fn fun_call_with_terminating_comma() {
    let input = InputSpan::new("ge(1, 2 ,\n)");
    assert_eq!(
        simple_expr::<FieldGrammar, Complete>(input).unwrap().1,
        sp(
            0,
            "ge(1, 2 ,\n)",
            Expr::Function {
                name: Box::new(sp(0, "ge", Expr::Variable)),
                args: vec![
                    sp(3, "1", Expr::Literal(Literal::Number)),
                    sp(6, "2", Expr::Literal(Literal::Number)),
                ],
            }
        )
    );
}

#[test]
fn fun_works_with_complex_called_values() {
    let input = InputSpan::new("ge(x, 3)(0x123456) + 5");
    let inner_fn = sp(
        0,
        "ge(x, 3)",
        Expr::Function {
            name: Box::new(sp(0, "ge", Expr::Variable)),
            args: vec![
                sp(3, "x", Expr::Variable),
                sp(6, "3", Expr::Literal(Literal::Number)),
            ],
        },
    );
    assert_eq!(
        simple_expr::<FieldGrammar, Complete>(input).unwrap().1,
        sp(
            0,
            "ge(x, 3)(0x123456)",
            Expr::Function {
                name: Box::new(inner_fn),
                args: vec![sp(
                    9,
                    "0x123456",
                    Expr::Literal(Literal::Bytes {
                        value: vec![0x12, 0x34, 0x56],
                        ty: LiteralType::Bytes,
                    })
                )]
            }
        )
    );

    let input = InputSpan::new("(|x| x + 1)(0xs_123456) + 5");
    let (_, function) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    let function_value = match function.extra {
        Expr::Function { name, .. } => name.extra,
        other => panic!("unexpected expr: {:?}", other),
    };
    assert_matches!(function_value, Expr::FnDefinition(_));

    let input = InputSpan::new("|x| { x + 1 }(0xs_123456) + 5");
    let (_, function) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    let function_value = match function.extra {
        Expr::Function { name, .. } => name.extra,
        other => panic!("unexpected expr: {:?}", other),
    };
    assert_matches!(function_value, Expr::FnDefinition(_));
}

#[test]
fn method_expr_works() {
    let input = InputSpan::new("x.sin();");
    let (_, call) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    assert_eq!(
        call,
        sp(
            0,
            "x.sin()",
            Expr::Method {
                name: span(2, "sin").into(),
                receiver: Box::new(sp(0, "x", Expr::Variable)),
                separator: sp(1, ".", MethodCallSeparator::Dot),
                args: vec![],
            }
        )
    );

    let input = InputSpan::new("(1, x, 2).foo(y) + 3;");
    let (_, call) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    let expected = sp(
        0,
        "(1, x, 2).foo(y)",
        Expr::Method {
            name: span(10, "foo").into(),
            receiver: Box::new(sp(
                0,
                "(1, x, 2)",
                Expr::Tuple(vec![
                    sp(1, "1", Expr::Literal(Literal::Number)),
                    sp(4, "x", Expr::Variable),
                    sp(7, "2", Expr::Literal(Literal::Number)),
                ]),
            )),
            separator: sp(9, ".", MethodCallSeparator::Dot),
            args: vec![sp(14, "y", Expr::Variable)],
        },
    );
    assert_eq!(call, expected);

    let input = InputSpan::new("(1, x, 2).foo(y)(7.bar());");
    let (_, call) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    assert_eq!(
        call,
        sp(
            0,
            "(1, x, 2).foo(y)(7.bar())",
            Expr::Function {
                name: Box::new(expected),
                args: vec![sp(
                    17,
                    "7.bar()",
                    Expr::Method {
                        name: span(19, "bar").into(),
                        receiver: Box::new(sp(17, "7", Expr::Literal(Literal::Number))),
                        separator: sp(18, ".", MethodCallSeparator::Dot),
                        args: vec![],
                    }
                )],
            }
        )
    );
}

#[test]
fn method_call_with_colon2_separator() {
    let input = InputSpan::new("Num::sin(x)");
    let (_, call) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    let expected_call = Expr::Method {
        name: span(5, "sin").into(),
        receiver: Box::new(sp(0, "Num", Expr::Variable)),
        separator: sp(3, "::", MethodCallSeparator::Colon2),
        args: vec![sp(9, "x", Expr::Variable)],
    };
    assert_eq!(call.extra, expected_call);

    let ws_input = InputSpan::new("Num   :: sin (  x  )");
    let (_, ws_call) = simple_expr::<FieldGrammar, Complete>(ws_input).unwrap();
    assert_matches!(ws_call.extra, Expr::Method { .. });

    let chained_input = InputSpan::new("Num::sin(x).cos()");
    let (_, chained_call) = expr::<FieldGrammar, Complete>(chained_input).unwrap();
    assert_eq!(
        chained_call.extra,
        Expr::Method {
            name: span(12, "cos").into(),
            receiver: Box::new(sp(0, "Num::sin(x)", expected_call)),
            separator: sp(11, ".", MethodCallSeparator::Dot),
            args: vec![],
        }
    );

    let chained_input = InputSpan::new("x.sin()::cos()");
    let (_, chained_call) = expr::<FieldGrammar, Complete>(chained_input).unwrap();
    let inner_call = Expr::Method {
        name: span(2, "sin").into(),
        receiver: Box::new(sp(0, "x", Expr::Variable)),
        separator: sp(1, ".", MethodCallSeparator::Dot),
        args: vec![],
    };
    assert_eq!(
        chained_call.extra,
        Expr::Method {
            name: span(9, "cos").into(),
            receiver: Box::new(sp(0, "x.sin()", inner_call)),
            separator: sp(7, "::", MethodCallSeparator::Colon2),
            args: vec![],
        }
    );
}

#[test]
fn errors_parsing_colon2_method_call() {
    let input = InputSpan::new("Num::sin + 1");
    let err = simple_expr::<FieldGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Failure(e) if e.span().location_offset() == 9);
}

#[test]
fn fn_definition_parsing() {
    let input = InputSpan::new("|x| x + z;");
    assert_eq!(
        fn_def::<FieldGrammar, Complete>(input).unwrap().1,
        FnDefinition {
            args: args(
                span(0, "|x|"),
                vec![lsp(1, "x", Lvalue::Variable { ty: None })]
            ),
            body: Block {
                statements: vec![],
                return_value: Some(Box::new(sp(
                    4,
                    "x + z",
                    Expr::Binary {
                        lhs: Box::new(sp(4, "x", Expr::Variable)),
                        op: BinaryOp::from_span(span(6, "+")),
                        rhs: Box::new(sp(8, "z", Expr::Variable)),
                    }
                ))),
            }
        }
    );

    let input = InputSpan::new("|x| { x + 3 }");
    assert_eq!(
        fn_def::<FieldGrammar, Complete>(input).unwrap().1,
        FnDefinition {
            args: args(
                span(0, "|x|"),
                vec![lsp(1, "x", Lvalue::Variable { ty: None })]
            ),
            body: Block {
                statements: vec![],
                return_value: Some(Box::new(sp(
                    6,
                    "x + 3",
                    Expr::Binary {
                        lhs: Box::new(sp(6, "x", Expr::Variable)),
                        op: BinaryOp::from_span(span(8, "+")),
                        rhs: Box::new(sp(10, "3", Expr::Literal(Literal::Number))),
                    }
                )))
            }
        }
    );

    let input = InputSpan::new("|x: Sc, (y, _: Ge)| { x + y }");
    let mut def = fn_def::<FieldGrammar, Complete>(input).unwrap().1;
    assert!(def.body.statements.is_empty());
    assert!(def.body.return_value.is_some());

    def.body = Block {
        statements: vec![],
        return_value: None,
    };
    assert_eq!(
        def,
        FnDefinition {
            args: args(
                span(0, "|x: Sc, (y, _: Ge)|"),
                vec![
                    lsp(
                        1,
                        "x",
                        Lvalue::Variable {
                            ty: Some(Spanned::new(span(4, "Sc"), ValueType::Scalar)),
                        }
                    ),
                    lsp(
                        8,
                        "(y, _: Ge)",
                        lvalue_tuple(vec![
                            lsp(9, "y", Lvalue::Variable { ty: None }),
                            lsp(
                                12,
                                "_",
                                Lvalue::Variable {
                                    ty: Some(Spanned::new(span(15, "Ge"), ValueType::Element)),
                                }
                            )
                        ])
                    ),
                ]
            ),
            body: Block {
                statements: vec![],
                return_value: None
            },
        }
    );
}

#[test]
fn function_call_with_literal_as_fn_name() {
    let input = InputSpan::new("1(2, 3);");
    let err = expr::<FieldGrammar, Complete>(input).unwrap_err();
    let spanned_err = match err {
        NomErr::Failure(spanned) => spanned,
        _ => panic!("Unexpected error: {}", err),
    };
    assert_matches!(spanned_err.kind(), ErrorKind::LiteralName);
}
