//! Tests for functions / methods.

use assert_matches::assert_matches;
use nom::Err as NomErr;

use super::{args, lsp, lvalue_tuple, sp, span, FieldGrammar, Literal, LiteralType, ValueType};
use crate::{
    parser::{
        expr::{expr, simple_expr},
        fn_def, Complete,
    },
    BinaryOp, Block, ErrorKind, Expr, FnDefinition, InputSpan, Lvalue, Spanned,
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
    let Expr::Function { name: fn_name, .. } = function.extra else {
        panic!("unexpected expr: {function:?}");
    };
    assert_matches!(fn_name.extra, Expr::FnDefinition(_));

    let input = InputSpan::new("|x| { x + 1 }(0xs_123456) + 5");
    let (_, function) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    let Expr::Function { name: fn_name, .. } = function.extra else {
        panic!("unexpected expr: {function:?}");
    };
    assert_matches!(fn_name.extra, Expr::FnDefinition(_));
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
                name: Box::new(sp(2, "sin", Expr::Variable)),
                receiver: Box::new(sp(0, "x", Expr::Variable)),
                separator: span(1, ".").into(),
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
            name: Box::new(sp(10, "foo", Expr::Variable)),
            receiver: Box::new(sp(
                0,
                "(1, x, 2)",
                Expr::Tuple(vec![
                    sp(1, "1", Expr::Literal(Literal::Number)),
                    sp(4, "x", Expr::Variable),
                    sp(7, "2", Expr::Literal(Literal::Number)),
                ]),
            )),
            separator: span(9, ".").into(),
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
                        name: Box::new(sp(19, "bar", Expr::Variable)),
                        receiver: Box::new(sp(17, "7", Expr::Literal(Literal::Number))),
                        separator: span(18, ".").into(),
                        args: vec![],
                    }
                )],
            }
        )
    );
}

#[test]
fn method_expr_with_complex_name() {
    let input = InputSpan::new("x.{Num.cmp}(y);");
    let (_, call) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    assert_eq!(*call.fragment(), "x.{Num.cmp}(y)");

    let Expr::Method {
        name,
        receiver,
        separator,
        args,
    } = &call.extra
    else {
        panic!("Unexpected expr: {:#?}", call.extra);
    };
    assert_eq!(*receiver.as_ref(), sp(0, "x", Expr::Variable));
    assert_eq!(*separator, sp(1, ".", ()));
    assert_eq!(args.as_slice(), [sp(12, "y", Expr::Variable)]);
    let Expr::Block(name) = &name.extra else {
        panic!("Unexpected name expr: {:#?}", name.extra);
    };
    assert!(name.statements.is_empty());
    assert_matches!(
        name.return_value.as_deref().unwrap().extra,
        Expr::FieldAccess { .. }
    );

    let input = InputSpan::new("x.{method(x)}(y);");
    let (_, call) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    assert_eq!(*call.fragment(), "x.{method(x)}(y)");

    let Expr::Method { name, .. } = call.extra else {
        panic!("Unexpected expr: {:#?}", call.extra);
    };
    let Expr::Block(Block {
        statements,
        return_value,
    }) = name.extra
    else {
        panic!("Unexpected name expr: {:#?}", name.extra);
    };
    assert!(statements.is_empty());
    assert_eq!(
        *return_value.unwrap(),
        sp(
            3,
            "method(x)",
            Expr::Function {
                name: Box::new(sp(3, "method", Expr::Variable)),
                args: vec![sp(10, "x", Expr::Variable)],
            }
        )
    );

    let ridiculous_inputs = [
        "x.{ a = 5; method(a) }(5, 123);",
        "x.{ #{ x: 3, y: 4 }.x }(#{ x }, y);",
        "x.{(1, |x| x + 3).1}();",
    ];
    for input in ridiculous_inputs {
        let input = InputSpan::new(input);
        let (_, call) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
        assert_eq!(*call.fragment(), input.strip_suffix(';').unwrap());
        assert_matches!(call.extra, Expr::Method { .. });
    }
}

#[test]
fn method_expr_with_complex_name_in_chain() {
    let inputs = [
        "(1 + 2).{Num.cmp}(y);",
        "#{ x: 3, y: 4}.{Point.normalize}();",
        "test.field.{access}();",
        "test.field.{method.access}();",
        "test.field.{method.access}(and).{other.access}();",
    ];

    for input in inputs {
        let input = InputSpan::new(input);
        let (_, call) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
        assert_eq!(*call.fragment(), input.strip_suffix(';').unwrap());
        assert_matches!(call.extra, Expr::Method { .. });
    }
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
    let NomErr::Failure(spanned_err) = err else {
        panic!("Unexpected error: {err}");
    };
    assert_matches!(spanned_err.kind(), ErrorKind::LiteralName);
}
