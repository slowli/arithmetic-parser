use assert_matches::assert_matches;
use nom::{
    bytes::complete::{escaped_transform, is_not},
    combinator::map_res,
    multi::fold_many1,
};

use core::fmt;

use super::*;
use crate::{alloc::String, grammars::F32Grammar, Features, Op};

#[derive(Debug, Clone, Copy, PartialEq)]
enum LiteralType {
    /// Literal is a generic buffer.
    Bytes,
    /// Literal is a group scalar.
    Scalar,
    /// Literal is a group element.
    Element,
}

#[derive(Debug, Clone, PartialEq)]
enum Literal {
    Number,
    Bytes { value: Vec<u8>, ty: LiteralType },
}

impl Literal {
    /// Parses an ASCII string like `"Hello, world!"`.
    fn string(input: InputSpan<'_>) -> NomResult<'_, String> {
        let parser = escaped_transform(
            is_not("\\\"\n"),
            '\\',
            alt((
                map(tag_char('\\'), |_| "\\"),
                map(tag_char('"'), |_| "\""),
                map(tag_char('n'), |_| "\n"),
            )),
        );
        map(
            preceded(tag_char('"'), cut(terminated(opt(parser), tag_char('"')))),
            Option::unwrap_or_default,
        )(input)
    }

    /// Hex-encoded buffer like `0x09abcd`.
    fn hex_buffer(input: InputSpan<'_>) -> NomResult<'_, Self> {
        let hex_parser = preceded(
            tag("0x"),
            cut(tuple((
                opt(alt((
                    map(tag_char('s'), |_| LiteralType::Scalar),
                    map(tag_char('S'), |_| LiteralType::Scalar),
                    map(tag_char('g'), |_| LiteralType::Element),
                    map(tag_char('G'), |_| LiteralType::Element),
                ))),
                fold_many1(
                    map_res(
                        preceded(
                            opt(tag_char('_')),
                            take_while1(|c: char| c.is_ascii_hexdigit()),
                        ),
                        |digits: InputSpan| {
                            hex::decode(digits.fragment()).map_err(anyhow::Error::from)
                        },
                    ),
                    vec![],
                    |mut acc, digits| {
                        acc.extend_from_slice(&digits);
                        acc
                    },
                ),
            ))),
        );

        map(hex_parser, |(maybe_ty, value)| Literal::Bytes {
            ty: maybe_ty.unwrap_or(LiteralType::Bytes),
            value,
        })(input)
    }

    fn parse(input: InputSpan<'_>) -> NomResult<'_, Self> {
        alt((
            map(Self::string, |s| Literal::Bytes {
                value: s.into_bytes(),
                ty: LiteralType::Bytes,
            }),
            Self::hex_buffer,
            // Numbers must be parsed after hex buffers because they share the same prefix `0`.
            map(take_while1(|c: char| c.is_ascii_digit()), |_| {
                Literal::Number
            }),
        ))(input)
    }
}

/// Possible value type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueType {
    /// Any type.
    Any,
    /// Boolean.
    Bool,
    /// Group scalar.
    Scalar,
    /// Group element.
    Element,
    /// Byte buffer.
    Bytes,
    /// Tuple.
    Tuple(Vec<ValueType>),
}

fn type_info<Ty: GrammarType>(input: InputSpan<'_>) -> NomResult<'_, ValueType> {
    alt((
        map(tag_char('_'), |_| ValueType::Any),
        map(tag("Sc"), |_| ValueType::Scalar),
        map(tag("Ge"), |_| ValueType::Element),
        map(tag("bool"), |_| ValueType::Bool),
        map(tag("bytes"), |_| ValueType::Bytes),
        map(
            delimited(
                terminated(tag_char('('), ws::<Ty>),
                separated_list(
                    delimited(ws::<Ty>, tag_char(','), ws::<Ty>),
                    type_info::<Ty>,
                ),
                preceded(ws::<Ty>, tag_char(')')),
            ),
            ValueType::Tuple,
        ),
    ))(input)
}

#[derive(Debug, Clone)]
struct FieldGrammar;

impl Grammar for FieldGrammar {
    type Lit = Literal;
    type Type = ValueType;

    const FEATURES: Features = Features::all();

    fn parse_literal(span: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
        Literal::parse(span)
    }

    fn parse_type(span: InputSpan<'_>) -> NomResult<'_, Self::Type> {
        type_info::<Streaming>(span)
    }
}

fn span(offset: usize, fragment: &str) -> InputSpan<'_> {
    span_on_line(offset, 1, fragment)
}

fn span_on_line(offset: usize, line: u32, fragment: &str) -> InputSpan<'_> {
    unsafe {
        // SAFETY: `offset` is small (hand-picked).
        InputSpan::new_from_raw_offset(offset, line, fragment, ())
    }
}

fn sp<T>(offset: usize, fragment: &str, value: T) -> Spanned<'_, T> {
    Spanned::new(span(offset, fragment), value)
}

fn lsp<'a>(
    offset: usize,
    fragment: &'a str,
    lvalue: Lvalue<'a, ValueType>,
) -> SpannedLvalue<'a, ValueType> {
    Spanned::new(span(offset, fragment), lvalue)
}

fn lvalue_tuple(elements: Vec<SpannedLvalue<'_, ValueType>>) -> Lvalue<'_, ValueType> {
    Lvalue::Tuple(Destructure {
        start: elements,
        middle: None,
        end: vec![],
    })
}

fn args<'a>(
    span: InputSpan<'a>,
    elements: Vec<SpannedLvalue<'a, ValueType>>,
) -> Spanned<'a, Destructure<'a, ValueType>> {
    Spanned::new(
        span,
        Destructure {
            start: elements,
            middle: None,
            end: vec![],
        },
    )
}

#[test]
fn is_valid_variable_name_works() {
    for &valid_name in &[
        "a",
        "abc",
        "abc_",
        "camelCase",
        "_dash_",
        "_",
        "a12",
        "e1e3t_hax0r",
    ] {
        assert!(
            is_valid_variable_name(valid_name),
            "failed at valid name: {}",
            valid_name
        );
    }

    for &invalid_name in &["", "1abc", "\u{43d}\u{435}\u{442}", "xy+", "a-b"] {
        assert!(
            !is_valid_variable_name(invalid_name),
            "failed at invalid name: {}",
            invalid_name
        );
    }
}

#[test]
fn whitespace_can_include_comments() {
    let input = InputSpan::new("ge(1)");
    assert_eq!(ws::<Complete>(input).unwrap().0, span(0, "ge(1)"));

    let input = InputSpan::new("   ge(1)");
    assert_eq!(ws::<Complete>(input).unwrap().0, span(3, "ge(1)"));

    let input = InputSpan::new("  \nge(1)");
    assert_eq!(
        ws::<Complete>(input).unwrap().0,
        span_on_line(3, 2, "ge(1)")
    );
    let input = InputSpan::new("// Comment\nge(1)");
    assert_eq!(
        ws::<Complete>(input).unwrap().0,
        span_on_line(11, 2, "ge(1)")
    );
    let input = InputSpan::new("//!\nge(1)");
    assert_eq!(
        ws::<Complete>(input).unwrap().0,
        span_on_line(4, 2, "ge(1)")
    );

    let input = InputSpan::new(
        "   // This is a comment.
             \t// This is a comment, too
             this_is_not // although this *is*",
    );
    assert_eq!(
        ws::<Complete>(input).unwrap().0,
        span_on_line(78, 3, "this_is_not // although this *is*")
    );
}

#[test]
fn non_ascii_input() {
    let input = InputSpan::new("\u{444}\u{44b}\u{432}\u{430}");
    let err = statements::<FieldGrammar>(input).unwrap_err();
    assert_matches!(err.kind(), ErrorKind::NonAsciiInput);

    let input = InputSpan::new("1 + \u{444}\u{44b}");
    let err = statements::<FieldGrammar>(input).unwrap_err();
    assert_matches!(err.kind(), ErrorKind::NonAsciiInput);
}

#[test]
fn hex_buffer_works() {
    let input = InputSpan::new("0xAbcd1234 + 5");
    assert_eq!(
        Literal::hex_buffer(input).unwrap().1,
        Literal::Bytes {
            value: vec![0xab, 0xcd, 0x12, 0x34],
            ty: LiteralType::Bytes,
        }
    );

    let input = InputSpan::new("0xg_Abcd_1234 + 5");
    assert_eq!(
        Literal::hex_buffer(input).unwrap().1,
        Literal::Bytes {
            value: vec![0xab, 0xcd, 0x12, 0x34],
            ty: LiteralType::Element,
        }
    );

    let erroneous_inputs = ["0xAbcd1234a", "0x", "0xP12", "0x__12", "0x_s12", "0xsA_BCD"];
    for &input in &erroneous_inputs {
        let input = InputSpan::new(input);
        assert_matches!(Literal::hex_buffer(input).unwrap_err(), NomErr::Failure(_));
    }
}

#[test]
fn string_literal_works() {
    let input = InputSpan::new(r#""abc";"#);
    assert_eq!(Literal::string(input).unwrap().1, "abc");
    let input = InputSpan::new(r#""Hello, \"world\"!";"#);
    assert_eq!(Literal::string(input).unwrap().1, r#"Hello, "world"!"#);
    let input = InputSpan::new(r#""Hello,\nworld!";"#);
    assert_eq!(Literal::string(input).unwrap().1, "Hello,\nworld!");
    let input = InputSpan::new(r#""";"#);
    assert_eq!(Literal::string(input).unwrap().1, "");

    // Unfinished string literal.
    let input = InputSpan::new("\"Hello, world!\n");
    assert_matches!(Literal::string(input).unwrap_err(), NomErr::Failure(_));
    // Unsupported escape sequence.
    let input = InputSpan::new(r#""Hello,\tworld!"#);
    assert_matches!(Literal::string(input).unwrap_err(), NomErr::Failure(_));
}

#[test]
fn var_name_works() {
    let input = InputSpan::new("A + B");
    assert_eq!(var_name(input).unwrap().1, span(0, "A"));
    let input = InputSpan::new("Abc_d + B");
    assert_eq!(var_name(input).unwrap().1, span(0, "Abc_d"));
    let input = InputSpan::new("_ + 3");
    assert_eq!(var_name(input).unwrap().1, span(0, "_"));
}

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
                        args: vec![],
                    }
                )],
            }
        )
    );
}

#[test]
fn element_expr_works() {
    fn simple_fn(offset: usize) -> Expr<'static, FieldGrammar> {
        Expr::Function {
            name: Box::new(sp(offset, "ge", Expr::Variable)),
            args: vec![sp(
                offset + 3,
                "0x1234",
                Expr::Literal(Literal::Bytes {
                    value: vec![0x12, 0x34],
                    ty: LiteralType::Bytes,
                }),
            )],
        }
    }

    let input = InputSpan::new("A;");
    assert_eq!(
        expr::<FieldGrammar, Complete>(input).unwrap().1,
        sp(0, "A", Expr::Variable)
    );

    let input = InputSpan::new("(ge(0x1234));");
    assert_eq!(
        expr::<FieldGrammar, Complete>(input).unwrap().1.extra,
        simple_fn(1)
    );

    let input = InputSpan::new("ge(0x1234) + A_b;");
    let sum_expr = Expr::Binary {
        lhs: Box::new(sp(0, "ge(0x1234)", simple_fn(0))),
        op: Spanned::new(span(11, "+"), BinaryOp::Add),
        rhs: Box::new(sp(13, "A_b", Expr::Variable)),
    };
    assert_eq!(
        expr::<FieldGrammar, Complete>(input).unwrap().1.extra,
        sum_expr
    );

    let input = InputSpan::new("ge(0x1234) + A_b - C;");
    assert_eq!(
        expr::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Expr::Binary {
            lhs: Box::new(sp(0, "ge(0x1234) + A_b", sum_expr)),
            op: Spanned::new(span(17, "-"), BinaryOp::Sub),
            rhs: Box::new(sp(19, "C", Expr::Variable)),
        }
    );

    let input = InputSpan::new("(ge(0x1234) + A_b) - (C + ge(0x00) + D);");
    assert_eq!(
        expr::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Expr::Binary {
            lhs: Box::new(sp(0, "(ge(0x1234) + A_b)", {
                Expr::Binary {
                    lhs: Box::new(sp(1, "ge(0x1234)", simple_fn(1))),
                    op: BinaryOp::from_span(span(12, "+")),
                    rhs: Box::new(sp(14, "A_b", Expr::Variable)),
                }
            })),
            op: BinaryOp::from_span(span(19, "-")),
            rhs: Box::new(sp(21, "(C + ge(0x00) + D)", {
                Expr::Binary {
                    lhs: Box::new(sp(
                        22,
                        "C + ge(0x00)",
                        Expr::Binary {
                            lhs: Box::new(sp(22, "C", Expr::Variable)),
                            op: BinaryOp::from_span(span(24, "+")),
                            rhs: Box::new(sp(
                                26,
                                "ge(0x00)",
                                Expr::Function {
                                    name: Box::new(sp(26, "ge", Expr::Variable)),
                                    args: vec![sp(
                                        29,
                                        "0x00",
                                        Expr::Literal(Literal::Bytes {
                                            value: vec![0],
                                            ty: LiteralType::Bytes,
                                        }),
                                    )],
                                },
                            )),
                        },
                    )),
                    op: BinaryOp::from_span(span(35, "+")),
                    rhs: Box::new(sp(37, "D", Expr::Variable)),
                }
            }))
        }
    );
}

#[test]
fn unary_expr_works() {
    let input = InputSpan::new("-3;");
    assert_eq!(
        expr::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Expr::Unary {
            op: Spanned::new(span(0, "-"), UnaryOp::Neg),
            inner: Box::new(sp(1, "3", Expr::Literal(Literal::Number))),
        }
    );

    let input = InputSpan::new("-x + 5;");
    assert_eq!(
        *expr::<FieldGrammar, Complete>(input)
            .unwrap()
            .1
            .extra
            .binary_lhs()
            .unwrap(),
        sp(
            0,
            "-x",
            Expr::Unary {
                op: Spanned::new(span(0, "-"), UnaryOp::Neg),
                inner: Box::new(sp(1, "x", Expr::Variable)),
            }
        ),
    );

    let input = InputSpan::new("-(x + y);");
    assert_eq!(
        expr::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Expr::Unary {
            op: Spanned::new(span(0, "-"), UnaryOp::Neg),
            inner: Box::new(sp(
                1,
                "(x + y)",
                Expr::Binary {
                    lhs: Box::new(sp(2, "x", Expr::Variable)),
                    op: BinaryOp::from_span(span(4, "+")),
                    rhs: Box::new(sp(6, "y", Expr::Variable)),
                }
            ))
        }
    );

    let input = InputSpan::new("2 * -3;");
    assert_eq!(
        expr::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Expr::Binary {
            lhs: Box::new(sp(0, "2", Expr::Literal(Literal::Number))),
            op: BinaryOp::from_span(span(2, "*")),
            rhs: Box::new(sp(
                4,
                "-3",
                Expr::Unary {
                    op: Spanned::new(span(4, "-"), UnaryOp::Neg),
                    inner: Box::new(sp(5, "3", Expr::Literal(Literal::Number))),
                }
            )),
        }
    );

    let input = InputSpan::new("!f && x == 2;");
    assert_eq!(
        *expr::<FieldGrammar, Complete>(input)
            .unwrap()
            .1
            .extra
            .binary_lhs()
            .unwrap()
            .fragment(),
        "!f"
    );
}

#[test]
fn expr_with_numbers_works() {
    let input = InputSpan::new("(2 + a) * b;");
    assert_eq!(
        expr::<FieldGrammar, Streaming>(input).unwrap().1.extra,
        Expr::Binary {
            lhs: Box::new(sp(0, "(2 + a)", {
                Expr::Binary {
                    lhs: Box::new(sp(1, "2", Expr::Literal(Literal::Number))),
                    op: BinaryOp::from_span(span(3, "+")),
                    rhs: Box::new(sp(5, "a", Expr::Variable)),
                }
            })),
            op: BinaryOp::from_span(span(8, "*")),
            rhs: Box::new(sp(10, "b", Expr::Variable)),
        }
    );
}

#[test]
fn comparison_expr_works() {
    let input = InputSpan::new("a == b && c > d;");
    assert_eq!(
        expr::<FieldGrammar, Streaming>(input).unwrap().1.extra,
        Expr::Binary {
            lhs: Box::new(sp(0, "a == b", {
                Expr::Binary {
                    lhs: Box::new(sp(0, "a", Expr::Variable)),
                    op: BinaryOp::from_span(span(3, "==")),
                    rhs: Box::new(sp(5, "b", Expr::Variable)),
                }
            })),
            op: BinaryOp::from_span(span(7, "&&")),
            rhs: Box::new(sp(10, "c > d", {
                Expr::Binary {
                    lhs: Box::new(sp(10, "c", Expr::Variable)),
                    op: BinaryOp::from_span(span(12, ">")),
                    rhs: Box::new(sp(14, "d", Expr::Variable)),
                }
            })),
        }
    );
}

#[test]
fn two_char_comparisons_are_parsed() {
    let input = InputSpan::new("a >= b;");
    assert_eq!(
        expr::<FieldGrammar, Streaming>(input).unwrap().1.extra,
        Expr::Binary {
            lhs: Box::new(sp(0, "a", Expr::Variable)),
            op: BinaryOp::from_span(span(2, ">=")),
            rhs: Box::new(sp(5, "b", Expr::Variable)),
        }
    );
}

#[test]
fn assignment_works() {
    let input = InputSpan::new("x = sc(0x1234);");
    assert_eq!(
        statement::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Statement::Assignment {
            lhs: lsp(0, "x", Lvalue::Variable { ty: None }),
            rhs: Box::new(sp(
                4,
                "sc(0x1234)",
                Expr::Function {
                    name: Box::new(sp(4, "sc", Expr::Variable)),
                    args: vec![sp(
                        7,
                        "0x1234",
                        Expr::Literal(Literal::Bytes {
                            value: vec![0x12, 0x34],
                            ty: LiteralType::Bytes,
                        })
                    )],
                }
            ))
        }
    );

    let input = InputSpan::new("yb = 7 * sc(0x0001) + k;");
    assert_eq!(
        statement::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Statement::Assignment {
            lhs: lsp(0, "yb", Lvalue::Variable { ty: None }),
            rhs: Box::new(sp(5, "7 * sc(0x0001) + k", {
                Expr::Binary {
                    lhs: Box::new(sp(
                        5,
                        "7 * sc(0x0001)",
                        Expr::Binary {
                            lhs: Box::new(sp(5, "7", Expr::Literal(Literal::Number))),
                            op: BinaryOp::from_span(span(7, "*")),
                            rhs: Box::new(sp(
                                9,
                                "sc(0x0001)",
                                Expr::Function {
                                    name: Box::new(sp(9, "sc", Expr::Variable)),
                                    args: vec![sp(
                                        12,
                                        "0x0001",
                                        Expr::Literal(Literal::Bytes {
                                            value: vec![0, 1],
                                            ty: LiteralType::Bytes,
                                        }),
                                    )],
                                },
                            )),
                        },
                    )),
                    op: BinaryOp::from_span(span(20, "+")),
                    rhs: Box::new(sp(22, "k", Expr::Variable)),
                }
            }))
        }
    );
}

#[test]
fn comparison_works() {
    let input = InputSpan::new("x == 3;");
    assert_eq!(
        statement::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Statement::Expr(sp(
            0,
            "x == 3",
            Expr::Binary {
                lhs: Box::new(sp(0, "x", Expr::Variable)),
                rhs: Box::new(sp(5, "3", Expr::Literal(Literal::Number))),
                op: BinaryOp::from_span(span(2, "==")),
            }
        ))
    );

    let input = InputSpan::new("s*G == R + h*A;");
    assert_eq!(
        statement::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Statement::Expr(sp(
            0,
            "s*G == R + h*A",
            Expr::Binary {
                lhs: Box::new(sp(0, "s*G", {
                    Expr::Binary {
                        lhs: Box::new(sp(0, "s", Expr::Variable)),
                        op: BinaryOp::from_span(span(1, "*")),
                        rhs: Box::new(sp(2, "G", Expr::Variable)),
                    }
                })),
                rhs: Box::new(sp(7, "R + h*A", {
                    Expr::Binary {
                        lhs: Box::new(sp(7, "R", Expr::Variable)),
                        op: BinaryOp::from_span(span(9, "+")),
                        rhs: Box::new(sp(
                            11,
                            "h*A",
                            Expr::Binary {
                                lhs: Box::new(sp(11, "h", Expr::Variable)),
                                op: BinaryOp::from_span(span(12, "*")),
                                rhs: Box::new(sp(13, "A", Expr::Variable)),
                            },
                        )),
                    }
                })),
                op: BinaryOp::from_span(span(4, "==")),
            }
        ))
    );

    let input = InputSpan::new("G^s != R + A^h;");
    assert_eq!(
        statement::<FieldGrammar, Complete>(input).unwrap().1.extra,
        Statement::Expr(sp(
            0,
            "G^s != R + A^h",
            Expr::Binary {
                lhs: Box::new(sp(
                    0,
                    "G^s",
                    Expr::Binary {
                        lhs: Box::new(sp(0, "G", Expr::Variable)),
                        op: BinaryOp::from_span(span(1, "^")),
                        rhs: Box::new(sp(2, "s", Expr::Variable)),
                    }
                )),
                rhs: Box::new(sp(
                    7,
                    "R + A^h",
                    Expr::Binary {
                        lhs: Box::new(sp(7, "R", Expr::Variable)),
                        op: BinaryOp::from_span(span(9, "+")),
                        rhs: Box::new(sp(
                            11,
                            "A^h",
                            Expr::Binary {
                                lhs: Box::new(sp(11, "A", Expr::Variable)),
                                op: BinaryOp::from_span(span(12, "^")),
                                rhs: Box::new(sp(13, "h", Expr::Variable)),
                            },
                        )),
                    }
                )),
                op: BinaryOp::from_span(span(4, "!=")),
            }
        ))
    );
}

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

    let input = InputSpan::new("..., end)");
    assert_eq!(
        destructure::<FieldGrammar, Complete>(input).unwrap().1,
        Destructure {
            start: vec![],
            middle: Some(Spanned::new(span(0, "..."), DestructureRest::Unnamed)),
            end: vec![lsp(5, "end", Lvalue::Variable { ty: None })],
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
fn block_parsing() {
    let input = InputSpan::new("{ x + y }");
    assert_eq!(
        block::<FieldGrammar, Complete>(input).unwrap().1,
        Block {
            statements: vec![],
            return_value: Some(Box::new(sp(
                2,
                "x + y",
                Expr::Binary {
                    lhs: Box::new(sp(2, "x", Expr::Variable)),
                    op: BinaryOp::from_span(span(4, "+")),
                    rhs: Box::new(sp(6, "y", Expr::Variable)),
                }
            ))),
        }
    );

    let input = InputSpan::new("{ x = 1 + 2; x * 3 }");
    let parsed = block::<FieldGrammar, Complete>(input).unwrap().1;
    assert_eq!(parsed.statements.len(), 1);
    let return_value = parsed.return_value.unwrap();
    assert_eq!(*return_value.fragment(), "x * 3");

    let input = InputSpan::new("{ x = 1 + 2; x * 3; }");
    let parsed = block::<FieldGrammar, Complete>(input).unwrap().1;
    assert_eq!(parsed.statements.len(), 2);
    assert!(parsed.return_value.is_none());
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
fn incomplete_fn() {
    let input = InputSpan::new("sc(1,");
    assert_matches!(
        simple_expr::<FieldGrammar, Streaming>(input).unwrap_err(),
        NomErr::Incomplete(_)
    );
}

#[test]
fn incomplete_expr() {
    const SNIPPETS: &[&str] = &[
        "1 +",
        "2 + 3*",
        "2 * sc(1,",
        "sc(x) +",
        "(",
        "(x, ",
        "(x, 3 +",
        "(x, 3) + ",
    ];
    for snippet in SNIPPETS {
        let input = InputSpan::new(snippet);
        assert_matches!(
            expr::<FieldGrammar, Streaming>(input).unwrap_err(),
            NomErr::Incomplete(_)
        );
    }
}

#[test]
fn incomplete_statement() {
    const SNIPPETS: &[&str] = &[
        "x ==",
        "x =",
        "(",
        "(x, y",
        "(x, y) =",
        "(\nx: Ge,",
        "x = 2 +",
        "x == 2 +",
    ];
    for snippet in SNIPPETS {
        let input = InputSpan::new(snippet);
        assert_matches!(
            statement::<FieldGrammar, Streaming>(input).unwrap_err(),
            NomErr::Incomplete(_)
        );
    }
}

#[test]
fn separated_statements_parse() {
    let input = InputSpan::new("x = 1 + 2; x");
    let block = separated_statements::<FieldGrammar, Complete>(input)
        .unwrap()
        .1;
    assert_eq!(block.statements.len(), 1);
    assert_eq!(*block.return_value.unwrap(), sp(11, "x", Expr::Variable));

    let input = InputSpan::new("foo = |x| { 2*x }; foo(3)");
    let block = separated_statements::<FieldGrammar, Complete>(input)
        .unwrap()
        .1;
    assert_eq!(block.statements.len(), 1);
    assert_eq!(*block.return_value.unwrap().fragment(), "foo(3)");

    let input = InputSpan::new("{ x = 2; }; foo(3)");
    let block = separated_statements::<FieldGrammar, Complete>(input)
        .unwrap()
        .1;
    assert_eq!(block.statements.len(), 1);
    assert_eq!(*block.return_value.unwrap().fragment(), "foo(3)");

    let input = InputSpan::new("y = { x = 2; x + 3 }; foo(y)");
    let block = separated_statements::<FieldGrammar, Complete>(input)
        .unwrap()
        .1;
    assert_eq!(block.statements.len(), 1);
    assert_eq!(*block.return_value.unwrap().fragment(), "foo(y)");
}

#[test]
fn type_hints_when_switched_off() {
    #[derive(Debug, Clone)]
    struct SimpleGrammar;

    impl Grammar for SimpleGrammar {
        type Lit = Literal;
        type Type = ValueType;

        const FEATURES: Features = Features {
            type_annotations: false,
            ..Features::all()
        };

        fn parse_literal(span: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
            Literal::parse(span)
        }

        fn parse_type(span: InputSpan<'_>) -> NomResult<'_, Self::Type> {
            type_info::<Complete>(span)
        }
    }

    // Check that expressions without type hints are parsed fine.
    let input = InputSpan::new("x = 1 + y");
    let (_, stmt) = statement::<SimpleGrammar, Complete>(input).unwrap();
    assert_eq!(
        stmt.extra,
        Statement::Assignment {
            lhs: lsp(0, "x", Lvalue::Variable { ty: None }),
            rhs: Box::new(sp(
                4,
                "1 + y",
                Expr::Binary {
                    lhs: Box::new(sp(4, "1", Expr::Literal(Literal::Number))),
                    rhs: Box::new(sp(8, "y", Expr::Variable)),
                    op: BinaryOp::from_span(span(6, "+")),
                }
            )),
        }
    );

    let input = InputSpan::new("x: Sc = 1 + 2");
    let (rem, _) = statement::<SimpleGrammar, Complete>(input).unwrap();
    assert!(rem.fragment().starts_with(": Sc"));

    let input = InputSpan::new("(x, y) = (1 + 2, 3 + 5)");
    let (rem, _) = statement::<SimpleGrammar, Complete>(input).unwrap();
    assert_eq!(*rem.fragment(), "");

    let input = InputSpan::new("(x, y: Sc) = (1 + 2, 3 + 5)");
    let err = statement::<SimpleGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Failure(ref spanned) if spanned.span().location_offset() == 5);

    let input = InputSpan::new("duplicate = |x| { x * (1, 2) }");
    let (rem, _) = statement::<SimpleGrammar, Complete>(input).unwrap();
    assert_eq!(*rem.fragment(), "");

    let input = InputSpan::new("duplicate = |x: Sc| { x * (1, 2) }");
    let err = statement::<SimpleGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Failure(ref spanned) if *spanned.span().fragment() == ":");
}

#[test]
fn fn_defs_when_switched_off() {
    #[derive(Debug, Clone)]
    struct SimpleGrammar;

    impl Grammar for SimpleGrammar {
        type Lit = Literal;
        type Type = ValueType;

        const FEATURES: Features = Features {
            fn_definitions: false,
            ..Features::all()
        };

        fn parse_literal(span: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
            Literal::parse(span)
        }

        fn parse_type(span: InputSpan<'_>) -> NomResult<'_, Self::Type> {
            type_info::<Complete>(span)
        }
    }

    let input = InputSpan::new("foo = |x| { x + 3 }");
    let err = statement::<SimpleGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Error(ref spanned) if *spanned.span().fragment() == "|");
}

#[test]
fn tuples_when_switched_off() {
    #[derive(Debug, Clone)]
    struct SimpleGrammar;

    impl Grammar for SimpleGrammar {
        type Lit = Literal;
        type Type = ValueType;

        const FEATURES: Features = Features {
            tuples: false,
            ..Features::all()
        };

        fn parse_literal(span: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
            Literal::parse(span)
        }

        fn parse_type(span: InputSpan<'_>) -> NomResult<'_, Self::Type> {
            type_info::<Complete>(span)
        }
    }

    let input = InputSpan::new("tup = (1 + 2, 3 + 5)");
    let err = statement::<SimpleGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Failure(ref spanned) if spanned.span().location_offset() == 6);

    let input = InputSpan::new("(x, y) = (1 + 2, 3 + 5)");
    let err = statement::<SimpleGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Failure(ref spanned) if spanned.span().location_offset() == 0);
}

#[test]
fn blocks_when_switched_off() {
    #[derive(Debug, Clone)]
    struct SimpleGrammar;

    impl Grammar for SimpleGrammar {
        type Lit = Literal;
        type Type = ValueType;

        const FEATURES: Features = Features {
            blocks: false,
            ..Features::all()
        };

        fn parse_literal(span: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
            Literal::parse(span)
        }

        fn parse_type(span: InputSpan<'_>) -> NomResult<'_, Self::Type> {
            type_info::<Complete>(span)
        }
    }

    let input = InputSpan::new("x = { y = 10; y * 2 }");
    let err = statement::<SimpleGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Error(ref spanned) if spanned.span().location_offset() == 4);

    let input = InputSpan::new("foo({ y = 10; y * 2 }, z)");
    let err = statement::<SimpleGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Failure(ref spanned) if spanned.span().location_offset() == 4);
}

#[test]
fn methods_when_switched_off() {
    #[derive(Debug, Clone)]
    struct SimpleGrammar;

    impl Grammar for SimpleGrammar {
        type Lit = Literal;
        type Type = ValueType;

        const FEATURES: Features = Features {
            methods: false,
            ..Features::all()
        };

        fn parse_literal(span: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
            Literal::parse(span)
        }

        fn parse_type(span: InputSpan<'_>) -> NomResult<'_, Self::Type> {
            type_info::<Complete>(span)
        }
    }

    let input = InputSpan::new("foo.bar(1)");
    let (rest, _) = statement::<SimpleGrammar, Complete>(input).unwrap();
    assert_eq!(*rest.fragment(), ".bar(1)");

    let input = InputSpan::new("(1, 2).bar(1)");
    let (rest, _) = statement::<SimpleGrammar, Complete>(input).unwrap();
    assert_eq!(*rest.fragment(), ".bar(1)");
}

#[test]
fn order_comparisons_when_switched_off() {
    #[derive(Debug, Clone)]
    struct SimpleGrammar;

    impl Grammar for SimpleGrammar {
        type Lit = Literal;
        type Type = ValueType;

        const FEATURES: Features = Features {
            boolean_ops: BooleanOps::Basic,
            ..Features::all()
        };

        fn parse_literal(span: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
            Literal::parse(span)
        }

        fn parse_type(span: InputSpan<'_>) -> NomResult<'_, Self::Type> {
            type_info::<Complete>(span)
        }
    }

    for &op in &[BinaryOp::Gt, BinaryOp::Lt, BinaryOp::Ge, BinaryOp::Le] {
        assert_binary_op_is_not_parsed::<SimpleGrammar>(op);
    }
}

fn assert_binary_op_is_not_parsed<T>(op: BinaryOp)
where
    T: Grammar + fmt::Debug,
{
    let input = format!("x {} 1;", op.as_str());
    let input = InputSpan::new(&input);
    let err = statement::<T, Complete>(input).unwrap_err();
    let spanned_err = match err {
        NomErr::Failure(spanned) => spanned,
        _ => panic!("Unexpected error: {}", err),
    };
    assert_eq!(spanned_err.span().location_offset(), 2);
    assert_eq!(*spanned_err.span().fragment(), op.as_str());
    assert_matches!(
        *spanned_err.kind(),
        ErrorKind::UnsupportedOp(Op::Binary(real_op)) if real_op == op
    );
}

#[test]
fn boolean_ops_when_switched_off() {
    #[derive(Debug, Clone)]
    struct SimpleGrammar;

    impl Grammar for SimpleGrammar {
        type Lit = Literal;
        type Type = ValueType;

        const FEATURES: Features = Features {
            boolean_ops: BooleanOps::None,
            ..Features::all()
        };

        fn parse_literal(span: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
            Literal::parse(span)
        }

        fn parse_type(span: InputSpan<'_>) -> NomResult<'_, Self::Type> {
            type_info::<Complete>(span)
        }
    }

    for &op in &[
        BinaryOp::Eq,
        BinaryOp::NotEq,
        BinaryOp::And,
        BinaryOp::Or,
        BinaryOp::Gt,
        BinaryOp::Le,
    ] {
        assert_binary_op_is_not_parsed::<SimpleGrammar>(op);
    }
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

#[test]
fn methods_have_higher_priority_than_unary_ops() {
    const INPUTS: &[&str] = &["-5.abs();", "-5.5.sin();", "-5.25.foo(1, PI)(1, x).cos();"];

    for &input in INPUTS {
        let input = InputSpan::new(input);
        let expr = expr::<F32Grammar, Complete>(input).unwrap().1.extra;
        assert_matches!(
            expr,
            Expr::Unary { op, .. } if op.extra == UnaryOp::Neg,
            "Errored input: {}",
            input.fragment()
        );
    }
}

#[test]
fn no_unary_op_if_literal_without_it_not_parsed() {
    #[derive(Debug)]
    struct ExclamationGrammar;

    impl Grammar for ExclamationGrammar {
        type Lit = String;
        type Type = ();

        const FEATURES: Features = Features {
            type_annotations: false,
            ..Features::all()
        };

        fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
            map(
                delimited(
                    tag_char('!'),
                    take_while1(|c: char| c != '!'),
                    tag_char('!'),
                ),
                |s: InputSpan<'_>| (*s.fragment()).to_owned(),
            )(input)
        }

        fn parse_type(_: InputSpan<'_>) -> NomResult<'_, Self::Type> {
            unreachable!("Never called per `FEATURES`")
        }
    }

    let input = InputSpan::new("!foo!.bar();");
    let expr = expr::<ExclamationGrammar, Complete>(input).unwrap().1.extra;
    match expr {
        Expr::Method { name, receiver, .. } => {
            assert_eq!(*name.fragment(), "bar");
            assert_matches!(receiver.extra, Expr::Literal(s) if s == "foo");
        }
        _ => panic!("Unexpected expr: {:?}", expr),
    };
}

#[test]
fn methods_and_multiple_unary_ops() {
    let input = InputSpan::new("--1.abs();");
    let expr = expr::<F32Grammar, Complete>(input).unwrap().1;
    assert_eq!(*expr.fragment(), "--1.abs()");
    let inner_expr = match expr.extra {
        Expr::Unary { inner, .. } => inner,
        _ => panic!("Unexpected expr: {:?}", expr),
    };
    assert_eq!(*inner_expr.fragment(), "-1.abs()");
    assert_matches!(inner_expr.extra, Expr::Unary { .. });
}
