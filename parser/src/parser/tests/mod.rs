use assert_matches::assert_matches;
use nom::{
    bytes::complete::{escaped_transform, is_not, take_while1},
    combinator::map_res,
    multi::{fold_many1, separated_list0},
};

use super::{expr::simple_expr, *};
use crate::{
    alloc::String,
    grammars::{Grammar, ParseLiteral, Typed, Untyped},
    spans::Spanned,
    BinaryOp, Destructure, Expr, Lvalue, SpannedLvalue, UnaryOp,
};

mod basics;
mod features;
mod function;
mod object;
mod order;
mod tuple;
mod type_cast;

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
                        |digits: InputSpan<'_>| {
                            hex::decode(digits.fragment()).map_err(ErrorKind::literal)
                        },
                    ),
                    Vec::new,
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
                separated_list0(
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
struct FieldGrammarBase;

impl ParseLiteral for FieldGrammarBase {
    type Lit = Literal;

    fn parse_literal(span: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
        Literal::parse(span)
    }
}

impl Grammar<'_> for FieldGrammarBase {
    type Type = ValueType;

    fn parse_type(span: InputSpan<'_>) -> NomResult<'_, Self::Type> {
        type_info::<Streaming>(span)
    }
}

type FieldGrammar = Typed<FieldGrammarBase>;

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
fn element_expr_works() {
    fn simple_fn(offset: usize) -> Expr<'static, FieldGrammarBase> {
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
fn separated_statements_parsing() {
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
fn no_unary_op_if_literal_without_it_not_parsed() {
    #[derive(Debug)]
    struct ExclamationGrammar;

    impl ParseLiteral for ExclamationGrammar {
        type Lit = String;

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
    }

    let input = InputSpan::new("!foo!.bar();");
    let expr = expr::<Untyped<ExclamationGrammar>, Complete>(input)
        .unwrap()
        .1
        .extra;
    match expr {
        Expr::Method { name, receiver, .. } => {
            assert_eq!(*name.fragment(), "bar");
            assert_matches!(receiver.extra, Expr::Literal(s) if s == "foo");
        }
        _ => panic!("Unexpected expr: {expr:?}"),
    };
}

#[test]
fn field_access_with_indexed_field() {
    let input = InputSpan::new("point.0");
    let (rest, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(
        expr.extra,
        Expr::FieldAccess {
            name: Box::new(sp(6, "0", Expr::Variable)),
            receiver: Box::new(sp(0, "point", Expr::Variable)),
        }
    );

    let input = InputSpan::new("point.122.sin(1)");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();

    let Expr::Method { receiver, .. } = expr.extra else {
        panic!("Unexpected expr: {expr:?}");
    };
    assert_eq!(
        receiver.extra,
        Expr::FieldAccess {
            name: Box::new(sp(6, "122", Expr::Variable)),
            receiver: Box::new(sp(0, "point", Expr::Variable)),
        }
    );
}

#[test]
fn bogus_indexed_field_access() {
    let input = InputSpan::new("1. 0");
    let err = simple_expr::<FieldGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Failure(e) if matches!(e.kind(), ErrorKind::LiteralName));

    let input = InputSpan::new("x.0(1, 2)");
    let err = simple_expr::<FieldGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Failure(e) if matches!(e.kind(), ErrorKind::LiteralName));

    let input = InputSpan::new("1.test");
    let err = simple_expr::<FieldGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Failure(e) if matches!(e.kind(), ErrorKind::LiteralName));
}
