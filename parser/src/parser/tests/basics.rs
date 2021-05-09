//! Tests for basic parsers.

use assert_matches::assert_matches;
use nom::Err as NomErr;

use super::{sp, span, span_on_line, FieldGrammar, Literal, LiteralType};
use crate::{
    parser::{expr, is_valid_variable_name, statements, var_name, ws, Complete, Streaming},
    ErrorKind, Expr, InputSpan,
};

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
fn multiline_comments() {
    let input = InputSpan::new("/* !! */ foo");
    assert_eq!(ws::<Complete>(input).unwrap().0, span(9, "foo"));
    let input = InputSpan::new("/*\nFoo\n*/ foo");
    assert_eq!(ws::<Complete>(input).unwrap().0, span_on_line(10, 3, "foo"));
    let input = InputSpan::new("/* Foo");
    assert!(ws::<Streaming>(input).unwrap_err().is_incomplete());

    let input = InputSpan::new("/* Foo");
    let err = ws::<Complete>(input).unwrap_err();
    let err = match &err {
        NomErr::Failure(err) => err.kind(),
        _ => panic!("Unexpected error: {:?}", err),
    };
    assert_matches!(err, ErrorKind::UnfinishedComment);
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
fn expr_with_inline_comment() {
    let input = InputSpan::new("foo(/* 1, */ x, 2)");
    let expr = expr::<FieldGrammar, Complete>(input).unwrap().1.extra;
    assert_eq!(
        expr,
        Expr::Function {
            name: Box::new(sp(0, "foo", Expr::Variable)),
            args: vec![
                sp(13, "x", Expr::Variable),
                sp(16, "2", Expr::Literal(Literal::Number)),
            ]
        }
    );
}
