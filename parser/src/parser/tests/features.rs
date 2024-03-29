//! Tests for switched off parser features.

use assert_matches::assert_matches;
use nom::Err as NomErr;

use super::{sp, span, FieldGrammarBase, Literal};
use crate::{
    grammars::{Features, Parse, Untyped},
    parser::{statement, Complete},
    BinaryOp, ErrorKind, Expr, InputSpan, Lvalue, Op, Statement,
};

#[test]
fn type_hints_when_switched_off() {
    type SimpleGrammar = Untyped<FieldGrammarBase>;

    // Check that expressions without type annotations are parsed fine.
    let input = InputSpan::new("x = 1 + y");
    let (_, stmt) = statement::<SimpleGrammar, Complete>(input).unwrap();
    assert_eq!(
        stmt.extra,
        Statement::Assignment {
            lhs: sp(0, "x", Lvalue::Variable { ty: None }),
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
    assert_matches!(err, NomErr::Failure(spanned) if spanned.location().location_offset() == 5);

    let input = InputSpan::new("duplicate = |x| { x * (1, 2) }");
    let (rem, _) = statement::<SimpleGrammar, Complete>(input).unwrap();
    assert_eq!(*rem.fragment(), "");

    let input = InputSpan::new("duplicate = |x: Sc| { x * (1, 2) }");
    let err = statement::<SimpleGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Failure(spanned) if spanned.location().span(&input) == ":");
}

#[test]
fn fn_defs_when_switched_off() {
    #[derive(Debug, Clone)]
    struct SimpleGrammar;

    impl Parse for SimpleGrammar {
        type Base = FieldGrammarBase;

        const FEATURES: Features = Features::all().without(Features::FN_DEFINITIONS);
    }

    let input = InputSpan::new("foo = |x| { x + 3 }");
    let err = statement::<SimpleGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Error(spanned) if spanned.location().span(&input) == "|");
}

#[test]
fn tuples_when_switched_off() {
    #[derive(Debug, Clone)]
    struct SimpleGrammar;

    impl Parse for SimpleGrammar {
        type Base = FieldGrammarBase;

        const FEATURES: Features = Features::all().without(Features::TUPLES);
    }

    let input = InputSpan::new("tup = (1 + 2, 3 + 5)");
    let err = statement::<SimpleGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Failure(spanned) if spanned.location().location_offset() == 6);

    let input = InputSpan::new("(x, y) = (1 + 2, 3 + 5)");
    let err = statement::<SimpleGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Failure(spanned) if spanned.location().location_offset() == 0);

    let input = InputSpan::new("{ x, y } = #{ x: 1, y: 2 }");
    let stmt = statement::<SimpleGrammar, Complete>(input).unwrap().1;
    assert_matches!(
        stmt.extra,
        Statement::Assignment { lhs, .. } if matches!(lhs.extra, Lvalue::Object(_))
    );
}

#[test]
fn blocks_when_switched_off() {
    #[derive(Debug, Clone)]
    struct SimpleGrammar;

    impl Parse for SimpleGrammar {
        type Base = FieldGrammarBase;

        const FEATURES: Features = Features::all().without(Features::BLOCKS);
    }

    let input = InputSpan::new("x = { y = 10; y * 2 }");
    let err = statement::<SimpleGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Error(spanned) if spanned.location().location_offset() == 4);

    let input = InputSpan::new("foo({ y = 10; y * 2 }, z)");
    let err = statement::<SimpleGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Failure(spanned) if spanned.location().location_offset() == 4);
}

#[test]
fn methods_when_switched_off() {
    #[derive(Debug, Clone)]
    struct SimpleGrammar;

    impl Parse for SimpleGrammar {
        type Base = FieldGrammarBase;

        const FEATURES: Features = Features::all().without(Features::METHODS);
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

    impl Parse for SimpleGrammar {
        type Base = FieldGrammarBase;

        const FEATURES: Features = Features::all().without(Features::ORDER_COMPARISONS);
    }

    for &op in &[BinaryOp::Gt, BinaryOp::Lt, BinaryOp::Ge, BinaryOp::Le] {
        assert_binary_op_is_not_parsed::<SimpleGrammar>(op);
    }
}

fn assert_binary_op_is_not_parsed<T>(op: BinaryOp)
where
    T: Parse,
{
    let input = format!("x {} 1;", op.as_str());
    let input = InputSpan::new(&input);
    let err = statement::<T, Complete>(input).map(drop).unwrap_err();
    let NomErr::Failure(spanned_err) = err else {
        panic!("Unexpected error: {err}");
    };
    assert_eq!(spanned_err.location().location_offset(), 2);
    assert_eq!(spanned_err.location().span(&input), op.as_str());
    assert_matches!(
        *spanned_err.kind(),
        ErrorKind::UnsupportedOp(Op::Binary(real_op)) if real_op == op
    );
}

#[test]
fn boolean_ops_when_switched_off() {
    #[derive(Debug, Clone)]
    struct SimpleGrammar;

    impl Parse for SimpleGrammar {
        type Base = FieldGrammarBase;

        const FEATURES: Features = Features::all().without(Features::BOOLEAN_OPS);
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
fn object_expressions_when_switched_off() {
    #[derive(Debug, Clone)]
    struct SimpleGrammar;

    impl Parse for SimpleGrammar {
        type Base = FieldGrammarBase;

        const FEATURES: Features = Features::all().without(Features::OBJECTS);
    }

    let input = InputSpan::new("#{ x = 1; y = 2; };");
    let err = statement::<SimpleGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Error(spanned) if spanned.location().span(&input) == "#");

    let input = InputSpan::new("{ x, y } = #{ x = 1; y = 2; }");
    let err = statement::<SimpleGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Failure(spanned) if spanned.location().location_offset() == 3);

    let input = InputSpan::new("(x, y) = (1 + 2, 3 + 5)");
    let stmt = statement::<SimpleGrammar, Complete>(input).unwrap().1;
    assert_matches!(
        stmt.extra,
        Statement::Assignment { lhs, .. } if matches!(lhs.extra, Lvalue::Tuple(_))
    );
}

#[test]
fn object_expressions_when_blocks_are_switched_off() {
    #[derive(Debug, Clone)]
    struct SimpleGrammar;

    impl Parse for SimpleGrammar {
        type Base = FieldGrammarBase;

        const FEATURES: Features = Features::all().without(Features::BLOCKS);
    }

    let input = InputSpan::new("#{ x: 1, y: 2 };");
    let (_, statement) = statement::<SimpleGrammar, Complete>(input).unwrap();
    assert_matches!(
        statement.extra,
        Statement::Expr(expr) if matches!(expr.extra, Expr::Object(_))
    );
}
