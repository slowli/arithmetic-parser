//! Tests for type casts.

use assert_matches::assert_matches;
use nom::Err as NomErr;

use super::{sp, FieldGrammar, Literal, ValueType};
use crate::{
    parser::{
        expr::{expr, simple_expr},
        Complete,
    },
    BinaryOp, Expr, InputSpan, UnaryOp,
};

#[test]
fn type_casts_simple() {
    let input = InputSpan::new("x as Sc");
    let expr = simple_expr::<FieldGrammar, Complete>(input).unwrap().1;
    assert_eq!(
        expr.extra,
        Expr::TypeCast {
            value: Box::new(sp(0, "x", Expr::Variable)),
            ty: sp(5, "Sc", ValueType::Scalar),
        }
    );

    let input = InputSpan::new("1as Sc");
    let expr = simple_expr::<FieldGrammar, Complete>(input).unwrap().1;
    assert_eq!(
        expr.extra,
        Expr::TypeCast {
            value: Box::new(sp(0, "1", Expr::Literal(Literal::Number))),
            ty: sp(4, "Sc", ValueType::Scalar),
        }
    );

    let input = InputSpan::new("-x as Sc");
    let expr = simple_expr::<FieldGrammar, Complete>(input).unwrap().1;
    assert_eq!(
        expr.extra,
        Expr::TypeCast {
            value: Box::new(sp(
                0,
                "-x",
                Expr::Unary {
                    op: sp(0, "-", UnaryOp::Neg),
                    inner: Box::new(sp(1, "x", Expr::Variable)),
                }
            )),
            ty: sp(6, "Sc", ValueType::Scalar),
        }
    );

    let input = InputSpan::new("x.foo() as Sc");
    let expr = simple_expr::<FieldGrammar, Complete>(input).unwrap().1;
    assert_matches!(
        expr.extra,
        Expr::TypeCast { value, ty }
            if matches!(value.extra, Expr::Method { .. }) && *ty.fragment() == "Sc"
    );

    let input = InputSpan::new("{ x = 5; x } as Sc");
    let expr = simple_expr::<FieldGrammar, Complete>(input).unwrap().1;
    assert_matches!(
        expr.extra,
        Expr::TypeCast { value, ty }
            if matches!(value.extra, Expr::Block(_)) && *ty.fragment() == "Sc"
    );
}

#[test]
fn multiple_type_casts() {
    let input = InputSpan::new("abs(x) as Sc as bool");
    let expr = simple_expr::<FieldGrammar, Complete>(input).unwrap().1;

    let Expr::TypeCast { value, ty } = expr.extra else {
        panic!("Unexpected expr: {expr:?}");
    };
    assert_eq!(ty, sp(16, "bool", ValueType::Bool));
    assert_eq!(
        value.extra,
        Expr::TypeCast {
            value: Box::new(sp(
                0,
                "abs(x)",
                Expr::Function {
                    name: Box::new(sp(0, "abs", Expr::Variable)),
                    args: vec![sp(4, "x", Expr::Variable)],
                }
            )),
            ty: sp(10, "Sc", ValueType::Scalar),
        }
    );
}

#[test]
fn type_casts_within_larger_context() {
    let input = InputSpan::new("test(x as Sc, y)");
    let parsed = expr::<FieldGrammar, Complete>(input).unwrap().1;
    assert_matches!(
        parsed.extra,
        Expr::Function { args, .. } if matches!(args[0].extra, Expr::TypeCast { .. })
    );

    let input = InputSpan::new("x as Sc+y");
    let parsed = expr::<FieldGrammar, Complete>(input).unwrap().1;
    assert_eq!(
        parsed.extra,
        Expr::Binary {
            lhs: Box::new(sp(
                0,
                "x as Sc",
                Expr::TypeCast {
                    value: Box::new(sp(0, "x", Expr::Variable)),
                    ty: sp(5, "Sc", ValueType::Scalar),
                }
            )),
            op: sp(7, "+", BinaryOp::Add),
            rhs: Box::new(sp(8, "y", Expr::Variable)),
        }
    );

    let input = InputSpan::new("x + y as Sc");
    let parsed = expr::<FieldGrammar, Complete>(input).unwrap().1;
    assert_eq!(
        parsed.extra,
        Expr::Binary {
            lhs: Box::new(sp(0, "x", Expr::Variable)),
            op: sp(2, "+", BinaryOp::Add),
            rhs: Box::new(sp(
                4,
                "y as Sc",
                Expr::TypeCast {
                    value: Box::new(sp(4, "y", Expr::Variable)),
                    ty: sp(9, "Sc", ValueType::Scalar),
                }
            )),
        }
    );

    let input = InputSpan::new("(x + y)as Sc");
    let parsed = expr::<FieldGrammar, Complete>(input).unwrap().1;
    assert_matches!(
        parsed.extra,
        Expr::TypeCast { value, .. } if *value.fragment() == "(x + y)"
    );

    let input = InputSpan::new("-(y as Sc)");
    let parsed = expr::<FieldGrammar, Complete>(input).unwrap().1;
    assert_matches!(
        parsed.extra,
        Expr::Unary { inner, .. } if matches!(inner.extra, Expr::TypeCast { .. })
    );

    let input = InputSpan::new("(fn as Sc)(1, x)");
    let parsed = expr::<FieldGrammar, Complete>(input).unwrap().1;
    assert_matches!(
        parsed.extra,
        Expr::Function { name, .. } if matches!(name.extra, Expr::TypeCast { .. })
    );
}

#[test]
fn type_cast_errors() {
    let input = InputSpan::new("x as Unknown");
    let err = simple_expr::<FieldGrammar, Complete>(input).unwrap_err();
    assert_matches!(err, NomErr::Failure(_));

    // Should not be parsed as a cast
    let input = InputSpan::new("x asSc");
    let rest = simple_expr::<FieldGrammar, Complete>(input).unwrap().0;
    assert_eq!(*rest.fragment(), " asSc");

    // Should not be parsed as function call
    let input = InputSpan::new("foo as Sc(1, x)");
    let rest = expr::<FieldGrammar, Complete>(input).unwrap().0;
    assert_eq!(*rest.fragment(), "(1, x)");
}
