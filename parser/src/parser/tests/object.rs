//! Tests focused on object / field access.

use assert_matches::assert_matches;
use nom::Err as NomErr;

use super::{lsp, sp, FieldGrammar, Literal, ValueType};
use crate::{
    parser::{
        expr, fn_def, lvalue, object_destructure, object_expr, simple_expr, statement, Complete,
    },
    BinaryOp, Expr, InputSpan, Lvalue, ObjectDestructure, ObjectDestructureField, ObjectExpr,
    Statement,
};

#[test]
fn field_access_basics() {
    let input = InputSpan::new("point.x");
    let (rest, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(
        expr.extra,
        Expr::FieldAccess {
            name: Box::new(sp(6, "x", Expr::Variable)),
            receiver: Box::new(sp(0, "point", Expr::Variable)),
        }
    );

    let input = InputSpan::new("(5, 2). s0");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    assert_eq!(
        expr.extra,
        Expr::FieldAccess {
            name: Box::new(sp(8, "s0", Expr::Variable)),
            receiver: Box::new(sp(
                0,
                "(5, 2)",
                Expr::Tuple(vec![
                    sp(1, "5", Expr::Literal(Literal::Number)),
                    sp(4, "2", Expr::Literal(Literal::Number)),
                ])
            )),
        }
    );

    let input = InputSpan::new("{ x = 5; x }.foo");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    let inner_expr = match expr.extra {
        Expr::FieldAccess { receiver, name } if *name.fragment() == "foo" => *receiver,
        other => panic!("Unexpected expr: {other:?}"),
    };
    assert_matches!(inner_expr.extra, Expr::Block(_));
}

#[test]
fn chained_field_access() {
    let input = InputSpan::new("point.x.sin()");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();

    let Expr::Method { receiver, .. } = expr.extra else {
        panic!("Unexpected expr: {expr:?}");
    };
    assert_eq!(
        receiver.extra,
        Expr::FieldAccess {
            name: Box::new(sp(6, "x", Expr::Variable)),
            receiver: Box::new(sp(0, "point", Expr::Variable)),
        }
    );

    let input = InputSpan::new("Point(1, 2).x");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();

    let inner_expr = match expr.extra {
        Expr::FieldAccess { receiver, name } if *name.fragment() == "x" => *receiver,
        other => panic!("Unexpected expr: {other:?}"),
    };
    assert_eq!(
        inner_expr.extra,
        Expr::Function {
            name: Box::new(sp(0, "Point", Expr::Variable)),
            args: vec![
                sp(6, "1", Expr::Literal(Literal::Number)),
                sp(9, "2", Expr::Literal(Literal::Number)),
            ],
        }
    );

    let input = InputSpan::new("point.x.sin");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();

    let inner_expr = match expr.extra {
        Expr::FieldAccess { receiver, name } if *name.fragment() == "sin" => *receiver,
        other => panic!("Unexpected expr: {other:?}"),
    };
    assert_matches!(inner_expr.extra, Expr::FieldAccess { name, .. } if *name.fragment() == "x");
}

#[test]
fn callable_field_access() {
    let input = InputSpan::new("(obj.fun)(1, 2)");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();

    let Expr::Function { name, args } = expr.extra else {
        panic!("Unexpected expr: {expr:?}");
    };
    assert_eq!(args.len(), 2);
    assert_eq!(*name.fragment(), "(obj.fun)");
    assert_eq!(
        name.extra,
        Expr::FieldAccess {
            name: Box::new(sp(5, "fun", Expr::Variable)),
            receiver: Box::new(sp(1, "obj", Expr::Variable)),
        }
    );
}

#[test]
fn field_access_in_larger_context() {
    let input = InputSpan::new("point.x.sin + (1, 2)");
    let (_, parsed) = expr::<FieldGrammar, Complete>(input).unwrap();

    let Expr::Binary { lhs, rhs, op } = parsed.extra else {
        panic!("Unexpected expr: {parsed:?}");
    };
    assert_eq!(op, sp(12, "+", BinaryOp::Add));
    assert_matches!(rhs.extra, Expr::Tuple(items) if items.len() == 2);
    assert_matches!(lhs.extra, Expr::FieldAccess { name, .. } if *name.fragment() == "sin");

    let input = InputSpan::new("foo(point.sin().x).y");
    let (_, parsed) = expr::<FieldGrammar, Complete>(input).unwrap();

    let inner_expr = match parsed.extra {
        Expr::FieldAccess { name, receiver } if *name.fragment() == "y" => *receiver,
        other => panic!("Unexpected expr: {other:?}"),
    };
    let Expr::Function { name, mut args } = inner_expr.extra else {
        panic!("Unexpected expr: {inner_expr:?}");
    };

    assert_eq!(*name, sp(0, "foo", Expr::Variable));
    assert_eq!(args.len(), 1);
    let inner_field_access = args.pop().unwrap();
    assert_matches!(
        inner_field_access.extra,
        Expr::FieldAccess { name, .. } if *name.fragment() == "x"
    );
}

#[test]
fn object_basics() {
    let input = InputSpan::new("#{ x: 1, y: 2 }");
    let (rest, obj) = object_expr::<FieldGrammar, Complete>(input).unwrap();
    assert!(rest.fragment().is_empty());
    assert_matches!(
        obj.extra,
        Expr::Object(ObjectExpr { fields }) if fields.len() == 2
    );

    let input = InputSpan::new("#{}");
    let (rest, obj) = object_expr::<FieldGrammar, Complete>(input).unwrap();
    assert!(rest.fragment().is_empty());
    assert_matches!(
        obj.extra,
        Expr::Object(ObjectExpr { fields }) if fields.is_empty()
    );

    let input = InputSpan::new("#{ x, y: x + 1, z, }");
    let (_, obj) = object_expr::<FieldGrammar, Complete>(input).unwrap();
    assert_matches!(
        obj.extra,
        Expr::Object(ObjectExpr { fields }) if fields.len() == 3
    );
}

#[test]
fn object_errors() {
    let input = InputSpan::new("#{ x = 1 }");
    let err = object_expr::<FieldGrammar, Complete>(input).unwrap_err();
    let NomErr::Failure(err) = err else {
        panic!("Unexpected error: {err:?}");
    };
    assert_eq!(err.span(), sp(5, "=", ()));
}

#[test]
fn objects_within_larger_context() {
    let input = InputSpan::new("#{ x: 1, y: 5 }.x");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    let Expr::FieldAccess { receiver, name } = expr.extra else {
        panic!("Unexpected expr: {expr:?}");
    };
    assert_eq!(*name, sp(16, "x", Expr::Variable));
    assert_matches!(receiver.extra, Expr::Object(ObjectExpr { fields }) if fields.len() == 2);

    let input = InputSpan::new("test(x, #{ y, z: 2 * y })");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    let Expr::Function { name, mut args } = expr.extra else {
        panic!("Unexpected expr: {expr:?}");
    };
    assert_eq!(*name, sp(0, "test", Expr::Variable));
    assert_eq!(args.len(), 2);
    assert_eq!(args[0], sp(5, "x", Expr::Variable));
    let inner_expr = args.pop().unwrap();
    assert_matches!(inner_expr.extra, Expr::Object(ObjectExpr { fields }) if fields.len() == 2);

    let input = InputSpan::new("{ x = gen(); #{ x, y: x * GEN } }");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    let Expr::Block(block) = expr.extra else {
        panic!("Unexpected expr: {expr:?}");
    };
    assert_eq!(block.statements.len(), 1);
    assert_eq!(*block.statements[0].fragment(), "x = gen()");
    let inner_expr = *block.return_value.unwrap();
    assert_matches!(inner_expr.extra, Expr::Object(ObjectExpr { fields }) if fields.len() == 2);

    let input = InputSpan::new("|xs| #{ x: xs.0 }");
    let (_, fn_def) = fn_def::<FieldGrammar, Complete>(input).unwrap();
    assert_eq!(*fn_def.args.fragment(), "|xs|");
    assert!(fn_def.body.statements.is_empty());
    let inner_expr = *fn_def.body.return_value.unwrap();
    assert_matches!(inner_expr.extra, Expr::Object(ObjectExpr { fields }) if fields.len() == 1);
}

#[test]
fn object_destructure_basics() {
    let input = InputSpan::new("{ x }");
    let (rest, obj) = object_destructure::<FieldGrammar, Complete>(input).unwrap();
    assert!(rest.fragment().is_empty());
    assert_eq!(obj.fields.len(), 1);
    assert_eq!(
        obj.fields[0],
        ObjectDestructureField {
            field_name: sp(2, "x", ()),
            binding: None,
        }
    );

    let input = InputSpan::new("{ x, y: new_y }");
    let (rest, obj) = object_destructure::<FieldGrammar, Complete>(input).unwrap();
    assert!(rest.fragment().is_empty());
    assert_eq!(
        obj.fields,
        vec![
            ObjectDestructureField {
                field_name: sp(2, "x", ()),
                binding: None,
            },
            ObjectDestructureField {
                field_name: sp(5, "y", ()),
                binding: Some(lsp(8, "new_y", Lvalue::Variable { ty: None })),
            },
        ]
    );

    let input = InputSpan::new("{ x, ys: (flag, ...ys) }");
    let (rest, obj) = object_destructure::<FieldGrammar, Complete>(input).unwrap();
    assert!(rest.fragment().is_empty());
    assert_eq!(obj.fields.len(), 2);
    assert_matches!(
        obj.fields[1].binding.as_ref().unwrap().extra,
        Lvalue::Tuple(_)
    );
}

#[test]
fn embedded_object_destructuring() {
    let input = InputSpan::new("{ x: { y: _y, z } }");
    let (rest, obj) = object_destructure::<FieldGrammar, Complete>(input).unwrap();
    assert!(rest.fragment().is_empty());

    let inner_obj = ObjectDestructure {
        fields: vec![
            ObjectDestructureField {
                field_name: sp(7, "y", ()),
                binding: Some(lsp(10, "_y", Lvalue::Variable { ty: None })),
            },
            ObjectDestructureField {
                field_name: sp(14, "z", ()),
                binding: None,
            },
        ],
    };
    assert_eq!(
        obj.fields,
        vec![ObjectDestructureField {
            field_name: sp(2, "x", ()),
            binding: Some(lsp(5, "{ y: _y, z }", Lvalue::Object(inner_obj))),
        }]
    );
}

#[test]
fn object_destructuring_in_tuple() {
    let input = InputSpan::new("({ x, y -> (z, ...), }, ...pts)");
    let (rest, lvalue) = lvalue::<FieldGrammar, Complete>(input).unwrap();

    assert!(rest.fragment().is_empty());
    let Lvalue::Tuple(lvalue) = lvalue.extra else {
        panic!("Unexpected lvalue: {lvalue:?}");
    };
    assert_eq!(lvalue.start.len(), 1);
    let Lvalue::Object(inner_lvalue) = &lvalue.start[0].extra else {
        panic!("Unexpected lvalue: {lvalue:?}");
    };
    assert_eq!(inner_lvalue.fields.len(), 2);
    assert_eq!(inner_lvalue.fields[0].field_name, sp(3, "x", ()));
    assert_eq!(inner_lvalue.fields[1].field_name, sp(6, "y", ()));
}

#[test]
fn type_annotations_in_object_destructuring() {
    let input = InputSpan::new("{ x -> x: (Sc, Ge), y }");
    let (rest, obj) = object_destructure::<FieldGrammar, Complete>(input).unwrap();
    assert!(rest.fragment().is_empty());

    let ty = ValueType::Tuple(vec![ValueType::Scalar, ValueType::Element]);
    assert_eq!(
        obj.fields,
        vec![
            ObjectDestructureField {
                field_name: sp(2, "x", ()),
                binding: Some(lsp(
                    7,
                    "x",
                    Lvalue::Variable {
                        ty: Some(sp(10, "(Sc, Ge)", ty))
                    }
                )),
            },
            ObjectDestructureField {
                field_name: sp(20, "y", ()),
                binding: None,
            }
        ]
    );
}

#[test]
fn object_destructuring_in_statement() {
    let sample_inputs = &[
        "{ x } = obj",
        "{ x, y -> _: Sc } = obj",
        "{ x: (_, ...xs) } = obj",
    ];

    for &input in sample_inputs {
        let input = InputSpan::new(input);
        let (rest, stmt) = statement::<FieldGrammar, Complete>(input).unwrap();
        assert!(rest.fragment().is_empty());

        let Statement::Assignment { lhs, .. } = stmt.extra else {
            panic!("Unexpected statement: {stmt:?}");
        };
        assert_matches!(lhs.extra, Lvalue::Object(_));
    }
}
