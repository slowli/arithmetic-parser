//! Tests focused on object / field access.

use assert_matches::assert_matches;
use nom::Err as NomErr;

use super::{lsp, sp, FieldGrammar, Literal, ValueType};
use crate::{
    parser::{
        expr, fn_def, lvalue, object_destructure, object_expr, simple_expr, statement, Complete,
    },
    BinaryOp, Expr, InputSpan, Lvalue, ObjectDestructure, ObjectDestructureField, Statement,
};

#[test]
fn field_access_basics() {
    let input = InputSpan::new("point.x");
    let (rest, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(
        expr.extra,
        Expr::FieldAccess {
            name: sp(6, "x", ()),
            receiver: Box::new(sp(0, "point", Expr::Variable)),
        }
    );

    let input = InputSpan::new("(5, 2). s0");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    assert_eq!(
        expr.extra,
        Expr::FieldAccess {
            name: sp(8, "s0", ()),
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
        other => panic!("Unexpected expr: {:?}", other),
    };
    assert_matches!(inner_expr.extra, Expr::Block(_));
}

#[test]
fn chained_field_access() {
    let input = InputSpan::new("point.x.sin()");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();

    let inner_expr = match expr.extra {
        Expr::Method { receiver, .. } => *receiver,
        other => panic!("Unexpected expr: {:?}", other),
    };
    assert_eq!(
        inner_expr.extra,
        Expr::FieldAccess {
            name: sp(6, "x", ()),
            receiver: Box::new(sp(0, "point", Expr::Variable)),
        }
    );

    let input = InputSpan::new("Point(1, 2).x");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();

    let inner_expr = match expr.extra {
        Expr::FieldAccess { receiver, name } if *name.fragment() == "x" => *receiver,
        other => panic!("Unexpected expr: {:?}", other),
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
        other => panic!("Unexpected expr: {:?}", other),
    };
    assert_matches!(inner_expr.extra, Expr::FieldAccess { name, .. } if *name.fragment() == "x");
}

#[test]
fn callable_field_access() {
    let input = InputSpan::new("(obj.fun)(1, 2)");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();

    let inner_expr = match expr.extra {
        Expr::Function { name, args } => {
            assert_eq!(args.len(), 2);
            *name
        }
        other => panic!("Unexpected expr: {:?}", other),
    };
    assert_eq!(*inner_expr.fragment(), "(obj.fun)");
    assert_eq!(
        inner_expr.extra,
        Expr::FieldAccess {
            name: sp(5, "fun", ()),
            receiver: Box::new(sp(1, "obj", Expr::Variable)),
        }
    );
}

#[test]
fn field_access_in_larger_context() {
    let input = InputSpan::new("point.x.sin + (1, 2)");
    let (_, parsed) = expr::<FieldGrammar, Complete>(input).unwrap();

    let lhs = match parsed.extra {
        Expr::Binary { lhs, rhs, op } => {
            assert_eq!(op, sp(12, "+", BinaryOp::Add));
            assert_matches!(rhs.extra, Expr::Tuple(items) if items.len() == 2);
            *lhs
        }
        other => panic!("Unexpected expr: {:?}", other),
    };
    assert_matches!(lhs.extra, Expr::FieldAccess { name, .. } if *name.fragment() == "sin");

    let input = InputSpan::new("foo(point.sin().x).y");
    let (_, parsed) = expr::<FieldGrammar, Complete>(input).unwrap();

    let inner_expr = match parsed.extra {
        Expr::FieldAccess { name, receiver } if *name.fragment() == "y" => *receiver,
        other => panic!("Unexpected expr: {:?}", other),
    };
    let inner_field_access = match inner_expr.extra {
        Expr::Function { name, mut args } => {
            assert_eq!(*name, sp(0, "foo", Expr::Variable));
            assert_eq!(args.len(), 1);
            args.pop().unwrap()
        }
        other => panic!("Unexpected expr: {:?}", other),
    };
    assert_matches!(
        inner_field_access.extra,
        Expr::FieldAccess { name, .. } if *name.fragment() == "x"
    );
}

#[test]
fn object_basics() {
    let input = InputSpan::new("#{ x = 1; y = 2; }");
    let (rest, obj) = object_expr::<FieldGrammar, Complete>(input).unwrap();
    assert!(rest.fragment().is_empty());
    assert_matches!(
        obj.extra,
        Expr::ObjectBlock(statements) if statements.len() == 2
    );

    let input = InputSpan::new("#{}");
    let (rest, obj) = object_expr::<FieldGrammar, Complete>(input).unwrap();
    assert!(rest.fragment().is_empty());
    assert_matches!(
        obj.extra,
        Expr::ObjectBlock(statements) if statements.is_empty()
    );

    let input = InputSpan::new("#{ x = 5; dbg(x); y = x + 2; }");
    let (_, obj) = object_expr::<FieldGrammar, Complete>(input).unwrap();
    assert_matches!(
        obj.extra,
        Expr::ObjectBlock(statements) if statements.len() == 3
    );
}

#[test]
fn object_errors() {
    let input = InputSpan::new("#{ x = 1; y = 2 }");
    let err = object_expr::<FieldGrammar, Complete>(input).unwrap_err();
    let err = match err {
        NomErr::Failure(err) => err,
        other => panic!("Unexpected error: {:?}", other),
    };
    assert_eq!(err.span(), sp(10, "y", ()));
}

#[test]
fn objects_within_larger_context() {
    let input = InputSpan::new("#{ x = 1; y = 5; }.x");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    let inner_expr = match expr.extra {
        Expr::FieldAccess { receiver, name } => {
            assert_eq!(name, sp(19, "x", ()));
            *receiver
        }
        other => panic!("Unexpected expr: {:?}", other),
    };
    assert_matches!(inner_expr.extra, Expr::ObjectBlock(statements) if statements.len() == 2);

    let input = InputSpan::new("test(x, #{ y = y; z = 2 * y; })");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    let inner_expr = match expr.extra {
        Expr::Function { name, mut args } => {
            assert_eq!(*name, sp(0, "test", Expr::Variable));
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], sp(5, "x", Expr::Variable));
            args.pop().unwrap()
        }
        other => panic!("Unexpected expr: {:?}", other),
    };
    assert_matches!(inner_expr.extra, Expr::ObjectBlock(statements) if statements.len() == 2);

    let input = InputSpan::new("{ x = gen(); #{ x = x; y = x * GEN; } }");
    let (_, expr) = simple_expr::<FieldGrammar, Complete>(input).unwrap();
    let inner_expr = match expr.extra {
        Expr::Block(block) => {
            assert_eq!(block.statements.len(), 1);
            assert_eq!(*block.statements[0].fragment(), "x = gen()");
            *block.return_value.unwrap()
        }
        other => panic!("Unexpected expr: {:?}", other),
    };
    assert_matches!(inner_expr.extra, Expr::ObjectBlock(statements) if statements.len() == 2);

    let input = InputSpan::new("|xs| #{ x = xs.0; }");
    let (_, fn_def) = fn_def::<FieldGrammar, Complete>(input).unwrap();
    assert_eq!(*fn_def.args.fragment(), "|xs|");
    assert!(fn_def.body.statements.is_empty());
    let inner_expr = *fn_def.body.return_value.unwrap();
    assert_matches!(inner_expr.extra, Expr::ObjectBlock(statements) if statements.len() == 1);
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
    let lvalue = match lvalue.extra {
        Lvalue::Tuple(tuple) => tuple,
        _ => panic!("Unexpected lvalue: {:?}", lvalue),
    };
    assert_eq!(lvalue.start.len(), 1);
    let inner_lvalue = match &lvalue.start[0].extra {
        Lvalue::Object(obj) => obj,
        _ => panic!("Unexpected lvalue: {:?}", lvalue),
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

        let lhs = match stmt.extra {
            Statement::Assignment { lhs, .. } => lhs.extra,
            _ => panic!("Unexpected statement: {:?}", stmt),
        };
        assert_matches!(lhs, Lvalue::Object(_));
    }
}
