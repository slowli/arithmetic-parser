//! Tests targeting objects / object field access.

use assert_matches::assert_matches;

use arithmetic_eval::{error::AuxErrorInfo, Assertions, Environment, ErrorKind, Prelude, Value};
use arithmetic_parser::BinaryOp;

use crate::{evaluate, expect_compilation_error, try_evaluate};

#[test]
fn object_basics() {
    let program = "#{ x: 1, y: (2, 3) }";
    let return_value = evaluate(&mut Environment::new(), program);
    let fields = match return_value {
        Value::Object(fields) => fields,
        _ => panic!("Unexpected return value: {:?}", return_value),
    };

    assert_eq!(fields.len(), 2);
    assert_eq!(fields["x"], Value::Prim(1.0));
    assert_eq!(
        fields["y"],
        Value::from(vec![Value::Prim(2.0), Value::Prim(3.0)])
    );
}

#[test]
fn object_field_access() {
    let program = r#"
        obj = #{ x: 1, y: (2, 3) };
        obj.x == 1 && obj.y.1 == 3
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn capturing_vars_in_object() {
    let program = r#"
        (x, y) = (1, 2);
        #{ x, y: y + 1 }
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    let fields = match return_value {
        Value::Object(fields) => fields,
        _ => panic!("Unexpected return value: {:?}", return_value),
    };

    assert_eq!(fields.len(), 2);
    assert_eq!(fields["x"], Value::Prim(1.0));
    assert_eq!(fields["y"], Value::Prim(3.0));
}

#[test]
fn object_expr_does_not_capture_surroundings() {
    let program = r#"
        z = 5;
        obj = #{ x: 1, y: z };
        v = (obj,);
        obj
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    let fields = match return_value {
        Value::Object(fields) => fields,
        _ => panic!("Unexpected return value: {:?}", return_value),
    };

    assert_eq!(fields.len(), 2);
    assert_eq!(fields["x"], Value::Prim(1.0));
    assert_eq!(fields["y"], Value::Prim(5.0));
}

#[test]
fn object_expr_does_not_capture_inner_scopes() {
    let program = r#"
        #{
            x: 1,
            y: { z = 5; z + 1 },
            vec: (1, 6),
        }
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    let fields = match return_value {
        Value::Object(fields) => fields,
        _ => panic!("Unexpected return value: {:?}", return_value),
    };

    assert_eq!(fields.len(), 3);
    assert_eq!(fields["x"], Value::Prim(1.0));
    assert_eq!(fields["y"], Value::Prim(6.0));
    assert_eq!(
        fields["vec"],
        Value::from(vec![Value::Prim(1.0), Value::Prim(6.0)])
    );
}

#[test]
fn object_in_object() {
    let program = r#"
        pt_x = 1;
        #{
            pt_x,
            pt: #{ x: pt_x, y: pt_x + 1 },
        }
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    let fields = match return_value {
        Value::Object(fields) => fields,
        _ => panic!("Unexpected return value: {:?}", return_value),
    };

    assert_eq!(fields.len(), 2);
    assert_eq!(fields["pt_x"], Value::Prim(1.0));

    let inner_fields = match &fields["pt"] {
        Value::Object(fields) => fields,
        other => panic!("Unexpected inner object: {:?}", other),
    };
    assert_eq!(inner_fields.len(), 2);
    assert_eq!(inner_fields["x"], Value::Prim(1.0));
    assert_eq!(inner_fields["y"], Value::Prim(2.0));
}

#[test]
fn accessing_embedded_objects() {
    let program = r#"
        pt_x = 1;
        obj = #{
            pt_x,
            pt: #{ x: pt_x, y: pt_x + 1 },
        };
        obj.pt.x == obj.pt_x && obj.pt.y == 2
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn accessing_fields_within_object() {
    let program = r#"
        pt = #{ x: 3, y: 4 };
        #{ pt, len_sq: pt.x * pt.x + pt.y * pt.y }
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    let fields = match return_value {
        Value::Object(fields) => fields,
        _ => panic!("Unexpected return value: {:?}", return_value),
    };

    assert_eq!(fields.len(), 2);
    assert_eq!(
        fields["pt"],
        Value::Object(
            vec![
                ("x".to_owned(), Value::Prim(3.0)),
                ("y".to_owned(), Value::Prim(4.0))
            ]
            .into_iter()
            .collect()
        )
    );
    assert_eq!(fields["len_sq"], Value::Prim(25.0));
}

#[test]
fn callable_object_field() {
    let program = r#"
        (x, y) = (3, 4);
        obj = #{ x, y, len_sq: || x * x + y * y };
        obj.x == 3 && obj.y == 4 && (obj.len_sq)() == 25
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));

    let program = r#"
        obj = { POW = 2; #{ sq: |x| x ^ POW } };
        sq = obj.sq;
        (obj.sq)(3) == 9 && sq(4) == 16
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn field_invalid_receiver_error() {
    let program = "xs = (1, 2, 3); xs.len == 3";
    let err = try_evaluate(&mut Environment::new(), program).unwrap_err();

    assert_eq!(*err.source().main_span().code().fragment(), "xs.len");
    assert_matches!(err.source().kind(), ErrorKind::CannotAccessFields);
}

#[test]
fn no_field_error() {
    let program = "pt = #{ x: 1, y: 2 }; pt.z";
    let err = try_evaluate(&mut Environment::new(), program).unwrap_err();

    assert_eq!(*err.source().main_span().code().fragment(), "pt.z");
    assert_matches!(
        err.source().kind(),
        ErrorKind::NoField { field, available_fields }
            if field == "z" && available_fields.len() == 2 &&
            available_fields.contains(&"x".to_owned()) &&
            available_fields.contains(&"y".to_owned())
    );
}

#[test]
fn object_comparison() {
    let program = r#"
        #{ x: 1 } == #{ x: 1 } &&
        #{ x: 1 } == #{ x: 2 - 1 } &&
        #{ x: 1 } != #{ x: 0 } &&
        #{ x: 1 } != #{} &&
        #{ x: 1 } != #{ x: 1, y: 1 } &&
        #{ x: 1 } != (1,) && (1,) != #{ x: 1 } &&
        #{ x: 1 } != 1 && 1 != #{ x: 1 }
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn object_destructuring() {
    let program = r#"
        { x } = #{ x: 1 };
        { x -> y } = #{ x: 2 };
        obj = #{ xs: (3, 4, 5), flag: 1 == 1 };
        { xs: (head, ...tail), flag } = obj;
        x == 1 && y == 2 && head == 3 && tail == (4, 5) && flag
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn embedded_object_destructuring() {
    let program = r#"
        ({ x, y }, ...pts) = (#{ x: 1, y: 2 }, #{ x: 2, y: 3 });
        x == 1 && y == 2 && pts.0.x == 2 && pts.0.y == 3
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn object_destructuring_in_fn_args() {
    let program = r#"
        manhattan = |{ x, y }| x + y;
        manhattan(#{ x: 1, y: 2 }) == 3
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn object_destructuring_in_pipeline() {
    let program = r#"
        minmax = |...xs| xs.fold(#{ min: INF, max: -INF }, |{ min, max }, x| #{
            min: if(x < min, x, min),
            max: if(x > max, x, max),
        });
        assert_eq(minmax(5, -4, 6, 9, 1), #{ min: -4, max: 9 });
    "#;
    let mut env = Environment::new();
    env.insert("INF", Value::Prim(f32::INFINITY))
        .insert_prototypes(Prelude.prototypes());
    env.extend(Prelude.iter());
    env.extend(Assertions.iter());
    evaluate(&mut env, program);
}

#[test]
fn object_destructuring_on_non_object() {
    let programs = &[
        "{ x } = 1;",
        "{ x } = 1 == 1;",
        "{ x } = (1, 2);",
        "{ x } = || 1;",
    ];
    for &program in programs {
        let err = try_evaluate(&mut Environment::new(), program).unwrap_err();
        assert_eq!(*err.source().main_span().code().fragment(), "x");
        assert_matches!(err.source().kind(), ErrorKind::CannotAccessFields);
    }
}

#[test]
fn object_destructuring_with_missing_field() {
    let program = "{ x, y: Y } = #{ x: 1 };";
    let err = try_evaluate(&mut Environment::new(), program).unwrap_err();
    assert_eq!(*err.source().main_span().code().fragment(), "y");
    assert_matches!(err.source().kind(), ErrorKind::NoField { field, .. } if field == "y");

    let program = "({ x, y }, ...pts) = (#{ x: 1 }, #{ x: 2 });";
    let err = try_evaluate(&mut Environment::new(), program).unwrap_err();
    assert_eq!(*err.source().main_span().code().fragment(), "y");
    assert_matches!(err.source().kind(), ErrorKind::NoField { field, .. } if field == "y");
}

#[test]
fn embedded_destructuring_error() {
    let program = "{ x, y: (y, ...) } = #{ x: 1, y: 2 };";
    let err = try_evaluate(&mut Environment::new(), program).unwrap_err();
    assert_eq!(*err.source().main_span().code().fragment(), "(y, ...)");
    assert_matches!(err.source().kind(), ErrorKind::CannotDestructure);
}

#[test]
fn object_initialization_repeated_fields() {
    let program = "#{ x: 1, x: 2 }";
    let err = expect_compilation_error(&mut Environment::new(), program);
    let err_span = err.main_span().code();

    assert_eq!(*err_span.fragment(), "x");
    assert_eq!(err_span.location_offset(), 9);
    assert_matches!(err.kind(), ErrorKind::RepeatedField);

    assert_eq!(err.aux_spans().len(), 1);
    let aux_span = err.aux_spans()[0].code();
    assert_eq!(*aux_span.fragment(), "x");
    assert_eq!(aux_span.location_offset(), 3);
    assert_matches!(aux_span.extra, AuxErrorInfo::PrevAssignment);
}

#[test]
fn object_initialization_repeated_fields_with_shorthand() {
    let program = "x = 2; #{ x: 5 + x, x }";
    let err = expect_compilation_error(&mut Environment::new(), program);
    let err_span = err.main_span().code();

    assert_eq!(*err_span.fragment(), "x");
    assert_eq!(err_span.location_offset(), 20);
    assert_matches!(err.kind(), ErrorKind::RepeatedField);
}

#[test]
fn object_destructuring_repeated_fields() {
    let program = "{ x, x: y } = #{ x: 1, y: 2 };";
    let err = expect_compilation_error(&mut Environment::new(), program);
    let err_span = err.main_span().code();

    assert_eq!(*err_span.fragment(), "x");
    assert_eq!(err_span.location_offset(), 5);
    assert_matches!(err.kind(), ErrorKind::RepeatedField);

    assert_eq!(err.aux_spans().len(), 1);
    let aux_span = err.aux_spans()[0].code();
    assert_eq!(*aux_span.fragment(), "x");
    assert_eq!(aux_span.location_offset(), 2);
    assert_matches!(aux_span.extra, AuxErrorInfo::PrevAssignment);
}

#[test]
fn object_destructuring_repeated_assignment() {
    let program = "{ x, y: x } = #{ x: 1, y: 2 };";
    let err = expect_compilation_error(&mut Environment::new(), program);
    let err_span = err.main_span().code();

    assert_eq!(*err_span.fragment(), "x");
    assert_eq!(err_span.location_offset(), 8);
    assert_matches!(err.kind(), ErrorKind::RepeatedAssignment { .. });

    assert_eq!(err.aux_spans().len(), 1);
    let aux_span = err.aux_spans()[0].code();
    assert_eq!(*aux_span.fragment(), "x");
    assert_eq!(aux_span.location_offset(), 2);
    assert_matches!(aux_span.extra, AuxErrorInfo::PrevAssignment);
}

#[test]
fn object_destructuring_repeated_assignment_complex() {
    let program = "{ x, ys: (x, ...) } = #{ x: 1, ys: (2, 3) };";
    let err = expect_compilation_error(&mut Environment::new(), program);
    let err_span = err.main_span().code();

    assert_eq!(*err_span.fragment(), "x");
    assert_eq!(err_span.location_offset(), 10);
    assert_matches!(err.kind(), ErrorKind::RepeatedAssignment { .. });

    assert_eq!(err.aux_spans().len(), 1);
    let aux_span = err.aux_spans()[0].code();
    assert_eq!(*aux_span.fragment(), "x");
    assert_eq!(aux_span.location_offset(), 2);
    assert_matches!(aux_span.extra, AuxErrorInfo::PrevAssignment);
}

#[test]
fn binary_ops_on_objects() {
    let program = r#"
        #{ x: 1 } - #{ x: 2 } == #{ x: -1 } &&
        #{ x: 6 } / #{ x: 2 } == #{ x: 3 } &&
        #{ x: 1, y: (2, 3) } + #{ x: (5, 7), y: (1, 2) } ==
            #{ x: (6, 8), y: (3, 5) }
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn binary_ops_on_objects_with_number_operand() {
    let program = r#"
        #{ x: 3, y: 2 } - 1 == #{ x: 2, y: 1 } &&
        5 - #{ x: 3 } == #{ x: 2 } &&
        2 + #{ x: 3, y: (4, 5) } == #{ x: 5, y: (6, 7) }
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn error_in_binary_ops_on_objects() {
    let program = "#{ x: 1 } + #{ y: 1 }";
    let err = try_evaluate(&mut Environment::new(), program).unwrap_err();
    assert_eq!(*err.source().main_span().code().fragment(), "#{ x: 1 }");
    assert_matches!(
        err.source().kind(),
        ErrorKind::FieldsMismatch { op: BinaryOp::Add, lhs_fields, rhs_fields }
            if lhs_fields.len() == 1 && lhs_fields.contains("x")
            && rhs_fields.len() == 1 && rhs_fields.contains("y")
    );
    assert_eq!(err.source().aux_spans().len(), 1);

    let aux_info = &err.source().aux_spans()[0].code().extra;
    assert_matches!(
        aux_info,
        AuxErrorInfo::UnbalancedRhsObject(fields) if fields.contains("y")
    );
}
