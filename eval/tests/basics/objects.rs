//! Tests targeting objects / object field access.

use arithmetic_eval::{Environment, ErrorKind, Value};
use assert_matches::assert_matches;

use crate::{evaluate, try_evaluate};

#[test]
fn object_basics() {
    let program = "#{ x = 1; y = (2, 3); }";
    let return_value = evaluate(&mut Environment::new(), program);
    let fields = match return_value {
        Value::Object(fields) => fields,
        _ => panic!("Unexpected return value: {:?}", return_value),
    };

    assert_eq!(fields.len(), 2);
    assert_eq!(fields["x"], Value::Number(1.0));
    assert_eq!(
        fields["y"],
        Value::Tuple(vec![Value::Number(2.0), Value::Number(3.0)])
    );
}

#[test]
fn object_field_access() {
    let program = r#"
        obj = #{ x = 1; y = (2, 3); };
        obj.x == 1 && obj.y.1 == 3
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn destructuring_in_object() {
    let program = r#"
        vec = (1, 2);
        #{ (x, y) = vec; }
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    let fields = match return_value {
        Value::Object(fields) => fields,
        _ => panic!("Unexpected return value: {:?}", return_value),
    };

    assert_eq!(fields.len(), 2);
    assert_eq!(fields["x"], Value::Number(1.0));
    assert_eq!(fields["y"], Value::Number(2.0));

    let program = r#"
        vec = (1, 2, 3);
        #{ (..., x, y) = vec; }
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    let fields = match return_value {
        Value::Object(fields) => fields,
        _ => panic!("Unexpected return value: {:?}", return_value),
    };

    assert_eq!(fields.len(), 2);
    assert_eq!(fields["x"], Value::Number(2.0));
    assert_eq!(fields["y"], Value::Number(3.0));
}

#[test]
fn object_expr_does_not_capture_surroundings() {
    let program = r#"
        z = 5;
        obj = #{ x = 1; y = z; };
        v = (obj,);
        obj
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    let fields = match return_value {
        Value::Object(fields) => fields,
        _ => panic!("Unexpected return value: {:?}", return_value),
    };

    assert_eq!(fields.len(), 2);
    assert_eq!(fields["x"], Value::Number(1.0));
    assert_eq!(fields["y"], Value::Number(5.0));
}

#[test]
fn object_expr_does_not_capture_inner_scopes() {
    let program = r#"
        #{
            x = 1;
            y = { z = 5; z + 1 };
            vec = (x, y);
        }
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    let fields = match return_value {
        Value::Object(fields) => fields,
        _ => panic!("Unexpected return value: {:?}", return_value),
    };

    assert_eq!(fields.len(), 3);
    assert_eq!(fields["x"], Value::Number(1.0));
    assert_eq!(fields["y"], Value::Number(6.0));
    assert_eq!(
        fields["vec"],
        Value::Tuple(vec![Value::Number(1.0), Value::Number(6.0)])
    );
}

#[test]
fn object_in_object() {
    let program = r#"
        #{
            pt_x = 1;
            pt = #{ x = pt_x; y = pt_x + 1; };
        }
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    let fields = match return_value {
        Value::Object(fields) => fields,
        _ => panic!("Unexpected return value: {:?}", return_value),
    };

    assert_eq!(fields.len(), 2);
    assert_eq!(fields["pt_x"], Value::Number(1.0));

    let inner_fields = match &fields["pt"] {
        Value::Object(fields) => fields,
        other => panic!("Unexpected inner object: {:?}", other),
    };
    assert_eq!(inner_fields.len(), 2);
    assert_eq!(inner_fields["x"], Value::Number(1.0));
    assert_eq!(inner_fields["y"], Value::Number(2.0));
}

#[test]
fn accessing_embedded_objects() {
    let program = r#"
        obj = #{
            pt_x = 1;
            pt = #{ x = pt_x; y = pt_x + 1; };
        };
        obj.pt.x == obj.pt_x && obj.pt.y == 2
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn field_redefinition_in_object() {
    let program = "#{ x = 1; x = x + 1; }";
    let return_value = evaluate(&mut Environment::new(), program);
    let fields = match return_value {
        Value::Object(fields) => fields,
        _ => panic!("Unexpected return value: {:?}", return_value),
    };

    assert_eq!(fields.len(), 1);
    assert_eq!(fields["x"], Value::Number(2.0));

    let program = "#{ x = 1; x = (x, x + 1); }";
    let return_value = evaluate(&mut Environment::new(), program);
    let fields = match return_value {
        Value::Object(fields) => fields,
        _ => panic!("Unexpected return value: {:?}", return_value),
    };

    assert_eq!(fields.len(), 1);
    assert_eq!(
        fields["x"],
        Value::Tuple(vec![Value::Number(1.0), Value::Number(2.0)])
    );
}

#[test]
fn field_redefinition_via_destructuring() {
    let program = "#{ xs = (1, 2, 3); (y, ...xs) = xs; }";
    let return_value = evaluate(&mut Environment::new(), program);
    let fields = match return_value {
        Value::Object(fields) => fields,
        _ => panic!("Unexpected return value: {:?}", return_value),
    };

    assert_eq!(fields.len(), 2);
    assert_eq!(
        fields["xs"],
        Value::Tuple(vec![Value::Number(2.0), Value::Number(3.0)])
    );
    assert_eq!(fields["y"], Value::Number(1.0));
}

#[test]
fn accessing_fields_within_object() {
    let program = r#"
        #{
            pt = #{ x = 3; y = x + 1; };
            len_sq = pt.x * pt.x + pt.y * pt.y;
        }
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
                ("x".to_owned(), Value::Number(3.0)),
                ("y".to_owned(), Value::Number(4.0))
            ]
            .into_iter()
            .collect()
        )
    );
    assert_eq!(fields["len_sq"], Value::Number(25.0));
}

#[test]
fn callable_object_field() {
    let program = r#"
        obj = #{ x = 3; y = 4; len_sq = || x * x + y * y; };
        obj.x == 3 && obj.y == 4 && (obj.len_sq)() == 25
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));

    let program = r#"
        obj = {
            POW = 2;
            #{ sq = |x| x ^ POW; }
        };
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
    let program = "pt = #{ x = 1; y = 2; }; pt.z";
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
