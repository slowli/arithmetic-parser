//! Tests targeting objects / object field access.

use arithmetic_eval::{Environment, Value};

use crate::evaluate;

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
