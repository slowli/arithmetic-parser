//! Tests focused on errors in object types.

use assert_matches::assert_matches;

use arithmetic_parser::grammars::Parse;
use arithmetic_typing::{
    arith::NumArithmetic,
    defs::Prelude,
    error::{ErrorContext, ErrorKind, ErrorLocation},
    Type, TypeEnvironment,
};

use crate::{errors::fn_arg, hash_fn_type, ErrorsExt, F32Grammar};

#[test]
fn recursive_object_constraint() {
    let code = "|obj| obj == obj.x";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_matches!(err.kind(), ErrorKind::RecursiveType(ty) if ty.to_string() == "{ x: 'T }");
}

#[test]
fn recursive_object_type() {
    let code = "|obj| obj == #{ x: obj }";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_matches!(err.kind(), ErrorKind::RecursiveType(ty) if ty.to_string() == "{ x: 'T }");
}

#[test]
fn tuple_as_object() {
    let code = "require_x = |obj| obj.x; require_x((1, 2))";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.main_span().fragment(), "(1, 2)");
    assert_matches!(err.kind(), ErrorKind::CannotAccessFields);
}

#[test]
fn calling_non_function_field() {
    let code = "pt = #{ x: 3 }; {pt.x}();";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.main_span().fragment(), "{pt.x}()");
    assert_matches!(
        err.kind(),
        ErrorKind::TypeMismatch(Type::Function(_), Type::Prim(_))
    );
}

#[test]
fn calling_field_on_non_object() {
    let code = "array = (1, 2, 3); {array.len}()";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.main_span().fragment(), "array.len");
    assert_matches!(err.kind(), ErrorKind::CannotAccessFields);
}

#[test]
fn object_and_tuple_constraints() {
    let code = "|obj| { obj.x; (x, ...) = obj; }";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.main_span().fragment(), "(x, ...)");
    assert_matches!(err.kind(), ErrorKind::CannotAccessFields);
}

#[test]
fn object_and_tuple_constraints_via_fields() {
    let code = "|obj| { obj.x == 1; obj.0 }";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.main_span().fragment(), "obj.0");
    assert_matches!(err.kind(), ErrorKind::CannotIndex);
    assert_matches!(
        err.context(),
        ErrorContext::TupleIndex { ty } if ty.to_string() == "{ x: Num }"
    );
}

#[test]
fn no_required_field() {
    let code = r#"
        require_x = |obj| obj.x == 1;
        require_x(#{ y: 2 });
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.main_span().fragment(), "#{ y: 2 }");
    assert_eq!(err.location(), [fn_arg(0)]);
    assert_matches!(
        err.kind(),
        ErrorKind::MissingFields { fields, available_fields }
            if fields.len() == 1 && fields.contains("x") &&
            available_fields.len() == 1 && available_fields.contains("y")
    );
}

#[test]
fn incompatible_field_types() {
    let code = r#"
        require_x = |obj| obj.x == 1;
        require_x(#{ x: (1, 2) })
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.main_span().fragment(), "#{ x: (1, 2) }");
    assert_eq!(err.location(), [fn_arg(0), ErrorLocation::from("x")]);
    assert_matches!(err.context(), ErrorContext::FnCall { .. });
    assert_matches!(
        err.kind(),
        ErrorKind::TypeMismatch(lhs, rhs) if *lhs == Type::NUM && rhs.to_string() == "(Num, Num)"
    );
}

#[test]
fn incompatible_field_types_via_accesses() {
    let code = "|obj| obj.x == 1 && !obj.x";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.main_span().fragment(), "!obj.x");
    assert_eq!(err.location(), []);
    assert_matches!(err.context(), ErrorContext::UnaryOp(_));
    assert_matches!(
        err.kind(),
        ErrorKind::TypeMismatch(lhs, rhs) if *lhs == Type::BOOL && *rhs == Type::NUM
    );
}

#[test]
fn incompatible_field_types_via_fn() {
    let code = r#"
        require_x = |obj| obj.x == 1;
        |obj| { !obj.x; require_x(obj) }
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.main_span().fragment(), "obj");
    assert_eq!(err.location(), [fn_arg(0), ErrorLocation::from("x")]);
    assert_matches!(err.context(), ErrorContext::FnCall { .. });
    assert_matches!(
        err.kind(),
        ErrorKind::TypeMismatch(lhs, rhs) if *lhs == Type::BOOL && *rhs == Type::NUM
    );
}

#[test]
fn incompatible_fields_via_constraints_for_concrete_object() {
    let code = "hash(#{ x: 1, y: || 2 })";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .insert("hash", hash_fn_type())
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.main_span().fragment(), "#{ x: 1, y: || 2 }");
    assert_eq!(err.location(), [fn_arg(0), ErrorLocation::from("y")]);
    assert_matches!(err.context(), ErrorContext::FnCall { .. });
    assert_matches!(
        err.kind(),
        ErrorKind::FailedConstraint { ty, constraint }
            if ty.to_string() == "() -> Num" && constraint.to_string() == "Hash"
    );
}

#[test]
fn incompatible_fields_via_constraints_for_object_constraint() {
    let code = "|obj| { hash(obj); (obj.run)() }";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .insert("hash", hash_fn_type())
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.main_span().fragment(), "(obj.run)()");
    assert_eq!(err.location(), []);
    assert_matches!(err.context(), ErrorContext::FnCall { .. });
    assert_matches!(
        err.kind(),
        ErrorKind::FailedConstraint { ty, constraint }
            if ty.to_string() == "() -> _" && constraint.to_string() == "Hash"
    );
}

#[test]
fn incompatible_fields_via_constraints_for_object_constraint_rev() {
    let code = "|obj| { (obj.run)(); hash(obj) }";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .insert("hash", hash_fn_type())
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.main_span().fragment(), "obj");
    assert_eq!(err.location(), [fn_arg(0), ErrorLocation::from("run")]);
    assert_matches!(err.context(), ErrorContext::FnCall { .. });
    assert_matches!(
        err.kind(),
        ErrorKind::FailedConstraint { ty, .. } if ty.to_string() == "() -> _"
    );
}

#[test]
fn incompatible_fields_in_embedded_obj() {
    let code_samples = &[
        "|obj| { hash(obj); (obj.some.run)() }",
        "|obj| { hash(obj); some = obj.some; (some.run)() }",
        "|obj| { hash(obj); run = obj.some.run; run() }",
        "|obj| { (obj.some.run)(); hash(obj); }",
        "|obj| { some = obj.some; (some.run)(); hash(obj); }",
        "|obj| { run = obj.some.run; run(); hash(obj); }",
    ];

    for &code in code_samples {
        let block = F32Grammar::parse_statements(code).unwrap();
        let err = TypeEnvironment::new()
            .insert("hash", hash_fn_type())
            .process_statements(&block)
            .unwrap_err()
            .single();

        assert_matches!(
            err.kind(),
            ErrorKind::FailedConstraint { ty, .. } if ty.to_string() == "() -> _"
        );
    }
}

#[test]
fn creating_and_consuming_object_in_closure() {
    let code = r#"
        (1, 2, 3).map(|x| #{ x, y: x + 1 }).fold(0, |acc, pt| acc + pt.x / pt.z)
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env
        .insert("map", Prelude::Map)
        .insert("fold", Prelude::Fold)
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_matches!(
        err.kind(),
        ErrorKind::MissingFields { fields, available_fields }
            if fields.len() == 1 && fields.contains("z") &&
            available_fields.len() == 2 && available_fields.contains("y")
    );
}

#[test]
fn folding_to_object_errors() {
    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("INF", Type::NUM)
        .insert("if", Prelude::If)
        .insert("fold", Prelude::Fold);

    let code = r#"
        |xs| xs.fold(#{ min: INF, max: -INF }, |acc, x| #{
            min: if(x < acc.min, x, acc.min),
            max: if(x > acc.max, x, acc.max),
        })
    "#;

    let bogus_code = code.replace("max: -INF", "ma: -INF");
    let block = F32Grammar::parse_statements(bogus_code.as_str()).unwrap();
    let errors = type_env
        .process_with_arithmetic(&NumArithmetic::with_comparisons(), &block)
        .unwrap_err();

    assert!(errors.iter().any(|err| matches!(
        err.kind(),
        ErrorKind::MissingFields { fields, .. }
            if fields.len() == 1 && fields.contains("max")
    )));

    let bogus_code = code.replace("max: if", "ma: if");
    let block = F32Grammar::parse_statements(bogus_code.as_str()).unwrap();
    let err = type_env
        .process_with_arithmetic(&NumArithmetic::with_comparisons(), &block)
        .unwrap_err()
        .single();

    assert_matches!(
        err.kind(),
        ErrorKind::FieldsMismatch { lhs_fields, rhs_fields }
            if rhs_fields.contains("ma") && lhs_fields.contains("max")
    );
}

#[test]
fn repeated_field_in_object_initialization() {
    let code = "obj = #{ x: 1, x: 2 == 3 }; !obj.x";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.main_span().fragment(), "x");
    assert_eq!(err.main_span().location_offset(), 15);
    assert_matches!(err.kind(), ErrorKind::RepeatedField(field) if field == "x");
}

#[test]
fn repeated_field_in_object_destructure() {
    let code = "{ x, x } = #{ x: 1 };";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.main_span().fragment(), "x");
    assert_eq!(err.main_span().location_offset(), 5);
    assert_matches!(err.kind(), ErrorKind::RepeatedField(field) if field == "x");
}
