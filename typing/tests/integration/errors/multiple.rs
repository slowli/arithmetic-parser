//! Tests for multiple errors.

use arithmetic_parser::grammars::{NumGrammar, Parse};
use arithmetic_typing::{
    defs::Prelude, error::ErrorKind, Annotated, TupleLen, Type, TypeEnvironment,
};
use assert_matches::assert_matches;

type F32Grammar = Annotated<NumGrammar<f32>>;

#[test]
fn multiple_independent_errors() {
    let code = r#"
        true + 1;
        (1, 2).filter(|x| x + 1);
        (1, false).map(6);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env: TypeEnvironment = Prelude::iter().collect();
    let errors: Vec<_> = type_env
        .process_statements(&block)
        .unwrap_err()
        .into_iter()
        .collect();

    assert_eq!(errors.len(), 4);

    assert_eq!(errors[0].main_location().span(code), "true");
    assert_matches!(
        errors[0].kind(),
        ErrorKind::FailedConstraint { ty, constraint }
            if *ty == Type::BOOL && constraint.to_string() == "Ops"
    );

    assert_eq!(errors[1].main_location().span(code), "|x| x + 1");
    assert_eq!(
        errors[1].root_location().span(code),
        "(1, 2).filter(|x| x + 1)"
    );
    assert_matches!(
        errors[1].kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::BOOL && *rhs == Type::NUM
    );

    assert_eq!(errors[2].main_location().span(code), "false");
    assert_eq!(errors[2].root_location().span(code), "(1, false).map(6)");
    assert_matches!(
        errors[2].kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::NUM && *rhs == Type::BOOL
    );

    assert_eq!(errors[3].main_location().span(code), "6");
    assert_eq!(errors[3].root_location().span(code), "(1, false).map(6)");
    assert_matches!(
        errors[3].kind(),
        ErrorKind::TypeMismatch(Type::Function(_), rhs)
            if *rhs == Type::NUM
    );
}

#[test]
fn recovery_after_error() {
    let code = r#"
         1 + (x == 3) == 2;
         (x, y, z) = (1, false).map(|x| x + 1) + 3;
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env: TypeEnvironment = Prelude::iter().collect();
    let errors: Vec<_> = type_env
        .process_statements(&block)
        .unwrap_err()
        .into_iter()
        .collect();

    assert_eq!(errors.len(), 4);

    assert_eq!(errors[0].main_location().span(code), "x");
    assert_matches!(
        errors[0].kind(),
        ErrorKind::UndefinedVar(id) if id == "x"
    );

    assert_eq!(errors[1].main_location().span(code), "(x == 3)");
    assert_matches!(
        errors[1].kind(),
        ErrorKind::FailedConstraint { ty, constraint }
            if *ty == Type::BOOL && constraint.to_string() == "Ops"
    );

    assert_eq!(errors[2].main_location().span(code), "false");
    assert_eq!(
        errors[2].root_location().span(code),
        "(1, false).map(|x| x + 1)"
    );
    assert_matches!(
        errors[2].kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::NUM && *rhs == Type::BOOL
    );

    assert_eq!(errors[3].main_location().span(code), "(x, y, z)");
    assert_matches!(
        errors[3].kind(),
        ErrorKind::TupleLenMismatch { lhs, rhs, .. }
            if *lhs == TupleLen::from(3) && *rhs == TupleLen::from(2)
    );
}

#[test]
fn recovery_in_fn_definition() {
    let code = r#"
        bogus = |...xs| xs.filter(|x| x + 1);
        bogus(1, 2, 3) == (2, 3);
        bogus(true);
        (1, 2, 3).bogus();
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env: TypeEnvironment = Prelude::iter().collect();
    let errors: Vec<_> = type_env
        .process_statements(&block)
        .unwrap_err()
        .into_iter()
        .collect();

    assert_eq!(errors.len(), 3);

    assert_eq!(errors[0].main_location().span(code), "|x| x + 1");
    assert_eq!(errors[0].root_location().span(code), "xs.filter(|x| x + 1)");
    assert_matches!(
        errors[0].kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::BOOL && *rhs == Type::NUM
    );

    assert_eq!(errors[1].main_location().span(code), "true");
    assert_eq!(errors[1].root_location().span(code), "bogus(true)");
    assert_matches!(
        errors[1].kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::NUM && *rhs == Type::BOOL
    );

    assert_eq!(errors[2].main_location().span(code), "(1, 2, 3)");
    assert_eq!(errors[2].root_location().span(code), "(1, 2, 3).bogus()");
    assert_matches!(
        errors[2].kind(),
        ErrorKind::TypeMismatch(lhs, Type::Tuple(_)) if *lhs == Type::NUM
    );
}

#[test]
fn recovery_in_mangled_fn_definition() {
    let code = r#"
        bogus = |...xs| xs.filter(|x| x, 1);
        bogus(1, 2) == (3,);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env: TypeEnvironment = Prelude::iter().collect();
    let errors: Vec<_> = type_env
        .process_statements(&block)
        .unwrap_err()
        .into_iter()
        .collect();

    assert_eq!(errors.len(), 4);

    assert_eq!(errors[0].main_location().span(code), "xs.filter(|x| x, 1)");
    assert_matches!(
        errors[0].kind(),
        ErrorKind::TupleLenMismatch { lhs, rhs, .. }
            if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(3)
    );

    assert_eq!(errors[1].main_location().span(code), "1");
    assert_eq!(errors[1].root_location().span(code), "bogus(1, 2)");
    assert_matches!(
        errors[1].kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::BOOL && *rhs == Type::NUM
    );

    assert_eq!(errors[2].main_location().span(code), "2");
    assert_eq!(errors[2].root_location().span(code), "bogus(1, 2)");
    assert_matches!(
        errors[2].kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::BOOL && *rhs == Type::NUM
    );

    assert_eq!(errors[3].main_location().span(code), "bogus(1, 2) == (3,)");
    assert_matches!(
        errors[3].kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::BOOL && *rhs == Type::NUM
    );
}

#[test]
fn recovery_in_fn_with_insufficient_args() {
    let code = r#"
        xs = (1, 2, 3).map();
        xs == (false, true);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env: TypeEnvironment = Prelude::iter().collect();
    let errors: Vec<_> = type_env
        .process_statements(&block)
        .unwrap_err()
        .into_iter()
        .collect();

    assert_eq!(errors.len(), 2);

    assert_eq!(errors[0].main_location().span(code), "(1, 2, 3).map()");
    assert_matches!(
        errors[0].kind(),
        ErrorKind::TupleLenMismatch { lhs, rhs, .. }
            if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(1)
    );

    assert_eq!(errors[1].main_location().span(code), "xs == (false, true)");
    assert_matches!(
        errors[1].kind(),
        ErrorKind::TupleLenMismatch { lhs, rhs, .. }
            if *lhs == TupleLen::from(3) && *rhs == TupleLen::from(2)
    );
}
