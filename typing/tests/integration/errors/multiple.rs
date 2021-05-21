//! Tests for multiple errors.

use assert_matches::assert_matches;

use arithmetic_parser::grammars::{NumGrammar, Parse};
use arithmetic_typing::{
    defs::Prelude, error::ErrorKind, Annotated, TupleLen, Type, TypeEnvironment,
};

type F32Grammar = Annotated<NumGrammar<f32>>;

#[test]
fn multiple_independent_errors() {
    let code = r#"
        true + 1;
        (1, 2).filter(|x| x + 1);
        (1, false).map(6);
    "#;
    let code = F32Grammar::parse_statements(code).unwrap();
    let mut type_env: TypeEnvironment = Prelude::iter().collect();
    let errors: Vec<_> = type_env
        .process_statements(&code)
        .unwrap_err()
        .into_iter()
        .collect();

    assert_eq!(errors.len(), 4);

    assert_eq!(*errors[0].main_span().fragment(), "true");
    assert_matches!(
        errors[0].kind(),
        ErrorKind::FailedConstraint { ty, constraint }
            if *ty == Type::BOOL && constraint.to_string() == "Ops"
    );

    assert_eq!(*errors[1].main_span().fragment(), "|x| x + 1");
    assert_eq!(
        *errors[1].root_span().fragment(),
        "(1, 2).filter(|x| x + 1)"
    );
    assert_matches!(
        errors[1].kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::BOOL && *rhs == Type::NUM
    );

    assert_eq!(*errors[2].main_span().fragment(), "false");
    assert_eq!(*errors[2].root_span().fragment(), "(1, false).map(6)");
    assert_matches!(
        errors[2].kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::NUM && *rhs == Type::BOOL
    );

    assert_eq!(*errors[3].main_span().fragment(), "6");
    assert_eq!(*errors[3].root_span().fragment(), "(1, false).map(6)");
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
    let code = F32Grammar::parse_statements(code).unwrap();

    let mut type_env: TypeEnvironment = Prelude::iter().collect();
    let errors: Vec<_> = type_env
        .process_statements(&code)
        .unwrap_err()
        .into_iter()
        .collect();

    assert_eq!(errors.len(), 4);

    assert_eq!(*errors[0].main_span().fragment(), "x");
    assert_matches!(
        errors[0].kind(),
        ErrorKind::UndefinedVar(id) if id == "x"
    );

    assert_eq!(*errors[1].main_span().fragment(), "(x == 3)");
    assert_matches!(
        errors[1].kind(),
        ErrorKind::FailedConstraint { ty, constraint }
            if *ty == Type::BOOL && constraint.to_string() == "Ops"
    );

    assert_eq!(*errors[2].main_span().fragment(), "false");
    assert_eq!(
        *errors[2].root_span().fragment(),
        "(1, false).map(|x| x + 1)"
    );
    assert_matches!(
        errors[2].kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::NUM && *rhs == Type::BOOL
    );

    assert_eq!(*errors[3].main_span().fragment(), "(x, y, z)");
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
    let code = F32Grammar::parse_statements(code).unwrap();

    let mut type_env: TypeEnvironment = Prelude::iter().collect();
    let errors: Vec<_> = type_env
        .process_statements(&code)
        .unwrap_err()
        .into_iter()
        .collect();

    assert_eq!(errors.len(), 3);

    assert_eq!(*errors[0].main_span().fragment(), "|x| x + 1");
    assert_eq!(*errors[0].root_span().fragment(), "xs.filter(|x| x + 1)");
    assert_matches!(
        errors[0].kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::BOOL && *rhs == Type::NUM
    );

    assert_eq!(*errors[1].main_span().fragment(), "true");
    assert_eq!(*errors[1].root_span().fragment(), "bogus(true)");
    assert_matches!(
        errors[1].kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::NUM && *rhs == Type::BOOL
    );

    assert_eq!(*errors[2].main_span().fragment(), "(1, 2, 3)");
    assert_eq!(*errors[2].root_span().fragment(), "(1, 2, 3).bogus()");
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
    let code = F32Grammar::parse_statements(code).unwrap();

    let mut type_env: TypeEnvironment = Prelude::iter().collect();
    let errors: Vec<_> = type_env
        .process_statements(&code)
        .unwrap_err()
        .into_iter()
        .collect();

    assert_eq!(errors.len(), 4);

    assert_eq!(*errors[0].main_span().fragment(), "xs.filter(|x| x, 1)");
    assert_matches!(
        errors[0].kind(),
        ErrorKind::TupleLenMismatch { lhs, rhs, .. }
            if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(3)
    );

    assert_eq!(*errors[1].main_span().fragment(), "1");
    assert_eq!(*errors[1].root_span().fragment(), "bogus(1, 2)");
    assert_matches!(
        errors[1].kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::BOOL && *rhs == Type::NUM
    );

    assert_eq!(*errors[2].main_span().fragment(), "2");
    assert_eq!(*errors[2].root_span().fragment(), "bogus(1, 2)");
    assert_matches!(
        errors[2].kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::BOOL && *rhs == Type::NUM
    );

    assert_eq!(*errors[3].main_span().fragment(), "bogus(1, 2) == (3,)");
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
    let code = F32Grammar::parse_statements(code).unwrap();

    let mut type_env: TypeEnvironment = Prelude::iter().collect();
    let errors: Vec<_> = type_env
        .process_statements(&code)
        .unwrap_err()
        .into_iter()
        .collect();

    assert_eq!(errors.len(), 2);

    assert_eq!(*errors[0].main_span().fragment(), "(1, 2, 3).map()");
    assert_matches!(
        errors[0].kind(),
        ErrorKind::TupleLenMismatch { lhs, rhs, .. }
            if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(1)
    );

    assert_eq!(*errors[1].main_span().fragment(), "xs == (false, true)");
    assert_matches!(
        errors[1].kind(),
        ErrorKind::TupleLenMismatch { lhs, rhs, .. }
            if *lhs == TupleLen::from(3) && *rhs == TupleLen::from(2)
    );
}
