//! Tests for multiple errors.

use assert_matches::assert_matches;

use arithmetic_parser::grammars::{NumGrammar, Parse, Typed};
use arithmetic_typing::{
    arith::NumConstraints, error::TypeErrorKind, Annotated, Prelude, TupleLen, Type,
    TypeEnvironment,
};

type F32Grammar = Typed<Annotated<NumGrammar<f32>>>;

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

    assert_eq!(*errors[0].span().fragment(), "true");
    assert_matches!(
        errors[0].kind(),
        TypeErrorKind::FailedConstraint { ty, constraint }
            if *ty == Type::BOOL && *constraint == NumConstraints::Ops
    );

    assert_eq!(*errors[1].span().fragment(), "(1, 2).filter(|x| x + 1)");
    assert_matches!(
        errors[1].kind(),
        TypeErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::BOOL && *rhs == Type::NUM
    );

    assert_eq!(*errors[2].span().fragment(), "(1, false).map(6)");
    assert_matches!(
        errors[2].kind(),
        TypeErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::NUM && *rhs == Type::BOOL
    );

    assert_eq!(*errors[3].span().fragment(), "(1, false).map(6)");
    assert_matches!(
        errors[3].kind(),
        TypeErrorKind::TypeMismatch(Type::Function(_), rhs)
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

    assert_eq!(*errors[0].span().fragment(), "x");
    assert_matches!(
        errors[0].kind(),
        TypeErrorKind::UndefinedVar(id) if id == "x"
    );

    assert_eq!(*errors[1].span().fragment(), "(x == 3)");
    assert_matches!(
        errors[1].kind(),
        TypeErrorKind::FailedConstraint { ty, constraint }
            if *ty == Type::BOOL && *constraint == NumConstraints::Ops
    );

    assert_eq!(*errors[2].span().fragment(), "(1, false).map(|x| x + 1)");
    assert_matches!(
        errors[2].kind(),
        TypeErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::NUM && *rhs == Type::BOOL
    );

    assert!(errors[3].span().fragment().starts_with("(x, y, z) ="));
    assert_matches!(
        errors[3].kind(),
        TypeErrorKind::TupleLenMismatch { lhs, rhs, .. }
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

    assert_eq!(*errors[0].span().fragment(), "xs.filter(|x| x + 1)");
    assert_matches!(
        errors[0].kind(),
        TypeErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::BOOL && *rhs == Type::NUM
    );

    assert_eq!(*errors[1].span().fragment(), "bogus(true)");
    assert_matches!(
        errors[1].kind(),
        TypeErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::NUM && *rhs == Type::BOOL
    );

    assert_eq!(*errors[2].span().fragment(), "(1, 2, 3).bogus()");
    assert_matches!(
        errors[2].kind(),
        TypeErrorKind::TypeMismatch(lhs, Type::Tuple(_)) if *lhs == Type::NUM
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

    assert_eq!(*errors[0].span().fragment(), "xs.filter(|x| x, 1)");
    assert_matches!(
        errors[0].kind(),
        TypeErrorKind::TupleLenMismatch { lhs, rhs, .. }
            if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(3)
    );

    assert_eq!(*errors[1].span().fragment(), "bogus(1, 2)");
    assert_matches!(
        errors[1].kind(),
        TypeErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::BOOL && *rhs == Type::NUM
    );

    assert_eq!(*errors[2].span().fragment(), "bogus(1, 2)");
    assert_matches!(
        errors[2].kind(),
        TypeErrorKind::TypeMismatch(lhs, rhs)
            if *lhs == Type::BOOL && *rhs == Type::NUM
    );

    assert_eq!(*errors[3].span().fragment(), "bogus(1, 2) == (3,)");
    assert_matches!(
        errors[3].kind(),
        TypeErrorKind::TypeMismatch(lhs, rhs)
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

    assert_eq!(*errors[0].span().fragment(), "(1, 2, 3).map()");
    assert_matches!(
        errors[0].kind(),
        TypeErrorKind::TupleLenMismatch { lhs, rhs, .. }
            if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(1)
    );

    assert_eq!(*errors[1].span().fragment(), "xs == (false, true)");
    assert_matches!(
        errors[1].kind(),
        TypeErrorKind::TupleLenMismatch { lhs, rhs, .. }
            if *lhs == TupleLen::from(3) && *rhs == TupleLen::from(2)
    );
}
