//! Tests error handling basics.

use assert_matches::assert_matches;

use super::F32Grammar;
use crate::{error::TypeErrorKind, Prelude, TypeEnvironment, ValueType};
use arithmetic_parser::grammars::Parse;

#[test]
fn vars_are_not_assigned_beyond_first_error() {
    let code = r#"
        x = (1, 2);
        y = x.map(x);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "x");
    assert_matches!(err.kind(), TypeErrorKind::TypeMismatch(..));
    assert_eq!(type_env["x"], ValueType::slice(ValueType::NUM, 2));
    assert!(type_env.get("y").is_none());
}

#[test]
fn vars_are_not_redefined_beyond_first_error() {
    let code = r#"
        x = (1, 2);
        x = x.map(x);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "x");
    assert_matches!(err.kind(), TypeErrorKind::TypeMismatch(..));
    assert_eq!(type_env["x"], ValueType::slice(ValueType::NUM, 2));
}

#[test]
fn vars_are_not_assigned_beyond_first_error_in_expr() {
    let code = r#"
        x = (1, 2);
        x.map(x);
        y = (3, 4);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "x");
    assert_matches!(err.kind(), TypeErrorKind::TypeMismatch(..));
    assert_eq!(type_env["x"], ValueType::slice(ValueType::NUM, 2));
    assert!(type_env.get("y").is_none());
}

#[test]
fn errors_in_inner_scopes_are_handled_adequately() {
    let code = r#"
        x = (1, 2);
        y = { bogus = 5; x.map(bogus) };
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "bogus");
    assert_matches!(err.kind(), TypeErrorKind::TypeMismatch(..));
    assert_eq!(type_env["x"], ValueType::slice(ValueType::NUM, 2));
    assert!(type_env.get("y").is_none());
    assert!(type_env.get("bogus").is_none());
}

#[test]
fn errors_in_functions_are_handled_adequately() {
    let code = r#"
        x = (1, 2);
        y = || { bogus = 5; x.map(bogus) };
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "bogus");
    assert_matches!(err.kind(), TypeErrorKind::TypeMismatch(..));
    assert_eq!(type_env["x"], ValueType::slice(ValueType::NUM, 2));
    assert!(type_env.get("y").is_none());
    assert!(type_env.get("bogus").is_none());
}
