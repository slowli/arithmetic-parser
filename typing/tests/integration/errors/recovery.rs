//! Tests error recovery basics.

use assert_matches::assert_matches;

use std::collections::HashSet;

use arithmetic_parser::grammars::Parse;
use arithmetic_typing::{
    arith::NumArithmetic,
    defs::{Assertions, Prelude},
    error::ErrorKind,
    Type, TypeEnvironment,
};

use crate::{ErrorsExt, F32Grammar};

#[test]
fn vars_are_not_assigned_beyond_first_error() {
    let code = r#"
        x = (1, 2);
        y = x.map(x);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);

    let errors = type_env.process_statements(&block).unwrap_err();
    assert_eq!(errors.first_failing_statement(), 1);

    let err = errors.single();
    assert_eq!(*err.main_span().fragment(), "x");
    assert_eq!(*err.root_span().fragment(), "x.map(x)");
    assert_matches!(err.kind(), ErrorKind::TypeMismatch(..));
    assert_eq!(type_env["x"], Type::slice(Type::NUM, 2));
    assert!(type_env.get("y").is_none());
}

#[test]
fn first_failing_statement_is_not_overwritten() {
    let code = "x = (1, 2); !x; x = x.map(x);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);

    let errors = type_env.process_statements(&block).unwrap_err();
    assert_eq!(errors.first_failing_statement(), 1);
}

#[test]
fn first_failing_statement_on_error_in_return_value() {
    let code = "x = (1, 2); x; x.map(x)";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);

    let errors = type_env.process_statements(&block).unwrap_err();
    assert_eq!(errors.first_failing_statement(), 2);
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

    let errors = type_env.process_statements(&block).unwrap_err();
    assert_eq!(errors.first_failing_statement(), 1);

    let err = errors.single();
    assert_eq!(*err.main_span().fragment(), "x");
    assert_matches!(err.kind(), ErrorKind::TypeMismatch(..));
    assert_eq!(type_env["x"], Type::slice(Type::NUM, 2));
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

    assert_eq!(*err.main_span().fragment(), "x");
    assert_matches!(err.kind(), ErrorKind::TypeMismatch(..));
    assert_eq!(type_env["x"], Type::slice(Type::NUM, 2));
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

    let errors = type_env.process_statements(&block).unwrap_err();
    assert_eq!(errors.first_failing_statement(), 1);

    let err = errors.single();
    assert_eq!(*err.main_span().fragment(), "bogus");
    assert_eq!(*err.root_span().fragment(), "x.map(bogus)");
    assert_matches!(err.kind(), ErrorKind::TypeMismatch(..));
    assert_eq!(type_env["x"], Type::slice(Type::NUM, 2));
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

    let errors = type_env.process_statements(&block).unwrap_err();
    assert_eq!(errors.first_failing_statement(), 1);

    let err = errors.single();
    assert_eq!(*err.main_span().fragment(), "bogus");
    assert_matches!(err.kind(), ErrorKind::TypeMismatch(..));
    assert_eq!(type_env["x"], Type::slice(Type::NUM, 2));
    assert!(type_env.get("y").is_none());
    assert!(type_env.get("bogus").is_none());
}

#[test]
fn recovery_after_bogus_annotations() {
    let code = r#"
        fun: for<'T: Bogus, 'U: Lin> ('T) -> () = |x| assert(x > 1 && x < 10);
        other_fun = |x: 'T| x + 1;
        other_fun((4, 5));
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("assert", Assertions::Assert);
    let errors = type_env
        .process_with_arithmetic(&NumArithmetic::with_comparisons(), &block)
        .unwrap_err();

    let expected_messages = &[
        "2:22: Error instantiating type from annotation: Unknown constraint `Bogus`",
        "2:30: Error instantiating type from annotation: Unused type param `U`",
        "2:14: Params in declared function types are not supported yet",
        "3:25: Error instantiating type from annotation: \
        Type param `T` is not scoped by function definition",
        "4:19: Type `(Num, Num)` is not assignable to type `Num`",
    ];
    let expected_messages: HashSet<_> = expected_messages.iter().copied().collect();
    let actual_messages: Vec<_> = errors.iter().map(ToString::to_string).collect();
    let actual_messages: HashSet<_> = actual_messages.iter().map(String::as_str).collect();
    assert_eq!(actual_messages, expected_messages);
}
