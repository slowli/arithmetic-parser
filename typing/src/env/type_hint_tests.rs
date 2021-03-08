//! Tests with explicit type hints.

use assert_matches::assert_matches;

use super::{
    tests::{map_fn_type, NumGrammar},
    *,
};
use crate::TupleLength;
use arithmetic_parser::grammars::{Parse, Typed};

fn assert_incompatible_types(err: &TypeError, first: &ValueType, second: &ValueType) {
    let (x, y) = match err {
        TypeError::IncompatibleTypes(x, y) => (x, y),
        _ => panic!("Unexpected error type: {:?}", err),
    };
    assert!(
        (x == first && y == second) || (x == second && y == first),
        "Unexpected incompatible types: {:?}, expected: {:?}",
        (x, y),
        (first, second)
    );
}

#[test]
fn type_hint_within_tuple() {
    let code = "foo = |x, fun| { (y: Num, z) = x; fun(y + z) };";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<T: ?Lin>((Num, Num), fn(Num) -> T) -> T"
    );
}

#[test]
fn type_hint_in_fn_arg() {
    let code = r#"
        foo = |tuple: (Num, _), fun| {
            (x, flag) = tuple;
            flag && fun() == x
        };
    "#;
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn((Num, Bool), fn() -> Num) -> Bool"
    );
}

#[test]
fn contradicting_type_hint() {
    let code = "x: (Num, _) = (1, 2, 3);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_matches!(
        err.extra,
        TypeError::IncompatibleLengths(TupleLength::Exact(2), TupleLength::Exact(3))
    );
}

#[test]
fn valid_type_hint_with_slice() {
    let code = "x: [Num; _] = (1, 2, 3);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(type_env["x"].to_string(), "(Num, Num, Num)");
}

#[test]
fn contradicting_type_hint_with_slice() {
    let code = "x: [Num; _] = (1, 2 == 3);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_incompatible_types(&err.extra, &ValueType::Number, &ValueType::Bool);
}

#[test]
fn valid_type_hint_with_fn_arg() {
    let code = "foo = |xs, map_fn: fn(Num) -> _| xs.map(map_fn);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", map_fn_type().into());
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<const N; T: ?Lin>([Num; N], fn(Num) -> T) -> [T; N]"
    );
}

#[test]
fn invalid_type_hint_with_fn_arg() {
    let code = "foo = |xs, map_fn: fn(_, _) -> _| xs.map(map_fn);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", map_fn_type().into());
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_matches!(
        err.extra,
        TypeError::ArgLenMismatch {
            expected: 1,
            actual: 2
        }
    );
}

#[test]
fn valid_type_hint_with_fn_declaration() {
    let code = "foo: fn(Num) -> _ = |x| x + 3;";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", map_fn_type().into());
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(type_env["foo"].to_string(), "fn(Num) -> Num");
}

#[test]
fn invalid_type_hint_with_fn_declaration() {
    let code = "foo: fn(_) -> Bool = |x| x + 3;";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", map_fn_type().into());
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_incompatible_types(&err.extra, &ValueType::Number, &ValueType::Bool);
}

#[test]
fn widening_type_hint_with_generic_slice_arg() {
    // Without type hint on `xs` it would be interpreted as a number.
    let code = "foo = |xs: [_; _]| xs + 1;";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", map_fn_type().into());
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<const N; T>([T; N]) -> [T; N]"
    );
}

#[test]
fn widening_type_hint_with_slice_arg() {
    // Without type hint on `xs` it would be interpreted as a number.
    let code = "foo = |xs: [Num; _]| xs + 1;";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", map_fn_type().into());
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<const N>([Num; N]) -> [Num; N]"
    );
}

#[test]
fn unsupported_type_param_in_generic_fn() {
    let code = "identity: fn<Arg>(Arg) -> Arg = |x| x;";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_matches!(err.extra, TypeError::UnsupportedParam);
}

#[test]
#[ignore]
// TODO: type param indexes are assigned incorrectly.
// TODO: to use such a function, it's necessary to know how to unify parametric fns
fn widening_fn_arg_via_type_hint() {
    let code = r#"
        foo = |xs, map_fn: fn<Z>(Z) -> Z| {
            map_fn(xs.map(map_fn))
        };
    "#;
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", map_fn_type().into());
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<const N; T>([T; N], fn<U>(U) -> U) -> [T; N]"
    );
}

#[test]
fn unsupported_const_param_in_generic_fn() {
    let code = "identity: fn<const N>([Num; N]) -> [Num; N] = |x| x;";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_matches!(err.extra, TypeError::UnsupportedParam);
}

#[test]
fn fn_narrowed_via_type_hint() {
    let code = r#"
        identity: fn(Num) -> _ = |x| x;
        identity((1, 2));
    "#;
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_eq!(type_env["identity"].to_string(), "fn(Num) -> Num");
    assert_incompatible_types(
        &err.extra,
        &ValueType::Number,
        &ValueType::Tuple(vec![ValueType::Number, ValueType::Number]),
    )
}

#[test]
fn fn_incorrectly_narrowed_via_type_hint() {
    let code = "identity: fn(Num) -> Bool = |x| x;";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_incompatible_types(&err.extra, &ValueType::Number, &ValueType::Bool);
}

#[test]
fn fn_instantiated_via_type_hint() {
    let code = r#"
        identity: fn(_) -> _ = |x| x;
        identity(5) == 5;
    "#;
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(type_env["identity"].to_string(), "fn(Num) -> Num");
}
