//! Tests with explicit type annotations.

use assert_matches::assert_matches;

use super::{
    tests::{assert_incompatible_types, F32Grammar},
    *,
};
use crate::{Num, Prelude, TupleLength};
use arithmetic_parser::grammars::Parse;

#[test]
fn type_hint_within_tuple() {
    let code = "foo = |x, fun| { (y: Num, z) = x; fun(y + z) };";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<T>((Num, Num), fn(Num) -> T) -> T"
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
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn((Num, Bool), fn() -> Num) -> Bool"
    );
}

#[test]
fn contradicting_type_hint() {
    let code = "x: (Num, _) = (1, 2, 3);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();

    assert_matches!(
        err.kind(),
        TypeErrorKind::IncompatibleLengths(TupleLength::Exact(2), TupleLength::Exact(3))
    );
}

#[test]
fn valid_type_hint_with_slice() {
    let code = "x: [Num; _] = (1, 2, 3);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["x"].to_string(), "(Num, Num, Num)");
}

#[test]
fn contradicting_type_hint_with_slice() {
    let code = "x: [Num; _] = (1, 2 == 3);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();

    assert_incompatible_types(&err.kind(), &ValueType::Lit(Num), &ValueType::Bool);
}

#[test]
fn valid_type_hint_with_fn_arg() {
    let code = "foo = |xs, map_fn: fn(Num) -> _| xs.map(map_fn);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", Prelude::map_type().into());
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<const N; T>([Num; N], fn(Num) -> T) -> [T; N]"
    );
}

#[test]
fn invalid_type_hint_with_fn_arg() {
    let code = "foo = |xs, map_fn: fn(_, _) -> _| xs.map(map_fn);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", Prelude::map_type().into());
    let err = type_env.process_statements(&block).unwrap_err();

    assert_matches!(
        err.kind(),
        TypeErrorKind::ArgLenMismatch {
            expected: 1,
            actual: 2
        }
    );
}

#[test]
fn valid_type_hint_with_fn_declaration() {
    let code = "foo: fn(Num) -> _ = |x| x + 3;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", Prelude::map_type().into());
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["foo"].to_string(), "fn(Num) -> Num");
}

#[test]
fn invalid_type_hint_with_fn_declaration() {
    let code = "foo: fn(_) -> Bool = |x| x + 3;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", Prelude::map_type().into());
    let err = type_env.process_statements(&block).unwrap_err();

    assert_incompatible_types(&err.kind(), &ValueType::Lit(Num), &ValueType::Bool);
}

#[test]
fn widening_type_hint_with_generic_slice_arg() {
    // Without a type annotation on `xs` it would be interpreted as a number.
    let code = "foo = |xs: [_; _]| xs + 1;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", Prelude::map_type().into());
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<const N; T: Lin>([T; N]) -> [T; N]"
    );
}

#[test]
fn widening_type_hint_with_slice_arg() {
    // Without a type annotation on `xs` it would be interpreted as a number.
    let code = "foo = |xs: [Num; _]| xs + 1;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", Prelude::map_type().into());
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<const N>([Num; N]) -> [Num; N]"
    );
}

#[test]
fn unsupported_type_param_in_generic_fn() {
    let code = "identity: fn<Arg>(Arg) -> Arg = |x| x;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();

    assert_eq!(*err.span().fragment(), "fn<Arg>(Arg) -> Arg");
    assert_matches!(err.kind(), TypeErrorKind::UnsupportedParam);
}

#[test]
fn unsupported_const_param_in_generic_fn() {
    let code = "identity: fn<const N>([Num; N]) -> [Num; N] = |x| x;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();

    assert_eq!(*err.span().fragment(), "fn<const N>([Num; N]) -> [Num; N]");
    assert_matches!(err.kind(), TypeErrorKind::UnsupportedParam);
}

#[test]
fn fn_narrowed_via_type_hint() {
    let code = r#"
        identity: fn(Num) -> _ = |x| x;
        identity((1, 2));
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();

    assert_eq!(type_env["identity"].to_string(), "fn(Num) -> Num");
    assert_incompatible_types(
        &err.kind(),
        &ValueType::Lit(Num),
        &ValueType::Tuple(vec![ValueType::Lit(Num); 2]),
    )
}

#[test]
fn fn_incorrectly_narrowed_via_type_hint() {
    let code = "identity: fn(Num) -> Bool = |x| x;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();

    assert_incompatible_types(&err.kind(), &ValueType::Lit(Num), &ValueType::Bool);
}

#[test]
fn fn_instantiated_via_type_hint() {
    let code = r#"
        identity: fn(_) -> _ = |x| x;
        identity(5) == 5;
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["identity"].to_string(), "fn(Num) -> Num");
}

#[test]
fn assigning_to_dynamically_sized_slice() {
    let code = "slice: [Num] = (1, 2, 3);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["slice"].to_string(), "[Num]");
}

#[test]
fn assigning_to_a_slice_and_then_narrowing() {
    // TODO: maybe should work without arg type annotation
    let code = r#"
        // The arg type annotation is required because otherwise `xs` type will be set
        // to `[Num]` by unifying it with the type var.
        slice_fn = |xs: [Num; _]| {
            _unused: [Num] = xs;
            (x, y, z) = xs;
            x + y * z
        };
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["slice_fn"].to_string(),
        "fn((Num, Num, Num)) -> Num"
    );
}
