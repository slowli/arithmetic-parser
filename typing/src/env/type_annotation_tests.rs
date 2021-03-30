//! Tests with explicit type annotations.

use assert_matches::assert_matches;

use super::{
    tests::{assert_incompatible_types, zip_fn_type, F32Grammar},
    *,
};
use crate::{Prelude, TupleLen, TupleLenMismatchContext};
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
        TypeErrorKind::TupleLenMismatch {
            lhs: TupleLen::Exact(2),
            rhs: TupleLen::Exact(3),
            context: TupleLenMismatchContext::Assignment,
        }
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

    assert_incompatible_types(&err.kind(), &ValueType::NUM, &ValueType::BOOL);
}

#[test]
fn valid_type_hint_with_fn_arg() {
    let code = "foo = |xs, map_fn: fn(Num) -> _| xs.map(map_fn);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::map_type().into());
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<len N; T>([Num; N], fn(Num) -> T) -> [T; N]"
    );
}

#[test]
fn invalid_type_hint_with_fn_arg() {
    let code = "foo = |xs, map_fn: fn(_, _) -> _| xs.map(map_fn);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::map_type().into());
    let err = type_env.process_statements(&block).unwrap_err();

    assert_matches!(
        err.kind(),
        TypeErrorKind::TupleLenMismatch {
            lhs: TupleLen::Exact(2),
            rhs: TupleLen::Exact(1),
            context: TupleLenMismatchContext::FnArgs,
        }
    );
}

#[test]
fn valid_type_hint_with_fn_declaration() {
    let code = "foo: fn(Num) -> _ = |x| x + 3;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::map_type().into());
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["foo"].to_string(), "fn(Num) -> Num");
}

#[test]
fn invalid_type_hint_with_fn_declaration() {
    let code = "foo: fn(_) -> Bool = |x| x + 3;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::map_type().into());
    let err = type_env.process_statements(&block).unwrap_err();

    assert_incompatible_types(&err.kind(), &ValueType::NUM, &ValueType::BOOL);
}

#[test]
fn widening_type_hint_with_generic_slice_arg() {
    // Without a type annotation on `xs` it would be interpreted as a number.
    let code = "foo = |xs: [_; _]| xs + 1;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::map_type().into());
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<len N; T: Lin>([T; N]) -> [T; N]"
    );
}

#[test]
fn widening_type_hint_with_slice_arg() {
    // Without a type annotation on `xs` it would be interpreted as a number.
    let code = "foo = |xs: [Num; _]| xs + 1;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::map_type().into());
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<len N>([Num; N]) -> [Num; N]"
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
    let code = "identity: fn<len N>([Num; N]) -> [Num; N] = |x| x;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();

    assert_eq!(*err.span().fragment(), "fn<len N>([Num; N]) -> [Num; N]");
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
        &ValueType::NUM,
        &ValueType::Tuple(vec![ValueType::NUM; 2].into()),
    )
}

#[test]
fn fn_incorrectly_narrowed_via_type_hint() {
    let code = "identity: fn(Num) -> Bool = |x| x;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();

    assert_incompatible_types(&err.kind(), &ValueType::NUM, &ValueType::BOOL);
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
    assert!(type_env["identity"].is_concrete());
}

#[test]
fn assigning_to_dynamically_sized_slice() {
    let code = "slice: [Num] = (1, 2, 3);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["slice"].to_string(), "[Num; _]");
    assert!(!type_env["slice"].is_concrete());

    let bogus_code = "(x, y) = slice;";
    let bogus_block = F32Grammar::parse_statements(bogus_code).unwrap();
    let err = type_env.process_statements(&bogus_block).unwrap_err();

    assert_matches!(err.kind(), TypeErrorKind::TupleLenMismatch { .. });
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

#[test]
fn adding_dynamically_typed_slices() {
    let code = r#"
        x: [Num] = (1, 2);
        y: [Num] = (3, 4, 5);
        x + y
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();

    assert_matches!(err.kind(), TypeErrorKind::TupleLenMismatch { .. });
}

#[test]
fn unifying_dynamic_slices_error() {
    let code = r#"
        x: [Num] = (1, 2);
        y: [Num] = (3, 4, 5);
        x.zip_with(y)
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("zip_with", zip_fn_type().into());
    let err = type_env.process_statements(&block).unwrap_err();

    assert_matches!(err.kind(), TypeErrorKind::TupleLenMismatch { .. });
}

#[test]
fn unifying_dynamic_slices_in_arithmetic_op_error() {
    let code = r#"
        xs: [Num] = (1, 2);
        ys: [Num] = (3, 4, 5);
        xs + xs; // should work (dyn lengths are the same)
        xs + xs.map(|x| x - 1); // should work (`map` retains length)
        xs + ys
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::map_type().into());
    let err = type_env.process_statements(&block).unwrap_err();

    assert_eq!(*err.span().fragment(), "xs + ys");
    assert_matches!(err.kind(), TypeErrorKind::TupleLenMismatch { .. });
}

#[test]
fn unifying_dynamic_slices_in_fn_error() {
    let code = r#"
        // This should work because the dynamic length is the same.
        xs = (1, 2, 3).filter(|x| x != 2);
        xs.zip_with(xs);
        // This one shouldn't because dynamic lengths are different.
        xs.zip_with(xs.filter(|x| x == 1));
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("zip_with", zip_fn_type().into())
        .insert("filter", Prelude::filter_type().into());
    let err = type_env.process_statements(&block).unwrap_err();

    assert_eq!(*err.span().fragment(), "xs.zip_with(xs.filter(|x| x == 1))");
    assert_matches!(err.kind(), TypeErrorKind::TupleLenMismatch { .. });
}

#[test]
fn unifying_tuples_with_middle() {
    let code = r#"
        xs: (Num, ...[Num; _]) = (1, 2, 3);
        ys: (...[_; _], _) = (4, 5, |x| x);
        zs: (_, ...[Num], _) = (6, 7, 8, 9);

        // Check basic destructuring.
        (...xs_head: Num, _) = xs;
        (_, _, _: fn(_) -> _) = ys;
        (_, ...zs_middle, _) = zs;
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["xs"].to_string(), "(Num, Num, Num)");
    assert_eq!(type_env["ys"].to_string(), "(Num, Num, fn<T>(T) -> T)");
    assert_eq!(type_env["zs"].to_string(), "(Num, ...[Num; _], Num)");
    assert_eq!(type_env["xs_head"].to_string(), "(Num, Num)");
    assert_eq!(type_env["zs_middle"].to_string(), "[Num; _]");
}

#[test]
fn unifying_tuples_with_dyn_lengths() {
    let code = r#"
        xs: (_, ...[_], _) = (true, 1, 2, 3, 4);
        (_, _, ...ys) = xs; // should work
        zs: (...[_; _], _, Num) = xs; // should not work (Bool and Num cannot be unified)
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("true", ValueType::BOOL);
    let err = type_env.process_statements(&block).unwrap_err();

    assert!(err.span().fragment().starts_with("zs:"));
    assert_incompatible_types(err.kind(), &ValueType::BOOL, &ValueType::NUM);
    assert_eq!(type_env["xs"].to_string(), "(Bool, ...[Num; _], Num)");
    assert_eq!(type_env["ys"].to_string(), "[Num; _]");
}

#[test]
fn fn_with_varargs() {
    let code = r#"
        sum = |...xs: Num| xs.fold(0, |acc, x| acc + x);
        sum_spec: fn(Num, Num, Num) -> _ = sum;
        tuple_sum = |init, ...xs: (_, _)| xs.fold((init, init), |acc, x| acc + x);
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("fold", Prelude::fold_type().into());
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["sum"].to_string(), "fn<len N>(...[Num; N]) -> Num");
    assert_eq!(type_env["sum_spec"].to_string(), "fn(Num, Num, Num) -> Num");
    assert_eq!(
        type_env["tuple_sum"].to_string(),
        "fn<len N; T: Lin>(T, ...[(T, T); N]) -> (T, T)"
    );
}
