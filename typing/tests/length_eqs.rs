//! Tests length equations.

use assert_matches::assert_matches;

use arithmetic_parser::grammars::{NumGrammar, Parse, Typed};
use arithmetic_typing::{
    Annotated, Prelude, TupleLen, TupleLenMismatchContext, TypeEnvironment, TypeErrorKind,
    ValueType,
};

pub type F32Grammar = Typed<Annotated<NumGrammar<f32>>>;

#[test]
fn push_fn_basics() {
    let code = "(1, 2).push(3).push(4)";
    let block = F32Grammar::parse_statements(code).unwrap();
    let tuple = TypeEnvironment::new()
        .insert("push", Prelude::Push)
        .process_statements(&block)
        .unwrap();

    assert_eq!(tuple, ValueType::slice(ValueType::NUM, TupleLen::from(4)));
}

#[test]
fn push_fn_in_other_fn_definition() {
    let code = r#"
        push_fork = |...xs, item| (xs, xs.push(item));
        (xs, ys) = push_fork(1, 2, 3, 4);
        (_, (_, z)) = push_fork(4);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env = TypeEnvironment::new();
    type_env.insert("push", Prelude::Push);
    let err = type_env.process_statements(&block).unwrap_err();

    assert_eq!(*err.span().fragment(), "(_, (_, z)) = push_fork(4)");
    assert_matches!(
        err.kind(),
        TypeErrorKind::TupleLenMismatch {
            lhs,
            rhs,
            context: TupleLenMismatchContext::Assignment,
        } if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(1)
    );

    assert_eq!(
        type_env["push_fork"].to_string(),
        "(...['T; N], 'T) -> (['T; N], ['T; N + 1])"
    );
    assert_eq!(
        type_env["xs"],
        ValueType::slice(ValueType::NUM, TupleLen::from(3))
    );
    assert_eq!(
        type_env["ys"],
        ValueType::slice(ValueType::NUM, TupleLen::from(4))
    );
}

#[test]
fn several_push_applications() {
    let code = r#"
        push2 = |xs, x, y| xs.push(x).push(y);
        (head, ...tail) = (1, 2).push2(3, 4);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env = TypeEnvironment::new();
    type_env.insert("push", Prelude::Push);
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["push2"].to_string(),
        "(['T; N], 'T, 'T) -> ['T; N + 2]"
    );
    assert_eq!(type_env["head"], ValueType::NUM);
    assert_eq!(
        type_env["tail"],
        ValueType::slice(ValueType::NUM, TupleLen::from(3))
    );
}

#[test]
fn comparing_lengths_after_push() {
    let code = r#"
        simple = |xs, ys| xs.push(0) == ys.push(1);
        _asymmetric = |xs, ys| xs.push(0) == ys;
        asymmetric = |xs, ys| xs == ys.push(0);
        complex = |xs, ys| xs.push(0) + ys.push(1).push(0);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env = TypeEnvironment::new();
    type_env.insert("push", Prelude::Push);
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["simple"].to_string(),
        "([Num; N], [Num; N]) -> Bool"
    );
    assert_eq!(
        type_env["asymmetric"].to_string(),
        "([Num; N + 1], [Num; N]) -> Bool"
    );
    assert_eq!(
        type_env["complex"].to_string(),
        "for<len! N> ([Num; N + 1], [Num; N]) -> [Num; N + 2]"
    );
}

#[test]
fn requirements_on_len_via_destructuring() {
    let code = r#"
        len_at_least2 = |xs| { (_, _, ...) = xs; xs };

        // Check propagation to other fns.
        test_fn = |xs: [_; _]| xs.len_at_least2().fold(0, |acc, x| acc + x);

        other_test_fn = |xs: [_; _]| {
           (head, ...tail) = xs.len_at_least2();
           head == (1, 1 == 1);
           tail.map(|(x, _)| x)
        };

        (1, 2).len_at_least2();
        (..., x) = (1, 2, 3, 4).len_at_least2();
        (1,).len_at_least2();
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("fold", Prelude::Fold)
        .insert("map", Prelude::Map);
    let err = type_env.process_statements(&block).unwrap_err();

    assert_eq!(*err.span().fragment(), "(1,).len_at_least2()");
    assert_matches!(
        err.kind(),
        TypeErrorKind::TupleLenMismatch {
            lhs,
            rhs,
            context: TupleLenMismatchContext::Assignment,
        } if lhs.to_string() == "_ + 2" && *rhs == TupleLen::from(1)
    );

    assert_eq!(
        type_env["len_at_least2"].to_string(),
        "(('T, 'U, ...['V; N])) -> ('T, 'U, ...['V; N])"
    );
    assert_eq!(type_env["test_fn"].to_string(), "([Num; N + 2]) -> Num");
    assert_eq!(
        type_env["other_test_fn"].to_string(),
        "([(Num, Bool); N + 2]) -> [Num; N + 1]"
    );
}

#[test]
fn reversing_a_slice() {
    let code = r#"
        reverse = |xs| {
            empty: [_] = ();
            xs.fold(empty, |acc, x| (x,).merge(acc))
        };
        ys = (2, 3, 4).reverse().map(|x| x == 1);
        (_, ...) = ys;
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("fold", Prelude::Fold)
        .insert("map", Prelude::Map)
        .insert("merge", Prelude::Merge);
    let err = type_env.process_statements(&block).unwrap_err();

    assert_eq!(*err.span().fragment(), "(_, ...) = ys");
    assert_matches!(err.kind(), TypeErrorKind::TupleLenMismatch { .. });
    assert_eq!(type_env["reverse"].to_string(), "(['T; N]) -> ['T]");
    assert_eq!(type_env["ys"].to_string(), "[Bool]");
}

#[test]
fn errors_when_adding_dynamic_slices() {
    let setup_code = r#"
        slice: [_] = (1, 2, 3);
        other_slice: [_] = (4, 5);
        slice = -slice;
        other_slice * 8; // works: dynamic slices are linear
    "#;
    let setup_block = F32Grammar::parse_statements(setup_code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&setup_block).unwrap();

    let invalid_code = r#"
        slice + slice;
        slice + other_slice;
        (7,) + slice;
    "#;
    for line in invalid_code.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let line = F32Grammar::parse_statements(line).unwrap();
        let err = type_env.process_statements(&line).unwrap_err();
        assert_matches!(err.kind(), TypeErrorKind::DynamicLen(_));
    }
}

// FIXME: test square [['T; N]; N]
