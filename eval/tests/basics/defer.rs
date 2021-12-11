//! Tests for `defer` function.

use arithmetic_eval::{fns, Assertions, Environment, ErrorKind};
use assert_matches::assert_matches;

use crate::{evaluate, try_evaluate};

#[test]
fn deferred_initialization_with_function() {
    let mut env = Environment::<f32>::new();
    env.extend(Assertions.iter());
    env.insert_native_fn("defer", fns::Defer)
        .insert_native_fn("if", fns::If);

    let program = r#"
        count_digits = defer(|count_digits| {
            |x, acc| if(x < 1, || acc, || count_digits(x / 10, acc + 1))()
        });
        assert_eq(count_digits(7, 0), 1);
        assert_eq(count_digits(10, 0), 2);
        assert_eq(count_digits(126475, 0), 6);
    "#;
    evaluate(&mut env, program);
}

#[test]
fn deferred_initialization_with_prototype() {
    let mut env = Environment::<f32>::new();
    env.extend(Assertions.iter());
    env.insert_native_fn("defer", fns::Defer)
        .insert_native_fn("impl", fns::CreatePrototype);

    let program = r#"
        Point = defer(|Self| {
            impl(#{
                ZERO: #{ x: 0, y: 0 },
                flip: |self| Self(Self.ZERO - self),
            })
        });
        assert_eq(Point.ZERO, #{ x: 0, y: 0 });

        flipped = Point(#{ x: 3, y: 4 }).flip();
        assert_eq(flipped, Point(#{ x: -3, y: -4 }));
        flipped = flipped.flip();
        assert_eq(flipped.x, 3); assert_eq(flipped.y, 4);
    "#;
    evaluate(&mut env, program);
}

#[test]
fn uninitialized_value_usage() {
    let mut env = Environment::<f32>::new();
    env.extend(Assertions.iter());
    env.insert_native_fn("defer", fns::Defer);

    let immediate_use_program = "defer(|uninit| uninit());";
    {
        let err = try_evaluate(&mut env, immediate_use_program).unwrap_err();
        let err = err.source();
        assert_matches!(err.kind(), ErrorKind::CannotCall);
        assert_eq!(*err.main_span().code().fragment(), "uninit()");
    }

    let use_via_fn_program = r#"
        defer(|uninit| {
            fun = |x| 1 + uninit(x / 2); // definition is OK
            fun(5) // immediately calling the function is not
        });
    "#;
    let err = try_evaluate(&mut env, use_via_fn_program).unwrap_err();
    let err = err.source();
    assert_matches!(err.kind(), ErrorKind::Uninitialized(name) if name == "uninit");
    assert_eq!(*err.main_span().code().fragment(), "fun(5)");
}
