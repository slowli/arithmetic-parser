//! Tests targeting functions / methods.

use assert_matches::assert_matches;

use arithmetic_eval::{
    fns::FromValueErrorKind, Environment, ErrorKind, Function, NativeFn, Tuple, Value, ValueType,
};
use arithmetic_parser::LvalueLen;

use crate::{evaluate, try_evaluate, SIN};

#[test]
fn program_with_interpreted_function() {
    let program = "foo = |x| x + 5; foo(3.0)";
    let mut env = Environment::new();
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Prim(8.0));
    assert!(env["foo"].is_function());
}

#[test]
fn destructuring_in_fn_args() {
    let program = r#"
        swap = |x, (y, z)| ((x, y), z);
        swap(1, (2, 3))
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    let inner_tuple = Value::from(vec![Value::Prim(1.0), Value::Prim(2.0)]);
    assert_eq!(
        return_value,
        Value::from(vec![inner_tuple, Value::Prim(3.0)])
    );
}

#[test]
fn destructuring_in_fn_args_with_wildcard() {
    let program = r#"
        add = |x, (_, z)| x + z;
        add(1, (2, 3))
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Prim(4.0));
}

#[test]
fn captures_in_function() {
    let program = r#"
        x = 5;
        foo = |a| a + x;
        foo(-3)
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Prim(2.0));
}

#[test]
fn captures_in_function_with_shadowing() {
    // All captures are by value, so that redefining the captured var does not influence
    // the result.
    let program = r#"
        x = 5;
        foo = |a| a + x;
        x = 10;
        foo(-3)
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Prim(2.0));
}

#[test]
fn fn_captures_in_function() {
    // Functions may be captured as well.
    let program = r#"
        add = |x, y| x + y;
        foo = |a| add(a, 5);
        add = 0;
        foo(-3)
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Prim(2.0));
}

#[test]
fn captured_function() {
    let program = r#"
        gen = |op| { |u, v| op(u, v) - op(v, u) };
        add = gen(|x, y| x + y);
        add((1, 2), (3, 4))
    "#;
    let mut env = Environment::new();
    let return_value = evaluate(&mut env, program);
    assert_eq!(
        return_value,
        Value::from(vec![Value::Prim(0.0), Value::Prim(0.0)])
    );

    {
        let add = &env["add"];
        let Value::Function(Function::Interpreted(add)) = add else {
            panic!("Unexpected `add` value: {add:?}");
        };
        let captures = add.captures();
        assert_eq!(captures.len(), 1);
        assert_matches!(captures["op"], Value::Function(_));
    }

    let continued_program = r#"
        div = gen(|x, y| x / y);
        div(1, 2) == -1.5 // 1/2 - 2/1
    "#;
    let return_flag = evaluate(&mut env, continued_program);
    assert_eq!(return_flag, Value::Bool(true));
}

#[test]
fn variadic_function() {
    let program = r#"
        call = |fn, ...xs| fn(xs);
        (call(|x| -x, 1, 2, -3.5), call(|x| 1/x, 4, -0.125))
    "#;
    let mut env = Environment::new();
    let return_value = evaluate(&mut env, program);

    let first = Value::from(vec![Value::Prim(-1.0), Value::Prim(-2.0), Value::Prim(3.5)]);
    let second = Value::from(vec![Value::Prim(0.25), Value::Prim(-8.0)]);
    assert_eq!(return_value, Value::from(vec![first, second]));
}

#[test]
fn variadic_function_with_both_sides() {
    let program = r#"
        call = |fn, ...xs, y| fn(xs, y);
        call(|x, y| x + y, 1, 2, -3.5) == (-2.5, -1.5)
    "#;
    let mut env = Environment::new();
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn indirectly_captured_function() {
    let program = r#"
        gen = {
            div = |x, y| x / y;
            |u| { |v| div(u, v) - div(v, u) }
        };
        fn = gen(4);
        fn(1) == 3.75
    "#;
    let mut env = Environment::new();
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Bool(true));

    // Check that `div` is captured both by the external and internal functions.
    let functions = [&env["fn"], &env["gen"]];
    for &function in &functions {
        let Value::Function(Function::Interpreted(function)) = function else {
            panic!("Unexpected `fn` value: {function:?}");
        };
        assert!(function.captures()["div"].is_function());
    }
}

#[test]
fn captured_var_in_returned_fn() {
    let program = r#"
        gen = |x| {
            y = (x, x^2);
            // Check that `x` below is not taken from the arg above, but rather
            // from the function argument. `y` though should be captured
            // from the surrounding function.
            |x| y - (x, x^2)
        };
        foo = gen(2);
        foo(1) == (1, 3) && foo(2) == (0, 0) && foo(3) == (-1, -5)
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn embedded_function() {
    let program = r#"
        gen_add = |x| |y| x + y;
        add = gen_add(5.0);
        add(-3) + add(-5)
    "#;
    let mut env = Environment::new();
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Prim(2.0));

    let other_program = "add = gen_add(-3); add(-1)";
    let return_value = evaluate(&mut env, other_program);
    assert_eq!(return_value, Value::Prim(-4.0));

    let Value::Function(Function::Interpreted(function)) = &env["add"] else {
        panic!("Unexpected `add` value: {env:?}");
    };
    let captures = function.captures();
    assert_eq!(
        captures.into_iter().collect::<Vec<_>>(),
        [("x", &Value::Prim(-3.0))]
    );
}

#[test]
fn first_class_functions_apply() {
    let program = r#"
        apply = |fn, x, y| (fn(x), fn(y));
        apply(|x| x + 3, 1, -2)
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(
        return_value,
        Value::from(vec![Value::Prim(4.0), Value::Prim(1.0)])
    );
}

#[test]
fn first_class_functions_repeat() {
    let program = r#"
        repeat = |fn, x| fn(fn(fn(x)));
        a = repeat(|x| x * 2, 1);
        b = {
            lambda = |x| x / 2 - 1;
            repeat(lambda, 2)
        };
        (a, b)
    "#;
    let mut env = Environment::new();

    let return_value = evaluate(&mut env, program);

    assert_eq!(
        return_value,
        Value::from(vec![Value::Prim(8.0), Value::Prim(-1.5)])
    );
    assert!(env.get("lambda").is_none());
}

#[test]
fn immediately_executed_function() {
    let program = "-|x| { x + 5 }(-3)";
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Prim(-2.0));
}

#[test]
fn immediately_executed_function_priority() {
    let program = "2 + |x| { x + 5 }(-3)";
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Prim(4.0));
}

#[test]
fn immediately_executed_function_in_other_call() {
    let program = "add = |x, y| x + y; add(10, |x| { x + 5 }(-3))";
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Prim(12.0));
}

#[test]
fn program_with_native_function() {
    let mut env = Environment::new();
    env.insert_native_fn("sin", SIN);

    let program = "sin(1.0) - 3";
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Prim(1.0_f32.sin() - 3.0));
}

#[test]
fn function_aliasing() {
    let mut env = Environment::new();
    env.insert_native_fn("sin", SIN);

    let program = "alias = sin; alias(1.0)";
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Prim(1.0_f32.sin()));

    let sin = &env["sin"];
    let Value::Function(Function::Native(sin)) = sin else {
        panic!("Unexpected `sin` value: {sin:?}");
    };
    let alias = &env["alias"];
    let Value::Function(Function::Native(alias)) = alias else {
        panic!("Unexpected `alias` value: {alias:?}");
    };

    // Compare pointers to data instead of pointers to `dyn NativeFn<_>` (which is unsound,
    // see https://rust-lang.github.io/rust-clippy/master/index.html#vtable_address_comparisons).
    assert_eq!(
        sin.as_ref() as *const dyn NativeFn<_> as *const (),
        alias.as_ref() as *const dyn NativeFn<_> as *const ()
    );
}

#[test]
fn method_call() {
    let program = "1.0.add(2)";
    let mut env = Environment::new();
    env.insert_wrapped_fn("add", |x: f32, y: f32| x + y);
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Prim(3.0));
}

#[test]
fn complex_method_call_name() {
    let program = r#"
        (Num, x) = (#{ add }, 1);
        (x.{Num.add}(2), 2.0.{Num.add}(5))
    "#;
    let mut env = Environment::new();
    env.insert_wrapped_fn("add", |x: f32, y: f32| x + y);
    let return_value = evaluate(&mut env, program);
    assert_eq!(
        return_value,
        Tuple::from(vec![Value::Prim(3.0), Value::Prim(7.0)]).into()
    );
}

#[test]
fn non_trivial_block_in_method_call_name() {
    let program = r#"
        slope = |k| { |x, y| x * k + y };
        x = 5; x.{slope(3)}(4)
    "#;
    let mut env = Environment::new();
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Prim(19.0));
}

#[test]
fn chained_method_call_name() {
    let program = r#"
        Point = #{
            new: |x, y| #{ x, y },
            len2: |{x, y}| x * x + y * y,
        };
        3.0.{Point.new}(4).{Point.len2}()
    "#;
    let mut env = Environment::new();
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Prim(25.0));
}

#[test]
fn function_call_on_returned_fn() {
    let program = r#"
        gen = |x| { |y| x + y };
        gen((1, -2))(3) == (4, 1)
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn native_fn_error() {
    let mut env = Environment::new();
    env.insert_native_fn("sin", SIN);

    let program = "1 + sin(-5.0, 2.0)";
    let err = try_evaluate(&mut env, program).unwrap_err();
    let err = err.source();
    assert_eq!(err.main_span().code().span_code(program), "sin(-5.0, 2.0)");
    assert_matches!(
        err.kind(),
        ErrorKind::ArgsLenMismatch {
            def: LvalueLen::Exact(1),
            call: 2
        }
    );

    let other_program = "1 + sin((-5, 2))";
    let err = try_evaluate(&mut env, other_program).unwrap_err();
    let err = err.source();
    assert_eq!(
        err.main_span().code().span_code(other_program),
        "sin((-5, 2))"
    );

    let expected_err_kind = FromValueErrorKind::InvalidType {
        expected: ValueType::Prim,
        actual: ValueType::Tuple(2),
    };
    assert_matches!(
        err.kind(),
        ErrorKind::Wrapper(err) if *err.kind() == expected_err_kind && err.arg_index() == 0
    );
}

#[test]
fn single_statement_fn() {
    let program = "(|| 5)()";
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Prim(5.0));

    let other_program = "x = 3; (|| x)()";
    let return_value = evaluate(&mut Environment::new(), other_program);
    assert_eq!(return_value, Value::Prim(3.0));
}

#[test]
fn function_with_non_linear_flow() {
    let program = "(|x| { y = x - 3; x })(2)";
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Prim(2.0));
}
