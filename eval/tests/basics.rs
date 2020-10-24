use assert_matches::assert_matches;
use hashbrown::HashMap;

use std::{iter::FromIterator, rc::Rc};

use arithmetic_eval::{
    error::{ErrorWithBacktrace, RepeatedAssignmentContext},
    fns::{self, FromValueErrorKind},
    Environment, Error, ErrorKind, Function, Value, ValueType, VariableMap, WildcardId,
};
use arithmetic_parser::{grammars::F32Grammar, BinaryOp, GrammarExt, LvalueLen, UnaryOp};

const SIN: fns::Unary<f32> = fns::Unary::new(f32::sin);

fn expect_compilation_error<'a>(
    env: &mut Environment<'a, F32Grammar>,
    program: &'a str,
) -> Error<'a> {
    let block = F32Grammar::parse_statements(program).unwrap();
    env.compile_module(WildcardId, &block).unwrap_err()
}

fn try_evaluate<'a>(
    env: &mut Environment<'a, F32Grammar>,
    program: &'a str,
) -> Result<Value<'a, F32Grammar>, ErrorWithBacktrace<'a>> {
    let block = F32Grammar::parse_statements(program).unwrap();
    env.compile_module(WildcardId, &block)
        .unwrap()
        .run_in_env(env)
}

fn evaluate<'a>(env: &mut Environment<'a, F32Grammar>, program: &'a str) -> Value<'a, F32Grammar> {
    try_evaluate(env, program).unwrap()
}

#[test]
fn basic_program() {
    let mut env = Environment::new();
    let return_value = evaluate(&mut env, "x = 1; y = 2; x + y");
    assert_eq!(return_value, Value::Number(3.0));
    assert_eq!(*env.get("x").unwrap(), Value::Number(1.0));
    assert_eq!(*env.get("y").unwrap(), Value::Number(2.0));
}

#[test]
fn basic_program_with_tuples() {
    let mut env = Environment::new();
    let return_value = evaluate(&mut env, "tuple = (1 - 3, 2); (x, _) = tuple;");
    assert_eq!(return_value, Value::void());
    assert_eq!(
        env["tuple"],
        Value::Tuple(vec![Value::Number(-2.0), Value::Number(2.0)])
    );
    assert_eq!(env["x"], Value::Number(-2.0));
}

#[test]
fn arithmetic_ops_on_tuples() {
    let program = r#"
        x = (1, 2) + (3, 4);
        (y, z) = (0, 3) * (2, 0.5) - x;
        u = (1, 2) + 3 * (0.5, z);
    "#;
    let mut env = Environment::new();
    evaluate(&mut env, program);
    assert_eq!(
        env["x"],
        Value::Tuple(vec![Value::Number(4.0), Value::Number(6.0)])
    );
    assert_eq!(env["y"], Value::Number(-4.0));
    assert_eq!(env["z"], Value::Number(-4.5));
    assert_eq!(
        env["u"],
        Value::Tuple(vec![Value::Number(2.5), Value::Number(-11.5)])
    );

    assert_eq!(
        evaluate(&mut env, "1 / (2, 4)"),
        Value::Tuple(vec![Value::Number(0.5), Value::Number(0.25)])
    );

    assert_eq!(
        evaluate(&mut env, "1 / (2, (4, 0.2))"),
        Value::Tuple(vec![
            Value::Number(0.5),
            Value::Tuple(vec![Value::Number(0.25), Value::Number(5.0)])
        ])
    );

    let err = try_evaluate(&mut env, "(1, 2) / |x| { x + 1 }").unwrap_err();
    assert_matches!(
        err.source().kind(),
        ErrorKind::UnexpectedOperand { ref op } if *op == BinaryOp::Div.into()
    );
}

#[test]
fn comparisons() {
    let mut env = Environment::new();
    env.insert("true", Value::Bool(true))
        .insert("false", Value::Bool(false))
        .insert_native_fn("sin", SIN);

    let program = r#"
        foo = |x| { x + 1 };
        alias = foo;

        1 == foo(0) && 1 != (1,) && (2, 3) != (2,) && (2, 3) != (2, 3, 4) && () == ()
            && true == true && true != false && (true, false) == (true, false)
            && foo == foo && foo == alias && foo != 1 && sin == sin && foo != sin
            && (foo, (-1, 3)) == (alias, (-1, 3))
    "#;
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn tuple_destructuring_with_middle() {
    let program = r#"
        (x, ...) = (1, 2, 3);
        (..., u, v) = (-5, -4, -3, -2, -1);
        (_, _, ...middle, y) = (1, 2, 3, 4, 5);
        ((_, a), ..., (b, ...)) = ((1, 2), (3, 4, 5));
        x == 1 && u == -2 && v == -1 && middle == (3, 4) && y == 5 && a == 2 && b == 3
    "#;
    let mut env = Environment::new();
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn program_with_blocks() {
    let mut env = Environment::new();
    let return_value = evaluate(&mut env, "z = { x = 1; x + 3 };");
    assert_eq!(return_value, Value::void());
    assert_eq!(env["z"], Value::Number(4.0));
    assert!(env.get("x").is_none());
}

#[test]
fn program_with_interpreted_function() {
    let program = "foo = |x| x + 5; foo(3.0)";
    let mut env = Environment::new();
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Number(8.0));
    assert!(env["foo"].is_function());
}

#[test]
fn destructuring_in_fn_args() {
    let program = r#"
        swap = |x, (y, z)| ((x, y), z);
        swap(1, (2, 3))
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    let inner_tuple = Value::Tuple(vec![Value::Number(1.0), Value::Number(2.0)]);
    assert_eq!(
        return_value,
        Value::Tuple(vec![inner_tuple, Value::Number(3.0)])
    );
}

#[test]
fn destructuring_in_fn_args_with_wildcard() {
    let program = r#"
        add = |x, (_, z)| x + z;
        add(1, (2, 3))
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Number(4.0));
}

#[test]
fn captures_in_function() {
    let program = r#"
        x = 5;
        foo = |a| a + x;
        foo(-3)
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Number(2.0));
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
    assert_eq!(return_value, Value::Number(2.0));
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
    assert_eq!(return_value, Value::Number(2.0));
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
        Value::Tuple(vec![Value::Number(0.0), Value::Number(0.0)])
    );

    {
        let add = &env["add"];
        let add = match add {
            Value::Function(Function::Interpreted(function)) => function,
            other => panic!("Unexpected `add` value: {:?}", other),
        };
        let captures = add.captures();
        assert_eq!(captures.len(), 1);
        assert_matches!(captures["op"], Value::Function(_));
    }

    let continued_program = r#"
        div = gen(|x, y| x / y);
        div(1, 2) == -1.5 # 1/2 - 2/1
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

    let first = Value::Tuple(vec![
        Value::Number(-1.0),
        Value::Number(-2.0),
        Value::Number(3.5),
    ]);
    let second = Value::Tuple(vec![Value::Number(0.25), Value::Number(-8.0)]);
    assert_eq!(return_value, Value::Tuple(vec![first, second]));
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
        let function = match function {
            Value::Function(Function::Interpreted(function)) => function,
            other => panic!("Unexpected `fn` value: {:?}", other),
        };
        assert!(function.captures()["div"].is_function());
    }
}

#[test]
fn captured_var_in_returned_fn() {
    let program = r#"
        gen = |x| {
            y = (x, x^2);
            # Check that `x` below is not taken from the arg above, but rather
            # from the function argument. `y` though should be captured
            # from the surrounding function.
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
    assert_eq!(return_value, Value::Number(2.0));

    let other_program = "add = gen_add(-3); add(-1)";
    let return_value = evaluate(&mut env, other_program);
    assert_eq!(return_value, Value::Number(-4.0));

    let function = match &env["add"] {
        Value::Function(Function::Interpreted(function)) => function,
        other => panic!("Unexpected `add` value: {:?}", other),
    };
    let captures = function.captures();
    assert_eq!(
        captures,
        HashMap::from_iter(vec![("x", &Value::Number(-3.0))])
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
        Value::Tuple(vec![Value::Number(4.0), Value::Number(1.0)])
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
        Value::Tuple(vec![Value::Number(8.0), Value::Number(-1.5)])
    );
    assert!(env.get("lambda").is_none());
}

#[test]
fn immediately_executed_function() {
    let program = "-|x| { x + 5 }(-3)";
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Number(-2.0));
}

#[test]
fn immediately_executed_function_priority() {
    let program = "2 + |x| { x + 5 }(-3)";
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Number(4.0));
}

#[test]
fn immediately_executed_function_in_other_call() {
    let program = "add = |x, y| x + y; add(10, |x| { x + 5 }(-3))";
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Number(12.0));
}

#[test]
fn program_with_native_function() {
    let mut env = Environment::new();
    env.insert_native_fn("sin", SIN);

    let program = "sin(1.0) - 3";
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Number(1.0_f32.sin() - 3.0));
}

#[test]
fn function_aliasing() {
    let mut env = Environment::new();
    env.insert_native_fn("sin", SIN);

    let program = "alias = sin; alias(1.0)";
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Number(1.0_f32.sin()));

    let sin = &env["sin"];
    let sin = match sin {
        Value::Function(Function::Native(function)) => function,
        _ => panic!("Unexpected `sin` value: {:?}", sin),
    };
    let alias = &env["alias"];
    let alias = match alias {
        Value::Function(Function::Native(function)) => function,
        _ => panic!("Unexpected `alias` value: {:?}", alias),
    };
    assert_eq!(Rc::as_ptr(sin) as *const (), Rc::as_ptr(alias) as *const ());
}

#[test]
fn method_call() {
    let program = "add = |x, y| x + y; 1.0.add(2)";
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Number(3.0));
}

#[test]
fn method_call_on_returned_fn() {
    let program = r#"
        gen = |x| { |y| x + y };
        (1, -2).gen()(3) == (4, 1)
    "#;
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn undefined_var() {
    let err = expect_compilation_error(&mut Environment::new(), "x + 3");
    assert_eq!(*err.main_span().code().fragment(), "x");
    assert_matches!(err.kind(), ErrorKind::Undefined(ref var) if var == "x");
}

#[test]
fn undefined_function() {
    let err = expect_compilation_error(&mut Environment::new(), "1 + sin(-5.0)");
    assert_eq!(*err.main_span().code().fragment(), "sin");
    assert_matches!(err.kind(), ErrorKind::Undefined(ref var) if var == "sin");
}

#[test]
fn arg_len_mismatch() {
    let mut env = Environment::new();

    let program = "foo = |x| x + 5; foo()";
    let err = try_evaluate(&mut env, program).unwrap_err();
    let err = err.source();
    assert_eq!(*err.main_span().code().fragment(), "foo()");
    assert_matches!(
        err.kind(),
        ErrorKind::ArgsLenMismatch {
            def: LvalueLen::Exact(1),
            call: 0,
        }
    );

    let other_program = "foo(1, 2) * 3.0";
    let err = try_evaluate(&mut env, other_program).unwrap_err();
    let err = err.source();
    assert_eq!(*err.main_span().code().fragment(), "foo(1, 2)");
    assert_matches!(
        err.kind(),
        ErrorKind::ArgsLenMismatch {
            def: LvalueLen::Exact(1),
            call: 2,
        }
    );
}

#[test]
fn arg_len_mismatch_with_variadic_function() {
    let program = "foo = |fn, ...xs| fn(xs); foo()";
    let err = try_evaluate(&mut Environment::new(), program).unwrap_err();
    let err = err.source();
    assert_eq!(*err.main_span().code().fragment(), "foo()");
    assert_matches!(
        err.kind(),
        ErrorKind::ArgsLenMismatch {
            def: LvalueLen::AtLeast(1),
            call: 0,
        }
    );
}

#[test]
fn repeated_args_in_fn_definition() {
    let mut env = Environment::new();

    let program = "add = |x, x| x + 2;";
    let err = expect_compilation_error(&mut env, program);
    assert_eq!(*err.main_span().code().fragment(), "x");
    assert_eq!(err.main_span().code().location_offset(), 10);
    assert_matches!(
        err.kind(),
        ErrorKind::RepeatedAssignment {
            context: RepeatedAssignmentContext::FnArgs
        }
    );

    let other_program = "add = |x, (y, x)| x + y;";
    let err = expect_compilation_error(&mut env, other_program);
    assert_eq!(*err.main_span().code().fragment(), "x");
    assert_eq!(err.main_span().code().location_offset(), 14);
    assert_matches!(err.kind(), ErrorKind::RepeatedAssignment { .. });
}

#[test]
fn repeated_var_in_lvalue() {
    let program = "(x, x) = (1, 2);";
    let err = expect_compilation_error(&mut Environment::new(), program);
    assert_eq!(*err.main_span().code().fragment(), "x");
    assert_eq!(err.main_span().code().location_offset(), 4);
    assert_matches!(
        err.kind(),
        ErrorKind::RepeatedAssignment {
            context: RepeatedAssignmentContext::Assignment,
        }
    );

    let other_program = "(x, ...x) = (1, 2);";
    let err = expect_compilation_error(&mut Environment::new(), other_program);
    assert_eq!(*err.main_span().code().fragment(), "x");
    assert_eq!(err.main_span().code().location_offset(), 7);
    assert_matches!(
        err.kind(),
        ErrorKind::RepeatedAssignment {
            context: RepeatedAssignmentContext::Assignment,
        }
    );
}

#[test]
fn error_in_function_args() {
    let program = r#"
        add = |x, (_, z)| x + z;
        add(1, 2)
    "#;

    let err = try_evaluate(&mut Environment::new(), program).unwrap_err();
    let err = err.source();
    assert_eq!(*err.main_span().code().fragment(), "(_, z)");
    assert_matches!(err.kind(), ErrorKind::CannotDestructure);
}

#[test]
fn cannot_call_error() {
    let program = "x = 5; x(1.0)";
    let err = try_evaluate(&mut Environment::new(), program).unwrap_err();
    let err = err.source();
    assert_eq!(err.main_span().code().location_offset(), 7);
    assert_matches!(err.kind(), ErrorKind::CannotCall);

    let other_program = "2 + 1.0(5)";
    let err = try_evaluate(&mut Environment::new(), other_program).unwrap_err();
    let err = err.source();
    assert_eq!(*err.main_span().code().fragment(), "1.0(5)");
    assert_matches!(err.kind(), ErrorKind::CannotCall);
}

#[test]
fn tuple_len_mismatch_error() {
    let program = "x = (1, 2) + (3, 4, 5);";
    let err = try_evaluate(&mut Environment::new(), program).unwrap_err();
    let err = err.source();
    assert_eq!(*err.main_span().code().fragment(), "(1, 2)");
    assert_matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch { lhs: LvalueLen::Exact(2), rhs: 3, .. }
    );
}

#[test]
fn cannot_destructure_error() {
    let program = "(x, y) = 1.0;";
    let err = try_evaluate(&mut Environment::new(), program).unwrap_err();
    let err = err.source();
    assert_eq!(*err.main_span().code().fragment(), "(x, y)");
    assert_matches!(err.kind(), ErrorKind::CannotDestructure);
}

#[test]
fn unexpected_operand() {
    let mut env = Environment::new();

    let program = "1 / || 2";
    let err = try_evaluate(&mut env, program).unwrap_err();
    let err = err.source();
    assert_eq!(*err.main_span().code().fragment(), "|| 2");
    assert_matches!(
        err.kind(),
        ErrorKind::UnexpectedOperand { ref op } if *op == BinaryOp::Div.into()
    );

    let other_program = "1 == 1 && !(2, 3)";
    let err = try_evaluate(&mut env, other_program).unwrap_err();
    let err = err.source();
    assert_eq!(*err.main_span().code().fragment(), "!(2, 3)");
    assert_matches!(
        err.kind(),
        ErrorKind::UnexpectedOperand { ref op } if *op == UnaryOp::Not.into()
    );

    let third_program = "|x| { x + 5 } + 10";
    let err = try_evaluate(&mut env, third_program).unwrap_err();
    assert_matches!(
        err.source().kind(),
        ErrorKind::UnexpectedOperand { ref op } if *op == BinaryOp::Add.into()
    );
}

#[test]
fn native_fn_error() {
    let mut env = Environment::new();
    env.insert_native_fn("sin", SIN);

    let program = "1 + sin(-5.0, 2.0)";
    let err = try_evaluate(&mut env, program).unwrap_err();
    let err = err.source();
    assert_eq!(*err.main_span().code().fragment(), "sin(-5.0, 2.0)");
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
    assert_eq!(*err.main_span().code().fragment(), "sin((-5, 2))");

    let expected_err_kind = FromValueErrorKind::InvalidType {
        expected: ValueType::Number,
        actual: ValueType::Tuple(2),
    };
    assert_matches!(
        err.kind(),
        ErrorKind::Wrapper(err) if *err.kind() == expected_err_kind && err.arg_index() == 0
    );
}

#[test]
fn comparison_desugaring_with_no_cmp() {
    let err = expect_compilation_error(&mut Environment::new(), "2 > 5");
    assert_matches!(
        err.kind(),
        ErrorKind::Undefined(ref name) if name == "cmp"
    );
    assert_eq!(*err.main_span().code().fragment(), ">");
}

#[test]
fn comparison_desugaring_with_invalid_cmp() {
    let program = "cmp = |_, _| 2; 1 > 3";
    let err = try_evaluate(&mut Environment::new(), program).unwrap_err();
    let err = err.source();
    assert_matches!(err.kind(), ErrorKind::InvalidCmpResult);
    assert_eq!(*err.main_span().code().fragment(), "1 > 3");
}

#[test]
fn single_statement_fn() {
    let program = "(|| 5)()";
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Number(5.0));

    let other_program = "x = 3; (|| x)()";
    let return_value = evaluate(&mut Environment::new(), other_program);
    assert_eq!(return_value, Value::Number(3.0));
}

#[test]
fn function_with_non_linear_flow() {
    let program = "(|x| { y = x - 3; x })(2)";
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Number(2.0));
}

#[test]
fn comparison_desugaring_with_capture() {
    let program = "ge = |x, y| x >= y; ge(2, 3)";
    let return_value = evaluate(
        &mut Environment::new().insert_native_fn("cmp", fns::Compare),
        program,
    );
    assert_eq!(return_value, Value::Bool(false));
}

#[test]
fn comparison_desugaring_with_capture_and_no_cmp() {
    let program = "ge = |x, y| x >= y; ge(2, 3)";
    let err = expect_compilation_error(&mut Environment::new(), program);
    assert_matches!(err.kind(), ErrorKind::Undefined(ref name) if name == "cmp");
    assert_eq!(*err.main_span().code().fragment(), ">=");
}
