//! Tests of basic functionality.

use assert_matches::assert_matches;

use arithmetic_eval::{
    env::{Comparisons, Environment, VariableMap},
    error::{Error, ErrorKind, ErrorWithBacktrace, RepeatedAssignmentContext},
    exec::WildcardId,
    fns, StandardPrototypes, Value,
};
use arithmetic_parser::{
    grammars::{F32Grammar, Parse, Untyped},
    BinaryOp, LvalueLen, UnaryOp,
};

mod defer;
mod functions;
mod objects;
mod prototypes;

const SIN: fns::Unary<f32> = fns::Unary::new(f32::sin);

fn expect_compilation_error<'a>(env: &mut Environment<'a, f32>, program: &'a str) -> Error<'a> {
    let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();
    env.compile_module(WildcardId, &block).unwrap_err()
}

fn try_evaluate<'a>(
    env: &mut Environment<'a, f32>,
    program: &'a str,
) -> Result<Value<'a, f32>, ErrorWithBacktrace<'a>> {
    let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();
    env.compile_module(WildcardId, &block)
        .unwrap()
        .run_in_env(env)
}

fn evaluate<'a>(env: &mut Environment<'a, f32>, program: &'a str) -> Value<'a, f32> {
    try_evaluate(env, program).unwrap()
}

#[test]
fn basic_program() {
    let mut env = Environment::new();
    let return_value = evaluate(&mut env, "x = 1; y = 2; x + y");
    assert_eq!(return_value, Value::Prim(3.0));
    assert_eq!(*env.get("x").unwrap(), Value::Prim(1.0));
    assert_eq!(*env.get("y").unwrap(), Value::Prim(2.0));
}

#[test]
fn basic_program_with_tuples() {
    let mut env = Environment::new();
    let return_value = evaluate(&mut env, "tuple = (1 - 3, 2); (x, _) = tuple;");
    assert_eq!(return_value, Value::void());
    assert_eq!(
        env["tuple"],
        Value::from(vec![Value::Prim(-2.0), Value::Prim(2.0)])
    );
    assert_eq!(env["x"], Value::Prim(-2.0));
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
        Value::from(vec![Value::Prim(4.0), Value::Prim(6.0)])
    );
    assert_eq!(env["y"], Value::Prim(-4.0));
    assert_eq!(env["z"], Value::Prim(-4.5));
    assert_eq!(
        env["u"],
        Value::from(vec![Value::Prim(2.5), Value::Prim(-11.5)])
    );

    assert_eq!(
        evaluate(&mut env, "1 / (2, 4)"),
        Value::from(vec![Value::Prim(0.5), Value::Prim(0.25)])
    );

    assert_eq!(
        evaluate(&mut env, "1 / (2, (4, 0.2))"),
        Value::from(vec![
            Value::Prim(0.5),
            Value::from(vec![Value::Prim(0.25), Value::Prim(5.0)])
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
fn tuple_destructuring_with_multiple_components() {
    let program = "(x, y, z) = (1, 12, 5); x == 1 && y == 12 && z == 5";
    let return_value = evaluate(&mut Environment::new(), program);
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
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn program_with_blocks() {
    let mut env = Environment::new();
    let return_value = evaluate(&mut env, "z = { x = 1; x + 3 };");
    assert_eq!(return_value, Value::void());
    assert_eq!(env["z"], Value::Prim(4.0));
    assert!(env.get("x").is_none());
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

    let other_program = "2 + true(5)";
    let mut env = Environment::new();
    env.insert("true", Value::Bool(true));
    let err = try_evaluate(&mut env, other_program).unwrap_err();
    let err = err.source();
    assert_eq!(*err.main_span().code().fragment(), "true(5)");
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
        ErrorKind::TupleLenMismatch {
            lhs: LvalueLen::Exact(2),
            rhs: 3,
            ..
        }
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
fn comparison_of_invalid_values() {
    let err = try_evaluate(&mut Environment::new(), "(2,) > 5").unwrap_err();
    assert_matches!(err.source().kind(), ErrorKind::CannotCompare);
    assert_eq!(*err.source().main_span().code().fragment(), "(2,)");

    let err = try_evaluate(&mut Environment::new(), "2 > (3, 5)").unwrap_err();
    assert_matches!(err.source().kind(), ErrorKind::CannotCompare);
    assert_eq!(*err.source().main_span().code().fragment(), "(3, 5)");
}

#[test]
fn comparison_return_value() {
    let program = "cmp(5, 3) == GREATER && cmp(3, 5) == LESS && cmp(4, 4) == EQUAL";
    let mut env = Comparisons.iter().collect();
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn comparison_constants_are_comparable() {
    let program = r#"
        EQUAL != LESS && cmp(5, 3) != LESS && (LESS, GREATER) == (cmp(3, 4), cmp(4, -5)) &&
            EQUAL != 0 && LESS != -1
    "#;
    let mut env = Comparisons.iter().collect();
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn comparison_within_capture() {
    let program = "ge = |x, y| x >= y; ge(2, 3)";
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(false));
}

#[test]
fn comparison_error_within_capture() {
    let program = "ge = |x, y| x >= (y,); ge(2, 3)";
    let err = try_evaluate(&mut Environment::new(), program).unwrap_err();
    assert_matches!(err.source().kind(), ErrorKind::CannotCompare);
    assert_eq!(*err.source().main_span().code().fragment(), "(y,)");
}

#[test]
fn priority_of_unary_ops_and_methods() {
    let program = "-1.abs() == -1 && -1.0.abs() == -1 && --1.abs() == 1 && (-1).abs() == 1";
    let mut env = Environment::new();
    let proto = Value::wrapped_fn(f32::abs).into_object("abs");
    env.insert_prototypes(StandardPrototypes::new().with_primitive_proto(proto));
    let return_value = evaluate(&mut env, program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn indexed_field_access() {
    let program = "x = 3; (x, 1).0 == 3";
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn sequential_indexed_field_access() {
    let program = "xs = ((1, 2), (3, 4)); xs.0.0 == 1 && xs.1.0 == 3";
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn callable_indexed_field() {
    let program = "rec = (|x| x + 5, 1); (rec.0)(1) == 6";
    let return_value = evaluate(&mut Environment::new(), program);
    assert_eq!(return_value, Value::Bool(true));
}

#[test]
fn indexed_field_out_of_bounds_error() {
    let program = "x = 3; (x,).1 == 3";
    let err = try_evaluate(&mut Environment::new(), program).unwrap_err();

    assert_eq!(*err.source().main_span().code().fragment(), "(x,).1");
    assert_matches!(
        err.source().kind(),
        ErrorKind::IndexOutOfBounds { index: 1, len: 1 }
    );
}

#[test]
fn indexed_field_invalid_receiver_error() {
    let program = "x = 3; x.1 == 3";
    let err = try_evaluate(&mut Environment::new(), program).unwrap_err();

    assert_eq!(*err.source().main_span().code().fragment(), "x.1");
    assert_matches!(err.source().kind(), ErrorKind::CannotIndex);
}

#[test]
fn overly_large_indexed_field() {
    let program = "x = (2, 5); x.123456789012345678901234567890";
    let err = expect_compilation_error(&mut Environment::new(), program);

    assert_eq!(
        *err.main_span().code().fragment(),
        "123456789012345678901234567890"
    );
    assert_matches!(
        err.kind(),
        ErrorKind::InvalidFieldName(name) if name == "123456789012345678901234567890"
    );
}
