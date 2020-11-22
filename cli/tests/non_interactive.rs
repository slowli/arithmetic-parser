//! E2E tests for a non-interactive binary usage.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use unindent::unindent;

use std::process::Command;

const ERROR_EXIT_CODE: i32 = 2;

fn create_command(program: &str, arithmetic: &str) -> Command {
    let mut command = Command::cargo_bin(env!("CARGO_PKG_NAME")).expect("CLI binary");
    command
        .env("TERM", "dumb")
        .arg("-a")
        .arg(arithmetic)
        .arg(program);
    command
}

#[test]
fn successful_execution_for_u64_arithmetic() {
    let assert = create_command("1 + 2 * 3", "u64").assert();
    assert.success().stderr("7\n");
}

#[test]
fn successful_execution_for_i64_arithmetic() {
    let assert = create_command("1 - 2 * 3", "i64").assert();
    assert.success().stderr("-5\n");
}

#[test]
fn successful_execution_for_u128_arithmetic() {
    let assert = create_command("1 + 2 * 3", "u128").assert();
    assert.success().stderr("7\n");
}

#[test]
fn successful_execution_for_i128_arithmetic() {
    let assert = create_command("1 - 2 * 3", "i128").assert();
    assert.success().stderr("-5\n");
}

#[test]
fn successful_execution_for_wrapping_int_arithmetic() {
    let assert = create_command("1 - 2 + 3", "u128")
        .arg("--wrapping")
        .assert();
    assert.success().stderr("2\n");
}

#[test]
fn successful_execution_for_f32_arithmetic() {
    let assert = create_command("1 + 3 / 2", "f32").assert();
    assert.success().stderr("2.5\n");
}

#[test]
fn successful_execution_for_f64_arithmetic() {
    let assert = create_command("1 + 3 / 2", "f64").assert();
    assert.success().stderr("2.5\n");
}

#[test]
fn successful_execution_for_c32_arithmetic() {
    let assert = create_command("2 / (1 - i)", "c32").assert();
    assert.success().stderr("1+1i\n");
}

#[test]
fn successful_execution_for_c64_arithmetic() {
    let assert = create_command("2 / (1 - i)", "c64").assert();
    assert.success().stderr("1+1i\n");
}

#[test]
fn successful_execution_for_u64_modular_arithmetic() {
    let assert = create_command("5 / 9", "u64/11").assert();
    assert.success().stderr("3\n");
}

#[test]
fn outputting_native_function() {
    let assert = create_command("if", "f64").assert();
    assert.success().stderr("(native fn)\n");
}

#[test]
fn outputting_interpreted_function() {
    const PROGRAM: &str = "is_positive = |x| x > 0; is_positive";

    let assert = create_command(PROGRAM, "f64").assert();
    assert.success().stderr("fn(1 arg)\n");
}

#[test]
fn syntax_error() {
    const EXPECTED_ERR: &str = r#"
        error[PARSE]: Uninterpreted characters after parsing
          ┌─ Snippet #1:1:5
          │
        1 │ let x = 5
          │     ^^^^^ Error occurred here
    "#;

    let assert = create_command("let x = 5", "f64").assert();
    assert
        .failure()
        .code(ERROR_EXIT_CODE)
        .stderr(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn undefined_variable_error() {
    const EXPECTED_ERR: &str = r#"
        error[EVAL]: Variable `x` is not defined
          ┌─ Snippet #1:1:9
          │
        1 │ 1 + 2 * x
          │         ^ Undefined variable occurrence
    "#;

    let assert = create_command("1 + 2 * x", "f64").assert();
    assert
        .failure()
        .code(ERROR_EXIT_CODE)
        .stderr(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn integer_overflow_for_u64_arithmetic() {
    const EXPECTED_ERR: &str = r#"
        error[EVAL]: Arithmetic error
          ┌─ Snippet #1:1:1
          │
        1 │ 1 - 3 + 5
          │ ^^^^^ Integer overflow or underflow
    "#;

    let assert = create_command("1 - 3 + 5", "u64").assert();
    assert
        .failure()
        .code(ERROR_EXIT_CODE)
        .stderr(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn integer_overflow_for_i64_arithmetic() {
    const EXPECTED_ERR: &str = r#"
        error[EVAL]: Arithmetic error
          ┌─ Snippet #1:1:1
          │
        1 │ 20 ^ 20
          │ ^^^^^^^ Integer overflow or underflow
    "#;

    let assert = create_command("20 ^ 20", "i64").assert();
    assert
        .failure()
        .code(ERROR_EXIT_CODE)
        .stderr(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn negative_exp_for_integer_arithmetic() {
    const EXPECTED_ERR: &str = r#"
        error[EVAL]: Arithmetic error
          ┌─ Snippet #1:1:1
          │
        1 │ 10 ^ -3
          │ ^^^^^^^ Exponent is too large or negative
    "#;

    let assert = create_command("10 ^ -3", "i128").assert();
    assert
        .failure()
        .code(ERROR_EXIT_CODE)
        .stderr(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn incompatible_arg_count_error() {
    const EXPECTED_ERR: &str = r#"
        error[EVAL]: Mismatch between the number of arguments in the function definition and its call
          ┌─ Snippet #1:1:1
          │
        1 │ if(2 > 1, 3)
          │ ^^^^^^^^^^^^ Called with 2 arg(s) here
    "#;

    let assert = create_command("if(2 > 1, 3)", "f64").assert();
    assert
        .failure()
        .code(ERROR_EXIT_CODE)
        .stderr(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn error_with_call_trace() {
    const PROGRAM: &str = r#"
        is_positive = |x| x > 0;
        is_positive(3) && !is_positive((1, 2))
    "#;
    const EXPECTED_ERR: &str = r#"
        error[EVAL]: Value is not comparable
          ┌─ Snippet #1:1:15
          │
        1 │ is_positive = |x| x > 0;
          │               ----^----
          │               │   │
          │               │   Cannot be compared
          │               The error occurred in function `is_positive`
        2 │ is_positive(3) && !is_positive((1, 2))
          │                    ------------------- Call at depth 1
          │
          = Only numbers can be compared; complex values cannot
    "#;

    let assert = create_command(&unindent(PROGRAM), "f64").assert();
    assert
        .failure()
        .code(ERROR_EXIT_CODE)
        .stderr(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn error_with_call_complex_call_trace() {
    const PROGRAM: &str = r#"
        double = |x| x * 2;
        quadruple = |x| double(double(x));
        quadruple(true)
    "#;
    const EXPECTED_ERR: &str = r#"
        error[EVAL]: Unexpected operand type for multiplication
          ┌─ Snippet #1:1:10
          │
        1 │ double = |x| x * 2;
          │          ----^----
          │          │   │
          │          │   Operand of wrong type
          │          The error occurred in function `double`
        2 │ quadruple = |x| double(double(x));
          │                        --------- Call at depth 1
        3 │ quadruple(true)
          │ --------------- Call at depth 2
          │
          = Operands of binary arithmetic ops must be numbers or tuples containing numbers
    "#;

    let assert = create_command(&unindent(PROGRAM), "f64").assert();
    assert
        .failure()
        .code(ERROR_EXIT_CODE)
        .stderr(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn error_with_call_complex_call_trace_and_native_fns() {
    const PROGRAM: &str = r#"
        all = |array, predicate| array.fold(true, |acc, x| acc && predicate(x));
        (1, 2, map).all(|x| 0 < x)
    "#;
    const EXPECTED_ERR: &str = r#"
        error[EVAL]: Value is not comparable
          ┌─ Snippet #1:1:26
          │
        1 │ all = |array, predicate| array.fold(true, |acc, x| acc && predicate(x));
          │                          ----------------------------------------------
          │                          │                                │
          │                          │                                Call at depth 1
          │                          Call at depth 2
        2 │ (1, 2, map).all(|x| 0 < x)
          │ ------------------------^-
          │ │               │       │
          │ │               │       Cannot be compared
          │ │               The error occurred in function `predicate`
          │ Call at depth 3
          │
          = Only numbers can be compared; complex values cannot
    "#;

    let assert = create_command(&unindent(PROGRAM), "f64").assert();
    assert
        .failure()
        .code(ERROR_EXIT_CODE)
        .stderr(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}
