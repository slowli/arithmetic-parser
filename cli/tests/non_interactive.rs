//! E2E tests for a non-interactive binary usage.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use unindent::unindent;

use std::process::Command;

const PATH_TO_BIN: &str = env!("CARGO_BIN_EXE_arithmetic-parser");

const ERROR_EXIT_CODE: i32 = 2;

fn create_ast_command(program: &str, arithmetic: &str) -> Command {
    let mut command = Command::new(PATH_TO_BIN);
    command
        .env("TERM", "dumb")
        .arg("ast")
        .arg("-a")
        .arg(arithmetic)
        .arg(program);
    command
}

fn create_command(program: &str, arithmetic: &str) -> Command {
    let mut command = Command::new(PATH_TO_BIN);
    command
        .env("TERM", "dumb")
        .arg("eval")
        .arg("-a")
        .arg(arithmetic)
        .arg(program);
    command
}

#[test]
fn successful_execution_for_ast() {
    let assert = create_ast_command("1 + 2 * 3", "u64").assert();
    assert
        .success()
        .stdout(predicate::str::contains("BinaryOp"));
}

#[test]
fn successful_execution_for_u64_arithmetic() {
    let assert = create_command("1 + 2 * 3", "u64").assert();
    assert.success().stdout("7\n");
}

#[test]
fn successful_execution_for_i64_arithmetic() {
    let assert = create_command("1 - 2 * 3", "i64").assert();
    assert.success().stdout("-5\n");
}

#[test]
fn successful_execution_for_u128_arithmetic() {
    let assert = create_command("1 + 2 * 3", "u128").assert();
    assert.success().stdout("7\n");
}

#[test]
fn successful_execution_for_i128_arithmetic() {
    let assert = create_command("1 - 2 * 3", "i128").assert();
    assert.success().stdout("-5\n");
}

#[test]
fn successful_execution_for_wrapping_int_arithmetic() {
    let assert = create_command("1 - 2 + 3", "u128")
        .arg("--wrapping")
        .assert();
    assert.success().stdout("2\n");
}

#[test]
fn successful_execution_for_f32_arithmetic() {
    let assert = create_command("1 + 3 / 2", "f32").assert();
    assert.success().stdout("2.5\n");
}

#[test]
fn successful_execution_for_f64_arithmetic() {
    let assert = create_command("1 + 3 / 2", "f64").assert();
    assert.success().stdout("2.5\n");
}

#[test]
fn successful_execution_for_c32_arithmetic() {
    let assert = create_command("2 / (1 - i)", "c32").assert();
    assert.success().stdout("1+1i\n");
}

#[test]
fn successful_execution_for_c64_arithmetic() {
    let assert = create_command("2 / (1 - i)", "c64").assert();
    assert.success().stdout("1+1i\n");
}

#[test]
fn successful_execution_for_u64_modular_arithmetic() {
    let assert = create_command("5 / 9", "u64/11").assert();
    assert.success().stdout("3\n");
}

#[test]
fn outputting_native_function() {
    let assert = create_command("if", "f64").assert();
    assert.success().stdout("(native fn)\n");
}

#[test]
fn outputting_interpreted_function() {
    const PROGRAM: &str = "is_positive = |x| x > 0; is_positive";

    let assert = create_command(PROGRAM, "f64").assert();
    assert.success().stdout("fn(1 arg)\n");
}

#[test]
fn syntax_error_with_ast_command() {
    const EXPECTED_ERR: &str = r#"
        error[PARSE]: Uninterpreted characters after parsing
          ┌─ Snippet #1:1:5
          │
        1 │ let x = 5
          │     ^^^^^ Error occurred here
    "#;
    let assert = create_ast_command("let x = 5", "f64").assert();
    assert
        .failure()
        .code(ERROR_EXIT_CODE)
        .stdout(predicate::str::starts_with(unindent(EXPECTED_ERR)));
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
        .stdout(predicate::str::starts_with(unindent(EXPECTED_ERR)));
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
        .stdout(predicate::str::starts_with(unindent(EXPECTED_ERR)));
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
        .stdout(predicate::str::starts_with(unindent(EXPECTED_ERR)));
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
        .stdout(predicate::str::starts_with(unindent(EXPECTED_ERR)));
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
        .stdout(predicate::str::starts_with(unindent(EXPECTED_ERR)));
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
        .stdout(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn error_with_call_trace() {
    const PROGRAM: &str = r#"
        is_positive = |x| x > 0;
        is_positive(3) && !is_positive((1, 2))
    "#;
    const EXPECTED_ERR: &str = r#"
        error[EVAL]: Value is not comparable
          ┌─ Snippet #1:1:19
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
        .stdout(predicate::str::starts_with(unindent(EXPECTED_ERR)));
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
          ┌─ Snippet #1:1:14
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
        .stdout(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn error_with_call_complex_call_trace_and_native_fns() {
    const PROGRAM: &str = r#"
        all = |array, predicate| array.fold(true, |acc, x| acc && predicate(x));
        (1, 2, map).all(|x| 0 < x)
    "#;
    const EXPECTED_ERR: &str = r#"
        error[EVAL]: Value is not comparable
          ┌─ Snippet #1:2:25
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
        .stdout(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn assertion_error() {
    const EXPECTED_ERR: &str = r#"
        error[EVAL]: Equality assertion failed
          ┌─ Snippet #1:1:1
          │
        1 │ assert_eq(1 + 2, 3 / 2)
          │ ^^^^^^^^^^^^^^^^^^^^^^^
          │ │         │      │
          │ │         │      Has value: 1.5
          │ │         Has value: 3
          │ Failed call
    "#;

    let assert = create_command("assert_eq(1 + 2, 3 / 2)", "f64").assert();
    assert
        .failure()
        .code(ERROR_EXIT_CODE)
        .stdout(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn typing_error_simple() {
    const EXPECTED_ERR: &str = r#"
        error[TYPE]: Function expects 2 args, but is called with 3 args
          ┌─ Snippet #1:1:1
          │
        1 │ (1, 2, 3).map(|x| x, 1)
          │ ^^^^^^^^^^^^^^^^^^^^^^^ Error occurred here
    "#;

    let assert = create_command("(1, 2, 3).map(|x| x, 1)", "f64")
        .arg("--types")
        .assert();
    assert
        .failure()
        .code(ERROR_EXIT_CODE)
        .stdout(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn typing_error_complex() {
    const PROGRAM: &str = r#"
        all = |array, predicate| array.fold(true, |acc, x| acc && predicate(x));
        (1, 2, map).all(|x| 0 < x)
    "#;
    const EXPECTED_ERR: &str = r#"
        error[TYPE]: Type `(['T; N], ('T) -> 'U) -> ['U; N]` is not assignable to type `Num`
          ┌─ Snippet #1:2:8
          │
        2 │ (1, 2, map).all(|x| 0 < x)
          │        ^^^ Error occurred here
    "#;

    let assert = create_command(&unindent(PROGRAM), "f64")
        .arg("--types")
        .assert();
    assert
        .failure()
        .code(ERROR_EXIT_CODE)
        .stdout(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn multiple_typing_errors() {
    const EXPECTED_ERR: &str = r#"
        error[TYPE]: Type `(Num, Num)` is not assignable to type `Num`
          ┌─ Snippet #1:1:5
          │
        1 │ (1, (2, 3)).filter(|x| x + 1)
          │     ^^^^^^ Error occurred here

        error[TYPE]: Type `Num` is not assignable to type `Bool`
          ┌─ Snippet #1:1:20
          │
        1 │ (1, (2, 3)).filter(|x| x + 1)
          │                    ^^^^^^^^^ Error occurred here
    "#;

    let assert = create_command("(1, (2, 3)).filter(|x| x + 1)", "f64")
        .arg("--types")
        .assert();
    assert
        .failure()
        .code(ERROR_EXIT_CODE)
        .stdout(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}
