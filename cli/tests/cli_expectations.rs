use assert_cmd::prelude::*;
use predicates::prelude::*;
use unindent::unindent;

use std::process::Command;

fn create_command(program: &str) -> Command {
    let mut command = Command::cargo_bin(env!("CARGO_PKG_NAME")).expect("CLI binary");
    command
        .env("TERM", "dumb")
        .arg("-a")
        .arg("f64")
        .arg(program);
    command
}

#[test]
fn successful_execution() {
    let assert = create_command("1 + 2 * 3").assert();
    assert.success().stderr("7\n");
}

#[test]
fn outputting_native_function() {
    let assert = create_command("if").assert();
    assert.success().stderr("(native fn)\n");
}

#[test]
fn outputting_interpreted_function() {
    const PROGRAM: &str = "is_positive = |x| x > 0; is_positive";

    let assert = create_command(PROGRAM).assert();
    assert
        .success()
        .stderr("fn(1 args)[\n  > = (native fn)\n]\n");
}

#[test]
fn syntax_error() {
    const EXPECTED_ERR: &str = r#"
        error[PARSE]: Uninterpreted characters after parsing
          ┌─ Snip #1:1:5
          │
        1 │ let x = 5
          │     ^^^^^ Error occurred here
    "#;

    let assert = create_command("let x = 5").assert();
    assert
        .failure()
        .code(1)
        .stderr(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn undefined_variable_error() {
    const EXPECTED_ERR: &str = r#"
        error[EVAL]: Variable `x` is not defined
          ┌─ Snip #1:1:9
          │
        1 │ 1 + 2 * x
          │         ^ Undefined variable occurrence
    "#;

    let assert = create_command("1 + 2 * x").assert();
    assert
        .failure()
        .code(1)
        .stderr(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn incompatible_arg_count_error() {
    const EXPECTED_ERR: &str = r#"
        error[EVAL]: Mismatch between the number of arguments in the function definition and its call
          ┌─ Snip #1:1:1
          │
        1 │ if(2 > 1, 3)
          │ ^^^^^^^^^^^^ Called with 2 arg(s) here
    "#;

    let assert = create_command("if(2 > 1, 3)").assert();
    assert.failure().code(1).stderr(
        predicate::str::starts_with(unindent(EXPECTED_ERR))
            .and(predicate::str::contains("definition requires 3 arg(s)")),
    );
}

#[test]
fn error_with_call_trace() {
    const PROGRAM: &str = r#"
        is_positive = |x| x > 0;
        is_positive(3) && !is_positive((1, 2))
    "#;
    const EXPECTED_ERR: &str = r#"
        error[EVAL]: Compare requires 2 number arguments
          ┌─ Snip #1:2:27
          │
        2 │         is_positive = |x| x > 0;
          │                           ^^^^^
          │                           │
          │                           Failed call
          │                           Invalid argument
        3 │         is_positive(3) && !is_positive((1, 2))
          │                            ------------------- Call at depth 1
    "#;

    let assert = create_command(PROGRAM).assert();
    assert
        .failure()
        .code(1)
        .stderr(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}

#[test]
fn error_with_call_complex_call_trace() {
    const PROGRAM: &str = r#"
        all = |array, predicate| array.fold(true, |acc, x| acc && predicate(x));
        (1, 2, map).all(|x| 0 < x)
    "#;
    const EXPECTED_ERR: &str = r#"
        error[EVAL]: Compare requires 2 number arguments
          ┌─ Snip #1:2:34
          │
        2 │         all = |array, predicate| array.fold(true, |acc, x| acc && predicate(x));
          │                                  ----------------------------------------------
          │                                  │                                │
          │                                  │                                Call at depth 1
          │                                  Call at depth 2
        3 │         (1, 2, map).all(|x| 0 < x)
          │         --------------------^^^^^-
          │         │                   │   │
          │         │                   │   Invalid argument
          │         │                   Failed call
          │         Call at depth 3
    "#;

    let assert = create_command(PROGRAM).assert();
    assert
        .failure()
        .code(1)
        .stderr(predicate::str::starts_with(unindent(EXPECTED_ERR)));
}
