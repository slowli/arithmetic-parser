//! E2E tests for interactive binary usage.

use term_transcript::{
    test::{MatchKind, TestConfig},
    ShellOptions, UserInput,
};

use std::{process::Command, time::Duration};

const PATH_TO_BIN: &str = env!("CARGO_BIN_EXE_arithmetic-parser");

fn test_config(with_types: bool) -> TestConfig {
    let mut command = Command::new(PATH_TO_BIN);
    command.arg("eval");
    if with_types {
        command.arg("--types");
    }
    command.arg("-a").arg("f64").arg("-i");

    let shell_options = ShellOptions::new(command)
        .with_env("COLOR", "always")
        .with_io_timeout(Duration::from_millis(250));
    TestConfig::new(shell_options).with_match_kind(MatchKind::Precise)
}

// Helper commands to create `UserInput`s.

#[inline]
fn repl(input: &str) -> UserInput {
    UserInput::repl(input)
}

#[inline]
fn cont(input: &str) -> UserInput {
    UserInput::repl_continuation(input)
}

#[test]
fn repl_basics() {
    test_config(false).test(
        "tests/snapshots/repl/basics.svg",
        vec![
            repl("sin"),
            repl("is_positive = |x| x > 0;"),
            repl("(1, 0, -1).map(is_positive)"),
            repl("is_positive"),
        ],
    );
}

#[test]
fn incomplete_statements() {
    test_config(false).test(
        "tests/snapshots/repl/incomplete.svg",
        vec![
            repl("sum = |...xs| {"),
            cont("  xs.fold(0, |acc, x| acc + x)"),
            cont("};"),
            repl("sum(3, -5, 1)"),
            repl("x = 1; /* Comment starts"),
            cont("You can put anything within a comment, really"),
            cont("Comment ends */ x"),
        ],
    );
}

#[test]
fn undefined_var_error() {
    test_config(false).test(
        "tests/snapshots/repl/errors-var.svg",
        vec![repl("foo(3)"), repl("foo = |x| x + 1;"), repl("foo(3)")],
    );
}

#[test]
fn getting_help() {
    test_config(false).test("tests/snapshots/repl/help.svg", vec![repl(".help")]);
}

// TODO: more complex input doesn't work because of non-deterministic var order
#[test]
fn dumping_vars() {
    test_config(false).test(
        "tests/snapshots/repl/dump.svg",
        vec![repl("xs = (1, 2, || PI + 3);"), repl(".dump")],
    );
}

#[test]
fn unknown_command() {
    test_config(false).test(
        "tests/snapshots/repl/errors-command.svg",
        vec![repl(".exit")],
    );
}

#[test]
fn variable_type() {
    test_config(true).test(
        "tests/snapshots/repl/type.svg",
        vec![
            repl("tuple = (1, #{ x: 3, y: 4 });"),
            repl(".type tuple"),
            repl("all = |xs, pred| xs.fold(true, |acc, x| acc && pred(x));"),
            repl(".type all"),
            repl(".type non_existing_var"),
        ],
    );
}

#[test]
fn error_recovery() {
    test_config(true).test(
        "tests/snapshots/repl/errors-recovery.svg",
        vec![
            repl("x = 1; y = !x;"),
            repl("x // Should be defined since an error occurs in a later stmt"),
            repl("y"),
        ],
    );
}
