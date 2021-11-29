//! E2E tests for a non-interactive binary usage.

// Some tests use multi-line formatting, which is awkward to achieve on Windows.
#![cfg(unix)]

use term_transcript::{
    svg::{ScrollOptions, Template, TemplateOptions},
    test::{MatchKind, TestConfig},
    ShellOptions,
};

fn test_config() -> TestConfig {
    let shell_options = ShellOptions::default()
        .with_env("COLOR", "always")
        .with_cargo_path();
    TestConfig::new(shell_options).with_match_kind(MatchKind::Precise)
}

fn scroll_template() -> Template {
    Template::new(TemplateOptions {
        scroll: Some(ScrollOptions::default()),
        ..TemplateOptions::default()
    })
}

#[test]
fn successful_execution_for_ast() {
    test_config().with_template(scroll_template()).test(
        "tests/snapshots/ast.svg",
        &["arithmetic-parser ast -a u64 '1 + 2 * 3'"],
    );
}

#[test]
fn successful_execution_for_arithmetics() {
    test_config().test(
        "tests/snapshots/simple.svg",
        &[
            "arithmetic-parser eval -a u64 '1 + 2 * 3'",
            "arithmetic-parser eval -a i64 '1 - 2 * 3'",
            "arithmetic-parser eval -a u128 '2 ^ 71 - 1'",
            "arithmetic-parser eval -a u64 --wrapping '1 - 2 + 3'",
            "arithmetic-parser eval -a f32 '1 + 3 / 2'",
            "arithmetic-parser eval -a c64 '2 / (1 - i)'",
            "arithmetic-parser eval -a u64/11 '5 / 9'",
        ],
    );
}

#[test]
fn evaluating_functions() {
    test_config().test(
        "tests/snapshots/functions.svg",
        &[
            "arithmetic-parser eval 'if(5 > 3, 1, -1)'",
            "arithmetic-parser eval 'if'",
            "arithmetic-parser eval 'is_positive = |x| x > 0; is_positive'",
        ],
    );
}

#[test]
fn syntax_errors() {
    test_config().with_template(scroll_template()).test(
        "tests/snapshots/errors-ast.svg",
        &[
            "arithmetic-parser ast 'let x = 5'",
            "arithmetic-parser eval 'let x = 5'",
            "arithmetic-parser ast 'x = {'",
            "arithmetic-parser eval 'x = {'",
        ],
    );
}

#[test]
fn eval_integer_errors() {
    test_config().test(
        "tests/snapshots/errors-int.svg",
        &[
            "arithmetic-parser eval -a u64 '1 - 3 + 5'",
            "arithmetic-parser eval -a i64 '20 ^ 20'",
            "arithmetic-parser eval -a i128 '10 ^ -3'",
        ],
    );
}

#[test]
fn eval_basic_errors() {
    test_config().test(
        "tests/snapshots/errors-basic.svg",
        &[
            "arithmetic-parser eval '1 + 2 * x'",
            "arithmetic-parser eval 'if(2 > 1, 3)'",
            "arithmetic-parser eval 'assert_eq(1 + 2, 3 / 2)'",
        ],
    );
}

#[test]
fn error_with_call_trace() {
    test_config().test(
        "tests/snapshots/errors-call-trace.svg",
        &["arithmetic-parser eval '\n  \
             is_positive = |x| x > 0;\n  \
             is_positive(3) && !is_positive((1, 2))'"],
    );
}

#[test]
fn error_with_complex_call_trace() {
    test_config().test(
        "tests/snapshots/errors-complex-call-trace.svg",
        &["arithmetic-parser eval '\n  \
             double = |x| x * 2;\n  \
             quadruple = |x| double(double(x));\n  \
             quadruple(true)'"],
    );
}

#[test]
fn error_with_call_complex_call_trace_and_native_fns() {
    test_config().test(
        "tests/snapshots/errors-native-call-trace.svg",
        &["arithmetic-parser eval '\n  \
             all = |array, pred| array.fold(true, |acc, x| acc && pred(x));\n  \
             (1, 2, map).all(|x| 0 < x)'"],
    );
}

#[test]
fn typing_errors_simple() {
    test_config().test(
        "tests/snapshots/errors-typing.svg",
        &[
            "arithmetic-parser eval --types '(1, 2, 3).map(|x| x, 1)'",
            "arithmetic-parser eval --types '\n  \
             all = |array, pred| array.fold(true, |acc, x| acc && pred(x));\n  \
             (1, 2, map).all(|x| 0 < x)'",
        ],
    );
}

#[test]
fn multiple_typing_errors() {
    test_config().test(
        "tests/snapshots/errors-typing-multiple.svg",
        &["arithmetic-parser eval --types '(1, (2, 3)).filter(|x| x + 1)'"],
    );
}
