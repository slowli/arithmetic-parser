//! E2E tests for interactive binary usage.

use unindent::unindent;

use std::time::Duration;
use std::{
    io::{BufRead, BufReader, LineWriter, Write},
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc,
    thread::{self, JoinHandle},
};

const PATH_TO_BIN: &str = env!("CARGO_BIN_EXE_arithmetic-parser");

#[derive(Debug)]
struct ReplTester {
    repl_process: Child,
    stdin: LineWriter<ChildStdin>,
    io_handle: Option<JoinHandle<()>>,
    err_lines_rx: mpsc::Receiver<String>,
}

impl ReplTester {
    const TIMEOUT: Duration = Duration::from_millis(20);

    fn new(with_types: bool) -> Self {
        let mut command = Command::new(PATH_TO_BIN);
        command.env("NO_COLOR", "1").arg("eval");
        if with_types {
            command.arg("--types");
        }

        let mut repl_process = command
            .arg("-a")
            .arg("f64")
            .arg("-i")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("Cannot spawn repl");

        let stdout = repl_process.stdout.take().expect("REPL stderr");
        let stdout = BufReader::new(stdout);
        let stdin = repl_process.stdin.take().expect("REPL stdin");
        let stdin = LineWriter::new(stdin);

        let (err_lines_sx, err_lines_rx) = mpsc::channel();
        let io_handle = thread::spawn(move || {
            let mut lines = stdout.lines();
            while let Some(Ok(line)) = lines.next() {
                if err_lines_sx.send(line).is_err() {
                    break; // the receiver was dropped, we don't care any more
                }
            }
        });

        Self {
            repl_process,
            stdin,
            err_lines_rx,
            io_handle: Some(io_handle),
        }
    }

    fn wait_line(&self) -> String {
        self.err_lines_rx.recv().expect("failed to receive line")
    }

    fn assert_no_lines(&self) {
        let recv_result = self.err_lines_rx.recv_timeout(Self::TIMEOUT);
        assert!(matches!(recv_result, Err(mpsc::RecvTimeoutError::Timeout)));
    }

    fn assert_intro(&self) {
        let line = self.wait_line();
        assert!(line.contains("arithmetic-parser REPL"), "{}", line);
        let line = self.wait_line();
        assert!(
            line.starts_with("CLI / REPL for arithmetic expressions"),
            "{}",
            line
        );
        let line = self.wait_line();
        assert!(
            line.starts_with("Use .help for more information"),
            "{}",
            line
        );
        self.assert_no_lines();
    }

    fn send_line(&mut self, line: &str) -> String {
        self.assert_no_lines();
        writeln!(self.stdin, "{}", line).expect("send line to REPL");

        let mut lines = String::new();
        while let Ok(line) = self.err_lines_rx.recv_timeout(Self::TIMEOUT) {
            lines.push_str(&line);
            lines.push('\n');
        }

        if lines.ends_with('\n') {
            lines.truncate(lines.len() - 1);
        }
        lines
    }
}

impl Drop for ReplTester {
    fn drop(&mut self) {
        self.repl_process.kill().ok();
        self.repl_process.wait().ok();
        self.io_handle.take().unwrap().join().ok();
    }
}

#[test]
fn repl_basics() {
    let mut repl = ReplTester::new(false);
    repl.assert_intro();
    let response = repl.send_line("sin");
    assert!(response.contains("(native fn)"));
    let response = repl.send_line("is_positive = |x| x > 0;");
    assert!(response.is_empty());

    let response = repl.send_line("(1, 0, -1).map(is_positive)");
    let expected_tuple = r#"
        (
          true,
          false,
          false
        )
    "#;
    assert_eq!(response, unindent(expected_tuple).trim());

    let response = repl.send_line("is_positive");
    let expected_fn = r#"
        fn(1 arg)
    "#;
    assert_eq!(response, unindent(expected_fn).trim());
}

#[test]
fn incomplete_statements() {
    let mut repl = ReplTester::new(false);
    repl.assert_intro();

    let response = repl.send_line("sum = |...xs| {");
    assert!(response.is_empty());
    let response = repl.send_line("  xs.fold(0, |acc, x| acc + x)");
    assert!(response.is_empty());
    let response = repl.send_line("};");
    assert!(response.is_empty());
    let response = repl.send_line("sum(3, -5, 1)");
    assert_eq!(response, "-1");
}

#[test]
fn multiline_comments() {
    let mut repl = ReplTester::new(false);
    repl.assert_intro();

    let response = repl.send_line("x = 1; /* Comment starts");
    assert!(response.is_empty());
    let response = repl.send_line("You can put anything within a comment, really");
    assert!(response.is_empty());
    let response = repl.send_line("Comment ends */ x");
    assert_eq!(response, "1");
}

#[test]
fn undefined_var_error() {
    let mut repl = ReplTester::new(false);
    repl.assert_intro();

    let response = repl.send_line("foo(3)");
    let expected_err = r#"
        error[EVAL]: Variable `foo` is not defined
          ┌─ Snippet #1:1:1
          │
        1 │ foo(3)
          │ ^^^ Undefined variable occurrence
    "#;
    assert_eq!(response, unindent(expected_err));

    // We can still define a missing variable.
    let response = repl.send_line("foo = |x| x + 1;");
    assert!(response.is_empty());
    let response = repl.send_line("foo(3)");
    assert_eq!(response, "4");
}

#[test]
fn getting_help() {
    let mut repl = ReplTester::new(false);
    repl.assert_intro();

    let response = repl.send_line(".help");
    assert!(response.contains("Syntax is similar to Rust"));
    assert!(response.contains("Several commands are supported."));
}

#[test]
fn dumping_all_vars() {
    let mut repl = ReplTester::new(false);
    repl.assert_intro();

    let response = repl.send_line(".dump all");
    assert!(response.contains("PI = 3.1415"));
    assert!(response.contains("sin = (native fn)"));
}

#[test]
fn unknown_command() {
    let mut repl = ReplTester::new(false);
    repl.assert_intro();

    let response = repl.send_line(".exit");
    let expected_err = r#"
        error[CMD]: Unknown command
          ┌─ Snippet #1:1:1
          │
        1 │ .exit
          │ ^^^^^ Use `.help` to find out commands
    "#;
    assert_eq!(response, unindent(expected_err));
}

#[test]
fn variable_type() {
    let mut repl = ReplTester::new(true);
    repl.assert_intro();
    let response = repl.send_line("all = |xs, pred| xs.fold(true, |acc, x| acc && pred(x));");
    assert!(response.is_empty(), "{}", response);
    let ty = repl.send_line(".type all");
    assert_eq!(ty, "(['T; N], ('T) -> Bool) -> Bool");
}

#[test]
fn error_recovery() {
    let mut repl = ReplTester::new(true);
    repl.assert_intro();
    let response = repl.send_line("x = 1; y = !x;");
    assert!(response.contains("error[TYPE]"));
    let x_response = repl.send_line("x");
    assert_eq!(x_response, "1");
    let y_response = repl.send_line("y");
    assert!(
        y_response.contains("error[TYPE]") && y_response.contains("`y` is not defined"),
        "{}",
        y_response
    );
}

#[test]
fn error_recovery_on_error_in_return_value() {
    let mut repl = ReplTester::new(true);
    repl.assert_intro();
    let response = repl.send_line("x = 1; x + false");
    assert!(response.contains("error[TYPE]"));
    let x_response = repl.send_line("x");
    assert_eq!(x_response, "1");
}
