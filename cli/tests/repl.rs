//! E2E tests for interactive binary usage.

use assert_cmd::cargo::CommandCargoExt;
use unindent::unindent;

use std::time::Duration;
use std::{
    io::{BufRead, BufReader, LineWriter, Write},
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc,
    thread::{self, JoinHandle},
};

#[derive(Debug)]
struct ReplTester {
    repl_process: Child,
    stdin: LineWriter<ChildStdin>,
    io_handle: Option<JoinHandle<()>>,
    err_lines_rx: mpsc::Receiver<String>,
}

impl ReplTester {
    const TIMEOUT: Duration = Duration::from_millis(20);

    fn new() -> Self {
        let mut repl_process = Command::cargo_bin(env!("CARGO_PKG_NAME"))
            .expect("CLI binary")
            .env("TERM", "dumb")
            .arg("-a")
            .arg("f64")
            .arg("-i")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Cannot spawn repl");

        let stderr = repl_process.stderr.take().expect("REPL stderr");
        let stderr = BufReader::new(stderr);
        let stdin = repl_process.stdin.take().expect("REPL stdin");
        let stdin = LineWriter::new(stdin);

        let (err_lines_sx, err_lines_rx) = mpsc::channel();
        let io_handle = thread::spawn(move || {
            let mut lines = stderr.lines();
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
    let mut repl = ReplTester::new();
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
        fn(1 arg)[
          cmp = (native fn)
        ]
    "#;
    assert_eq!(response, unindent(expected_fn).trim());
}

#[test]
fn incomplete_statements() {
    let mut repl = ReplTester::new();
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
    let mut repl = ReplTester::new();
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
    let mut repl = ReplTester::new();
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
