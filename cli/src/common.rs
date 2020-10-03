//! Common utils.

use codespan::{FileId, Files};
use codespan_reporting::{
    diagnostic::{Diagnostic, Label, Severity},
    term::termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor},
    term::{emit, Config as ReportingConfig},
};
use num_complex::{Complex, Complex32, Complex64};
use unindent::unindent;

use std::{
    collections::BTreeMap,
    fmt,
    io::{self, Write},
    ops::Range,
};

use arithmetic_eval::{
    fns, BacktraceElement, ErrorWithBacktrace, Function, Interpreter, Number, Value,
};
use arithmetic_parser::{grammars::NumGrammar, Error, Grammar, GrammarExt, Span, Spanned};

/// Exit code on parse or evaluation error.
pub const ERROR_EXIT_CODE: i32 = 2;

/// Code map containing evaluated code snippets.
#[derive(Debug, Default)]
pub struct CodeMap<'a> {
    files: Files<&'a str>,
    code_positions: BTreeMap<usize, FileId>,
    next_position: usize,
}

impl<'a> CodeMap<'a> {
    fn add(&mut self, source: &'a str) -> (FileId, usize) {
        let file_name = format!("Snip #{}", self.code_positions.len() + 1);
        let file_id = self.files.add(file_name, source);
        let start_position = self.next_position;
        self.code_positions.insert(start_position, file_id);
        self.next_position += source.len();
        (file_id, start_position)
    }

    fn locate<T>(&self, span: &Spanned<'_, T>) -> Option<(FileId, Range<usize>)> {
        if span.location_offset() > self.next_position {
            return None;
        }
        let (&file_start, &file_id) = self
            .code_positions
            .range(..=span.location_offset())
            .rev()
            .next()?;
        let start = span.location_offset() - file_start;
        let range = start..(start + span.fragment().len());
        Some((file_id, range))
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ParseAndEvalResult<T = ()> {
    Ok(T),
    Incomplete,
    Errored,
}

impl<T> ParseAndEvalResult<T> {
    fn map<U>(self, map_fn: impl FnOnce(T) -> U) -> ParseAndEvalResult<U> {
        match self {
            Self::Ok(value) => ParseAndEvalResult::Ok(map_fn(value)),
            Self::Incomplete => ParseAndEvalResult::Incomplete,
            Self::Errored => ParseAndEvalResult::Errored,
        }
    }
}

pub struct Env<'a> {
    code_map: CodeMap<'a>,
    writer: StandardStream,
    config: ReportingConfig,
}

impl<'a> Env<'a> {
    pub fn new() -> Self {
        Self {
            code_map: CodeMap::default(),
            writer: StandardStream::stderr(ColorChoice::Auto),
            config: ReportingConfig::default(),
        }
    }

    pub fn non_interactive(code: &'a str) -> Self {
        let mut this = Self::new();
        this.code_map.add(code);
        this
    }

    pub fn print_greeting(&mut self) -> io::Result<()> {
        let mut writer = self.writer.lock();
        writer.set_color(ColorSpec::new().set_bold(true))?;
        writeln!(
            writer,
            "arithmetic-parser REPL v{}",
            env!("CARGO_PKG_VERSION")
        )?;
        writer.reset()?;
        writeln!(writer, "{}", env!("CARGO_PKG_DESCRIPTION"))?;
        write!(writer, "Use ")?;
        writer.set_color(ColorSpec::new().set_bold(true))?;
        write!(writer, ".help")?;
        writer.reset()?;
        writeln!(
            writer,
            " for more information about supported commands / operations."
        )
    }

    fn print_help(&mut self) -> io::Result<()> {
        const HELP: &str = "
            REPL supports functions, blocks, methods, comparisons, etc.
            Syntax is similar to Rust; see `arithmetic-parser` docs for details.
            Use Ctrl+C / Cmd+C to exit the REPL.

            EXAMPLE
            Input each line separately.

                sins = (1, 2, 3).map(sin); sins
                min_sin = sins.fold(INF, min); min_sin
                assert(min_sin > 0);

            COMMANDS
            Several commands are supported. All commands start with a dot '.'.

                .help     Displays help.
                .dump     Outputs all defined variables. Use '.dump all' to include
                          built-in vars.
                .clear    Resets the interpreter state to the original one.
        ";

        let mut writer = self.writer.lock();
        writeln!(writer, "{}", unindent(HELP))
    }

    /// Reports a parsing error.
    pub fn report_parse_error(&self, err: Spanned<'_, Error<'_>>) -> io::Result<()> {
        let (file, range) = self
            .code_map
            .locate(&err)
            .expect("Cannot locate parse error span");
        let label = Label::primary(file, range).with_message("Error occurred here");
        let diagnostic = Diagnostic::error()
            .with_message(err.extra.to_string())
            .with_code("PARSE")
            .with_labels(vec![label]);

        emit(
            &mut self.writer.lock(),
            &self.config,
            &self.code_map.files,
            &diagnostic,
        )
    }

    pub fn report_eval_error(&self, e: ErrorWithBacktrace) -> io::Result<()> {
        let severity = Severity::Error;
        let (file, range) = self
            .code_map
            .locate(&e.main_span())
            .expect("Cannot locate main error span");
        let main_label = Label::primary(file, range);
        let message = e.source().main_span_info();

        let mut labels = vec![main_label.with_message(message)];
        for aux_span in e.aux_spans() {
            let (file, range) = self
                .code_map
                .locate(&aux_span)
                .expect("Cannot locate aux error span");
            let label = Label::primary(file, range).with_message(aux_span.extra.to_string());
            labels.push(label);
        }

        let mut calls_iter = e.backtrace().calls();
        if let Some(BacktraceElement {
            fn_name, def_span, ..
        }) = calls_iter.next()
        {
            if let Some(def_span) = def_span {
                let (file_id, def_range) = self
                    .code_map
                    .locate(&def_span)
                    .expect("Cannot locate span in previously recorded snippets");
                let def_label = Label::secondary(file_id, def_range)
                    .with_message(format!("The error occurred in function `{}`", fn_name));
                labels.push(def_label);
            }

            let mut call_site;
            for (depth, call) in calls_iter.enumerate() {
                call_site = call.call_span;
                let (file_id, call_range) = self
                    .code_map
                    .locate(&call_site)
                    .expect("Cannot locate span in previously recorded snippets");
                let call_label = Label::secondary(file_id, call_range)
                    .with_message(format!("Call at depth {}", depth + 1));
                labels.push(call_label);
            }
        }

        let mut diagnostic = Diagnostic::new(severity)
            .with_message(e.source().to_short_string())
            .with_code("EVAL")
            .with_labels(labels);

        if let Some(help) = e.source().help() {
            diagnostic = diagnostic.with_notes(vec![help]);
        }

        emit(
            &mut self.writer.lock(),
            &self.config,
            &self.code_map.files,
            &diagnostic,
        )
    }

    pub fn writeln_value<T>(&mut self, value: &Value<T>) -> io::Result<()>
    where
        T: Grammar,
        T::Lit: fmt::Display,
    {
        self.dump_value(value, 0)?;
        writeln!(self.writer)
    }

    fn dump_value<T>(&mut self, value: &Value<T>, indent: usize) -> io::Result<()>
    where
        T: Grammar,
        T::Lit: fmt::Display,
    {
        let bool_color = ColorSpec::new().set_fg(Some(Color::Cyan)).clone();
        let num_color = ColorSpec::new().set_fg(Some(Color::Green)).clone();

        match value {
            Value::Bool(b) => {
                self.writer.set_color(&bool_color)?;
                write!(self.writer, "{}", if *b { "true" } else { "false" })?;
                self.writer.reset()
            }

            Value::Function(Function::Native(_)) => write!(self.writer, "(native fn)"),

            Value::Function(Function::Interpreted(function)) => {
                write!(self.writer, "fn({} args)", function.arg_count())?;
                let captures = function.captures();
                if !captures.is_empty() {
                    writeln!(self.writer, "[")?;
                    for (i, (var_name, capture)) in captures.iter().enumerate() {
                        write!(self.writer, "{}{} = ", " ".repeat(indent + 2), var_name)?;
                        self.dump_value(capture, indent + 2)?;
                        if i + 1 < captures.len() {
                            writeln!(self.writer, ",")?;
                        } else {
                            writeln!(self.writer)?;
                        }
                    }
                    write!(self.writer, "{}]", " ".repeat(indent))?;
                }
                Ok(())
            }

            Value::Number(num) => {
                self.writer.set_color(&num_color)?;
                write!(self.writer, "{}", num)?;
                self.writer.reset()
            }

            Value::Tuple(fragments) => {
                writeln!(self.writer, "(")?;
                for (i, fragment) in fragments.iter().enumerate() {
                    write!(self.writer, "{}", " ".repeat(indent + 2))?;
                    self.dump_value(fragment, indent + 2)?;
                    if i + 1 < fragments.len() {
                        writeln!(self.writer, ",")?;
                    } else {
                        writeln!(self.writer)?;
                    }
                }
                write!(self.writer, "{})", " ".repeat(indent))
            }
        }
    }

    fn dump_scope<T>(
        &mut self,
        scope: &Interpreter<'a, T>,
        original_scope: &Interpreter<'a, T>,
        dump_original_scope: bool,
    ) -> io::Result<()>
    where
        T: Grammar,
        T::Lit: PartialEq + fmt::Display,
    {
        for (name, var) in scope.variables() {
            if let Some(original_var) = original_scope.get_var(name) {
                if !dump_original_scope && original_var == var {
                    // The variable is present in the original scope, no need to output it.
                    continue;
                }
            }

            write!(self.writer, "{} = ", name)?;
            self.dump_value(var, 0)?;
            writeln!(self.writer)?;
        }
        Ok(())
    }

    pub fn parse_and_eval<T>(
        &mut self,
        line: &'a str,
        interpreter: &mut Interpreter<'a, T>,
        original_interpreter: &Interpreter<'a, T>,
    ) -> io::Result<ParseAndEvalResult>
    where
        T: Grammar,
        T::Lit: fmt::Display + Number,
    {
        let (file, start_position) = self.code_map.add(line);
        let visible_span = 0..line.len();

        if line.starts_with('.') {
            match line {
                ".clear" => interpreter.clone_from(original_interpreter),
                ".dump" => self.dump_scope(interpreter, original_interpreter, false)?,
                ".dump all" => self.dump_scope(interpreter, original_interpreter, true)?,
                ".help" => self.print_help()?,

                _ => {
                    let label = Label::primary(file, visible_span)
                        .with_message("Use `.help commands` to find out commands");
                    let diagnostic = Diagnostic::error()
                        .with_message("Unknown command")
                        .with_code("CMD")
                        .with_labels(vec![label]);

                    emit(
                        &mut self.writer.lock(),
                        &self.config,
                        &self.code_map.files,
                        &diagnostic,
                    )?;
                }
            }

            return Ok(ParseAndEvalResult::Ok(()));
        }

        let span = unsafe {
            // SAFETY: We do not traverse the portion of the program preceding the `span`
            // (this could lead to UB since `line` is not necessarily sliced from a larger program).
            // Instead, the span offset is used for diagnostic messages only.
            Span::new_from_raw_offset(start_position, 1, line, ())
        };
        let parse_result = T::parse_streaming_statements(span)
            .map(ParseAndEvalResult::Ok)
            .or_else(|e| {
                if let Error::Incomplete = e.extra {
                    Ok(ParseAndEvalResult::Incomplete)
                } else {
                    self.report_parse_error(e)
                        .map(|()| ParseAndEvalResult::Errored)
                }
            })?;

        Ok(if let ParseAndEvalResult::Ok(statements) = parse_result {
            match interpreter.evaluate(&statements) {
                Ok(value) => {
                    if !value.is_void() {
                        self.dump_value(&value, 0)?;
                    }
                    ParseAndEvalResult::Ok(())
                }
                Err(err) => {
                    self.report_eval_error(err)?;
                    ParseAndEvalResult::Errored
                }
            }
        } else {
            parse_result.map(drop)
        })
    }
}

fn init_interpreter<'a, T: Number>() -> Interpreter<'a, NumGrammar<T>> {
    Interpreter::<NumGrammar<T>>::with_prelude()
}

pub trait ReplLiteral: Number + fmt::Display {
    fn create_interpreter<'a>() -> Interpreter<'a, NumGrammar<Self>>;
}

#[derive(Debug, Clone, Copy)]
#[allow(clippy::type_complexity)] // not that complex, really
pub struct StdLibrary<T: 'static> {
    constants: &'static [(&'static str, T)],
    unary: &'static [(&'static str, fn(T) -> T)],
    binary: &'static [(&'static str, fn(T, T) -> T)],
}

impl<T: Number> StdLibrary<T> {
    fn add_to_interpreter(self, interpreter: &mut Interpreter<NumGrammar<T>>) {
        for (name, c) in self.constants {
            interpreter.insert_var(name, Value::Number(*c));
        }
        for (name, unary_fn) in self.unary {
            interpreter.insert_native_fn(name, fns::Unary::new(*unary_fn));
        }
        for (name, binary_fn) in self.binary {
            interpreter.insert_native_fn(name, fns::Binary::new(*binary_fn));
        }
    }
}

macro_rules! declare_real_functions {
    ($functions:ident : $type:ident) => {
        const $functions: StdLibrary<$type> = StdLibrary {
            constants: &[("E", std::$type::consts::E), ("PI", std::$type::consts::PI)],

            unary: &[
                // Rounding functions.
                ("floor", $type::floor),
                ("ceil", $type::ceil),
                ("round", $type::round),
                ("frac", $type::fract),
                // Exponential functions.
                ("exp", $type::exp),
                ("ln", $type::ln),
                ("sinh", $type::sinh),
                ("cosh", $type::cosh),
                ("tanh", $type::tanh),
                ("asinh", $type::asinh),
                ("acosh", $type::acosh),
                ("atanh", $type::atanh),
                // Trigonometric functions.
                ("sin", $type::sin),
                ("cos", $type::cos),
                ("tan", $type::tan),
                ("asin", $type::asin),
                ("acos", $type::acos),
                ("atan", $type::atan),
            ],

            binary: &[
                ("min", |x, y| if x < y { x } else { y }),
                ("max", |x, y| if x > y { x } else { y }),
            ],
        };
    };
}

declare_real_functions!(F32_FUNCTIONS: f32);
declare_real_functions!(F64_FUNCTIONS: f64);

impl ReplLiteral for f32 {
    fn create_interpreter<'a>() -> Interpreter<'a, NumGrammar<Self>> {
        let mut interpreter = init_interpreter::<f32>();
        F32_FUNCTIONS.add_to_interpreter(&mut interpreter);
        interpreter.insert_native_fn("cmp", fns::Compare);
        interpreter
    }
}

impl ReplLiteral for f64 {
    fn create_interpreter<'a>() -> Interpreter<'a, NumGrammar<Self>> {
        let mut interpreter = init_interpreter::<f64>();
        F64_FUNCTIONS.add_to_interpreter(&mut interpreter);
        interpreter.insert_native_fn("cmp", fns::Compare);
        interpreter
    }
}

macro_rules! declare_complex_functions {
    ($functions:ident : $type:ident, $real:ident) => {
        const $functions: StdLibrary<$type> = StdLibrary {
            constants: &[
                ("E", Complex::new(std::$real::consts::E, 0.0)),
                ("PI", Complex::new(std::$real::consts::PI, 0.0)),
            ],

            unary: &[
                ("norm", |x| Complex::new(x.norm(), 0.0)),
                ("arg", |x| Complex::new(x.arg(), 0.0)),
                // Exponential functions.
                ("exp", |x| x.exp()),
                ("ln", |x| x.ln()),
                ("sinh", |x| x.sinh()),
                ("cosh", |x| x.cosh()),
                ("tanh", |x| x.tanh()),
                ("asinh", |x| x.asinh()),
                ("acosh", |x| x.acosh()),
                ("atanh", |x| x.atanh()),
                // Trigonometric functions.
                ("sin", |x| x.sin()),
                ("cos", |x| x.cos()),
                ("tan", |x| x.tan()),
                ("asin", |x| x.asin()),
                ("acos", |x| x.acos()),
                ("atan", |x| x.atan()),
            ],

            binary: &[],
        };
    };
}

declare_complex_functions!(COMPLEX32_FUNCTIONS: Complex32, f32);
declare_complex_functions!(COMPLEX64_FUNCTIONS: Complex64, f64);

impl ReplLiteral for Complex32 {
    fn create_interpreter<'a>() -> Interpreter<'a, NumGrammar<Self>> {
        let mut interpreter = init_interpreter::<Complex32>();
        COMPLEX32_FUNCTIONS.add_to_interpreter(&mut interpreter);
        interpreter
    }
}

impl ReplLiteral for Complex64 {
    fn create_interpreter<'a>() -> Interpreter<'a, NumGrammar<Self>> {
        let mut interpreter = init_interpreter::<Complex64>();
        COMPLEX64_FUNCTIONS.add_to_interpreter(&mut interpreter);
        interpreter
    }
}
