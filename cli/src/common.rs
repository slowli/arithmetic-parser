//! Common utils.

use codespan::{FileId, Files};
use codespan_reporting::{
    diagnostic::{Diagnostic, Label, Severity},
    term::termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor},
    term::{emit, Config as ReportingConfig},
};
use unindent::unindent;

use std::{
    io::{self, Write},
    ops::Range,
};

use arithmetic_eval::{
    arith::OrdArithmetic,
    error::{BacktraceElement, CodeInModule, ErrorWithBacktrace},
    Environment, Error as EvalError, Function, IndexedId, ModuleId, Value, VariableMap,
};
use arithmetic_parser::{
    grammars::{Grammar, NumGrammar, Parse, Untyped},
    Block, CodeFragment, Error as ParseError, LocatedSpan, LvalueLen, StripCode,
};

use crate::library::ReplLiteral;

/// Exit code on parse or evaluation error.
pub const ERROR_EXIT_CODE: i32 = 2;

/// Helper trait for possible `LocationSpan.fragment` types.
trait SizedFragment {
    fn fragment_len(&self) -> usize;
}

impl SizedFragment for &str {
    fn fragment_len(&self) -> usize {
        self.len()
    }
}

impl SizedFragment for CodeFragment<'_> {
    fn fragment_len(&self) -> usize {
        self.len()
    }
}

/// Code map containing evaluated code snippets.
#[derive(Debug, Default)]
pub struct CodeMap {
    files: Files<String>,
    file_ids: Vec<FileId>, // necessary because `FileId` is opaque
}

impl CodeMap {
    const SNIPPET_PREFIX: &'static str = "Snippet";

    fn add(&mut self, source: String) -> FileId {
        let file_name = IndexedId::new(Self::SNIPPET_PREFIX, self.file_ids.len()).to_string();
        let file_id = self.files.add(file_name, source);
        self.file_ids.push(file_id);
        file_id
    }

    fn locate<Span: SizedFragment, T>(
        &self,
        module_id: &dyn ModuleId,
        span: &LocatedSpan<Span, T>,
    ) -> (FileId, Range<usize>) {
        let snippet = module_id
            .downcast_ref::<IndexedId>()
            .expect("Module ID is not an IndexedId");
        let file_id = self.file_ids[snippet.index];

        let start = span.location_offset();
        let range = start..(start + span.fragment().fragment_len());
        (file_id, range)
    }

    fn locate_in_most_recent_file<Span: SizedFragment, T>(
        &self,
        span: &LocatedSpan<Span, T>,
    ) -> (FileId, Range<usize>) {
        let file_id = *self.file_ids.last().expect("No files");
        let start = span.location_offset();
        let range = start..(start + span.fragment().fragment_len());
        (file_id, range)
    }

    fn latest_module_id(&self) -> IndexedId {
        IndexedId::new(Self::SNIPPET_PREFIX, self.file_ids.len() - 1)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ParseAndEvalResult<T = ()> {
    Ok(T),
    Incomplete,
    Errored,
}

impl<T> ParseAndEvalResult<T> {
    pub fn map<U>(self, map_fn: impl FnOnce(T) -> U) -> ParseAndEvalResult<U> {
        match self {
            Self::Ok(value) => ParseAndEvalResult::Ok(map_fn(value)),
            Self::Incomplete => ParseAndEvalResult::Incomplete,
            Self::Errored => ParseAndEvalResult::Errored,
        }
    }
}

pub struct Env<T> {
    code_map: CodeMap,
    writer: StandardStream,
    config: ReportingConfig,
    original_env: Environment<'static, T>,
    env: Environment<'static, T>,
    arithmetic: Box<dyn OrdArithmetic<T>>,
}

impl<T: ReplLiteral> Env<T> {
    pub fn new(arithmetic: Box<dyn OrdArithmetic<T>>, env: Environment<'static, T>) -> Self {
        Self {
            code_map: CodeMap::default(),
            writer: StandardStream::stderr(ColorChoice::Auto),
            config: ReportingConfig::default(),
            original_env: env.clone(),
            env,
            arithmetic,
        }
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
    pub fn report_parse_error(&self, err: ParseError<'_>) -> io::Result<()> {
        // Parsing errors are always reported for the most recently added snippet.
        let (file, range) = self.code_map.locate_in_most_recent_file(&err.span());

        let label = Label::primary(file, range).with_message("Error occurred here");
        let diagnostic = Diagnostic::error()
            .with_message(err.kind().to_string())
            .with_code("PARSE")
            .with_labels(vec![label]);

        emit(
            &mut self.writer.lock(),
            &self.config,
            &self.code_map.files,
            &diagnostic,
        )
    }

    fn create_diagnostic(&self, e: &EvalError<'_>) -> Diagnostic<FileId> {
        let main_span = e.main_span();
        let (file, range) = self
            .code_map
            .locate(main_span.module_id(), main_span.code());
        let main_label = Label::primary(file, range);
        let message = e.kind().main_span_info();

        let mut labels = vec![main_label.with_message(message)];
        for aux_span in e.aux_spans() {
            let (file, range) = self.code_map.locate(aux_span.module_id(), aux_span.code());
            let label = Label::primary(file, range).with_message(aux_span.code().extra.to_string());
            labels.push(label);
        }

        let mut diagnostic = Diagnostic::new(Severity::Error)
            .with_message(e.kind().to_short_string())
            .with_code("EVAL")
            .with_labels(labels);

        if let Some(help) = e.kind().help() {
            diagnostic = diagnostic.with_notes(vec![help]);
        }

        diagnostic
    }

    pub fn report_compile_error(&self, e: &EvalError<'_>) -> io::Result<()> {
        emit(
            &mut self.writer.lock(),
            &self.config,
            &self.code_map.files,
            &self.create_diagnostic(e),
        )
    }

    pub fn report_eval_error(&self, e: &ErrorWithBacktrace<'_>) -> io::Result<()> {
        let mut diagnostic = self.create_diagnostic(e.source());

        let mut calls_iter = e.backtrace().peekable();
        if let Some(BacktraceElement {
            fn_name, def_span, ..
        }) = calls_iter.peek()
        {
            if let Some(def_span) = def_span {
                let (file_id, def_range) =
                    self.code_map.locate(def_span.module_id(), &def_span.code());
                let def_label = Label::secondary(file_id, def_range)
                    .with_message(format!("The error occurred in function `{}`", fn_name));
                diagnostic.labels.push(def_label);
            }

            for (depth, call) in calls_iter.enumerate() {
                let call_span = &call.call_span;
                if Self::spans_are_equal(call_span, e.source().main_span()) {
                    // The span is already output.
                    continue;
                }

                let (file_id, call_range) = self
                    .code_map
                    .locate(call_span.module_id(), &call_span.code());
                let call_label = Label::secondary(file_id, call_range)
                    .with_message(format!("Call at depth {}", depth + 1));
                diagnostic.labels.push(call_label);
            }
        }

        emit(
            &mut self.writer.lock(),
            &self.config,
            &self.code_map.files,
            &diagnostic,
        )
    }

    fn spans_are_equal(span: &CodeInModule<'_>, other: &CodeInModule<'_>) -> bool {
        span.code() == other.code()
            && span.module_id().downcast_ref::<IndexedId>()
                == other.module_id().downcast_ref::<IndexedId>()
    }

    pub fn newline(&mut self) -> io::Result<()> {
        writeln!(self.writer)
    }

    fn dump_value(
        writer: &mut StandardStream,
        value: &Value<'_, T>,
        indent: usize,
    ) -> io::Result<()> {
        let bool_color = ColorSpec::new().set_fg(Some(Color::Cyan)).clone();
        let num_color = ColorSpec::new().set_fg(Some(Color::Green)).clone();
        let opaque_ref_color = ColorSpec::new()
            .set_fg(Some(Color::Black))
            .set_bg(Some(Color::White))
            .clone();

        match value {
            Value::Bool(b) => {
                writer.set_color(&bool_color)?;
                write!(writer, "{}", if *b { "true" } else { "false" })?;
                writer.reset()
            }

            Value::Function(Function::Native(_)) => write!(writer, "(native fn)"),

            Value::Function(Function::Interpreted(function)) => {
                let plurality = if function.arg_count() == LvalueLen::Exact(1) {
                    ""
                } else {
                    "s"
                };
                write!(writer, "fn({} arg{})", function.arg_count(), plurality)?;

                let captures = function.captures();
                if !captures.is_empty() {
                    writeln!(writer, "[")?;
                    for (i, (var_name, capture)) in captures.iter().enumerate() {
                        write!(writer, "{}{} = ", " ".repeat(indent + 2), var_name)?;
                        Self::dump_value(writer, capture, indent + 2)?;
                        if i + 1 < captures.len() {
                            writeln!(writer, ",")?;
                        } else {
                            writeln!(writer)?;
                        }
                    }
                    write!(writer, "{}]", " ".repeat(indent))?;
                }
                Ok(())
            }

            Value::Number(num) => {
                writer.set_color(&num_color)?;
                write!(writer, "{}", num)?;
                writer.reset()
            }

            Value::Tuple(fragments) => {
                writeln!(writer, "(")?;
                for (i, fragment) in fragments.iter().enumerate() {
                    write!(writer, "{}", " ".repeat(indent + 2))?;
                    Self::dump_value(writer, fragment, indent + 2)?;
                    if i + 1 < fragments.len() {
                        writeln!(writer, ",")?;
                    } else {
                        writeln!(writer)?;
                    }
                }
                write!(writer, "{})", " ".repeat(indent))
            }

            Value::Ref(opaque_ref) => {
                writer.set_color(&opaque_ref_color)?;
                write!(writer, "{}", opaque_ref)?;
                writer.reset()
            }

            _ => unreachable!(),
        }
    }

    fn dump_scope(&mut self, dump_original_scope: bool) -> io::Result<()> {
        for (name, var) in &self.env {
            if let Some(original_var) = self.original_env.get(name) {
                if !dump_original_scope && original_var == var {
                    // The variable is present in the original scope, no need to output it.
                    continue;
                }
            }

            write!(self.writer, "{} = ", name)?;
            Self::dump_value(&mut self.writer, var, 0)?;
            writeln!(self.writer)?;
        }
        Ok(())
    }

    pub fn parse<'a, P: Parse>(
        &mut self,
        line: &'a str,
    ) -> io::Result<ParseAndEvalResult<Block<'a, P::Base>>> {
        self.code_map.add(line.to_owned());

        P::parse_streaming_statements(line)
            .map(ParseAndEvalResult::Ok)
            .or_else(|e| {
                if e.kind().is_incomplete() {
                    Ok(ParseAndEvalResult::Incomplete)
                } else {
                    self.report_parse_error(e)
                        .map(|()| ParseAndEvalResult::Errored)
                }
            })
    }

    pub fn parse_and_eval(&mut self, line: &str) -> io::Result<ParseAndEvalResult> {
        if line.starts_with('.') {
            self.process_command(line)?;
            return Ok(ParseAndEvalResult::Ok(()));
        }

        let parse_result = self.parse::<Untyped<NumGrammar<T>>>(line)?;
        Ok(if let ParseAndEvalResult::Ok(statements) = parse_result {
            self.compile_and_execute(&statements)?
        } else {
            parse_result.map(drop)
        })
    }

    fn process_command(&mut self, line: &str) -> io::Result<()> {
        let file_id = *self.code_map.file_ids.last().expect("no files");

        match line {
            ".clear" => self.env.clone_from(&self.original_env),
            ".dump" => self.dump_scope(false)?,
            ".dump all" => self.dump_scope(true)?,
            ".help" => self.print_help()?,

            _ => {
                // FIXME: incorrect span on error!
                let label = Label::primary(file_id, 0..line.len())
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

        Ok(())
    }

    fn compile_and_execute<G>(
        &mut self,
        statements: &Block<'_, G>,
    ) -> io::Result<ParseAndEvalResult>
    where
        G: Grammar<Lit = T>,
    {
        let module_id = self.code_map.latest_module_id();
        let module = match self.env.compile_module(module_id, statements) {
            Ok(builder) => builder,
            Err(err) => {
                self.report_compile_error(&err)?;
                return Ok(ParseAndEvalResult::Errored);
            }
        };

        let value = match module
            .strip_code()
            .with_arithmetic(self.arithmetic.as_ref())
            .run_in_env(&mut self.env)
        {
            Ok(value) => value,
            Err(err) => {
                self.report_eval_error(&err)?;
                return Ok(ParseAndEvalResult::Errored);
            }
        };

        if !value.is_void() {
            Self::dump_value(&mut self.writer, &value, 0)?;
            self.newline()?;
        }
        Ok(ParseAndEvalResult::Ok(()))
    }
}
