//! Common utils.

use codespan::{FileId, Files};
use codespan_reporting::{
    diagnostic::{Diagnostic, Label, Severity},
    files::Error as FilesError,
    term::termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor},
    term::{emit, Config as ReportingConfig},
};
use unindent::unindent;

use std::{
    io::{self, Write},
    ops::Range,
};

use arithmetic_eval::{
    error::{BacktraceElement, Error as EvalError, ErrorWithBacktrace, LocationInModule},
    exec::{IndexedId, ModuleId},
    Environment, ExecutableModule, Function, Object, Value,
};
use arithmetic_parser::{
    grammars::{Grammar, NumGrammar, Parse},
    Block, Error as ParseError, LocatedSpan, LvalueLen,
};
use arithmetic_typing::{
    arith::{Num, NumArithmetic},
    error::Errors as TypingErrors,
    Annotated, Type, TypeEnvironment,
};

use crate::library::ReplLiteral;

/// Exit code on parse or evaluation error.
pub const ERROR_EXIT_CODE: i32 = 2;
/// Width of help notes in chars.
const HELP_WIDTH: usize = 72;

/// Helper trait for possible `LocationSpan.fragment` types.
trait SizedFragment {
    fn fragment_len(&self) -> usize;
}

impl SizedFragment for &str {
    fn fragment_len(&self) -> usize {
        self.len()
    }
}

impl SizedFragment for usize {
    fn fragment_len(&self) -> usize {
        *self
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

pub struct Reporter {
    code_map: CodeMap,
    writer: StandardStream,
    config: ReportingConfig,
}

impl Reporter {
    pub fn new(color_choice: ColorChoice) -> Self {
        Self {
            code_map: CodeMap::default(),
            writer: StandardStream::stdout(color_choice),
            config: ReportingConfig::default(),
        }
    }

    fn print_greeting(&self) -> io::Result<()> {
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
                .type     Outputs type of a variable. Requires `--types` flag.
                .clear    Resets the interpreter state to the original one.
        ";

        let mut writer = self.writer.lock();
        writeln!(writer, "{}", unindent(HELP))
    }

    /// Reports a parsing error.
    fn report_parse_error(&self, err: ParseError) -> io::Result<()> {
        // Parsing errors are always reported for the most recently added snippet.
        let (file, range) = self.code_map.locate_in_most_recent_file(&err.location());

        let label = Label::primary(file, range).with_message("Error occurred here");
        let diagnostic = Diagnostic::error()
            .with_message(err.kind().to_string())
            .with_code("PARSE")
            .with_labels(vec![label]);

        self.report_error(&diagnostic)
    }

    fn create_diagnostic(&self, err: &EvalError) -> Diagnostic<FileId> {
        let main_span = err.location();
        let (file, range) = self
            .code_map
            .locate(main_span.module_id(), main_span.in_module());
        let main_label = Label::primary(file, range);
        let message = err.kind().main_span_info();

        let mut labels = vec![main_label.with_message(message)];
        for aux_span in err.aux_spans() {
            let (file, range) = self
                .code_map
                .locate(aux_span.module_id(), aux_span.in_module());
            let label =
                Label::primary(file, range).with_message(aux_span.in_module().extra.to_string());
            labels.push(label);
        }

        let mut diagnostic = Diagnostic::new(Severity::Error)
            .with_message(err.kind().to_short_string())
            .with_code("EVAL")
            .with_labels(labels);

        if let Some(help) = err.kind().help() {
            let help = textwrap::fill(&help, HELP_WIDTH);
            diagnostic = diagnostic.with_notes(vec![help]);
        }

        diagnostic
    }

    fn report_error(&self, diagnostic: &Diagnostic<FileId>) -> io::Result<()> {
        emit(
            &mut self.writer.lock(),
            &self.config,
            &self.code_map.files,
            diagnostic,
        )
        .map_err(|err| match err {
            FilesError::Io(err) => err,
            _ => io::Error::new(io::ErrorKind::Other, err),
        })
    }

    fn report_eval_error(&self, e: &ErrorWithBacktrace) -> io::Result<()> {
        let mut diagnostic = self.create_diagnostic(e.source());

        let mut calls_iter = e.backtrace().peekable();
        if let Some(BacktraceElement {
            fn_name,
            def_location: def_span,
            ..
        }) = calls_iter.peek()
        {
            if let Some(def_span) = def_span {
                let (file_id, def_range) = self
                    .code_map
                    .locate(def_span.module_id(), def_span.in_module());
                let def_label = Label::secondary(file_id, def_range)
                    .with_message(format!("The error occurred in function `{}`", fn_name));
                diagnostic.labels.push(def_label);
            }

            for (depth, call) in calls_iter.enumerate() {
                let call_span = &call.call_location;
                if Self::spans_are_equal(call_span, e.source().location()) {
                    // The span is already output.
                    continue;
                }

                let (file_id, call_range) = self
                    .code_map
                    .locate(call_span.module_id(), call_span.in_module());
                let call_label = Label::secondary(file_id, call_range)
                    .with_message(format!("Call at depth {}", depth + 1));
                diagnostic.labels.push(call_label);
            }
        }

        self.report_error(&diagnostic)
    }

    fn report_typing_errors(&self, errors: &TypingErrors<'_, Num>) -> io::Result<()> {
        for err in errors.iter() {
            let (file, range) = self.code_map.locate_in_most_recent_file(&err.main_span());

            let label = Label::primary(file, range).with_message("Error occurred here");
            let diagnostic = Diagnostic::error()
                .with_message(err.kind().to_string())
                .with_code("TYPE")
                .with_labels(vec![label]);

            self.report_error(&diagnostic)?;
        }
        Ok(())
    }

    fn spans_are_equal(span: &LocationInModule, other: &LocationInModule) -> bool {
        span.in_module() == other.in_module()
            && span.module_id().downcast_ref::<IndexedId>()
                == other.module_id().downcast_ref::<IndexedId>()
    }

    fn parse_streaming<'a, T: ReplLiteral>(
        &mut self,
        line: &'a str,
    ) -> io::Result<ParseAndEvalResult<Block<'a, Annotated<NumGrammar<T>>>>> {
        self.code_map.add(line.to_owned());

        Annotated::<NumGrammar<T>>::parse_streaming_statements(line)
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

    pub fn parse<'a, T: ReplLiteral>(
        &mut self,
        line: &'a str,
    ) -> io::Result<ParseAndEvalResult<Block<'a, Annotated<NumGrammar<T>>>>> {
        self.code_map.add(line.to_owned());

        Annotated::<NumGrammar<T>>::parse_statements(line)
            .map(ParseAndEvalResult::Ok)
            .or_else(|e| {
                self.report_parse_error(e)
                    .map(|()| ParseAndEvalResult::Errored)
            })
    }

    fn dump_value<T: ReplLiteral>(
        writer: &mut StandardStream,
        value: &Value<T>,
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
                    let captures_count = captures.len();
                    let captures = order_vars(captures);

                    writeln!(writer, "[")?;
                    for (i, (var_name, capture)) in captures.enumerate() {
                        write!(writer, "{}{} = ", " ".repeat(indent + 2), var_name)?;
                        Self::dump_value(writer, capture, indent + 2)?;
                        if i + 1 < captures_count {
                            writeln!(writer, ",")?;
                        } else {
                            writeln!(writer)?;
                        }
                    }
                    write!(writer, "{}]", " ".repeat(indent))?;
                }
                Ok(())
            }

            Value::Prim(num) => {
                writer.set_color(&num_color)?;
                write!(writer, "{num}")?;
                writer.reset()
            }

            Value::Tuple(tuple) => {
                writeln!(writer, "(")?;
                for (i, element) in tuple.iter().enumerate() {
                    write!(writer, "{}", " ".repeat(indent + 2))?;
                    Self::dump_value(writer, element, indent + 2)?;
                    if i + 1 < tuple.len() {
                        writeln!(writer, ",")?;
                    } else {
                        writeln!(writer)?;
                    }
                }
                write!(writer, "{})", " ".repeat(indent))
            }

            Value::Object(object) => Self::dump_object(writer, object, indent),

            Value::Ref(opaque_ref) => {
                writer.set_color(&opaque_ref_color)?;
                write!(writer, "{opaque_ref}")?;
                writer.reset()
            }

            _ => unreachable!(),
        }
    }

    fn dump_object<T: ReplLiteral>(
        writer: &mut StandardStream,
        object: &Object<T>,
        indent: usize,
    ) -> io::Result<()> {
        let fields_count = object.len();
        let fields = order_vars(object.iter());
        writeln!(writer, "#{{")?;
        for (i, (name, value)) in fields.enumerate() {
            write!(writer, "{}{}: ", " ".repeat(indent + 2), name)?;
            Self::dump_value(writer, value, indent + 2)?;
            if i + 1 < fields_count {
                writeln!(writer, ",")?;
            } else {
                writeln!(writer)?;
            }
        }
        write!(writer, "{}}}", " ".repeat(indent))
    }

    fn report_value<T: ReplLiteral>(&mut self, value: &Value<T>) -> io::Result<()> {
        Self::dump_value(&mut self.writer, value, 0)?;
        writeln!(self.writer)
    }
}

fn order_vars<'a, T: 'a>(
    values: impl IntoIterator<Item = (&'a str, &'a Value<T>)>,
) -> impl Iterator<Item = (&'a str, &'a Value<T>)> {
    let mut values: Vec<_> = values.into_iter().collect();
    values.sort_unstable_by_key(|(name, _)| *name);
    values.into_iter()
}

pub struct Env<T> {
    reporter: Reporter,
    original_env: Environment<T>,
    original_type_env: Option<TypeEnvironment>,
    env: Environment<T>,
    type_env: Option<TypeEnvironment>,
}

impl<T: ReplLiteral> Env<T> {
    pub fn new(
        env: Environment<T>,
        type_env: Option<TypeEnvironment>,
        color_choice: ColorChoice,
    ) -> Self {
        Self {
            reporter: Reporter::new(color_choice),
            original_env: env.clone(),
            original_type_env: type_env.clone(),
            env,
            type_env,
        }
    }

    pub fn print_greeting(&self) -> io::Result<()> {
        self.reporter.print_greeting()
    }

    fn dump_scope(&mut self, dump_original_scope: bool) -> io::Result<()> {
        for (name, var) in order_vars(&self.env) {
            if let Some(original_var) = self.original_env.get(name) {
                if !dump_original_scope && original_var == var {
                    // The variable is present in the original scope, no need to output it.
                    continue;
                }
            }

            write!(self.reporter.writer, "{name} = ")?;
            self.reporter.report_value(var)?;
        }
        Ok(())
    }

    pub fn parse_and_eval(
        &mut self,
        line: &str,
        streaming: bool,
    ) -> io::Result<ParseAndEvalResult> {
        if line.starts_with('.') {
            self.process_command(line)?;
            return Ok(ParseAndEvalResult::Ok(()));
        }

        let parse_result = if streaming {
            self.reporter.parse_streaming::<T>(line)?
        } else {
            self.reporter.parse::<T>(line)?
        };
        Ok(if let ParseAndEvalResult::Ok(mut block) = parse_result {
            match self.process_types(&block)? {
                Ok(()) => self.compile_and_execute(&block)?,
                Err(errors) => {
                    // We still want to execute non-failing statements in the block
                    // so that vars in ordinary and type envs align.
                    block.return_value = None;
                    block.statements.truncate(errors.first_failing_statement());

                    self.compile_and_execute(&block)?;
                    ParseAndEvalResult::Errored
                }
            }
        } else {
            parse_result.map(drop)
        })
    }

    fn process_command(&mut self, line: &str) -> io::Result<()> {
        let line = line.trim();
        self.reporter.code_map.add(line.to_owned());
        let file_id = *self.reporter.code_map.file_ids.last().expect("no files");

        match line {
            ".clear" => {
                self.env.clone_from(&self.original_env);
                self.type_env.clone_from(&self.original_type_env);
            }
            ".dump" => self.dump_scope(false)?,
            ".dump all" => self.dump_scope(true)?,
            ".help" => self.reporter.print_help()?,

            line if line.starts_with(".type ") && self.type_env.is_some() => {
                let ident = line[6..].trim_start();
                let ty = self.type_env.as_ref().unwrap().get(ident);
                if let Some(ty) = ty {
                    writeln!(self.reporter.writer, "{ty}")?;
                } else {
                    let label =
                        Label::primary(file_id, 6..line.len()).with_message("Undefined variable");
                    let diagnostic = Diagnostic::error()
                        .with_message(format!("Variable `{ident}` is not defined"))
                        .with_code("CMD")
                        .with_labels(vec![label]);
                    self.reporter.report_error(&diagnostic)?;
                }
            }

            _ => {
                let label = Label::primary(file_id, 0..line.len())
                    .with_message("Use `.help` to find out commands");
                let diagnostic = Diagnostic::error()
                    .with_message("Unknown command")
                    .with_code("CMD")
                    .with_labels(vec![label]);
                self.reporter.report_error(&diagnostic)?;
            }
        }

        Ok(())
    }

    fn process_types<'a>(
        &mut self,
        block: &Block<'a, Annotated<NumGrammar<T>>>,
    ) -> io::Result<Result<(), TypingErrors<'a, Num>>> {
        let res = self.type_env.as_mut().map_or(Ok(Type::Any), |type_env| {
            type_env.process_with_arithmetic(&NumArithmetic::with_comparisons(), block)
        });
        if let Err(errors) = &res {
            self.reporter.report_typing_errors(errors)?;
        }
        Ok(res.map(drop))
    }

    fn compile_and_execute<G>(&mut self, block: &Block<'_, G>) -> io::Result<ParseAndEvalResult>
    where
        G: Grammar<Lit = T>,
    {
        let module_id = self.reporter.code_map.latest_module_id();
        let module = ExecutableModule::new(module_id, block);
        let module = match module {
            Ok(module) => module,
            Err(err) => {
                self.reporter
                    .report_error(&self.reporter.create_diagnostic(&err))?;
                return Ok(ParseAndEvalResult::Errored);
            }
        };

        let module = module.with_mutable_env(&mut self.env);
        let module = match module {
            Ok(module) => module,
            Err(err) => {
                self.reporter
                    .report_error(&self.reporter.create_diagnostic(&err))?;
                return Ok(ParseAndEvalResult::Errored);
            }
        };

        let value = match module.run() {
            Ok(value) => value,
            Err(err) => {
                self.reporter.report_eval_error(&err)?;
                return Ok(ParseAndEvalResult::Errored);
            }
        };

        if !value.is_void() {
            self.reporter.report_value(&value)?;
        }
        Ok(ParseAndEvalResult::Ok(()))
    }
}
