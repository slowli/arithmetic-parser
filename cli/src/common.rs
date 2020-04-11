//! Common utils.

use anyhow::format_err;
use codespan::{FileId, Files};
use codespan_reporting::{
    diagnostic::{Diagnostic, Label, Severity},
    term::termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor},
    term::{emit, Config as ReportingConfig},
};

use std::{
    collections::BTreeMap,
    fmt,
    io::{self, Write},
    ops::Range,
};

use arithmetic_parser::{
    grammars::NumLiteral,
    interpreter::{BacktraceElement, ErrorWithBacktrace, Function, Interpreter, Scope, Value},
    Block, Error, Grammar, GrammarExt, Span, Spanned,
};

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
        if span.offset > self.next_position {
            return None;
        }
        let (&file_start, &file_id) = self.code_positions.range(..=span.offset).rev().next()?;
        let start = span.offset - file_start;
        let range = start..(start + span.fragment.len());
        Some((file_id, range))
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
        writeln!(writer, "{}", env!("CARGO_PKG_DESCRIPTION"))
    }

    fn print_help(&mut self) -> io::Result<()> {
        unimplemented!()
    }

    /// Reports a parsing error.
    pub fn report_parse_error(&self, err: Spanned<'_, Error<'_>>) -> anyhow::Error {
        let (file, range) = self.code_map.locate(&err).unwrap();
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
        .unwrap();

        format_err!("{}", err.extra.to_string())
    }

    pub fn report_eval_error(&self, e: ErrorWithBacktrace) -> anyhow::Error {
        let severity = Severity::Error;
        let (file, range) = self.code_map.locate(&e.main_span()).unwrap();
        let main_label = Label::primary(file, range);
        let message = e.source().main_span_info();

        let mut labels = vec![main_label.with_message(message)];
        for aux_span in e.aux_spans() {
            let (file, range) = self.code_map.locate(&aux_span).unwrap();
            let label = Label::primary(file, range).with_message(aux_span.extra.to_string());
            labels.push(label);
        }

        let mut calls_iter = e.backtrace().calls();
        if let Some(BacktraceElement {
            fn_name,
            def_span,
            call_span,
        }) = calls_iter.next()
        {
            let (file_id, def_range) = self
                .code_map
                .locate(&def_span)
                .expect("Cannot locate span in previously recorded snippets");
            let def_label = Label::secondary(file_id, def_range)
                .with_message(format!("The error occurred in function `{}`", fn_name));
            labels.push(def_label);

            let mut call_site = call_span;
            for call in calls_iter {
                let (file_id, call_range) = self
                    .code_map
                    .locate(&call_site)
                    .expect("Cannot locate span in previously recorded snippets");
                let call_label = Label::secondary(file_id, call_range).with_message(format!(
                    "...which was called from function `{}`",
                    call.fn_name
                ));
                labels.push(call_label);
                call_site = call.call_span;
            }

            let (file_id, call_range) = self
                .code_map
                .locate(&call_site)
                .expect("Cannot locate span in previously recorded snippets");
            let call_label =
                Label::secondary(file_id, call_range).with_message("...which was called from here");
            labels.push(call_label);
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
        .unwrap();

        format_err!("{}", e.source())
    }

    pub fn dump_value<T>(&mut self, value: &Value<T>, indent: usize) -> io::Result<()>
    where
        T: Grammar,
        T::Lit: fmt::Display,
    {
        let bool_color = ColorSpec::new().set_fg(Some(Color::Cyan)).clone();
        let num_color = ColorSpec::new().set_fg(Some(Color::Green)).clone();
        let fn_color = ColorSpec::new().set_fg(Some(Color::Magenta)).clone();

        match value {
            Value::Bool(b) => {
                self.writer.set_color(&bool_color)?;
                write!(self.writer, "{}", if *b { "true" } else { "false" })?;
                self.writer.reset()
            }

            Value::Function(Function::Native(_)) => {
                self.writer.set_color(&fn_color)?;
                write!(self.writer, "(native fn)")?;
                self.writer.reset()
            }

            Value::Function(Function::Interpreted(function)) => {
                self.writer.set_color(&fn_color)?;
                write!(self.writer, "fn({} args)", function.arg_count())?;
                self.writer.reset()?;
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

    fn dump_scope<T>(&mut self, scope: &Scope<'_, T>) -> io::Result<()>
    where
        T: Grammar,
        T::Lit: fmt::Display,
    {
        for (name, var) in scope.variables() {
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
    ) -> Result<bool, ()>
    where
        T: Grammar,
        T::Lit: fmt::Display + NumLiteral,
    {
        let (file, start_position) = self.code_map.add(line);
        let visible_span = 0..line.len();

        if line.starts_with('.') {
            match line {
                ".clear" => interpreter.innermost_scope().clear(),
                ".dump" => self.dump_scope(interpreter.innermost_scope()).unwrap(),
                ".help" => self.print_help().unwrap(),

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
                    )
                    .unwrap();
                }
            }

            return Ok(false);
        }

        let span = Span {
            offset: start_position,
            line: 0,
            fragment: line,
            extra: (),
        };
        let mut incomplete = false;
        let statements = T::parse_streaming_statements(span).or_else(|e| {
            if let Error::Incomplete = e.extra {
                incomplete = true;
                Ok(Block::empty())
            } else {
                self.report_parse_error(e);
                Err(())
            }
        })?;

        if !incomplete {
            let output = interpreter.evaluate(&statements).map_err(|e| {
                self.report_eval_error(e);
            })?;
            if !output.is_void() {
                self.dump_value(&output, 0).unwrap();
            }
        }
        Ok(incomplete)
    }
}
