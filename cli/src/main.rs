//! Simple CLI / REPL for evaluating arithmetic expressions.

use anyhow::format_err;
use num_complex::{Complex32, Complex64};
use structopt::StructOpt;

use std::{fmt, str::FromStr};

use arithmetic_parser::{
    grammars::{NumGrammar, NumLiteral},
    interpreter::{If, Interpreter, Loop, Value},
    GrammarExt, Span,
};

mod common;
use crate::common::Env;

#[derive(Debug, Clone, Copy)]
enum Arithmetic {
    F32,
    F64,
    Complex32,
    Complex64,
}

impl FromStr for Arithmetic {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            "c32" | "complex32" => Ok(Self::Complex32),
            "c64" | "complex64" => Ok(Self::Complex64),
            _ => Err(format_err!(
                "Invalid arithmetic spec. Use one of  `f32`, `f64`, `complex32`, `complex64`"
            )),
        }
    }
}

#[derive(Debug, StructOpt)]
struct Args {
    /// Launch the REPL for arithmetic expressions.
    #[structopt(name = "interactive", long, short = "i")]
    interactive: bool,

    /// Only output parsed AST without interpreting it.
    #[structopt(name = "ast", long, conflicts_with = "interactive")]
    ast: bool,

    /// Arithmetic system to use. Available values are `f32`, `f64`, `complex32`, `complex64`.
    #[structopt(name = "arithmetic", long, short = "a", default_value = "f32")]
    arithmetic: Arithmetic,

    /// Command to parse and interpret. The line will be ignored in the interactive mode.
    #[structopt(name = "command")]
    command: String,
}

impl Args {
    fn run(&self) -> anyhow::Result<()> {
        if self.interactive {
            unimplemented!();
        }

        match self.arithmetic {
            Arithmetic::F32 => self.run_command::<f32>(),
            Arithmetic::F64 => self.run_command::<f64>(),
            Arithmetic::Complex32 => self.run_command::<Complex32>(),
            Arithmetic::Complex64 => self.run_command::<Complex64>(),
        }
    }

    fn run_command<T>(&self) -> anyhow::Result<()>
    where
        T: NumLiteral + fmt::Display,
    {
        let mut env = Env::non_interactive(&self.command);
        let command = Span::new(&self.command);
        let parsed =
            NumGrammar::<T>::parse_statements(command).map_err(|e| env.report_parse_error(e))?;
        if self.ast {
            println!("{:#?}", parsed);
            Ok(())
        } else {
            let mut interpreter = Interpreter::<NumGrammar<T>>::new();
            interpreter
                .innermost_scope()
                .insert_var("false", Value::Bool(false))
                .insert_var("true", Value::Bool(true))
                .insert_native_fn("if", If)
                .insert_native_fn("loop", Loop);
            let value = interpreter
                .evaluate(&parsed)
                .map_err(|e| env.report_eval_error(e))?;
            env.dump_value(&value, 0)?;
            Ok(())
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::from_args();
    args.run()
}
