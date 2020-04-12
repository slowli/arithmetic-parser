//! Simple CLI / REPL for evaluating arithmetic expressions.

use anyhow::format_err;
use num_complex::{Complex32, Complex64};
use structopt::StructOpt;

use std::str::FromStr;

use arithmetic_parser::{grammars::NumGrammar, GrammarExt, Span};

mod common;
mod repl;
use crate::{
    common::{Env, ReplLiteral},
    repl::repl,
};

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
    command: Option<String>,
}

impl Args {
    fn run(self) -> anyhow::Result<()> {
        if self.interactive {
            match self.arithmetic {
                Arithmetic::F32 => repl::<f32>(),
                Arithmetic::F64 => repl::<f64>(),
                Arithmetic::Complex32 => repl::<Complex32>(),
                Arithmetic::Complex64 => repl::<Complex64>(),
            }
        } else {
            match self.arithmetic {
                Arithmetic::F32 => self.run_command::<f32>(),
                Arithmetic::F64 => self.run_command::<f64>(),
                Arithmetic::Complex32 => self.run_command::<Complex32>(),
                Arithmetic::Complex64 => self.run_command::<Complex64>(),
            }
        }
    }

    fn run_command<T>(self) -> anyhow::Result<()>
    where
        T: ReplLiteral,
    {
        let command = self.command.unwrap_or_default();
        let mut env = Env::non_interactive(&command);
        let command = Span::new(&command);
        let parsed =
            NumGrammar::<T>::parse_statements(command).map_err(|e| env.report_parse_error(e))?;
        if self.ast {
            println!("{:#?}", parsed);
            Ok(())
        } else {
            let mut interpreter = T::create_interpreter();
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
