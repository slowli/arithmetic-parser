//! Simple CLI / REPL for evaluating arithmetic expressions.

use anyhow::format_err;
use num_complex::{Complex32, Complex64};
use structopt::StructOpt;

use std::{io, process, str::FromStr};

mod common;
mod repl;

use crate::{
    common::{Env, ParseAndEvalResult, ReplLiteral, ERROR_EXIT_CODE},
    repl::repl,
};
use arithmetic_parser::{grammars::NumGrammar, Untyped};

const ABOUT: &str = "CLI and REPL for parsing and evaluating arithmetic expressions.";

const AFTER_HELP: &str = "\
EXIT CODES:
    0    Normal exit
    1    Invalid command-line option
    2    Parsing or evaluation error in non-interactive mode";

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
#[structopt(about = ABOUT, after_help = AFTER_HELP)]
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
    fn run(self) -> io::Result<()> {
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

    fn run_command<T: ReplLiteral>(self) -> io::Result<()> {
        let command = self.command.unwrap_or_default();
        let mut env = Env::new();

        let res = if self.ast {
            env.parse::<Untyped<NumGrammar<T>>>(&command)?
                .map(|parsed| println!("{:#?}", parsed))
        } else {
            env.parse_and_eval(&command, &mut T::create_env(), &T::create_env())?
        };

        match res {
            ParseAndEvalResult::Ok(()) => Ok(()),
            ParseAndEvalResult::Incomplete | ParseAndEvalResult::Errored => {
                process::exit(ERROR_EXIT_CODE);
            }
        }
    }
}

fn main() -> io::Result<()> {
    let args = Args::from_args();
    args.run()
}
