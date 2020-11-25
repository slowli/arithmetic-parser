//! Simple CLI / REPL for evaluating arithmetic expressions.

use anyhow::format_err;
use num_complex::{Complex32, Complex64};
use structopt::StructOpt;

use std::{io, process, str::FromStr};

mod common;
mod library;
mod repl;

use crate::{
    common::{Env, ParseAndEvalResult, ERROR_EXIT_CODE},
    library::{
        create_complex_env, create_float_env, create_int_env, create_modular_env, ReplLiteral,
    },
    repl::repl,
};
use arithmetic_eval::{
    arith::{
        ArithmeticExt, CheckedArithmetic, ModularArithmetic, OrdArithmetic, StdArithmetic,
        WrappingArithmetic,
    },
    Environment,
};
use arithmetic_parser::grammars::{NumGrammar, Untyped};

const ABOUT: &str = "CLI and REPL for parsing and evaluating arithmetic expressions.";

const AFTER_HELP: &str = "\
EXIT CODES:
    0    Normal exit
    1    Invalid command-line option
    2    Parsing or evaluation error in non-interactive mode";

#[derive(Debug, Clone, Copy)]
enum ArithmeticType {
    U64,
    I64,
    U128,
    I128,
    F32,
    F64,
    Complex32,
    Complex64,
    Modular64(u64),
}

impl FromStr for ArithmeticType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "u64" => Ok(Self::U64),
            "i64" => Ok(Self::I64),
            "u128" => Ok(Self::U128),
            "i128" => Ok(Self::I128),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),

            "c32" | "complex32" => Ok(Self::Complex32),
            "c64" | "complex64" => Ok(Self::Complex64),

            s if s.starts_with("u64/") => {
                let modulus: u64 = s[4..].parse()?;
                Ok(Self::Modular64(modulus))
            }

            _ => Err(format_err!(
                "Invalid arithmetic spec. Use one of `u64`, `i64`, `u128`, `i128`, \
                 `f32`, `f64`, `complex32`, `complex64`, or `u64/$modulus`"
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

    /// Type of numbers to use. Available values are `u64`, `i64`, `u128`, `i128`,
    /// `f32`, `f64`, `complex32`, `complex64`, and `u64/$mod`, where `$mod` is the modulus
    /// for modular arithmetic.
    #[structopt(name = "arithmetic", long, short = "a", default_value = "f32")]
    arithmetic: ArithmeticType,

    /// Use wrapping semantics for integer arithmetic instead of the default checked semantics.
    /// Ignored if a non-integer number type is selected.
    #[structopt(name = "wrapping", long, short = "w")]
    wrapping: bool,

    /// Command to parse and interpret. The line will be ignored in the interactive mode.
    #[structopt(name = "command")]
    command: Option<String>,
}

impl Args {
    fn int_arithmetic<T>(wrapping: bool) -> Box<dyn OrdArithmetic<T>>
    where
        WrappingArithmetic: OrdArithmetic<T>,
        CheckedArithmetic: OrdArithmetic<T>,
    {
        if wrapping {
            Box::new(WrappingArithmetic)
        } else {
            Box::new(CheckedArithmetic::new())
        }
    }

    fn run(self) -> io::Result<()> {
        let wrapping = self.wrapping;

        match self.arithmetic {
            ArithmeticType::U64 => self.run_inner(
                Self::int_arithmetic(wrapping),
                create_int_env::<u64>(wrapping),
            ),
            ArithmeticType::I64 => self.run_inner(
                Self::int_arithmetic(wrapping),
                create_int_env::<i64>(wrapping),
            ),
            ArithmeticType::U128 => self.run_inner(
                Self::int_arithmetic(wrapping),
                create_int_env::<u128>(wrapping),
            ),
            ArithmeticType::I128 => self.run_inner(
                Self::int_arithmetic(wrapping),
                create_int_env::<i128>(wrapping),
            ),

            ArithmeticType::Modular64(modulus) => {
                let arithmetic = ModularArithmetic::new(modulus).without_comparisons();
                self.run_inner(Box::new(arithmetic), create_modular_env(modulus))
            }

            ArithmeticType::F32 => {
                self.run_inner(Box::new(StdArithmetic), create_float_env::<f32>())
            }
            ArithmeticType::F64 => {
                self.run_inner(Box::new(StdArithmetic), create_float_env::<f32>())
            }
            ArithmeticType::Complex32 => self.run_inner(
                Box::new(StdArithmetic.without_comparisons()),
                create_complex_env::<Complex32>(),
            ),
            ArithmeticType::Complex64 => self.run_inner(
                Box::new(StdArithmetic.without_comparisons()),
                create_complex_env::<Complex64>(),
            ),
        }
    }

    fn run_inner<T: ReplLiteral>(
        self,
        arithmetic: Box<dyn OrdArithmetic<T>>,
        env: Environment<'static, T>,
    ) -> io::Result<()> {
        if self.interactive {
            repl(arithmetic, env)
        } else {
            self.run_command(arithmetic, env)
        }
    }

    fn run_command<T: ReplLiteral>(
        self,
        arithmetic: Box<dyn OrdArithmetic<T>>,
        env: Environment<'static, T>,
    ) -> io::Result<()> {
        let command = self.command.unwrap_or_default();
        let mut env = Env::new(arithmetic, env);

        let res = if self.ast {
            env.parse::<Untyped<NumGrammar<T>>>(&command)?
                .map(|parsed| println!("{:#?}", parsed))
        } else {
            env.parse_and_eval(&command)?
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
