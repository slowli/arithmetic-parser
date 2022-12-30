//! Simple CLI / REPL for evaluating arithmetic expressions.

use anyhow::format_err;
use clap::{Args, Parser, Subcommand};
use codespan_reporting::term::{termcolor::ColorChoice, ColorArg};
use num_complex::{Complex32, Complex64};

use std::{
    io::{self, Read},
    process,
    str::FromStr,
};

use arithmetic_eval::Environment;
use arithmetic_typing::TypeEnvironment;

mod common;
mod library;
mod repl;

use crate::{
    common::{Env, ParseAndEvalResult, Reporter, ERROR_EXIT_CODE},
    library::{
        create_complex_env, create_float_env, create_int_env, create_modular_env, ReplLiteral,
    },
    repl::repl,
};

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

/// CLI for capturing and snapshot-testing terminal output.
#[derive(Debug, Parser)]
#[command(author, version, about = ABOUT, long_about = None, after_help = AFTER_HELP)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Parse the input and output the AST.
    Ast {
        /// Type of numbers to use. Available values are `u64`, `i64`, `u128`, `i128`,
        /// `f32`, `f64`, `complex32`, `complex64`, and `u64/$mod`, where `$mod` is the modulus
        /// for modular arithmetic.
        #[arg(name = "arithmetic", long, short = 'a', default_value = "f32")]
        arithmetic: ArithmeticType,
        /// Command to interpret. If omitted, the command will be read from stdin.
        #[arg(name = "command")]
        command: Option<String>,
        /// Controls coloring of the output.
        #[arg(long, default_value = "auto", value_enum, ignore_case = true, env)]
        color: ColorArg,
    },
    /// Evaluate the input, optionally checking types beforehand.
    Eval(EvalArgs),
}

#[derive(Debug, Args)]
struct EvalArgs {
    /// Launch the REPL for arithmetic expressions.
    #[arg(name = "interactive", long, short = 'i')]
    interactive: bool,
    /// Type of numbers to use. Available values are `u64`, `i64`, `u128`, `i128`,
    /// `f32`, `f64`, `complex32`, `complex64`, and `u64/$mod`, where `$mod` is the modulus
    /// for modular arithmetic.
    #[arg(name = "arithmetic", long, short = 'a', default_value = "f32")]
    arithmetic: ArithmeticType,
    /// Use wrapping semantics for integer arithmetic instead of the default checked semantics.
    /// Ignored if a non-integer number type is selected.
    #[arg(name = "wrapping", long, short = 'w')]
    wrapping: bool,
    /// Check / infer types before evaluation.
    #[arg(name = "types", long)]
    types: bool,
    /// Command to interpret. If omitted, the command will be read from stdin.
    #[arg(name = "command", conflicts_with = "interactive")]
    command: Option<String>,
    /// Controls coloring of the output.
    #[arg(long, default_value = "auto", value_enum, ignore_case = true, env)]
    color: ColorArg,
}

impl Command {
    fn run(self) -> io::Result<()> {
        match self {
            Self::Ast {
                arithmetic,
                command,
                color,
            } => Self::output_ast(arithmetic, command, Self::color_choice(color)),

            Self::Eval(eval_args) => eval_args.run(),
        }
    }

    fn color_choice(color_arg: ColorArg) -> ColorChoice {
        match color_arg.into() {
            ColorChoice::Auto => {
                if atty::is(atty::Stream::Stdout) {
                    ColorChoice::Auto
                } else {
                    ColorChoice::Never
                }
            }
            other => other,
        }
    }

    fn output_ast(
        arithmetic: ArithmeticType,
        command: Option<String>,
        color_choice: ColorChoice,
    ) -> io::Result<()> {
        let command = match command {
            Some(command) => command,
            None => {
                let mut buffer = String::new();
                io::stdin().read_to_string(&mut buffer)?;
                buffer
            }
        };

        match arithmetic {
            ArithmeticType::U64 | ArithmeticType::Modular64(_) => {
                Self::parse_and_output_block::<u64>(&command, color_choice)
            }
            ArithmeticType::I64 => Self::parse_and_output_block::<i64>(&command, color_choice),
            ArithmeticType::U128 => Self::parse_and_output_block::<u128>(&command, color_choice),
            ArithmeticType::I128 => Self::parse_and_output_block::<i128>(&command, color_choice),
            ArithmeticType::F32 => Self::parse_and_output_block::<f32>(&command, color_choice),
            ArithmeticType::F64 => Self::parse_and_output_block::<f64>(&command, color_choice),
            ArithmeticType::Complex32 => {
                Self::parse_and_output_block::<Complex32>(&command, color_choice)
            }
            ArithmeticType::Complex64 => {
                Self::parse_and_output_block::<Complex64>(&command, color_choice)
            }
        }
    }

    fn parse_and_output_block<T: ReplLiteral>(
        command: &str,
        color_choice: ColorChoice,
    ) -> io::Result<()> {
        let mut reporter = Reporter::new(color_choice);
        let block = reporter.parse::<T>(command)?;
        if let ParseAndEvalResult::Ok(block) = &block {
            println!("{:#?}", block);
            Ok(())
        } else {
            process::exit(ERROR_EXIT_CODE);
        }
    }
}

impl EvalArgs {
    fn run(self) -> io::Result<()> {
        let wrapping = self.wrapping;
        match self.arithmetic {
            ArithmeticType::U64 => self.run_inner(create_int_env::<u64>(wrapping)),
            ArithmeticType::I64 => self.run_inner(create_int_env::<i64>(wrapping)),
            ArithmeticType::U128 => self.run_inner(create_int_env::<u128>(wrapping)),
            ArithmeticType::I128 => self.run_inner(create_int_env::<i128>(wrapping)),
            ArithmeticType::Modular64(modulus) => self.run_inner(create_modular_env(modulus)),
            ArithmeticType::F32 => self.run_inner(create_float_env::<f32>(1e-5)),
            ArithmeticType::F64 => self.run_inner(create_float_env::<f64>(1e-5)),
            ArithmeticType::Complex32 => self.run_inner(create_complex_env::<Complex32>()),
            ArithmeticType::Complex64 => self.run_inner(create_complex_env::<Complex64>()),
        }
    }

    fn run_inner<T: ReplLiteral>(
        self,
        (env, type_env): (Environment<'static, T>, TypeEnvironment),
    ) -> io::Result<()> {
        let type_env = if self.types { Some(type_env) } else { None };
        if self.interactive {
            repl(env, type_env, Command::color_choice(self.color))
        } else {
            self.run_command(env, type_env)
        }
    }

    fn run_command<T: ReplLiteral>(
        self,
        env: Environment<'static, T>,
        type_env: Option<TypeEnvironment>,
    ) -> io::Result<()> {
        let command = match self.command {
            Some(command) => command,
            None => {
                let mut buffer = String::new();
                io::stdin().read_to_string(&mut buffer)?;
                buffer
            }
        };
        let mut env = Env::new(env, type_env, Command::color_choice(self.color));

        match env.parse_and_eval(&command, false)? {
            ParseAndEvalResult::Ok(()) => Ok(()),
            ParseAndEvalResult::Incomplete | ParseAndEvalResult::Errored => {
                process::exit(ERROR_EXIT_CODE);
            }
        }
    }
}

fn main() -> io::Result<()> {
    Cli::parse().command.run()
}
