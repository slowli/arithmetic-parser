//! REPL for arithmetic expressions.

use codespan_reporting::term::termcolor::ColorChoice;
use rustyline::{error::ReadlineError, Editor};

use std::io;

use arithmetic_eval::{arith::OrdArithmetic, Environment};
use arithmetic_typing::TypeEnvironment;

use crate::{
    common::{Env, ParseAndEvalResult},
    library::ReplLiteral,
};

pub fn repl<T: ReplLiteral>(
    arithmetic: Box<dyn OrdArithmetic<T>>,
    env: Environment<'static, T>,
    type_env: Option<TypeEnvironment>,
    color_choice: ColorChoice,
) -> io::Result<()> {
    let mut rl = Editor::<()>::new();
    let mut env = Env::new(arithmetic, env, type_env, color_choice);
    env.print_greeting()?;

    let mut snippet = String::new();
    let mut prompt = ">>> ";

    loop {
        let line = rl.readline(prompt);
        match line {
            Ok(line) => {
                snippet.push_str(&line);
                let result = env.parse_and_eval(&snippet, true)?;
                match result {
                    ParseAndEvalResult::Ok(_) => {
                        prompt = ">>> ";
                        snippet.clear();
                        rl.add_history_entry(line);
                    }
                    ParseAndEvalResult::Incomplete => {
                        prompt = "... ";
                        snippet.push('\n');
                        rl.add_history_entry(line);
                    }
                    ParseAndEvalResult::Errored => {
                        prompt = ">>> ";
                        snippet.clear();
                    }
                }
            }

            Err(ReadlineError::Interrupted) => {
                println!("Bye");
                break Ok(());
            }

            Err(ReadlineError::Eof) => {
                break Ok(());
            }

            Err(e) => panic!("Error reading command: {}", e),
        }
    }
}
