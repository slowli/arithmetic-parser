//! REPL for arithmetic expressions.

use codespan_reporting::term::termcolor::ColorChoice;
use rustyline::{error::ReadlineError, Editor};

use std::io;

use arithmetic_eval::Environment;
use arithmetic_typing::TypeEnvironment;

use crate::{
    common::{Env, ParseAndEvalResult},
    library::ReplLiteral,
};

fn into_io_error(err: ReadlineError) -> io::Error {
    match err {
        ReadlineError::Io(err) => err,
        other => io::Error::new(io::ErrorKind::Other, other),
    }
}

pub fn repl<T: ReplLiteral>(
    env: Environment<T>,
    type_env: Option<TypeEnvironment>,
    color_choice: ColorChoice,
) -> io::Result<()> {
    let mut rl = Editor::<(), _>::new().map_err(into_io_error)?;
    let mut env = Env::new(env, type_env, color_choice);
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
                        rl.add_history_entry(line).map_err(into_io_error)?;
                    }
                    ParseAndEvalResult::Incomplete => {
                        prompt = "... ";
                        snippet.push('\n');
                        rl.add_history_entry(line).map_err(into_io_error)?;
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

            Err(err) => panic!("Error reading command: {err}"),
        }
    }
}
