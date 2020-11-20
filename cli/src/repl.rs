//! REPL for arithmetic expressions.

use rustyline::{error::ReadlineError, Editor};

use std::io;

use crate::common::{Env, ParseAndEvalResult, ReplLiteral};
use arithmetic_eval::arith::{Arithmetic, StdArithmetic};

pub fn repl<T: ReplLiteral>() -> io::Result<()>
where
    StdArithmetic: Arithmetic<T>,
{
    let mut rl = Editor::<()>::new();
    let mut env = Env::new();
    env.print_greeting()?;

    let mut interpreter = T::create_env();
    let original_interpreter = interpreter.clone();
    let mut snippet = String::new();
    let mut prompt = ">>> ";

    loop {
        let line = rl.readline(prompt);
        match line {
            Ok(line) => {
                snippet.push_str(&line);
                let result =
                    env.parse_and_eval(&snippet, &mut interpreter, &original_interpreter)?;
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
