//! REPL for arithmetic expressions.

use rustyline::{error::ReadlineError, Editor};

use std::io;

use crate::common::{Env, ParseAndEvalResult, ReplLiteral};

pub fn repl<T: ReplLiteral>() -> io::Result<()> {
    let mut rl = Editor::<()>::new();
    let mut env = Env::new();
    env.print_greeting()?;

    let mut interpreter = T::create_interpreter();
    let original_interpreter = interpreter.clone();
    let mut snippet = String::new();
    let mut prompt = ">>> ";

    loop {
        let line = rl.readline(prompt);
        match line {
            Ok(line) => {
                snippet.push_str(&line);
                let result = env.parse_and_eval(&line, &mut interpreter, &original_interpreter)?;
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

            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                println!("Bye");
                break Ok(());
            }

            Err(e) => panic!("Error reading command: {}", e),
        }
    }
}
