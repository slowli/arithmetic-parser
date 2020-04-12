//! REPL for arithmetic expressions.

use rustyline::{error::ReadlineError, Editor};
use typed_arena::Arena;

use crate::common::{Env, ReplLiteral};

pub fn repl<T: ReplLiteral>() -> anyhow::Result<()> {
    let mut rl = Editor::<()>::new();
    let mut env = Env::new();
    env.print_greeting()?;

    let mut interpreter = T::create_interpreter();
    interpreter.create_scope();
    let snippet_arena = Arena::new();
    let mut snippet = String::new();
    let mut prompt = ">>> ";

    loop {
        let line = rl.readline(prompt);
        match line {
            Ok(line) => {
                snippet.push_str(&line);
                let arena_ref = &*snippet_arena.alloc(snippet.clone());

                if let Ok(incomplete) = env.parse_and_eval(arena_ref, &mut interpreter) {
                    if incomplete {
                        prompt = "... ";
                        snippet.push('\n');
                    } else {
                        prompt = ">>> ";
                        snippet.clear();
                    }
                    rl.add_history_entry(line);
                } else {
                    prompt = ">>> ";
                    snippet.clear();
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