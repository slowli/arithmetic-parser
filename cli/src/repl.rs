//! REPL for arithmetic expressions.

use rustyline::{error::ReadlineError, Editor};
use typed_arena::Arena;

use crate::common::{Env, ReplLiteral};

pub fn repl<T: ReplLiteral>() -> anyhow::Result<()> {
    let mut rl = Editor::<()>::new();
    let mut env = Env::new();
    env.print_greeting()?;

    let snippet_arena = Arena::new();
    let mut interpreter = T::create_interpreter();
    let original_interpreter = interpreter.clone();
    let mut snippet = String::new();
    let mut prompt = ">>> ";

    loop {
        let line = rl.readline(prompt);
        match line {
            Ok(line) => {
                snippet.push_str(&line);
                let arena_ref = &*snippet_arena.alloc(snippet.clone());
                let res = env.parse_and_eval(arena_ref, &mut interpreter, &original_interpreter);

                if let Ok(incomplete) = res {
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
