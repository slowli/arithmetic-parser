//! Tests that the README code samples actually work.

use pulldown_cmark::{CodeBlockKind, Event, Parser, Tag};
use rand::{thread_rng, Rng};

use std::fs;

use arithmetic_eval::{fns, Assertions, Environment, Prelude, Value, VariableMap};
use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};

fn read_file(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|err| panic!("Cannot read file {}: {}", path, err))
}

fn check_sample(code_sample: &str) {
    let program = Untyped::<F32Grammar>::parse_statements(code_sample).unwrap();

    let mut env: Environment<'_, f32> = Prelude.iter().chain(Assertions.iter()).collect();
    env.insert("INF", Value::Prim(f32::INFINITY));
    env.insert("array", Value::native_fn(fns::Array)).insert(
        "rand_num",
        Value::wrapped_fn(|min: f32, max: f32| thread_rng().gen_range(min..max)),
    );
    let module = env.compile_module("test", &program).unwrap();
    module.run().unwrap();
}

#[test]
fn code_samples_in_readme_are_valid() {
    let readme = read_file("README.md");

    let parser = Parser::new(&readme);
    let mut code: Option<String> = None;
    for event in parser {
        match event {
            Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(lang)))
                if lang.as_ref() == "text" =>
            {
                assert!(code.is_none(), "Embedded code samples");
                code = Some(String::with_capacity(1_024));
            }
            Event::End(Tag::CodeBlock(CodeBlockKind::Fenced(lang))) if lang.as_ref() == "text" => {
                let code_sample = code.take().unwrap();
                assert!(!code_sample.is_empty());
                check_sample(&code_sample);
            }
            Event::Text(text) => {
                if let Some(code) = &mut code {
                    code.push_str(text.as_ref());
                }
            }
            _ => { /* Do nothing */ }
        }
    }
}
