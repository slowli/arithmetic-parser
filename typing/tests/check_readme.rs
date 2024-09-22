//! Tests that the README code samples actually work.

use std::fs;

use arithmetic_parser::grammars::{F32Grammar, Parse};
use arithmetic_typing::{
    arith::{Num, NumArithmetic},
    defs::{Assertions, Prelude},
    Annotated, Type, TypeEnvironment,
};
use pulldown_cmark::{CodeBlockKind, Event, Parser, Tag, TagEnd};

type Grammar = Annotated<F32Grammar>;

fn read_file(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|err| panic!("Cannot read file {}: {}", path, err))
}

fn check_sample(code_sample: &str) {
    let program = Grammar::parse_statements(code_sample).unwrap();

    let mut env: TypeEnvironment = Prelude::iter().chain(Assertions::iter()).collect();
    env.insert("INF", Type::NUM)
        .insert("array", Prelude::array(Num::Num));
    env.process_with_arithmetic(&NumArithmetic::with_comparisons(), &program)
        .unwrap();
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
            Event::End(TagEnd::CodeBlock) => {
                if let Some(code_sample) = code.take() {
                    assert!(!code_sample.is_empty());
                    check_sample(&code_sample);
                }
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
