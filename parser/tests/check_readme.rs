//! Tests that the README code sample actually parses.

use std::fs;

use arithmetic_parser::grammars::{F64Grammar, MockTypes, Parse, WithMockedTypes};
use pulldown_cmark::{CodeBlockKind, Event, Parser, Tag, TagEnd};

struct MockedTypesList;

impl MockTypes for MockedTypesList {
    const MOCKED_TYPES: &'static [&'static str] = &["Num"];
}

type Grammar = WithMockedTypes<F64Grammar, MockedTypesList>;

fn read_file(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|err| panic!("Cannot read file {path}: {err}"))
}

fn check_sample(code_sample: &str) {
    Grammar::parse_statements(code_sample).unwrap();
}

#[test]
fn code_sample_in_readme_is_parsed() {
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
