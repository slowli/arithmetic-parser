//! Tests that the README code sample actually parses.

use pulldown_cmark::{CodeBlockKind, Event, Parser, Tag};

use arithmetic_parser::grammars::{F64Grammar, MockTypes, Parse, WithMockedTypes};

struct MockedTypesList;

impl MockTypes for MockedTypesList {
    const MOCKED_TYPES: &'static [&'static str] = &["Num"];
}

type Grammar = WithMockedTypes<F64Grammar, MockedTypesList>;

fn check_sample(code_sample: &str) {
    Grammar::parse_statements(code_sample).unwrap();
}

#[test]
fn code_sample_in_readme_is_parsed() {
    const README: &str = include_str!("../README.md");

    let parser = Parser::new(README);
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
