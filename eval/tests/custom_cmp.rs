//! Demonstrates how to use custom comparison functions.

use num_complex::Complex64;

use core::cmp::Ordering;

use arithmetic_eval::Interpreter;
use arithmetic_parser::{grammars::NumGrammar, GrammarExt, InputSpan};

const PROGRAM: &str = r#"
    # The original comparison function compares numbers by their real part.
    assert(1 > -1);
    assert(-1 + 2i < 1 + i);

    # This function will capture the original comparison function.
    is_positive = |x| x > 0;
    assert(is_positive(1));
    assert(!is_positive(-1));
    assert(!is_positive(0));

    # Override the comparison function so that it compares imaginary parts of numbers.
    # It immediately influences all following comparisons.
    cmp = |x, y| cmp(-i * x, -i * y);
    assert(!(1 > -1));
    assert(-1 + 2i >= 1 + i);

    # ...but does not influence the comparisons in `is_positive`.
    assert(is_positive(1 - i) && !is_positive(-1 + 2i));
"#;

type ComplexGrammar = NumGrammar<Complex64>;

#[test]
fn custom_cmp_function() {
    let mut interpreter = Interpreter::<ComplexGrammar>::with_prelude();
    // There is no "natural" comparison function on complex numbers. Here, we'll define one
    // using comparison of real parts.
    interpreter.insert_wrapped_fn("cmp", |x: Complex64, y: Complex64| {
        x.re.partial_cmp(&y.re).unwrap_or(Ordering::Equal)
    });

    let block = ComplexGrammar::parse_statements(InputSpan::new(PROGRAM)).unwrap();
    interpreter.evaluate(&block).unwrap();
}
