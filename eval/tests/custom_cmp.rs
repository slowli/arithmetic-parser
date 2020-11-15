//! Demonstrates how to use custom comparison functions.

use num_complex::Complex64;

use core::cmp::Ordering;

use arithmetic_eval::{ExecutableModule, Prelude, Value};
use arithmetic_parser::grammars::{GrammarExt, NumGrammar, Untyped};

const PROGRAM: &str = r#"
    // The original comparison function compares numbers by their real part.
    assert(1 > -1);
    assert(-1 + 2i < 1 + i);

    // This function will capture the original comparison function.
    is_positive = |x| x > 0;
    assert(is_positive(1));
    assert(!is_positive(-1));
    assert(!is_positive(0));

    // Override the comparison function so that it compares imaginary parts of numbers.
    // It immediately influences all following comparisons.
    cmp = |x, y| cmp(-i * x, -i * y);
    assert(!(1 > -1));
    assert(-1 + 2i >= 1 + i);

    // ...but does not influence the comparisons in `is_positive`.
    assert(is_positive(1 - i) && !is_positive(-1 + 2i));
"#;

type ComplexGrammar = NumGrammar<Complex64>;

#[test]
fn custom_cmp_function() {
    let block = Untyped::<ComplexGrammar>::parse_statements(PROGRAM).unwrap();

    let module = ExecutableModule::builder("custom_cmp", &block)
        .unwrap()
        // There is no "natural" comparison function on complex numbers. Here, we'll define one
        // using comparison of real parts.
        .with_import(
            "cmp",
            Value::wrapped_fn(|x: Complex64, y: Complex64| {
                x.re.partial_cmp(&y.re).unwrap_or(Ordering::Equal)
            }),
        )
        .with_imports_from(&Prelude)
        .build();
    module.run().unwrap();
}
