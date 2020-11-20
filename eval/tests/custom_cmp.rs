//! Demonstrates how to use custom comparison functions.

use num_complex::Complex64;

use arithmetic_eval::{
    arith::{PreArithmeticExt, StdArithmetic},
    ExecutableModule, Prelude,
};
use arithmetic_parser::grammars::{NumGrammar, Parse, Untyped};

const PROGRAM: &str = r#"
    // The defined arithmetic compares numbers by their real part.
    assert(1 > -1);
    assert(-1 + 2i < 1 + i);

    // This function will capture the original comparison function.
    is_positive = |x| x > 0;
    assert(is_positive(1));
    assert(!is_positive(-1));
    assert(!is_positive(0));
"#;

type ComplexGrammar = NumGrammar<Complex64>;

#[test]
fn custom_cmp_function() {
    let block = Untyped::<ComplexGrammar>::parse_statements(PROGRAM).unwrap();

    let module = ExecutableModule::builder("custom_cmp", &block)
        .unwrap()
        .with_imports_from(&Prelude)
        .build();
    let arithmetic =
        StdArithmetic.with_comparison(|x: &Complex64, y: &Complex64| x.re.partial_cmp(&y.re));
    module.with_arithmetic(&arithmetic).run().unwrap();
}
