//! Demonstrates how to use custom comparison functions.

use num_complex::Complex64;

use arithmetic_eval::{
    arith::{ArithmeticExt, StdArithmetic},
    env::{Assertions, Prelude},
    Environment, ExecutableModule,
};
use arithmetic_parser::grammars::{NumGrammar, Parse, Untyped};

type ComplexGrammar = NumGrammar<Complex64>;

fn compile_module(program: &str) -> ExecutableModule<Complex64> {
    let block = Untyped::<ComplexGrammar>::parse_statements(program).unwrap();
    ExecutableModule::new("custom_cmp", &block).unwrap()
}

#[test]
fn no_comparisons() {
    const PROGRAM: &str = r#"
        // Without comparisons, all comparison ops will return `false`.
        assert(!(1 < -1 || 1 <= -1 || 1 > -1 || 1 >= -1));
        assert(!(-1 + 2i < 1 + i));
    "#;
    let module = compile_module(PROGRAM);

    let mut env = Environment::with_arithmetic(StdArithmetic.without_comparisons());
    env.extend(Prelude::iter().chain(Assertions::iter()));
    module.with_env(&env).unwrap().run().unwrap();
}

#[test]
fn custom_cmp_function() {
    //! Defines comparisons by the real part of the number.

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
    let module = compile_module(PROGRAM);

    let arithmetic =
        StdArithmetic.with_comparison(|x: &Complex64, y: &Complex64| x.re.partial_cmp(&y.re));
    let mut env = Environment::with_arithmetic(arithmetic);
    env.extend(Prelude::iter().chain(Assertions::iter()));
    module.with_env(&env).unwrap().run().unwrap();
}

#[test]
fn partial_cmp_function() {
    //! Defines comparisons on real numbers, leaving numbers with imaginary parts non-comparable.

    const PROGRAM: &str = r#"
        // Real numbers can be compared.
        assert(-1 < 1 && 2 > 1);
        // Numbers with an imaginary part are not comparable.
        assert(!(-1 < i || -1 <= i || -1 > i || -1 >= i));
        assert(!(2i > 3 || 2i <= 3));
    "#;
    let module = compile_module(PROGRAM);

    let arithmetic = StdArithmetic.with_comparison(|x: &Complex64, y: &Complex64| {
        if x.im == 0.0 && y.im == 0.0 {
            x.re.partial_cmp(&y.re)
        } else {
            None
        }
    });
    let mut env = Environment::with_arithmetic(arithmetic);
    env.extend(Prelude::iter().chain(Assertions::iter()));
    module.with_env(&env).unwrap().run().unwrap();
}
