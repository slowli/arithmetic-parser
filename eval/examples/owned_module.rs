//! Shows how to use owned modules.

use assert_matches::assert_matches;

use arithmetic_eval::{Environment, ErrorKind, ExecutableModule, Prelude, Value};
use arithmetic_parser::{grammars::F64Grammar, BinaryOp, GrammarExt, StripCode, StripResultExt};

fn create_module<'a>(
    module_name: &'static str,
    program: &'a str,
) -> anyhow::Result<ExecutableModule<'a, F64Grammar>> {
    let block = F64Grammar::parse_statements(program).strip_err()?;
    Ok(ExecutableModule::builder(module_name, &block)
        .strip_err()?
        .with_imports_from(&Prelude)
        .set_imports(|_| Value::void()))
}

fn create_static_module(
    module_name: &'static str,
    program: &str,
) -> anyhow::Result<ExecutableModule<'static, F64Grammar>> {
    // By default, the module is tied by its lifetime to the `program`. However,
    // we can break this tie using the `StripCode` trait.
    create_module(module_name, program).map(StripCode::strip_code)
}

fn main() -> anyhow::Result<()> {
    let sum_module = {
        let dynamic_program = String::from("|var| var.fold(0, |acc, x| acc + x)");
        create_static_module("sum", &dynamic_program)?
        // Ensure that the program is indeed dropped by using a separate scope.
    };

    // The code is dropped here, but the module is still usable.
    let sum_fn = sum_module.run()?;
    assert!(sum_fn.is_function());

    // Let's import the function into another module and check that it works.
    let mut test_module = create_module("test", "(1, 2, -5).sum()")?;
    test_module.set_import("sum", sum_fn.clone());
    let sum_value = test_module.run()?;
    assert_eq!(sum_value, Value::Number(-2.0)); // 1 + 2 - 5

    // Errors are handled as well.
    let bogus_module = create_module("bogus", "(1, true, -5).sum()")?;
    let mut env = Environment::new();
    env.extend(bogus_module.imports());
    env["sum"] = sum_fn;

    let err = bogus_module.run_in_env(&mut env).unwrap_err();
    println!("Expected error:\n{:#}", err);
    assert_matches!(
        err.source().kind(),
        ErrorKind::UnexpectedOperand { op } if *op == BinaryOp::Add.into()
    );

    // Naturally, spans in the stripped module do not retain refs to source code,
    // but rather contain info sufficient to be recoverable.
    assert_eq!(
        err.source().main_span().code().code_or_location("call"),
        "call at 1:34"
    );

    // Importing into a stripped module also works. Let's redefine the `fold` import.
    let fold_program = r#"
        # Implement right fold instead of standard left one.
        rfold = |xs, acc, fn| {
            (_, acc) = (xs, acc).while(|(xs, _)| xs != (), |(xs, acc)| {
                (...head, tail) = xs;
                (head, fn(acc, tail))
            });
            acc
        };

        # Check that it works.
        folded = (1, 2, 3).rfold((), |acc, elem| acc.push(elem));
        assert(folded == (3, 2, 1));

        rfold
    "#;
    let fold_program = String::from(fold_program);
    let fold_module = create_module("rfold", &fold_program)?;
    let rfold_fn = fold_module.run().strip_err()?;

    let mut env = Environment::new();
    env.extend(sum_module.imports());
    env["fold"] = rfold_fn;

    let rfold_sum = sum_module.run_in_env(&mut env).strip_err()?;
    // Due to lifetime checks, we need to re-assign `test_module`, since the original one
    // is inferred to have `'static` lifetime.
    let mut test_module = test_module;
    test_module.set_import("sum", rfold_sum);
    let sum_value = test_module.run().strip_err()?;
    assert_eq!(sum_value, Value::Number(-2.0));

    Ok(())
}
