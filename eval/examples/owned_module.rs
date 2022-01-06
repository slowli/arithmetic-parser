//! Shows how to use owned modules.

use assert_matches::assert_matches;

use arithmetic_eval::{
    env::{Assertions, Environment, Prelude},
    fns, ErrorKind, ExecutableModule, Value,
};
use arithmetic_parser::{
    grammars::{F64Grammar, MockTypes, Parse, WithMockedTypes},
    BinaryOp, StripCode, StripResultExt,
};

/// We need to process some type annotations, but don't want to depend
/// on the typing crate for that. Hence, we define a grammar that gobbles up the exact
/// type annotations used in the script.
struct MockedTypesList;

impl MockTypes for MockedTypesList {
    const MOCKED_TYPES: &'static [&'static str] = &["Num", "[_]", "any"];
}

type Grammar = WithMockedTypes<F64Grammar, MockedTypesList>;

fn create_module<'a>(
    module_name: &'static str,
    program: &'a str,
) -> anyhow::Result<ExecutableModule<'a, f64>> {
    let block = Grammar::parse_statements(program).strip_err()?;
    Ok(ExecutableModule::new(module_name, &block).strip_err()?)
}

fn create_static_module(
    module_name: &'static str,
    program: &str,
) -> anyhow::Result<ExecutableModule<'static, f64>> {
    // By default, the module is tied by its lifetime to the `program`. However,
    // we can break this tie using the `StripCode` trait.
    create_module(module_name, program).map(StripCode::strip_code)
}

fn main() -> anyhow::Result<()> {
    let mut env = Environment::new();
    env.extend(Prelude::vars().chain(Assertions::vars()));
    env.insert_native_fn("fold", fns::Fold)
        .insert_native_fn("push", fns::Push)
        .insert("INF", Value::Prim(f64::INFINITY));

    let sum_module = {
        let dynamic_program = String::from("|...vars| fold(vars, 0, |acc, x| acc + x)");
        create_static_module("sum", &dynamic_program)?
        // Ensure that the program is indeed dropped by using a separate scope.
    };

    // The code is dropped here, but the module is still usable.
    let sum_fn = sum_module.with_env(&env)?.run()?;
    assert!(sum_fn.is_function());

    // Let's import the function into another module and check that it works.
    let test_module = create_module("test", "sum(1, 2, -5)")?;
    let mut non_static_env = env.clone();
    non_static_env.insert("sum", sum_fn);
    let sum_value = test_module.with_env(&non_static_env)?.run()?;
    assert_eq!(sum_value, Value::Prim(-2.0)); // 1 + 2 - 5

    // Errors are handled as well.
    let bogus_module = create_module("bogus", "sum(1, true, -5)")?;

    let err = bogus_module.with_env(&non_static_env)?.run().unwrap_err();
    println!("Expected error:\n{:#}", err);
    assert_matches!(
        err.source().kind(),
        ErrorKind::UnexpectedOperand { op } if *op == BinaryOp::Add.into()
    );

    // Naturally, spans in the stripped module do not retain refs to source code,
    // but rather contain info sufficient to be recoverable.
    assert_eq!(
        err.source().main_span().code().code_or_location("call"),
        "call at 1:40"
    );

    // Importing into a stripped module also works. Let's redefine the `fold` import.
    let fold_program = include_str!("rfold.script");
    let fold_program = String::from(fold_program);
    let fold_module = create_module("rfold", &fold_program)?;
    let rfold_fn = fold_module.with_env(&env).strip_err()?.run().strip_err()?;

    // Due to lifetime checks, we need to re-assign `env`, since the original one
    // is inferred to have `'static` lifetime.
    let mut env = env;
    env.insert("fold", rfold_fn);
    let rfold_sum = sum_module.with_env(&env).strip_err()?.run().strip_err()?;
    env.insert("sum", rfold_sum);
    let sum_value = test_module.with_env(&env).strip_err()?.run().strip_err()?;
    assert_eq!(sum_value, Value::Prim(-2.0));

    Ok(())
}
