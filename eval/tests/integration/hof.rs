//! Demonstrates how to define high-order native functions.

use arithmetic_eval::{
    fns, wrap_fn, wrap_fn_with_context, CallContext, Environment, ErrorKind, EvalResult,
    ExecutableModule, Function, NativeFn, SpannedValue, Value,
};
use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};

/// Function that applies the `inner` function the specified amount of times to the result of
/// the previous execution.
#[derive(Debug, Clone)]
struct Repeated<T> {
    inner: Function<T>,
    times: usize,
}

impl<T: 'static + Clone> NativeFn<T> for Repeated<T> {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<T>>,
        context: &mut CallContext<'_, T>,
    ) -> EvalResult<T> {
        if args.len() != 1 {
            let err = ErrorKind::native("Should be called with single argument");
            return Err(context.call_site_error(err));
        }
        let mut arg = args.pop().unwrap();
        for _ in 0..self.times {
            let result = self.inner.evaluate(vec![arg], context)?;
            arg = context.apply_call_location(result);
        }
        Ok(arg.extra)
    }
}

fn repeat(function: Function<f32>, times: f32) -> Result<Function<f32>, String> {
    if times <= 0.0 {
        Err("`times` should be positive".to_owned())
    } else {
        let function = Repeated {
            inner: function,
            times: times as usize,
        };
        Ok(Function::native(function))
    }
}

fn eager_repeat(
    context: &mut CallContext<'_, f32>,
    function: Function<f32>,
    times: f32,
    mut arg: Value<f32>,
) -> EvalResult<f32> {
    if times <= 0.0 {
        Err(context.call_site_error(ErrorKind::native("`times` should be positive")))
    } else {
        for _ in 0..times as usize {
            arg = function.evaluate(vec![context.apply_call_location(arg)], context)?;
        }
        Ok(arg)
    }
}

#[test]
fn repeated_function() -> anyhow::Result<()> {
    let program = r#"
        fn = |x| 2 * x + 1;
        repeated = repeat(fn, 3);
        // 2 * 1 + 1 = 3 -> 2 * 3 + 1 = 7 -> 2 * 7 + 1 = 15
        assert_eq(repeated(1), 15);
        // -1 is the immovable point of the transform
        assert_eq(repeated(-1), -1);
    "#;
    let program = Untyped::<F32Grammar>::parse_statements(program)?;
    let program = ExecutableModule::new("repeat", &program)?;

    let mut env = Environment::new();
    env.insert_native_fn("repeat", wrap_fn!(2, repeat))
        .insert_native_fn("assert_eq", fns::AssertEq);
    program.with_env(&env)?.run()?;
    Ok(())
}

#[test]
fn eager_repeated_function() -> anyhow::Result<()> {
    let program = r#"
        fn = |x| 2 * x + 1;
        // 2 * 1 + 1 = 3 -> 2 * 3 + 1 = 7 -> 2 * 7 + 1 = 15
        assert_eq(repeat(fn, 3, 1), 15);
        // -1 is the immovable point of the transform
        assert_eq(repeat(fn, 3, -1), -1);
    "#;
    let program = Untyped::<F32Grammar>::parse_statements(program)?;
    let program = ExecutableModule::new("repeat", &program)?;

    let mut env = Environment::new();
    env.insert_native_fn("repeat", wrap_fn_with_context!(3, eager_repeat))
        .insert_native_fn("assert_eq", fns::AssertEq);
    program.with_env(&env)?.run()?;
    Ok(())
}
