//! Demonstrates how to define high-order native functions.

use arithmetic_eval::{
    fns, wrap_fn, wrap_fn_with_context, CallContext, ErrorKind, EvalResult, ExecutableModule,
    Function, NativeFn, SpannedValue, Value,
};
use arithmetic_parser::{
    grammars::{F32Grammar, Parse, Untyped},
    StripCode,
};

/// Function that applies the `inner` function the specified amount of times to the result of
/// the previous execution.
#[derive(Debug, Clone)]
struct Repeated<T> {
    inner: Function<'static, T>,
    times: usize,
}

impl<T: Clone> NativeFn<T> for Repeated<T> {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        context: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        if args.len() != 1 {
            let err = ErrorKind::native("Should be called with single argument");
            return Err(context.call_site_error(err));
        }
        let mut arg = args.pop().unwrap();
        for _ in 0..self.times {
            let result = self.inner.evaluate(vec![arg], context)?;
            arg = context.apply_call_span(result);
        }
        Ok(arg.extra)
    }
}

fn repeat(function: Function<'_, f32>, times: f32) -> Result<Function<'_, f32>, String> {
    if times <= 0.0 {
        Err("`times` should be positive".to_owned())
    } else {
        let function = Repeated {
            inner: function.strip_code(),
            times: times as usize,
        };
        Ok(Function::native(function))
    }
}

fn eager_repeat<'a>(
    context: &mut CallContext<'_, 'a, f32>,
    function: Function<'a, f32>,
    times: f32,
    mut arg: Value<'a, f32>,
) -> EvalResult<'a, f32> {
    if times <= 0.0 {
        Err(context.call_site_error(ErrorKind::native("`times` should be positive")))
    } else {
        for _ in 0..times as usize {
            arg = function.evaluate(vec![context.apply_call_span(arg)], context)?;
        }
        Ok(arg)
    }
}

#[test]
fn repeated_function() {
    let program = r#"
        fn = |x| 2 * x + 1;
        repeated = repeat(fn, 3);
        // 2 * 1 + 1 = 3 -> 2 * 3 + 1 = 7 -> 2 * 7 + 1 = 15
        assert_eq(repeated(1), 15);
        // -1 is the immovable point of the transform
        assert_eq(repeated(-1), -1);
    "#;
    let program = Untyped::<F32Grammar>::parse_statements(program).unwrap();

    let program = ExecutableModule::builder("repeat", &program)
        .unwrap()
        .with_import("repeat", Value::native_fn(wrap_fn!(2, repeat)))
        .with_import("assert_eq", Value::native_fn(fns::AssertEq))
        .build();
    program.run().unwrap();
}

#[test]
fn eager_repeated_function() {
    let program = r#"
        fn = |x| 2 * x + 1;
        // 2 * 1 + 1 = 3 -> 2 * 3 + 1 = 7 -> 2 * 7 + 1 = 15
        assert_eq(repeat(fn, 3, 1), 15);
        // -1 is the immovable point of the transform
        assert_eq(repeat(fn, 3, -1), -1);
    "#;
    let program = Untyped::<F32Grammar>::parse_statements(program).unwrap();

    let program = ExecutableModule::builder("repeat", &program)
        .unwrap()
        .with_import(
            "repeat",
            Value::native_fn(wrap_fn_with_context!(3, eager_repeat)),
        )
        .with_import("assert_eq", Value::native_fn(fns::AssertEq))
        .build();
    program.run().unwrap();
}
