//! Demonstrates how to define high-order native functions.

use arithmetic_eval::{
    fns, wrap_fn, wrap_fn_with_context, CallContext, EvalError, EvalResult, Function, Interpreter,
    NativeFn, Number, SpannedValue, Value,
};
use arithmetic_parser::{grammars::F32Grammar, Grammar, GrammarExt, InputSpan, StripCode};

/// Function that applies the `inner` function the specified amount of times to the result of
/// the previous execution.
#[derive(Debug, Clone)]
struct Repeated<G: Grammar> {
    inner: Function<'static, G>,
    times: usize,
}

impl<G: Grammar> NativeFn<G> for Repeated<G>
where
    G::Lit: Number,
{
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, G>>,
        context: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, G> {
        if args.len() != 1 {
            let err = EvalError::native("Should be called with single argument");
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

fn repeat<G: Grammar<Lit = f32>>(
    function: Function<'_, G>,
    times: f32,
) -> Result<Function<'_, G>, String> {
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

fn eager_repeat<'a, G: Grammar<Lit = f32>>(
    context: &mut CallContext<'_, 'a>,
    function: Function<'a, G>,
    times: f32,
    mut arg: Value<'a, G>,
) -> EvalResult<'a, G> {
    if times <= 0.0 {
        Err(context.call_site_error(EvalError::native("`times` should be positive")))
    } else {
        for _ in 0..times as usize {
            arg = function.evaluate(vec![context.apply_call_span(arg)], context)?;
        }
        Ok(arg)
    }
}

#[test]
fn repeated_function() {
    let mut interpreter = Interpreter::new();
    interpreter
        .insert_native_fn("repeat", wrap_fn!(2, repeat))
        .insert_native_fn("assert", fns::Assert);

    let program = r#"
        fn = |x| 2 * x + 1;
        repeated = fn.repeat(3);
        # 2 * 1 + 1 = 3 -> 2 * 3 + 1 = 7 -> 2 * 7 + 1 = 15
        assert(repeated(1) == 15);
        # -1 is the immovable point of the transform
        assert(repeated(-1) == -1);
    "#;
    let program = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
    interpreter.evaluate(&program).unwrap();
}

#[test]
fn eager_repeated_function() {
    let mut interpreter = Interpreter::new();
    interpreter
        .insert_native_fn("repeat", wrap_fn_with_context!(3, eager_repeat))
        .insert_native_fn("assert", fns::Assert);

    let program = r#"
        fn = |x| 2 * x + 1;
        # 2 * 1 + 1 = 3 -> 2 * 3 + 1 = 7 -> 2 * 7 + 1 = 15
        assert(fn.repeat(3, 1) == 15);
        # -1 is the immovable point of the transform
        assert(fn.repeat(3, -1) == -1);
    "#;
    let program = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
    interpreter.evaluate(&program).unwrap();
}
