//! Demonstrates how to define high-order native functions.

use arithmetic_eval::{
    fns, CallContext, EvalError, EvalResult, Function, Interpreter, NativeFn, Number, SpannedValue,
    Value,
};
use arithmetic_parser::{grammars::F32Grammar, Grammar, GrammarExt, Span};

/// Function that applies the `inner` function the specified amount of times to the result of
/// the previous execution.
#[derive(Debug, Clone)]
struct Repeated<'a, G: Grammar> {
    inner: Function<'a, G>,
    times: usize,
}

impl<'a, G: Grammar> NativeFn<'a, G> for Repeated<'a, G>
where
    G::Lit: Number,
{
    fn evaluate(
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
    if times < 0.0 {
        Err("`times` should be positive".to_owned())
    } else {
        let function = Repeated {
            inner: function,
            times: times as usize,
        };
        Ok(Function::native(function))
    }
}

#[test]
fn repeated_function() {
    let mut interpreter = Interpreter::new();
    interpreter.insert_native_fn("repeat", fns::wrap(repeat));

    let program = r#"
        fn = |x| 2*x + 1;
        repeated = fn.repeat(3);
        repeated(1) == 15 &&    # 2 * 1 + 1 = 3 -> 2 * 3 + 1 = 7 -> 2 * 7 + 1 = 15
            repeated(-1) == -1  # -1 is the immovable point of the transform
    "#;
    let program = F32Grammar::parse_statements(Span::new(program)).unwrap();
    let ret = interpreter.evaluate(&program).unwrap();
    assert_eq!(ret, Value::Bool(true));
}
