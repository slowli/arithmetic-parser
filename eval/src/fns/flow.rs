//! Flow control functions.

use crate::{
    alloc::{vec, Vec},
    error::AuxErrorInfo,
    fns::extract_fn,
    CallContext, ErrorKind, EvalResult, NativeFn, SpannedValue, Value,
};

/// `if` function that eagerly evaluates "if" / "else" terms.
///
/// # Type
///
/// (using [`arithmetic-typing`](https://docs.rs/arithmetic-typing/) notation)
///
/// ```text
/// (Bool, 'T, 'T) -> 'T
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = "x = 3; if(x == 2, -1, x + 1)";
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("if", fns::If)
///     .compile_module("if_test", &program)?;
/// assert_eq!(module.run()?, Value::Prim(4.0));
/// # Ok(())
/// # }
/// ```
///
/// You can also use the lazy evaluation by returning a function and evaluating it
/// afterwards:
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = "x = 3; if(x == 2, || -1, || x + 1)()";
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("if", fns::If)
///     .compile_module("if_test", &program)?;
/// assert_eq!(module.run()?, Value::Prim(4.0));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct If;

impl<T> NativeFn<T> for If {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 3)?;
        let else_val = args.pop().unwrap().extra;
        let then_val = args.pop().unwrap().extra;

        if let Value::Bool(condition) = &args[0].extra {
            Ok(if *condition { then_val } else { else_val })
        } else {
            let err = ErrorKind::native("`if` requires first arg to be boolean");
            Err(ctx
                .call_site_error(err)
                .with_span(&args[0], AuxErrorInfo::InvalidArg))
        }
    }
}

/// Loop function that evaluates the provided closure one or more times.
///
/// # Type
///
/// (using [`arithmetic-typing`](https://docs.rs/arithmetic-typing/) notation with custom
/// notation for union types; they are not supported in the typing crate)
///
/// ```text
/// ('T, ('T) -> (false, 'R) | (true, 'T)) -> 'R
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     factorial = |x| {
///         loop((x, 1), |(i, acc)| {
///             continue = i >= 1;
///             (continue, if(continue, (i - 1, acc * i), acc))
///         })
///     };
///     factorial(5) == 120 && factorial(10) == 3628800
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("if", fns::If)
///     .insert_native_fn("loop", fns::Loop)
///     .compile_module("test_loop", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Loop;

impl Loop {
    const ITER_ERROR: &'static str =
        "iteration function should return a 2-element tuple with first bool value";
}

impl<T: 'static + Clone> NativeFn<T> for Loop {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 2)?;
        let iter = args.pop().unwrap();
        let iter = if let Value::Function(iter) = iter.extra {
            iter
        } else {
            let err = ErrorKind::native("Second argument of `loop` should be an iterator function");
            return Err(ctx
                .call_site_error(err)
                .with_span(&iter, AuxErrorInfo::InvalidArg));
        };

        let mut arg = args.pop().unwrap();
        loop {
            if let Value::Tuple(tuple) = iter.evaluate(vec![arg], ctx)? {
                let (ret_or_next_arg, flag) = if tuple.len() == 2 {
                    let mut tuple = Vec::from(tuple);
                    (tuple.pop().unwrap(), tuple.pop().unwrap())
                } else {
                    let err = ErrorKind::native(Self::ITER_ERROR);
                    break Err(ctx.call_site_error(err));
                };

                match (flag, ret_or_next_arg) {
                    (Value::Bool(false), ret) => break Ok(ret),
                    (Value::Bool(true), next_arg) => {
                        arg = ctx.apply_call_span(next_arg);
                    }
                    _ => {
                        let err = ErrorKind::native(Self::ITER_ERROR);
                        break Err(ctx.call_site_error(err));
                    }
                }
            } else {
                let err = ErrorKind::native(Self::ITER_ERROR);
                break Err(ctx.call_site_error(err));
            }
        }
    }
}

/// Loop function that evaluates the provided closure while a certain condition is true.
/// Returns the loop state afterwards.
///
/// # Type
///
/// (using [`arithmetic-typing`](https://docs.rs/arithmetic-typing/) notation)
///
/// ```text
/// ('T, ('T) -> Bool, ('T) -> 'T) -> 'T
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     factorial = |x| {
///         (_, acc) = (x, 1).while(
///             |(i, _)| i >= 1,
///             |(i, acc)| (i - 1, acc * i),
///         );
///         acc
///     };
///     factorial(5) == 120 && factorial(10) == 3628800
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("while", fns::While)
///     .compile_module("test_while", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct While;

impl<T: 'static + Clone> NativeFn<T> for While {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 3)?;

        let step_fn = extract_fn(
            ctx,
            args.pop().unwrap(),
            "`while` requires third arg to be a step function",
        )?;
        let condition_fn = extract_fn(
            ctx,
            args.pop().unwrap(),
            "`while` requires second arg to be a condition function",
        )?;
        let mut state = args.pop().unwrap();
        let state_span = state.copy_with_extra(());

        loop {
            let condition_value = condition_fn.evaluate(vec![state.clone()], ctx)?;
            match condition_value {
                Value::Bool(true) => {
                    let new_state = step_fn.evaluate(vec![state], ctx)?;
                    state = state_span.copy_with_extra(new_state);
                }
                Value::Bool(false) => break Ok(state.extra),
                _ => {
                    let err =
                        ErrorKind::native("`while` requires condition function to return booleans");
                    return Err(ctx.call_site_error(err));
                }
            }
        }
    }
}
