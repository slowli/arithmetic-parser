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
/// # use arithmetic_eval::{fns, Environment, ExecutableModule, Value};
/// # fn main() -> anyhow::Result<()> {
/// let program = "x = 3; if(x == 2, -1, x + 1)";
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_if", &program)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("if", fns::If);
/// assert_eq!(module.with_env(&env)?.run()?, Value::Prim(4.0));
/// # Ok(())
/// # }
/// ```
///
/// You can also use the lazy evaluation by returning a function and evaluating it
/// afterwards:
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ExecutableModule, Value};
/// # fn main() -> anyhow::Result<()> {
/// let program = "x = 3; if(x == 2, || -1, || x + 1)()";
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_if", &program)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("if", fns::If);
/// assert_eq!(module.with_env(&env)?.run()?, Value::Prim(4.0));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct If;

impl<T> NativeFn<T> for If {
    fn evaluate(
        &self,
        mut args: Vec<SpannedValue<T>>,
        ctx: &mut CallContext<'_, T>,
    ) -> EvalResult<T> {
        ctx.check_args_count(&args, 3)?;
        let else_val = args.pop().unwrap().extra;
        let then_val = args.pop().unwrap().extra;

        if let Value::Bool(condition) = &args[0].extra {
            Ok(if *condition { then_val } else { else_val })
        } else {
            let err = ErrorKind::native("`if` requires first arg to be boolean");
            Err(ctx
                .call_site_error(err)
                .with_location(&args[0], AuxErrorInfo::InvalidArg))
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
/// # use arithmetic_eval::{fns, Environment, ExecutableModule, Value};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     factorial = |x| {
///         (_, acc) = while(
///             (x, 1),
///             |(i, _)| i >= 1,
///             |(i, acc)| (i - 1, acc * i),
///         );
///         acc
///     };
///     factorial(5) == 120 && factorial(10) == 3628800
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_while", &program)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("while", fns::While);
/// assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct While;

impl<T: 'static + Clone> NativeFn<T> for While {
    fn evaluate(
        &self,
        mut args: Vec<SpannedValue<T>>,
        ctx: &mut CallContext<'_, T>,
    ) -> EvalResult<T> {
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
