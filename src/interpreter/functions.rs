//! Standard functions for the interpreter.

use num_traits::{Num, One, Pow, Zero};

use std::ops;

use super::{CallContext, EvalError, EvalResult, NativeFn, Value};
use crate::Grammar;

/// Assertion function.
#[derive(Debug, Clone, Copy)]
pub struct Assert;

impl<T: Grammar> NativeFn<T> for Assert {
    fn evaluate<'a>(
        &self,
        args: &[Value<'a, T>],
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(args, 1)?;
        match args[0] {
            Value::Bool(true) => Ok(Value::void()),
            Value::Bool(false) => {
                let err = EvalError::native("Assertion failed");
                Err(ctx.call_site_error(err))
            }
            _ => {
                let err = EvalError::native("`assert` requires a single boolean argument");
                Err(ctx.call_site_error(err))
            }
        }
    }
}

/// `if` function that eagerly evaluates "if" / "else" terms.
#[derive(Debug, Clone, Copy)]
pub struct EagerIf;

impl<T> NativeFn<T> for EagerIf
where
    T: Grammar,
    T::Lit: Num + ops::Neg<Output = T::Lit> + Pow<T::Lit, Output = T::Lit>,
{
    fn evaluate<'a>(
        &self,
        args: &[Value<'a, T>],
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(args, 3)?;
        match args {
            [Value::Bool(condition), then_val, else_val] => {
                Ok(if *condition {
                    then_val.to_owned()
                } else {
                    else_val.to_owned()
                })
            }
            _ => {
                let err = EvalError::native("`if` requires 3 arguments");
                Err(ctx.call_site_error(err))
            }
        }
    }
}

/// `if` function that lazily evaluates "if" / "else" terms.
#[derive(Debug, Clone, Copy)]
pub struct LazyIf;

impl<T> NativeFn<T> for LazyIf
where
    T: Grammar,
    T::Lit: Num + ops::Neg<Output = T::Lit> + Pow<T::Lit, Output = T::Lit>,
{
    fn evaluate<'a>(
        &self,
        args: &[Value<'a, T>],
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(args, 3)?;
        match args {
            [Value::Bool(condition), Value::Function(then_fn), Value::Function(else_fn)] => {
                if *condition {
                    then_fn.evaluate(&[], ctx)
                } else {
                    else_fn.evaluate(&[], ctx)
                }
            }
            _ => {
                let err = EvalError::native("`if` requires 3 arguments");
                Err(ctx.call_site_error(err))
            }
        }
    }
}

/// Type signature: `fn(T, fn(T) -> (false, R) | (true, T)) -> R
#[derive(Debug, Clone, Copy)]
pub struct Loop;

impl Loop {
    const ITER_ERROR: &'static str =
        "iteration function should return a 2-element tuple with first bool value";
}

impl<T> NativeFn<T> for Loop
where
    T: Grammar,
    T::Lit: Num + ops::Neg<Output = T::Lit> + Pow<T::Lit, Output = T::Lit>,
{
    fn evaluate<'a>(
        &self,
        args: &[Value<'a, T>],
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(args, 2)?;
        match args {
            [init, Value::Function(iter)] => {
                let mut arg = init.clone();
                loop {
                    match iter.evaluate(&[arg], ctx)? {
                        Value::Tuple(mut tuple) => {
                            let (ret_or_next_arg, flag) = if tuple.len() == 2 {
                                (tuple.pop().unwrap(), tuple.pop().unwrap())
                            } else {
                                let err = EvalError::native(Self::ITER_ERROR);
                                break Err(ctx.call_site_error(err));
                            };

                            match (flag, ret_or_next_arg) {
                                (Value::Bool(false), ret) => break Ok(ret),
                                (Value::Bool(true), next_arg) => {
                                    arg = next_arg;
                                }
                                _ => {
                                    let err = EvalError::native(Self::ITER_ERROR);
                                    break Err(ctx.call_site_error(err));
                                }
                            }
                        }
                        _ => {
                            let err = EvalError::native(Self::ITER_ERROR);
                            break Err(ctx.call_site_error(err));
                        }
                    }
                }
            }
            _ => {
                let err = EvalError::native(
                    "loop requires two arguments: an initializer and iteration fn",
                );
                Err(ctx.call_site_error(err))
            }
        }
    }
}

/// Comparator function on two arguments. Returns `-1` if the first argument is lesser than
/// the second, `1` if the first argument is greater, and `0` in other cases.
#[derive(Debug, Clone, Copy)]
pub struct Compare;

impl<T> NativeFn<T> for Compare
where
    T: Grammar,
    T::Lit: Num + ops::Neg<Output = T::Lit> + Pow<T::Lit, Output = T::Lit> + PartialOrd,
{
    fn evaluate<'a>(
        &self,
        args: &[Value<'a, T>],
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(args, 2)?;
        match args {
            [Value::Simple(x), Value::Simple(y)] => Ok(Value::Simple(if *x < *y {
                -<T::Lit as One>::one()
            } else if *x > *y {
                <T::Lit as One>::one()
            } else {
                <T::Lit as Zero>::zero()
            })),
            _ => {
                let err = EvalError::native("Compare requires 2 primitive arguments");
                Err(ctx.call_site_error(err))
            }
        }
    }
}

/// Unary function wrapper.
#[derive(Debug, Clone, Copy)]
pub struct UnaryFn<F> {
    function: F,
}

impl<F> UnaryFn<F> {
    /// Creates a new function.
    pub const fn new(function: F) -> Self {
        Self { function }
    }
}

impl<F, T> NativeFn<T> for UnaryFn<F>
where
    T: Grammar,
    F: Fn(T::Lit) -> T::Lit,
{
    fn evaluate<'a>(
        &self,
        args: &[Value<'a, T>],
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(args, 1)?;
        match args {
            [Value::Simple(x)] => {
                let output = (self.function)(x.to_owned());
                Ok(Value::Simple(output))
            }
            _ => {
                let err = EvalError::native("Unary function requires one primitive argument");
                Err(ctx.call_site_error(err))
            }
        }
    }
}

/// Binary function wrapper.
#[derive(Debug, Clone, Copy)]
pub struct BinaryFn<F> {
    function: F,
}

impl<F> BinaryFn<F> {
    /// Creates a new function.
    pub const fn new(function: F) -> Self {
        Self { function }
    }
}

impl<F, T> NativeFn<T> for BinaryFn<F>
where
    T: Grammar,
    F: Fn(T::Lit, T::Lit) -> T::Lit,
{
    fn evaluate<'a>(
        &self,
        args: &[Value<'a, T>],
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(args, 2)?;
        match args {
            [Value::Simple(x), Value::Simple(y)] => {
                let output = (self.function)(x.to_owned(), y.to_owned());
                Ok(Value::Simple(output))
            }
            _ => {
                let err = EvalError::native("Binary function requires two primitive arguments");
                Err(ctx.call_site_error(err))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grammars::F32Grammar, interpreter::Context, GrammarExt, Span};

    #[test]
    fn lazy_if_basic() {
        let mut context = Context::new();
        context
            .innermost_scope()
            .insert_native_fn("if", LazyIf)
            .insert_native_fn("cmp", Compare);

        let program = r#"
            x = 1.0;
            if(cmp(x, 2) == -1, || x + 5, || 3 - x)
        "#;
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let ret = context.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Simple(6.0));

        let program = r#"
            x = 4.5;
            if(cmp(x, 2) == -1, || x + 5, || 3 - x)
        "#;
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let ret = context.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Simple(-1.5));
    }

    #[test]
    fn loop_basic() {
        let mut context = Context::new();
        context
            .innermost_scope()
            .insert_native_fn("loop", Loop)
            .insert_native_fn("if", EagerIf)
            .insert_native_fn("cmp", Compare);

        let program = r#"
            x = 9.0;
            # Finds the greatest power of 2 lesser on equal to the value.
            loop((0, 1), |(i, pow)| {
                continue = cmp(pow, x) != 1;
                (continue, if(continue, (i + 1, pow * 2), i - 1))
            })
        "#;
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let ret = context.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Simple(3.0));
    }
}
