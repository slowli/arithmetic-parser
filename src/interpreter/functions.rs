//! Standard functions for the interpreter.

use num_traits::{Num, One, Pow, Zero};

use std::ops;

use super::{CallContext, EvalError, EvalResult, NativeFn, Value};
use crate::Grammar;

/// Assertion function.
///
/// # Type
///
/// ```text
/// fn(bool)
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::{
/// #     interpreter::{Assert, EvalError, Interpreter}, grammars::F32Grammar, GrammarExt, Span,
/// # };
/// # use assert_matches::assert_matches;
/// let program = r#"
///     assert(1 + 2 == 3); # this assertion is fine
///     assert(3^2 == 10); # this one will fail
/// "#;
/// let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
///
/// let mut interpreter = Interpreter::new();
/// interpreter.innermost_scope().insert_native_fn("assert", Assert);
/// let err = interpreter.evaluate(&block).unwrap_err();
/// assert_eq!(err.inner.fragment, "assert(3^2 == 10)");
/// assert_matches!(
///     err.inner.extra,
///     EvalError::NativeCall(ref msg) if msg == "Assertion failed"
/// );
/// ```
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
///
/// # Type
///
/// ```text
/// fn<T>(bool, T, T) -> T
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::{
/// #     interpreter::{If, EvalError, Interpreter, Value}, grammars::F32Grammar,
/// #     GrammarExt, Span,
/// # };
/// let program = "x = 3; if(x == 2, -1, x + 1)";
/// let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
///
/// let mut interpreter = Interpreter::new();
/// interpreter.innermost_scope().insert_native_fn("if", If);
/// let ret = interpreter.evaluate(&block).unwrap();
/// assert_eq!(ret, Value::Number(4.0));
/// ```
///
/// You can also use the lazy evaluation by returning a function and evaluating it
/// afterwards:
///
/// ```
/// # use arithmetic_parser::{
/// #     interpreter::{If, EvalError, Interpreter, Value}, grammars::F32Grammar,
/// #     GrammarExt, Span,
/// # };
/// let program = "x = 3; if(x == 2, || -1, || x + 1)()";
/// let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
///
/// let mut interpreter = Interpreter::new();
/// interpreter.innermost_scope().insert_native_fn("if", If);
/// let ret = interpreter.evaluate(&block).unwrap();
/// assert_eq!(ret, Value::Number(4.0));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct If;

impl<T> NativeFn<T> for If
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
            [Value::Bool(condition), then_val, else_val] => Ok(if *condition {
                then_val.to_owned()
            } else {
                else_val.to_owned()
            }),
            _ => {
                let err = EvalError::native("`if` requires 3 arguments");
                Err(ctx.call_site_error(err))
            }
        }
    }
}

/// Loop function that evaluates the provided closure one or more times.
///
/// # Type
///
/// ```text
/// fn<T, R>(T, fn(T) -> (false, R) | (true, T)) -> R
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::{
/// #     interpreter::{Compare, EvalError, If, Interpreter, Loop, Value}, grammars::F32Grammar,
/// #     GrammarExt, Span,
/// # };
/// let program = r#"
///     factorial = |x| {
///         loop((x, 1), |(i, acc)| {
///             continue = cmp(i, 1) != -1; # i >= 1
///             (continue, if(continue, (i - 1, acc * i), acc))
///         })
///     };
///     factorial(5) == 120 && factorial(10) == 3628800
/// "#;
/// let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
///
/// let mut interpreter = Interpreter::new();
/// interpreter
///     .innermost_scope()
///     .insert_native_fn("cmp", Compare)
///     .insert_native_fn("if", If)
///     .insert_native_fn("loop", Loop);
/// let ret = interpreter.evaluate(&block).unwrap();
/// assert_eq!(ret, Value::Bool(true));
/// ```
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
///
/// # Type
///
/// ```text
/// fn(Num, Num) -> Num
/// ```
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
            [Value::Number(x), Value::Number(y)] => Ok(Value::Number(if *x < *y {
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
            [Value::Number(x)] => {
                let output = (self.function)(x.to_owned());
                Ok(Value::Number(output))
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
            [Value::Number(x), Value::Number(y)] => {
                let output = (self.function)(x.to_owned(), y.to_owned());
                Ok(Value::Number(output))
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
    use crate::{grammars::F32Grammar, interpreter::Interpreter, GrammarExt, Span};

    #[test]
    fn if_basic() {
        let mut interpreter = Interpreter::new();
        interpreter
            .innermost_scope()
            .insert_native_fn("if", If)
            .insert_native_fn("cmp", Compare);

        let program = r#"
            x = 1.0;
            if(cmp(x, 2) == -1, x + 5, 3 - x)
        "#;
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let ret = interpreter.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Number(6.0));

        let program = r#"
            x = 4.5;
            if(cmp(x, 2) == -1, || x + 5, || 3 - x)()
        "#;
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let ret = interpreter.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Number(-1.5));
    }

    #[test]
    fn loop_basic() {
        let mut interpreter = Interpreter::new();
        interpreter
            .innermost_scope()
            .insert_native_fn("loop", Loop)
            .insert_native_fn("if", If)
            .insert_native_fn("cmp", Compare);

        let program = r#"
            # Finds the greatest power of 2 lesser or equal to the value.
            discrete_log2 = |x| {
                loop(0, |i| {
                    continue = cmp(2^i, x) != 1;
                    (continue, if(continue, i + 1, i - 1))
                })
            };
            discrete_log2(9)
        "#;
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let ret = interpreter.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Number(3.0));

        let program = "(discrete_log2(1), discrete_log2(2), \
            discrete_log2(4), discrete_log2(6.5), discrete_log2(1000))";
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let ret = interpreter.evaluate(&block).unwrap();
        assert_eq!(
            ret,
            Value::Tuple(vec![
                Value::Number(0.0),
                Value::Number(1.0),
                Value::Number(2.0),
                Value::Number(2.0),
                Value::Number(9.0),
            ])
        );
    }
}
