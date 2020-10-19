//! Standard functions for the interpreter, and the tools to define new native functions.
//!
//! # Defining native functions
//!
//! There are several ways to define new native functions:
//!
//! - Implement [`NativeFn`] manually. This is the most versatile approach, but it can be overly
//!   verbose.
//! - Use [`FnWrapper`] or the [`wrap`] function. This allows specifying arguments / output
//!   with custom types (such as `bool` or a [`Number`]), but does not work for non-`'static`
//!   types.
//! - Use [`wrap_fn`] or [`wrap_fn_with_context`] macros. These macros support
//!   the same eloquent interface as `wrap`, and also do not have `'static` requirement for args.
//!   As a downside, debugging compile-time errors when using macros can be rather painful.
//!
//! ## Why multiple ways to do the same thing?
//!
//! In the ideal world, `FnWrapper` would be used for all cases, since it does not involve
//! macro magic. Unfortunately, stable Rust currently does not provide means to describe
//! lifetime restrictions on args / return type of wrapped functions in the general case
//! (this requires [generic associated types][GAT]). As such, the (implicit) `'static` requirement
//! is a temporary measure, and macros fill the gaps in their usual clunky manner.
//!
//! [`FnWrapper`]: struct.FnWrapper.html
//! [`wrap`]: fn.wrap.html
//! [`Number`]: ../trait.Number.html
//! [`wrap_fn`]: ../macro.wrap_fn.html
//! [`wrap_fn_with_context`]: ../macro.wrap_fn_with_context.html
//! [GAT]: https://github.com/rust-lang/rust/issues/44265

use arithmetic_parser::Grammar;
use num_traits::{One, Zero};

use crate::{
    alloc::{vec, Vec},
    error::AuxErrorInfo,
    CallContext, Error, ErrorKind, EvalResult, Function, NativeFn, Number, SpannedValue, Value,
};

mod wrapper;
pub use self::wrapper::{
    enforce_closure_type, wrap, Binary, ErrorOutput, FnWrapper, FromValueError, FromValueErrorKind,
    FromValueErrorLocation, IntoEvalResult, Quaternary, Ternary, TryFromValue, Unary,
};

fn extract_number<'a, T: Grammar>(
    ctx: &CallContext<'_, 'a>,
    value: SpannedValue<'a, T>,
    error_msg: &str,
) -> Result<T::Lit, Error<'a>> {
    match value.extra {
        Value::Number(value) => Ok(value),
        _ => Err(ctx
            .call_site_error(ErrorKind::native(error_msg))
            .with_span(ctx.enrich_call_site_span(&value), AuxErrorInfo::InvalidArg)),
    }
}

fn extract_array<'a, T: Grammar>(
    ctx: &CallContext<'_, 'a>,
    value: SpannedValue<'a, T>,
    error_msg: &str,
) -> Result<Vec<Value<'a, T>>, Error<'a>> {
    if let Value::Tuple(array) = value.extra {
        Ok(array)
    } else {
        let err = ErrorKind::native(error_msg);
        Err(ctx
            .call_site_error(err)
            .with_span(ctx.enrich_call_site_span(&value), AuxErrorInfo::InvalidArg))
    }
}

fn extract_fn<'a, T: Grammar>(
    ctx: &CallContext<'_, 'a>,
    value: SpannedValue<'a, T>,
    error_msg: &str,
) -> Result<Function<'a, T>, Error<'a>> {
    if let Value::Function(function) = value.extra {
        Ok(function)
    } else {
        let err = ErrorKind::native(error_msg);
        Err(ctx
            .call_site_error(err)
            .with_span(ctx.enrich_call_site_span(&value), AuxErrorInfo::InvalidArg))
    }
}

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
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt, InputSpan};
/// # use arithmetic_eval::{fns, ErrorKind, Interpreter};
/// # use assert_matches::assert_matches;
/// let program = r#"
///     assert(1 + 2 == 3); # this assertion is fine
///     assert(3^2 == 10); # this one will fail
/// "#;
/// let block = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
///
/// let mut interpreter = Interpreter::new();
/// interpreter.insert_native_fn("assert", fns::Assert);
/// let err = interpreter.evaluate(&block).unwrap_err();
/// assert_eq!(*err.source().main_span().code().fragment(), "assert(3^2 == 10)");
/// assert_matches!(
///     err.source().kind(),
///     ErrorKind::NativeCall(ref msg) if msg == "Assertion failed"
/// );
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Assert;

impl<T: Grammar> NativeFn<T> for Assert {
    fn evaluate<'a>(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 1)?;
        match args[0].extra {
            Value::Bool(true) => Ok(Value::void()),
            Value::Bool(false) => {
                let err = ErrorKind::native("Assertion failed");
                Err(ctx.call_site_error(err))
            }
            _ => {
                let err = ErrorKind::native("`assert` requires a single boolean argument");
                Err(ctx.call_site_error(err).with_span(
                    ctx.enrich_call_site_span(&args[0]),
                    AuxErrorInfo::InvalidArg,
                ))
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
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt, InputSpan};
/// # use arithmetic_eval::{fns::If, Interpreter, Value};
/// let program = "x = 3; if(x == 2, -1, x + 1)";
/// let block = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
///
/// let mut interpreter = Interpreter::new();
/// interpreter.insert_native_fn("if", If);
/// let ret = interpreter.evaluate(&block).unwrap();
/// assert_eq!(ret, Value::Number(4.0));
/// ```
///
/// You can also use the lazy evaluation by returning a function and evaluating it
/// afterwards:
///
/// ```
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt, InputSpan};
/// # use arithmetic_eval::{fns::If, Interpreter, Value};
/// let program = "x = 3; if(x == 2, || -1, || x + 1)()";
/// let block = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
///
/// let mut interpreter = Interpreter::new();
/// interpreter.insert_native_fn("if", If);
/// let ret = interpreter.evaluate(&block).unwrap();
/// assert_eq!(ret, Value::Number(4.0));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct If;

impl<T> NativeFn<T> for If
where
    T: Grammar,
    T::Lit: Number,
{
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 3)?;
        let else_val = args.pop().unwrap().extra;
        let then_val = args.pop().unwrap().extra;

        if let Value::Bool(condition) = &args[0].extra {
            Ok(if *condition { then_val } else { else_val })
        } else {
            let err = ErrorKind::native("`if` requires first arg to be boolean");
            Err(ctx.call_site_error(err).with_span(
                ctx.enrich_call_site_span(&args[0]),
                AuxErrorInfo::InvalidArg,
            ))
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
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt, InputSpan};
/// # use arithmetic_eval::{fns, Interpreter, Value};
/// let program = r#"
///     factorial = |x| {
///         loop((x, 1), |(i, acc)| {
///             continue = i >= 1;
///             (continue, if(continue, (i - 1, acc * i), acc))
///         })
///     };
///     factorial(5) == 120 && factorial(10) == 3628800
/// "#;
/// let block = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
///
/// let mut interpreter = Interpreter::new();
/// interpreter
///     .insert_native_fn("cmp", fns::Compare)
///     .insert_native_fn("if", fns::If)
///     .insert_native_fn("loop", fns::Loop);
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
    T::Lit: Number,
{
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 2)?;
        let iter = args.pop().unwrap();
        let iter = if let Value::Function(iter) = iter.extra {
            iter
        } else {
            let err = ErrorKind::native("Second argument of `loop` should be an iterator function");
            return Err(ctx
                .call_site_error(err)
                .with_span(ctx.enrich_call_site_span(&iter), AuxErrorInfo::InvalidArg));
        };

        let mut arg = args.pop().unwrap();
        loop {
            if let Value::Tuple(mut tuple) = iter.evaluate(vec![arg], ctx)? {
                let (ret_or_next_arg, flag) = if tuple.len() == 2 {
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
/// ```text
/// fn<T>(T, fn(T) -> bool, fn(T) -> T) -> T
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt, InputSpan};
/// # use arithmetic_eval::{fns, Interpreter, Value};
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
/// let block = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
///
/// let mut interpreter = Interpreter::new();
/// interpreter
///     .insert_native_fn("cmp", fns::Compare)
///     .insert_native_fn("while", fns::While);
/// let ret = interpreter.evaluate(&block).unwrap();
/// assert_eq!(ret, Value::Bool(true));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct While;

impl<T> NativeFn<T> for While
where
    T: Grammar,
    T::Lit: Number,
{
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a>,
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

/// Map function that evaluates the provided function on each item of the tuple.
///
/// # Type
///
/// ```text
/// fn<T, U>([T], fn(T) -> U) -> [U]
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt, InputSpan};
/// # use arithmetic_eval::{fns, Interpreter, Value};
/// let program = r#"
///     xs = (1, -2, 3, -0.3);
///     map(xs, |x| if(x > 0, x, 0)) == (1, 0, 3, 0)
/// "#;
/// let block = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
///
/// let mut interpreter = Interpreter::new();
/// interpreter
///     .insert_native_fn("cmp", fns::Compare)
///     .insert_native_fn("if", fns::If)
///     .insert_native_fn("map", fns::Map);
/// let ret = interpreter.evaluate(&block).unwrap();
/// assert_eq!(ret, Value::Bool(true));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Map;

impl<T> NativeFn<T> for Map
where
    T: Grammar,
    T::Lit: Number,
{
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 2)?;
        let map_fn = extract_fn(
            ctx,
            args.pop().unwrap(),
            "`map` requires second arg to be a mapping function",
        )?;
        let array = extract_array(
            ctx,
            args.pop().unwrap(),
            "`map` requires first arg to be a tuple",
        )?;

        let mapped: Result<Vec<_>, _> = array
            .into_iter()
            .map(|value| {
                let spanned = ctx.apply_call_span(value);
                map_fn.evaluate(vec![spanned], ctx)
            })
            .collect();
        mapped.map(Value::Tuple)
    }
}

/// Filter function that evaluates the provided function on each item of the tuple and retains
/// only elements for which the function returned `true`.
///
/// # Type
///
/// ```text
/// fn<T>([T], fn(T) -> bool) -> [T]
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt, InputSpan};
/// # use arithmetic_eval::{fns, Interpreter, Value};
/// let program = r#"
///     xs = (1, -2, 3, -7, -0.3);
///     filter(xs, |x| x > -1) == (1, 3, -0.3)
/// "#;
/// let block = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
///
/// let mut interpreter = Interpreter::new();
/// interpreter
///     .insert_native_fn("cmp", fns::Compare)
///     .insert_native_fn("filter", fns::Filter);
/// let ret = interpreter.evaluate(&block).unwrap();
/// assert_eq!(ret, Value::Bool(true));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Filter;

impl<T> NativeFn<T> for Filter
where
    T: Grammar,
    T::Lit: Number,
{
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 2)?;
        let filter_fn = extract_fn(
            ctx,
            args.pop().unwrap(),
            "`filter` requires second arg to be a filter function",
        )?;
        let array = extract_array(
            ctx,
            args.pop().unwrap(),
            "`filter` requires first arg to be a tuple",
        )?;

        let mut filtered = vec![];
        for value in array {
            let spanned = ctx.apply_call_span(value.clone());
            match filter_fn.evaluate(vec![spanned], ctx)? {
                Value::Bool(true) => filtered.push(value),
                Value::Bool(false) => { /* do nothing */ }
                _ => {
                    let err = ErrorKind::native(
                        "`filter` requires filtering function to return booleans",
                    );
                    return Err(ctx.call_site_error(err));
                }
            }
        }
        Ok(Value::Tuple(filtered))
    }
}

/// Reduce (aka fold) function that reduces the provided tuple to a single value.
///
/// # Type
///
/// ```text
/// fn<T, Acc>([T], Acc, fn(Acc, T) -> Acc) -> Acc
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt, InputSpan};
/// # use arithmetic_eval::{fns, Interpreter, Value};
/// let program = r#"
///     xs = (1, -2, 3, -7);
///     fold(xs, 1, |acc, x| acc * x) == 42
/// "#;
/// let block = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
///
/// let mut interpreter = Interpreter::new();
/// interpreter
///     .insert_native_fn("fold", fns::Fold);
/// let ret = interpreter.evaluate(&block).unwrap();
/// assert_eq!(ret, Value::Bool(true));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Fold;

impl<T> NativeFn<T> for Fold
where
    T: Grammar,
    T::Lit: Number,
{
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 3)?;
        let fold_fn = extract_fn(
            ctx,
            args.pop().unwrap(),
            "`fold` requires third arg to be a folding function",
        )?;
        let acc = args.pop().unwrap().extra;
        let array = extract_array(
            ctx,
            args.pop().unwrap(),
            "`fold` requires first arg to be a tuple",
        )?;

        array.into_iter().try_fold(acc, |acc, value| {
            let spanned_args = vec![ctx.apply_call_span(acc), ctx.apply_call_span(value)];
            fold_fn.evaluate(spanned_args, ctx)
        })
    }
}

/// Function that appends a value onto a tuple.
///
/// # Type
///
/// ```text
/// fn<T>([T], T) -> [T]
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt, InputSpan};
/// # use arithmetic_eval::{fns, Interpreter, Value};
/// let program = r#"
///     repeat = |x, times| {
///         (_, acc) = (0, ()).while(
///             |(i, _)| i < times,
///             |(i, acc)| (i + 1, push(acc, x)),
///         );
///         acc
///     };
///     repeat(-2, 3) == (-2, -2, -2) && repeat((7,), 4) == ((7,), (7,), (7,), (7,))
/// "#;
/// let block = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
///
/// let mut interpreter = Interpreter::new();
/// interpreter
///     .insert_native_fn("cmp", fns::Compare)
///     .insert_native_fn("while", fns::While)
///     .insert_native_fn("push", fns::Push);
/// let ret = interpreter.evaluate(&block).unwrap();
/// assert_eq!(ret, Value::Bool(true));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Push;

impl<T> NativeFn<T> for Push
where
    T: Grammar,
    T::Lit: Number,
{
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 2)?;
        let elem = args.pop().unwrap().extra;
        let mut array = extract_array(
            ctx,
            args.pop().unwrap(),
            "`fold` requires first arg to be a tuple",
        )?;

        array.push(elem);
        Ok(Value::Tuple(array))
    }
}

/// Function that merges two tuples.
///
/// # Type
///
/// ```text
/// fn<T>([T], [T]) -> [T]
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt, InputSpan};
/// # use arithmetic_eval::{fns, Interpreter, Value};
/// let program = r#"
///     ## Merges all arguments (which should be tuples) into a single tuple.
///     super_merge = |...xs| fold(xs, (), merge);
///     super_merge((1, 2), (3,), (), (4, 5, 6)) == (1, 2, 3, 4, 5, 6)
/// "#;
/// let block = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
///
/// let mut interpreter = Interpreter::new();
/// interpreter
///     .insert_native_fn("fold", fns::Fold)
///     .insert_native_fn("merge", fns::Merge);
/// let ret = interpreter.evaluate(&block).unwrap();
/// assert_eq!(ret, Value::Bool(true));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Merge;

impl<T> NativeFn<T> for Merge
where
    T: Grammar,
    T::Lit: Number,
{
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 2)?;
        let second = extract_array(
            ctx,
            args.pop().unwrap(),
            "`merge` requires second arg to be a tuple",
        )?;
        let mut first = extract_array(
            ctx,
            args.pop().unwrap(),
            "`merge` requires first arg to be a tuple",
        )?;

        first.extend_from_slice(&second);
        Ok(Value::Tuple(first))
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

const COMPARE_ERROR_MSG: &str = "Compare requires 2 number arguments";

impl<T> NativeFn<T> for Compare
where
    T: Grammar,
    T::Lit: Number + PartialOrd,
{
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 2)?;
        let y = args.pop().unwrap();
        let x = args.pop().unwrap();

        let x = extract_number(ctx, x, COMPARE_ERROR_MSG)?;
        let y = extract_number(ctx, y, COMPARE_ERROR_MSG)?;

        Ok(Value::Number(if x < y {
            -<T::Lit as One>::one()
        } else if x > y {
            <T::Lit as One>::one()
        } else {
            <T::Lit as Zero>::zero()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Interpreter, WildcardId};

    use arithmetic_parser::{grammars::F32Grammar, GrammarExt, InputSpan};
    use assert_matches::assert_matches;

    use core::f32;

    #[test]
    fn if_basic() {
        let mut interpreter = Interpreter::new();
        interpreter
            .insert_native_fn("if", If)
            .insert_native_fn("cmp", Compare);

        let program = r#"
            x = 1.0;
            if(x < 2, x + 5, 3 - x)
        "#;
        let block = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
        let ret = interpreter.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Number(6.0));

        let program_with_closures = r#"
            x = 4.5;
            if(x < 2, || x + 5, || 3 - x)()
        "#;
        let block_with_closures =
            F32Grammar::parse_statements(InputSpan::new(program_with_closures)).unwrap();
        let ret_with_closures = interpreter.evaluate(&block_with_closures).unwrap();
        assert_eq!(ret_with_closures, Value::Number(-1.5));
    }

    #[test]
    fn cmp_sugar() {
        let mut interpreter = Interpreter::new();
        interpreter.insert_native_fn("cmp", Compare);

        let program = "x = 1.0; x > 0 && x <= 3";
        let block = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
        let ret = interpreter.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Bool(true));

        let bogus_program = "x = 1.0; x > (1, 2)";
        let bogus_block = F32Grammar::parse_statements(InputSpan::new(bogus_program)).unwrap();
        let err = interpreter.evaluate(&bogus_block).unwrap_err();
        let err = err.source();
        assert_matches!(
            err.kind(),
            ErrorKind::NativeCall(ref message) if message.contains("Compare requires")
        );
        assert_eq!(*err.main_span().code().fragment(), "x > (1, 2)");
        assert_eq!(err.aux_spans().len(), 1);
        assert_eq!(*err.aux_spans()[0].code().fragment(), "(1, 2)");
    }

    #[test]
    fn loop_basic() {
        let mut interpreter = Interpreter::new();
        interpreter
            .insert_native_fn("loop", Loop)
            .insert_native_fn("if", If)
            .insert_native_fn("cmp", Compare);

        let program = r#"
            # Finds the greatest power of 2 lesser or equal to the value.
            discrete_log2 = |x| {
                loop(0, |i| {
                    continue = 2^i <= x;
                    (continue, if(continue, i + 1, i - 1))
                })
            };
            discrete_log2(9)
        "#;
        let block = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
        let ret = interpreter.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Number(3.0));

        let test_program = "(discrete_log2(1), discrete_log2(2), \
            discrete_log2(4), discrete_log2(6.5), discrete_log2(1000))";
        let test_block = F32Grammar::parse_statements(InputSpan::new(test_program)).unwrap();
        let test_ret = interpreter.evaluate(&test_block).unwrap();
        assert_eq!(
            test_ret,
            Value::Tuple(vec![
                Value::Number(0.0),
                Value::Number(1.0),
                Value::Number(2.0),
                Value::Number(2.0),
                Value::Number(9.0),
            ])
        );
    }

    #[test]
    fn max_value_with_fold() {
        let mut interpreter = Interpreter::new();
        interpreter
            .insert_var("Inf", Value::Number(f32::INFINITY))
            .insert_native_fn("cmp", Compare)
            .insert_native_fn("if", If)
            .insert_native_fn("fold", Fold);

        let program = r#"
            max_value = |...xs| {
                fold(xs, -Inf, |acc, x| if(x > acc, x, acc))
            };
            max_value(1, -2, 7, 2, 5) == 7 && max_value(3, -5, 9) == 9
        "#;
        let block = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
        let ret = interpreter.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Bool(true));
    }

    #[test]
    fn reverse_list_with_fold() {
        const SAMPLES: &[(&[f32], &[f32])] = &[
            (&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]),
            (&[], &[]),
            (&[1.0], &[1.0]),
        ];

        let mut interpreter = Interpreter::new();
        interpreter
            .insert_native_fn("merge", Merge)
            .insert_native_fn("fold", Fold);

        let program = r#"
            reverse = |xs| {
                fold(xs, (), |acc, x| merge((x,), acc))
            };
            xs = (-4, 3, 0, 1);
            xs.reverse() == (1, 0, 3, -4)
        "#;
        let block = F32Grammar::parse_statements(InputSpan::new(program)).unwrap();
        let ret = interpreter.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Bool(true));

        let module_block = F32Grammar::parse_statements(InputSpan::new("xs.reverse()")).unwrap();
        let mut module = interpreter.compile(WildcardId, &module_block).unwrap();

        for &(input, expected) in SAMPLES {
            let input = input.iter().copied().map(Value::Number).collect();
            let expected = expected.iter().copied().map(Value::Number).collect();
            module.set_import("xs", Value::Tuple(input));
            assert_eq!(module.run().unwrap(), Value::Tuple(expected));
        }
    }
}
