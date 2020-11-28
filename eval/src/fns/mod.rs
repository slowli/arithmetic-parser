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
//! - Use [`wrap_fn`](crate::wrap_fn) or [`wrap_fn_with_context`](crate::wrap_fn_with_context)
//!   macros. These macros support
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
//! [`Number`]: crate::Number
//! [GAT]: https://github.com/rust-lang/rust/issues/44265

use core::cmp::Ordering;

use crate::{
    alloc::{vec, Vec},
    error::AuxErrorInfo,
    CallContext, Error, ErrorKind, EvalResult, Function, NativeFn, SpannedValue, Value,
};

#[cfg(feature = "std")]
mod std;
mod wrapper;

#[cfg(feature = "std")]
pub use self::std::Dbg;
pub use self::wrapper::{
    enforce_closure_type, wrap, Binary, ErrorOutput, FnWrapper, FromValueError, FromValueErrorKind,
    FromValueErrorLocation, IntoEvalResult, Quaternary, Ternary, TryFromValue, Unary,
};

fn extract_number<'a, T, A>(
    ctx: &CallContext<'_, 'a, A>,
    value: SpannedValue<'a, T>,
    error_msg: &str,
) -> Result<T, Error<'a>> {
    match value.extra {
        Value::Number(value) => Ok(value),
        _ => Err(ctx
            .call_site_error(ErrorKind::native(error_msg))
            .with_span(&value, AuxErrorInfo::InvalidArg)),
    }
}

fn extract_array<'a, T, A>(
    ctx: &CallContext<'_, 'a, A>,
    value: SpannedValue<'a, T>,
    error_msg: &str,
) -> Result<Vec<Value<'a, T>>, Error<'a>> {
    if let Value::Tuple(array) = value.extra {
        Ok(array)
    } else {
        let err = ErrorKind::native(error_msg);
        Err(ctx
            .call_site_error(err)
            .with_span(&value, AuxErrorInfo::InvalidArg))
    }
}

fn extract_fn<'a, T, A>(
    ctx: &CallContext<'_, 'a, A>,
    value: SpannedValue<'a, T>,
    error_msg: &str,
) -> Result<Function<'a, T>, Error<'a>> {
    if let Value::Function(function) = value.extra {
        Ok(function)
    } else {
        let err = ErrorKind::native(error_msg);
        Err(ctx
            .call_site_error(err)
            .with_span(&value, AuxErrorInfo::InvalidArg))
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
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ErrorKind, VariableMap};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     assert(1 + 2 == 3); // this assertion is fine
///     assert(3^2 == 10); // this one will fail
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = Environment::new()
///     .insert_native_fn("assert", fns::Assert)
///     .compile_module("test_assert", &program)?;
///
/// let err = module.run().unwrap_err();
/// assert_eq!(*err.source().main_span().code().fragment(), "assert(3^2 == 10)");
/// assert_matches!(
///     err.source().kind(),
///     ErrorKind::NativeCall(ref msg) if msg == "Assertion failed"
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Assert;

impl<T> NativeFn<T> for Assert {
    fn evaluate<'a>(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
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
                Err(ctx
                    .call_site_error(err)
                    .with_span(&args[0], AuxErrorInfo::InvalidArg))
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
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = "x = 3; if(x == 2, -1, x + 1)";
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("if", fns::If)
///     .compile_module("if_test", &program)?;
/// assert_eq!(module.run()?, Value::Number(4.0));
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
/// assert_eq!(module.run()?, Value::Number(4.0));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
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
/// ```text
/// fn<T, R>(T, fn(T) -> (false, R) | (true, T)) -> R
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
#[derive(Debug, Clone, Copy)]
pub struct Loop;

impl Loop {
    const ITER_ERROR: &'static str =
        "iteration function should return a 2-element tuple with first bool value";
}

impl<T: Clone> NativeFn<T> for Loop {
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
#[derive(Debug, Clone, Copy)]
pub struct While;

impl<T: Clone> NativeFn<T> for While {
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
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     xs = (1, -2, 3, -0.3);
///     map(xs, |x| if(x > 0, x, 0)) == (1, 0, 3, 0)
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("if", fns::If)
///     .insert_native_fn("map", fns::Map)
///     .compile_module("test_map", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Map;

impl<T: Clone> NativeFn<T> for Map {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
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
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     xs = (1, -2, 3, -7, -0.3);
///     filter(xs, |x| x > -1) == (1, 3, -0.3)
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("filter", fns::Filter)
///     .compile_module("test_filter", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Filter;

impl<T: Clone> NativeFn<T> for Filter {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
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
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     xs = (1, -2, 3, -7);
///     fold(xs, 1, |acc, x| acc * x) == 42
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("fold", fns::Fold)
///     .compile_module("test_fold", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Fold;

impl<T: Clone> NativeFn<T> for Fold {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
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
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     repeat = |x, times| {
///         (_, acc) = (0, ()).while(
///             |(i, _)| i < times,
///             |(i, acc)| (i + 1, push(acc, x)),
///         );
///         acc
///     };
///     repeat(-2, 3) == (-2, -2, -2) &&
///         repeat((7,), 4) == ((7,), (7,), (7,), (7,))
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("while", fns::While)
///     .insert_native_fn("push", fns::Push)
///     .compile_module("test_push", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Push;

impl<T> NativeFn<T> for Push {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
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
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     // Merges all arguments (which should be tuples) into a single tuple.
///     super_merge = |...xs| fold(xs, (), merge);
///     super_merge((1, 2), (3,), (), (4, 5, 6)) == (1, 2, 3, 4, 5, 6)
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("fold", fns::Fold)
///     .insert_native_fn("merge", fns::Merge)
///     .compile_module("test_merge", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Merge;

impl<T: Clone> NativeFn<T> for Merge {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
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

/// Comparator functions on two number arguments. All functions use [`Arithmetic`] to determine
/// ordering between the args.
///
/// # Type
///
/// ```text
/// fn(Num, Num) -> Ordering // for `Compare::Raw`
/// fn(Num, Num) -> Num // for `Compare::Min` and `Compare::Max`
/// ```
///
/// [`Arithmetic`]: crate::arith::Arithmetic
///
/// # Examples
///
/// Using `min` function:
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     // Finds a minimum number in an array.
///     extended_min = |...xs| xs.fold(INFINITY, min);
///     extended_min(2, -3, 7, 1, 3) == -3
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert("INFINITY", Value::Number(f32::INFINITY))
///     .insert_native_fn("fold", fns::Fold)
///     .insert_native_fn("min", fns::Compare::Min)
///     .compile_module("test_min", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
///
/// Using `cmp` function with [`Comparisons`](crate::Comparisons).
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Comparisons, Environment, Value, VariableMap};
/// # use core::iter::FromIterator;
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     (1, -7, 0, 2).map(|x| cmp(x, 0)) == (GREATER, LESS, EQUAL, GREATER)
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::from_iter(Comparisons.iter())
///     .insert_native_fn("map", fns::Map)
///     .compile_module("test_cmp", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Compare {
    /// Returns an [`Ordering`] wrapped into an [`OpaqueRef`](crate::OpaqueRef),
    /// or [`Value::void()`] if the provided values are not comparable.
    Raw,
    /// Returns the minimum of the two numbers. If the numbers are equal, returns the first one.
    Min,
    /// Returns the maximum of the two numbers. If the numbers are equal, returns the first one.
    Max,
}

impl Compare {
    fn extract_numbers<'a, T>(
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> Result<(T, T), Error<'a>> {
        ctx.check_args_count(&args, 2)?;
        let y = args.pop().unwrap();
        let x = args.pop().unwrap();
        let x = extract_number(ctx, x, COMPARE_ERROR_MSG)?;
        let y = extract_number(ctx, y, COMPARE_ERROR_MSG)?;
        Ok((x, y))
    }
}

const COMPARE_ERROR_MSG: &str = "Compare requires 2 number arguments";

impl<T> NativeFn<T> for Compare {
    fn evaluate<'a>(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        let (x, y) = Self::extract_numbers(args, ctx)?;
        let maybe_ordering = ctx.arithmetic().partial_cmp(&x, &y);

        if let Self::Raw = self {
            Ok(maybe_ordering.map_or_else(Value::void, Value::opaque_ref))
        } else {
            let ordering =
                maybe_ordering.ok_or_else(|| ctx.call_site_error(ErrorKind::CannotCompare))?;
            let value = match (ordering, self) {
                (Ordering::Equal, _)
                | (Ordering::Less, Self::Min)
                | (Ordering::Greater, Self::Max) => x,
                _ => y,
            };
            Ok(Value::Number(value))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{alloc::ToOwned, Environment, ExecutableModule, WildcardId};

    use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
    use assert_matches::assert_matches;

    use core::{f32, iter::FromIterator};

    #[test]
    fn if_basic() {
        let block = r#"
            x = 1.0;
            if(x < 2, x + 5, 3 - x)
        "#;
        let block = Untyped::<F32Grammar>::parse_statements(block).unwrap();
        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("if", Value::native_fn(If))
            .build();
        assert_eq!(module.run().unwrap(), Value::Number(6.0));
    }

    #[test]
    fn if_with_closures() {
        let block = r#"
            x = 4.5;
            if(x < 2, || x + 5, || 3 - x)()
        "#;
        let block = Untyped::<F32Grammar>::parse_statements(block).unwrap();
        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("if", Value::native_fn(If))
            .build();
        assert_eq!(module.run().unwrap(), Value::Number(-1.5));
    }

    #[test]
    fn cmp_sugar() {
        let program = "x = 1.0; x > 0 && x <= 3";
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();
        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .build();
        assert_eq!(module.run().unwrap(), Value::Bool(true));

        let bogus_program = "x = 1.0; x > (1, 2)";
        let bogus_block = Untyped::<F32Grammar>::parse_statements(bogus_program).unwrap();
        let bogus_module = ExecutableModule::builder(WildcardId, &bogus_block)
            .unwrap()
            .build();

        let err = bogus_module.run().unwrap_err();
        let err = err.source();
        assert_matches!(err.kind(), ErrorKind::CannotCompare);
        assert_eq!(*err.main_span().code().fragment(), "(1, 2)");
    }

    #[test]
    fn loop_basic() {
        let program = r#"
            // Finds the greatest power of 2 lesser or equal to the value.
            discrete_log2 = |x| {
                loop(0, |i| {
                    continue = 2^i <= x;
                    (continue, if(continue, i + 1, i - 1))
                })
            };

            (discrete_log2(1), discrete_log2(2),
                discrete_log2(4), discrete_log2(6.5), discrete_log2(1000))
        "#;
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();

        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("loop", Value::native_fn(Loop))
            .with_import("if", Value::native_fn(If))
            .build();

        assert_eq!(
            module.run().unwrap(),
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
        let program = r#"
            max_value = |...xs| {
                fold(xs, -Inf, |acc, x| if(x > acc, x, acc))
            };
            max_value(1, -2, 7, 2, 5) == 7 && max_value(3, -5, 9) == 9
        "#;
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();

        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("Inf", Value::Number(f32::INFINITY))
            .with_import("fold", Value::native_fn(Fold))
            .with_import("if", Value::native_fn(If))
            .build();

        assert_eq!(module.run().unwrap(), Value::Bool(true));
    }

    #[test]
    fn reverse_list_with_fold() {
        const SAMPLES: &[(&[f32], &[f32])] = &[
            (&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]),
            (&[], &[]),
            (&[1.0], &[1.0]),
        ];

        let program = r#"
            reverse = |xs| {
                fold(xs, (), |acc, x| merge((x,), acc))
            };
            xs = (-4, 3, 0, 1);
            xs.reverse() == (1, 0, 3, -4)
        "#;
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();

        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("merge", Value::native_fn(Merge))
            .with_import("fold", Value::native_fn(Fold))
            .build();

        let mut env = Environment::from_iter(module.imports());
        assert_eq!(module.run_in_env(&mut env).unwrap(), Value::Bool(true));

        let test_block = Untyped::<F32Grammar>::parse_statements("xs.reverse()").unwrap();
        let mut test_module = ExecutableModule::builder("test", &test_block)
            .unwrap()
            .with_import("reverse", env["reverse"].to_owned())
            .set_imports(|_| Value::void());

        for &(input, expected) in SAMPLES {
            let input = input.iter().copied().map(Value::Number).collect();
            let expected = expected.iter().copied().map(Value::Number).collect();
            test_module.set_import("xs", Value::Tuple(input));
            assert_eq!(test_module.run().unwrap(), Value::Tuple(expected));
        }
    }

    #[test]
    fn error_with_min_function_args() {
        let program = "5 - min(1, (2, 3))";
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();
        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("min", Value::native_fn(Compare::Min))
            .build();

        let err = module.run().unwrap_err();
        let err = err.source();
        assert_eq!(*err.main_span().code().fragment(), "min(1, (2, 3))");
        assert_matches!(
            err.kind(),
            ErrorKind::NativeCall(ref msg) if msg.contains("requires 2 number arguments")
        );
    }

    #[test]
    fn error_with_min_function_incomparable_args() {
        let program = "5 - min(1, NAN)";
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();
        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("NAN", Value::Number(f32::NAN))
            .with_import("min", Value::native_fn(Compare::Min))
            .build();

        let err = module.run().unwrap_err();
        let err = err.source();
        assert_eq!(*err.main_span().code().fragment(), "min(1, NAN)");
        assert_matches!(err.kind(), ErrorKind::CannotCompare);
    }
}
