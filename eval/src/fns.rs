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
//! [`NativeFn`]: ../trait.NativeFn.html
//! [`FnWrapper`]: struct.FnWrapper.html
//! [`wrap`]: fn.wrap.html
//! [`Number`]: ../trait.Number.html
//! [`wrap_fn`]: ../macro.wrap_fn.html
//! [`wrap_fn_with_context`]: ../macro.wrap_fn_with_context.html
//! [GAT]: https://github.com/rust-lang/rust/issues/44265

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

fn extract_number<'a, T>(
    ctx: &CallContext<'_, 'a>,
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

fn extract_array<'a, T>(
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
            .with_span(&value, AuxErrorInfo::InvalidArg))
    }
}

fn extract_fn<'a, T>(
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
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt};
/// # use arithmetic_eval::{fns, Environment, ErrorKind, VariableMap};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     assert(1 + 2 == 3); // this assertion is fine
///     assert(3^2 == 10); // this one will fail
/// "#;
/// let program = F32Grammar::parse_statements(program)?;
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
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = "x = 3; if(x == 2, -1, x + 1)";
/// let program = F32Grammar::parse_statements(program)?;
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
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = "x = 3; if(x == 2, || -1, || x + 1)()";
/// let program = F32Grammar::parse_statements(program)?;
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

impl<T: Number> NativeFn<T> for If {
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
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt};
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
/// let program = F32Grammar::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("cmp", fns::Compare)
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

impl<T: Number> NativeFn<T> for Loop {
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
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt};
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
/// let program = F32Grammar::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("cmp", fns::Compare)
///     .insert_native_fn("while", fns::While)
///     .compile_module("test_while", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct While;

impl<T: Number> NativeFn<T> for While {
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
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     xs = (1, -2, 3, -0.3);
///     map(xs, |x| if(x > 0, x, 0)) == (1, 0, 3, 0)
/// "#;
/// let program = F32Grammar::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("cmp", fns::Compare)
///     .insert_native_fn("if", fns::If)
///     .insert_native_fn("map", fns::Map)
///     .compile_module("test_map", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Map;

impl<T: Number> NativeFn<T> for Map {
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
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     xs = (1, -2, 3, -7, -0.3);
///     filter(xs, |x| x > -1) == (1, 3, -0.3)
/// "#;
/// let program = F32Grammar::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("cmp", fns::Compare)
///     .insert_native_fn("filter", fns::Filter)
///     .compile_module("test_filter", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Filter;

impl<T: Number> NativeFn<T> for Filter {
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
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     xs = (1, -2, 3, -7);
///     fold(xs, 1, |acc, x| acc * x) == 42
/// "#;
/// let program = F32Grammar::parse_statements(program)?;
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

impl<T: Number> NativeFn<T> for Fold {
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
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt};
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
/// let program = F32Grammar::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("cmp", fns::Compare)
///     .insert_native_fn("while", fns::While)
///     .insert_native_fn("push", fns::Push)
///     .compile_module("test_push", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Push;

impl<T: Number> NativeFn<T> for Push {
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
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     // Merges all arguments (which should be tuples) into a single tuple.
///     super_merge = |...xs| fold(xs, (), merge);
///     super_merge((1, 2), (3,), (), (4, 5, 6)) == (1, 2, 3, 4, 5, 6)
/// "#;
/// let program = F32Grammar::parse_statements(program)?;
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

impl<T: Number> NativeFn<T> for Merge {
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

impl<T: Number + PartialOrd> NativeFn<T> for Compare {
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
            -<T as One>::one()
        } else if x > y {
            <T as One>::one()
        } else {
            <T as Zero>::zero()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Environment, ExecutableModule, WildcardId};

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
            .with_import("cmp", Value::native_fn(Compare))
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
            .with_import("cmp", Value::native_fn(Compare))
            .build();
        assert_eq!(module.run().unwrap(), Value::Number(-1.5));
    }

    #[test]
    fn cmp_sugar() {
        let compare_fn = Value::native_fn(Compare);

        let program = "x = 1.0; x > 0 && x <= 3";
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();
        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("cmp", compare_fn.clone())
            .build();
        assert_eq!(module.run().unwrap(), Value::Bool(true));

        let bogus_program = "x = 1.0; x > (1, 2)";
        let bogus_block = Untyped::<F32Grammar>::parse_statements(bogus_program).unwrap();
        let bogus_module = ExecutableModule::builder(WildcardId, &bogus_block)
            .unwrap()
            .with_import("cmp", compare_fn)
            .build();

        let err = bogus_module.run().unwrap_err();
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
            .with_import("cmp", Value::native_fn(Compare))
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
            .with_import("cmp", Value::native_fn(Compare))
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
}
