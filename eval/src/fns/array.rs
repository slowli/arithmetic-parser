//! Functions on arrays.

use crate::{
    alloc::{vec, Vec},
    fns::{extract_array, extract_fn},
    CallContext, ErrorKind, EvalResult, NativeFn, SpannedValue, Value,
};

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
#[derive(Debug, Clone, Copy, Default)]
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
#[derive(Debug, Clone, Copy, Default)]
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
#[derive(Debug, Clone, Copy, Default)]
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
#[derive(Debug, Clone, Copy, Default)]
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
#[derive(Debug, Clone, Copy, Default)]
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
