//! Functions on arrays.

use core::cmp::Ordering;

use num_traits::{FromPrimitive, One, Zero};

use crate::{
    alloc::{format, vec, Vec},
    error::AuxErrorInfo,
    fns::{extract_array, extract_fn, extract_primitive},
    CallContext, ErrorKind, EvalResult, NativeFn, SpannedValue, Tuple, Value,
};

/// Function generating an array by mapping its indexes.
///
/// # Type
///
/// (using [`arithmetic-typing`](https://docs.rs/arithmetic-typing/) notation)
///
/// ```text
/// (Num, (Num) -> 'T) -> ['T]
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ExecutableModule, Value};
/// # fn main() -> anyhow::Result<()> {
/// let program = "array(3, |i| 2 * i + 1) == (1, 3, 5)";
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_array", &program)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("array", fns::Array);
/// assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Array;

impl<T> NativeFn<T> for Array
where
    T: 'static + Clone + Zero + One,
{
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 2)?;
        let generation_fn = extract_fn(
            ctx,
            args.pop().unwrap(),
            "`array` requires second arg to be a generation function",
        )?;
        let len = extract_primitive(
            ctx,
            args.pop().unwrap(),
            "`array` requires first arg to be a number",
        )?;

        let mut index = T::zero();
        let mut array = vec![];
        loop {
            let next_index = ctx
                .arithmetic()
                .add(index.clone(), T::one())
                .map_err(|err| ctx.call_site_error(ErrorKind::Arithmetic(err)))?;

            let cmp = ctx.arithmetic().partial_cmp(&next_index, &len);
            if matches!(cmp, Some(Ordering::Less | Ordering::Equal)) {
                let spanned = ctx.apply_call_span(Value::Prim(index));
                array.push(generation_fn.evaluate(vec![spanned], ctx)?);
                index = next_index;
            } else {
                break;
            }
        }
        Ok(Value::Tuple(array.into()))
    }
}

/// Function returning array / object length.
///
/// # Type
///
/// (using [`arithmetic-typing`](https://docs.rs/arithmetic-typing/) notation)
///
/// ```text
/// ([T]) -> Num
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ExecutableModule, Value};
/// # fn main() -> anyhow::Result<()> {
/// let program = "len(()) == 0 && len((1, 2, 3)) == 3";
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("tes_len", &program)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("len", fns::Len);
/// assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Len;

impl<T: FromPrimitive> NativeFn<T> for Len {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 1)?;
        let arg = args.pop().unwrap();

        let len = match arg.extra {
            Value::Tuple(array) => array.len(),
            Value::Object(object) => object.len(),
            _ => {
                let err = ErrorKind::native("`len` requires object or tuple arg");
                return Err(ctx
                    .call_site_error(err)
                    .with_span(&arg, AuxErrorInfo::InvalidArg));
            }
        };
        let len = T::from_usize(len).ok_or_else(|| {
            let err = ErrorKind::native("Cannot convert length to number");
            ctx.call_site_error(err)
        })?;
        Ok(Value::Prim(len))
    }
}

/// Map function that evaluates the provided function on each item of the tuple.
///
/// # Type
///
/// (using [`arithmetic-typing`](https://docs.rs/arithmetic-typing/) notation)
///
/// ```text
/// (['T; N], ('T) -> 'U) -> ['U; N]
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ExecutableModule, Value};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     xs = (1, -2, 3, -0.3);
///     map(xs, |x| if(x > 0, x, 0)) == (1, 0, 3, 0)
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_map", &program)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("if", fns::If).insert_native_fn("map", fns::Map);
/// assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Map;

impl<T: 'static + Clone> NativeFn<T> for Map {
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

        let mapped: Result<Tuple<_>, _> = array
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
/// (using [`arithmetic-typing`](https://docs.rs/arithmetic-typing/) notation)
///
/// ```text
/// (['T; N], ('T) -> Bool) -> ['T]
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ExecutableModule, Value};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     xs = (1, -2, 3, -7, -0.3);
///     filter(xs, |x| x > -1) == (1, 3, -0.3)
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_filter", &program)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("filter", fns::Filter);
/// assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Filter;

impl<T: 'static + Clone> NativeFn<T> for Filter {
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
        Ok(Value::Tuple(filtered.into()))
    }
}

/// Reduce (aka fold) function that reduces the provided tuple to a single value.
///
/// # Type
///
/// (using [`arithmetic-typing`](https://docs.rs/arithmetic-typing/) notation)
///
/// ```text
/// (['T], 'Acc, ('Acc, 'T) -> 'Acc) -> 'Acc
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ExecutableModule, Value};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     xs = (1, -2, 3, -7);
///     fold(xs, 1, |acc, x| acc * x) == 42
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_fold", &program)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("fold", fns::Fold);
/// assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Fold;

impl<T: 'static + Clone> NativeFn<T> for Fold {
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
/// (using [`arithmetic-typing`](https://docs.rs/arithmetic-typing/) notation)
///
/// ```text
/// (['T; N], 'T) -> ['T; N + 1]
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ExecutableModule, Value};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     repeat = |x, times| {
///         (_, acc) = while(
///             (0, ()),
///             |(i, _)| i < times,
///             |(i, acc)| (i + 1, push(acc, x)),
///         );
///         acc
///     };
///     repeat(-2, 3) == (-2, -2, -2) &&
///         repeat((7,), 4) == ((7,), (7,), (7,), (7,))
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_push", &program)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("while", fns::While)
///     .insert_native_fn("push", fns::Push);
/// assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
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
            "`push` requires first arg to be a tuple",
        )?;

        array.push(elem);
        Ok(Value::Tuple(array.into()))
    }
}

/// Function that merges two tuples.
///
/// # Type
///
/// (using [`arithmetic-typing`](https://docs.rs/arithmetic-typing/) notation)
///
/// ```text
/// (['T], ['T]) -> ['T]
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ExecutableModule, Value};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     // Merges all arguments (which should be tuples) into a single tuple.
///     super_merge = |...xs| fold(xs, (), merge);
///     super_merge((1, 2), (3,), (), (4, 5, 6)) == (1, 2, 3, 4, 5, 6)
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_merge", &program)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("fold", fns::Fold)
///     .insert_native_fn("merge", fns::Merge);
/// assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
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
        Ok(Value::Tuple(first.into()))
    }
}

/// Function that checks whether any of array items satisfy the provided predicate.
///
/// # Type
///
/// (using [`arithmetic-typing`](https://docs.rs/arithmetic-typing/) notation)
///
/// ```text
/// (['T], ('T) -> Bool) -> Bool
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ExecutableModule, Value};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     assert(any((1, 3, -1), |x| x < 0));
///     assert(!any((1, 2, 3), |x| x < 0));
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_any", &program)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("any", fns::Any)
///     .insert_native_fn("assert", fns::Assert);
/// module.with_env(&env)?.run()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Any;

impl<T: Clone + 'static> NativeFn<T> for Any {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 2)?;
        let predicate = extract_fn(
            ctx,
            args.pop().unwrap(),
            "`any` requires second arg to be a predicate function",
        )?;
        let array = extract_array(
            ctx,
            args.pop().unwrap(),
            "`any` requires first arg to be a tuple",
        )?;

        for value in array {
            let spanned = ctx.apply_call_span(value);
            let result = predicate.evaluate(vec![spanned], ctx)?;
            match result {
                Value::Bool(false) => { /* continue */ }
                Value::Bool(true) => return Ok(Value::Bool(true)),
                _ => {
                    let err = ErrorKind::native(format!(
                        "Incorrect return type of a predicate: expected Boolean, got {}",
                        result.value_type()
                    ));
                    ctx.call_site_error(err);
                }
            }
        }
        Ok(Value::Bool(false))
    }
}

/// Function that checks whether all of array items satisfy the provided predicate.
///
/// # Type
///
/// (using [`arithmetic-typing`](https://docs.rs/arithmetic-typing/) notation)
///
/// ```text
/// (['T], ('T) -> Bool) -> Bool
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ExecutableModule, Value};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     assert(all((1, 2, 3, 5), |x| x > 0));
///     assert(!all((1, -2, 3), |x| x > 0));
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_all", &program)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("all", fns::All)
///     .insert_native_fn("assert", fns::Assert);
/// module.with_env(&env)?.run()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct All;

impl<T: Clone + 'static> NativeFn<T> for All {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 2)?;
        let predicate = extract_fn(
            ctx,
            args.pop().unwrap(),
            "`all` requires second arg to be a predicate function",
        )?;
        let array = extract_array(
            ctx,
            args.pop().unwrap(),
            "`all` requires first arg to be a tuple",
        )?;

        for value in array {
            let spanned = ctx.apply_call_span(value);
            let result = predicate.evaluate(vec![spanned], ctx)?;
            match result {
                Value::Bool(false) => return Ok(Value::Bool(false)),
                Value::Bool(true) => { /* continue */ }
                _ => {
                    let err = ErrorKind::native(format!(
                        "Incorrect return type of a predicate: expected Boolean, got {}",
                        result.value_type()
                    ));
                    ctx.call_site_error(err);
                }
            }
        }
        Ok(Value::Bool(true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        arith::{OrdArithmetic, StdArithmetic, WrappingArithmetic},
        Environment, ExecutableModule,
    };

    use arithmetic_parser::grammars::{F32Grammar, NumGrammar, NumLiteral, Parse, Untyped};
    use assert_matches::assert_matches;

    fn test_len_function<T: NumLiteral, A>(arithmetic: A)
    where
        Len: NativeFn<T>,
        A: OrdArithmetic<T> + 'static,
    {
        let code = r#"
            len((1, 2, 3)) == 3 && len(()) == 0 &&
            len(#{}) == 0 && len(#{ x: 1 }) == 1 && len(#{ x: 1, y: 2 }) == 2
        "#;
        let block = Untyped::<NumGrammar<T>>::parse_statements(code).unwrap();
        let module = ExecutableModule::new("len", &block).unwrap();
        let mut env = Environment::with_arithmetic(arithmetic);
        env.insert_native_fn("len", Len);

        let output = module.with_env(&env).unwrap().run().unwrap();
        assert_matches!(output, Value::Bool(true));
    }

    #[test]
    fn len_function_in_floating_point_arithmetic() {
        test_len_function::<f32, _>(StdArithmetic);
        test_len_function::<f64, _>(StdArithmetic);
    }

    #[test]
    fn len_function_in_int_arithmetic() {
        test_len_function::<u8, _>(WrappingArithmetic);
        test_len_function::<i8, _>(WrappingArithmetic);
        test_len_function::<u64, _>(WrappingArithmetic);
        test_len_function::<i64, _>(WrappingArithmetic);
    }

    #[test]
    fn len_function_with_number_overflow() -> anyhow::Result<()> {
        let code = "len(xs)";
        let block = Untyped::<NumGrammar<i8>>::parse_statements(code)?;
        let module = ExecutableModule::new("len", &block)?;

        let mut env = Environment::with_arithmetic(WrappingArithmetic);
        env.insert("xs", Value::from(vec![Value::Bool(true); 128]))
            .insert_native_fn("len", Len);

        let err = module.with_env(&env)?.run().unwrap_err();
        assert_matches!(
            err.source().kind(),
            ErrorKind::NativeCall(msg) if msg.contains("length to number")
        );
        Ok(())
    }

    #[test]
    fn array_function_in_floating_point_arithmetic() -> anyhow::Result<()> {
        let code = r#"
            array(0, |_| 1) == () && array(-1, |_| 1) == () &&
            array(0.1, |_| 1) == () && array(0.999, |_| 1) == () &&
            array(1, |_| 1) == (1,) && array(1.5, |_| 1) == (1,) &&
            array(2, |_| 1) == (1, 1) && array(3, |i| i) == (0, 1, 2)
        "#;
        let block = Untyped::<NumGrammar<f32>>::parse_statements(code)?;
        let module = ExecutableModule::new("array", &block)?;

        let mut env = Environment::new();
        env.insert_native_fn("array", Array);

        let output = module.with_env(&env)?.run()?;
        assert_matches!(output, Value::Bool(true));
        Ok(())
    }

    #[test]
    fn array_function_in_unsigned_int_arithmetic() -> anyhow::Result<()> {
        let code = r#"
            array(0, |_| 1) == () && array(1, |_| 1) == (1,) && array(3, |i| i) == (0, 1, 2)
        "#;
        let block = Untyped::<NumGrammar<u32>>::parse_statements(code)?;
        let module = ExecutableModule::new("array", &block)?;

        let mut env = Environment::with_arithmetic(WrappingArithmetic);
        env.insert_native_fn("array", Array);

        let output = module.with_env(&env)?.run()?;
        assert_matches!(output, Value::Bool(true));
        Ok(())
    }

    #[test]
    fn all_and_any_are_short_circuit() -> anyhow::Result<()> {
        let code = r#"
            !all((1, 5 == 5), |x| x < 0) && any((-1, 1, 5 == 4), |x| x > 0)
        "#;
        let block = Untyped::<F32Grammar>::parse_statements(code)?;
        let module = ExecutableModule::new("array", &block)?;

        let mut env = Environment::new();
        env.insert_native_fn("all", All)
            .insert_native_fn("any", Any);

        let output = module.with_env(&env)?.run()?;
        assert_matches!(output, Value::Bool(true));
        Ok(())
    }
}
