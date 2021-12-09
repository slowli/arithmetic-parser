//! Functions on arrays.

use core::cmp::Ordering;

use num_traits::{FromPrimitive, One, Zero};

use crate::{
    alloc::{vec, Vec},
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
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"array(3, |i| 2 * i + 1) == (1, 3, 5)"#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("array", fns::Array)
///     .compile_module("test_array", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Array;

impl<T> NativeFn<T> for Array
where
    T: Clone + Zero + One,
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
            if matches!(cmp, Some(Ordering::Less) | Some(Ordering::Equal)) {
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
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"().len() == 0 && (1, 2, 3).len() == 3"#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("len", fns::Len)
///     .compile_module("test_len", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
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
        Ok(Value::Tuple(first.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        arith::{OrdArithmetic, StdArithmetic, WrappingArithmetic},
        Environment, VariableMap,
    };

    use arithmetic_parser::grammars::{NumGrammar, NumLiteral, Parse, Untyped};
    use assert_matches::assert_matches;

    fn test_len_function<T: NumLiteral>(arithmetic: &dyn OrdArithmetic<T>)
    where
        Len: NativeFn<T>,
    {
        let code = r#"
            (1, 2, 3).len() == 3 && ().len() == 0 &&
            #{}.len() == 0 && #{ x: 1 }.len() == 1 && #{ x: 1, y: 2 }.len() == 2
        "#;
        let block = Untyped::<NumGrammar<T>>::parse_statements(code).unwrap();
        let mut env = Environment::new();
        let module = env
            .insert("len", Value::native_fn(Len))
            .compile_module("len", &block)
            .unwrap();

        let output = module.with_arithmetic(arithmetic).run().unwrap();
        assert_matches!(output, Value::Bool(true));
    }

    #[test]
    fn len_function_in_floating_point_arithmetic() {
        test_len_function::<f32>(&StdArithmetic);
        test_len_function::<f64>(&StdArithmetic);
    }

    #[test]
    fn len_function_in_int_arithmetic() {
        test_len_function::<u8>(&WrappingArithmetic);
        test_len_function::<i8>(&WrappingArithmetic);
        test_len_function::<u64>(&WrappingArithmetic);
        test_len_function::<i64>(&WrappingArithmetic);
    }

    #[test]
    fn len_function_with_number_overflow() {
        let code = "xs.len()";
        let block = Untyped::<NumGrammar<i8>>::parse_statements(code).unwrap();
        let mut env = Environment::new();
        let module = env
            .insert("xs", Value::from(vec![Value::Bool(true); 128]))
            .insert("len", Value::native_fn(Len))
            .compile_module("len", &block)
            .unwrap();

        let err = module
            .with_arithmetic(&WrappingArithmetic)
            .run()
            .unwrap_err();
        assert_matches!(
            err.source().kind(),
            ErrorKind::NativeCall(msg) if msg.contains("length to number")
        );
    }

    #[test]
    fn array_function_in_floating_point_arithmetic() {
        let code = r#"
            array(0, |_| 1) == () && array(-1, |_| 1) == () &&
            array(0.1, |_| 1) == () && array(0.999, |_| 1) == () &&
            array(1, |_| 1) == (1,) && array(1.5, |_| 1) == (1,) &&
            array(2, |_| 1) == (1, 1) && array(3, |i| i) == (0, 1, 2)
        "#;
        let block = Untyped::<NumGrammar<f32>>::parse_statements(code).unwrap();
        let mut env = Environment::new();
        let module = env
            .insert("array", Value::native_fn(Array))
            .compile_module("array", &block)
            .unwrap();

        let output = module.with_arithmetic(&StdArithmetic).run().unwrap();
        assert_matches!(output, Value::Bool(true));
    }

    #[test]
    fn array_function_in_unsigned_int_arithmetic() {
        let code = r#"
            array(0, |_| 1) == () && array(1, |_| 1) == (1,) && array(3, |i| i) == (0, 1, 2)
        "#;
        let block = Untyped::<NumGrammar<u32>>::parse_statements(code).unwrap();
        let mut env = Environment::new();
        let module = env
            .insert("array", Value::native_fn(Array))
            .compile_module("array", &block)
            .unwrap();

        let output = module.with_arithmetic(&WrappingArithmetic).run().unwrap();
        assert_matches!(output, Value::Bool(true));
    }
}
