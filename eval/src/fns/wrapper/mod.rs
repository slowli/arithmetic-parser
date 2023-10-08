//! Wrapper for eloquent `NativeFn` definitions.

use core::{fmt, marker::PhantomData};

use crate::{
    alloc::Vec, error::AuxErrorInfo, CallContext, ErrorKind, EvalResult, NativeFn, SpannedValue,
};

mod traits;

pub use self::traits::{
    ErrorOutput, FromValueError, FromValueErrorKind, FromValueErrorLocation, IntoEvalResult,
    TryFromValue,
};

/// Wraps a function enriching it with the information about its arguments.
/// This is a slightly shorter way to create wrappers compared to calling [`FnWrapper::new()`].
///
/// See [`FnWrapper`] for more details on function requirements.
pub const fn wrap<const CTX: bool, T, F>(function: F) -> FnWrapper<T, F, CTX> {
    FnWrapper::new(function)
}

/// Wrapper of a function containing information about its arguments.
///
/// Using `FnWrapper` allows to define [native functions](NativeFn) with minimum boilerplate
/// and with increased type safety. `FnWrapper`s can be constructed explicitly or indirectly
/// via [`Environment::insert_wrapped_fn()`], [`Value::wrapped_fn()`], or [`wrap()`].
///
/// Arguments of a wrapped function must implement [`TryFromValue`] trait for the applicable
/// grammar, and the output type must implement [`IntoEvalResult`]. If you need [`CallContext`] (e.g.,
/// to call functions provided as an argument), provide it as a first argument.
///
/// [`Environment::insert_wrapped_fn()`]: crate::Environment::insert_wrapped_fn()
/// [`wrap_fn`]: crate::wrap_fn
/// [`wrap_fn_with_context`]: crate::wrap_fn_with_context
/// [`Value::wrapped_fn()`]: crate::Value::wrapped_fn()
///
/// # Examples
///
/// ## Basic function
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// use arithmetic_eval::{fns, Environment, ExecutableModule, Value};
///
/// # fn main() -> anyhow::Result<()> {
/// let max = fns::wrap(|x: f32, y: f32| if x > y { x } else { y });
///
/// let program = "max(1, 3) == 3 && max(-1, -3) == -1";
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_max", &program)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("max", max);
/// assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
///
/// ## Fallible function with complex args
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns::FnWrapper, Environment, ExecutableModule, Value};
/// fn zip_arrays(xs: Vec<f32>, ys: Vec<f32>) -> Result<Vec<(f32, f32)>, String> {
///     if xs.len() == ys.len() {
///         Ok(xs.into_iter().zip(ys).map(|(x, y)| (x, y)).collect())
///     } else {
///         Err("Arrays must have the same size".to_owned())
///     }
/// }
///
/// # fn main() -> anyhow::Result<()> {
/// let program = "zip((1, 2, 3), (4, 5, 6)) == ((1, 4), (2, 5), (3, 6))";
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_zip", &program)?;
///
/// let mut env = Environment::new();
/// env.insert_wrapped_fn("zip", zip_arrays);
/// assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
///
/// FIXME: example with context
pub struct FnWrapper<T, F, const CTX: bool = false> {
    function: F,
    _arg_types: PhantomData<T>,
}

impl<T, F, const CTX: bool> fmt::Debug for FnWrapper<T, F, CTX>
where
    F: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FnWrapper")
            .field("function", &self.function)
            .field("context", &CTX)
            .finish()
    }
}

impl<T, F: Clone, const CTX: bool> Clone for FnWrapper<T, F, CTX> {
    fn clone(&self) -> Self {
        Self {
            function: self.function.clone(),
            _arg_types: PhantomData,
        }
    }
}

impl<T, F: Copy, const CTX: bool> Copy for FnWrapper<T, F, CTX> {}

// Ideally, we would want to constrain `T` and `F`, but this would make it impossible to declare
// the constructor as `const fn`; see https://github.com/rust-lang/rust/issues/57563.
impl<T, F, const CTX: bool> FnWrapper<T, F, CTX> {
    /// Creates a new wrapper.
    ///
    /// Note that the created wrapper is not guaranteed to be usable as [`NativeFn`]. For this
    /// to be the case, `function` needs to be a function or an [`Fn`] closure,
    /// and the `T` type argument needs to be a tuple with the function return type
    /// and the argument types (in this order).
    pub const fn new(function: F) -> Self {
        Self {
            function,
            _arg_types: PhantomData,
        }
    }
}

macro_rules! arity_fn {
    ($arity:tt, $with_ctx:tt $(, $ctx_name:ident : $ctx_t:ty)? => $($arg_name:ident : $t:ident),*) => {
        impl<Num, F, Ret, $($t,)*> NativeFn<Num> for FnWrapper<(Ret, $($t,)*), F, $with_ctx>
        where
            F: Fn($($ctx_t,)? $($t,)*) -> Ret,
            $($t: TryFromValue<Num>,)*
            Ret: IntoEvalResult<Num>,
        {
            #[allow(clippy::shadow_unrelated)] // makes it easier to write macro
            #[allow(unused_variables, unused_mut)] // `args_iter` is unused for 0-ary functions
            fn evaluate(
                &self,
                args: Vec<SpannedValue<Num>>,
                context: &mut CallContext<'_, Num>,
            ) -> EvalResult<Num> {
                context.check_args_count(&args, $arity)?;
                let mut args_iter = args.into_iter().enumerate();

                $(
                    let (index, $arg_name) = args_iter.next().unwrap();
                    let span = $arg_name.with_no_extra();
                    let $arg_name = $t::try_from_value($arg_name.extra).map_err(|mut err| {
                        err.set_arg_index(index);
                        context
                            .call_site_error(ErrorKind::Wrapper(err))
                            .with_location(&span, AuxErrorInfo::InvalidArg)
                    })?;
                )*

                $(let $ctx_name = &mut *context;)?
                let output = (self.function)($($ctx_name,)? $($arg_name,)*);
                output.into_eval_result().map_err(|err| err.into_spanned(context))
            }
        }
    };
}

arity_fn!(0, false =>);
arity_fn!(0, true, ctx: &mut CallContext<'_, Num> =>);
arity_fn!(1, false => x0: T);
arity_fn!(1, true, ctx: &mut CallContext<'_, Num> => x0: T);
arity_fn!(2, false => x0: T, x1: U);
arity_fn!(2, true, ctx: &mut CallContext<'_, Num> => x0: T, x1: U);
arity_fn!(3, false => x0: T, x1: U, x2: V);
arity_fn!(3, true, ctx: &mut CallContext<'_, Num> => x0: T, x1: U, x2: V);
arity_fn!(4, false => x0: T, x1: U, x2: V, x3: W);
arity_fn!(4, true, ctx: &mut CallContext<'_, Num> => x0: T, x1: U, x2: V, x3: W);
arity_fn!(5, false => x0: T, x1: U, x2: V, x3: W, x4: X);
arity_fn!(5, true, ctx: &mut CallContext<'_, Num> => x0: T, x1: U, x2: V, x3: W, x4: X);
arity_fn!(6, false => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y);
arity_fn!(6, true, ctx: &mut CallContext<'_, Num> => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y);
arity_fn!(7, false => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y, x6: Z);
arity_fn!(7, true, ctx: &mut CallContext<'_, Num> => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y, x6: Z);
arity_fn!(8, false => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y, x6: Z, x7: A);
arity_fn!(8, true, ctx: &mut CallContext<'_, Num> => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y, x6: Z, x7: A);
arity_fn!(9, false => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y, x6: Z, x7: A, x8: B);
arity_fn!(9, true, ctx: &mut CallContext<'_, Num> => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y, x6: Z, x7: A, x8: B);
arity_fn!(10, false => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y, x6: Z, x7: A, x8: B, x9: C);
arity_fn!(10, true, ctx: &mut CallContext<'_, Num> => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y, x6: Z, x7: A, x8: B, x9: C);

/// Unary function wrapper.
pub type Unary<T> = FnWrapper<(T, T), fn(T) -> T>;

/// Binary function wrapper.
pub type Binary<T> = FnWrapper<(T, T, T), fn(T, T) -> T>;

/// Ternary function wrapper.
pub type Ternary<T> = FnWrapper<(T, T, T, T), fn(T, T, T) -> T>;

/// Quaternary function wrapper.
pub type Quaternary<T> = FnWrapper<(T, T, T, T, T), fn(T, T, T, T) -> T>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        alloc::{format, ToOwned},
        env::{Environment, Prelude},
        exec::{ExecutableModule, WildcardId},
        Function, Object, Tuple, Value,
    };

    use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
    use assert_matches::assert_matches;

    #[test]
    fn functions_with_primitive_args() -> anyhow::Result<()> {
        let unary_fn = Unary::new(|x: f32| x + 3.0);
        let binary_fn = Binary::new(f32::min);
        let ternary_fn = Ternary::new(|x: f32, y, z| if x > 0.0 { y } else { z });

        let program = r#"
            unary_fn(2) == 5 && binary_fn(1, -3) == -3 &&
                ternary_fn(1, 2, 3) == 2 && ternary_fn(-1, 2, 3) == 3
        "#;
        let block = Untyped::<F32Grammar>::parse_statements(program)?;
        let module = ExecutableModule::new(WildcardId, &block)?;

        let mut env = Environment::new();
        env.insert_native_fn("unary_fn", unary_fn)
            .insert_native_fn("binary_fn", binary_fn)
            .insert_native_fn("ternary_fn", ternary_fn);

        assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
        Ok(())
    }

    fn array_min_max(values: Vec<f32>) -> (f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        for value in values {
            if value < min {
                min = value;
            }
            if value > max {
                max = value;
            }
        }
        (min, max)
    }

    fn overly_convoluted_fn(xs: Vec<(f32, f32)>, ys: (Vec<f32>, f32)) -> f32 {
        xs.into_iter().map(|(a, b)| a + b).sum::<f32>() + ys.0.into_iter().sum::<f32>() + ys.1
    }

    #[test]
    fn functions_with_composite_args() -> anyhow::Result<()> {
        let program = r#"
            array_min_max((1, 5, -3, 2, 1)) == (-3, 5) &&
                total_sum(((1, 2), (3, 4)), ((5, 6, 7), 8)) == 36
        "#;
        let block = Untyped::<F32Grammar>::parse_statements(program)?;
        let module = ExecutableModule::new(WildcardId, &block)?;

        let mut env = Environment::new();
        env.insert_wrapped_fn("array_min_max", array_min_max)
            .insert_wrapped_fn("total_sum", overly_convoluted_fn);

        assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
        Ok(())
    }

    fn sum_arrays(xs: Vec<f32>, ys: Vec<f32>) -> Result<Vec<f32>, String> {
        if xs.len() == ys.len() {
            Ok(xs.into_iter().zip(ys).map(|(x, y)| x + y).collect())
        } else {
            Err("Summed arrays must have the same size".to_owned())
        }
    }

    #[test]
    fn fallible_function() -> anyhow::Result<()> {
        let program = "sum_arrays((1, 2, 3), (4, 5, 6)) == (5, 7, 9)";
        let block = Untyped::<F32Grammar>::parse_statements(program)?;
        let module = ExecutableModule::new(WildcardId, &block)?;

        let mut env = Environment::new();
        env.insert_wrapped_fn("sum_arrays", sum_arrays);
        assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
        Ok(())
    }

    #[test]
    fn fallible_function_with_bogus_program() -> anyhow::Result<()> {
        let program = "sum_arrays((1, 2, 3), (4, 5))";
        let block = Untyped::<F32Grammar>::parse_statements(program)?;
        let module = ExecutableModule::new(WildcardId, &block)?;

        let mut env = Environment::new();
        env.insert_wrapped_fn("sum_arrays", sum_arrays);

        let err = module.with_env(&env)?.run().unwrap_err();
        assert!(err
            .source()
            .kind()
            .to_short_string()
            .contains("Summed arrays must have the same size"));
        Ok(())
    }

    #[test]
    fn function_with_bool_return_value() -> anyhow::Result<()> {
        let contains = wrap(|(a, b): (f32, f32), x: f32| (a..=b).contains(&x));

        let program = "contains((-1, 2), 0) && !contains((1, 3), 0)";
        let block = Untyped::<F32Grammar>::parse_statements(program)?;
        let module = ExecutableModule::new(WildcardId, &block)?;

        let mut env = Environment::new();
        env.insert_native_fn("contains", contains);
        assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
        Ok(())
    }

    #[test]
    fn function_with_void_return_value() -> anyhow::Result<()> {
        let program = "assert_eq(3, 1 + 2)";
        let block = Untyped::<F32Grammar>::parse_statements(program)?;
        let module = ExecutableModule::new(WildcardId, &block)?;

        let mut env = Environment::new();
        env.insert_wrapped_fn("assert_eq", |expected: f32, actual: f32| {
            if (expected - actual).abs() < f32::EPSILON {
                Ok(())
            } else {
                Err(format!(
                    "Assertion failed: expected {expected}, got {actual}"
                ))
            }
        });

        assert!(module.with_env(&env)?.run()?.is_void());

        let bogus_program = "assert_eq(3, 1 - 2)";
        let bogus_block = Untyped::<F32Grammar>::parse_statements(bogus_program)?;
        let err = ExecutableModule::new(WildcardId, &bogus_block)?
            .with_env(&env)?
            .run()
            .unwrap_err();

        assert_matches!(
            err.source().kind(),
            ErrorKind::NativeCall(ref msg) if msg.contains("Assertion failed")
        );
        Ok(())
    }

    #[test]
    fn function_with_bool_argument() -> anyhow::Result<()> {
        let program = "flip_sign(-1, true) == 1 && flip_sign(-1, false) == -1";
        let block = Untyped::<F32Grammar>::parse_statements(program)?;
        let module = ExecutableModule::new(WildcardId, &block)?;

        let mut env = Environment::new();
        env.extend(Prelude::iter());
        env.insert_wrapped_fn(
            "flip_sign",
            |val: f32, flag: bool| if flag { -val } else { val },
        );

        assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
        Ok(())
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // fine for this test
    fn function_with_object_and_tuple() -> anyhow::Result<()> {
        fn test_function(tuple: Tuple<f32>) -> Object<f32> {
            let mut obj = Object::default();
            obj.insert("len", Value::Prim(tuple.len() as f32));
            obj.insert("tuple", tuple.into());
            obj
        }

        let program = r#"
            { len, tuple } = test((1, 1, 1));
            len == 3 && tuple == (1, 1, 1)
        "#;
        let block = Untyped::<F32Grammar>::parse_statements(program)?;
        let module = ExecutableModule::new(WildcardId, &block)?;

        let test_function = Value::native_fn(wrap(test_function));
        let mut env = Environment::new();
        env.insert("test", test_function).extend(Prelude::iter());

        assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
        Ok(())
    }

    #[test]
    fn error_reporting_with_destructuring() -> anyhow::Result<()> {
        let program = "destructure(((true, 1), (2, 3)))";
        let block = Untyped::<F32Grammar>::parse_statements(program)?;
        let module = ExecutableModule::new(WildcardId, &block)?;

        let mut env = Environment::new();
        env.extend(Prelude::iter());
        env.insert_wrapped_fn("destructure", |values: Vec<(bool, f32)>| {
            values
                .into_iter()
                .map(|(flag, x)| if flag { x } else { 0.0 })
                .sum::<f32>()
        });

        let err = module.with_env(&env)?.run().unwrap_err();
        let err_message = err.source().kind().to_short_string();
        assert!(err_message.contains("Cannot convert primitive value to bool"));
        assert!(err_message.contains("location: arg0[1].0"));
        Ok(())
    }

    #[test]
    fn function_with_context() -> anyhow::Result<()> {
        #[allow(clippy::needless_pass_by_value)] // required for wrapping to work
        fn call(
            ctx: &mut CallContext<'_, f32>,
            func: Function<f32>,
            value: f32,
        ) -> EvalResult<f32> {
            let args = vec![ctx.apply_call_location(Value::Prim(value))];
            func.evaluate(args, ctx)
        }

        let program = "(|x| { x + 1 }).call(1)";
        let block = Untyped::<F32Grammar>::parse_statements(program)?;
        let module = ExecutableModule::new(WildcardId, &block)?;

        let mut env = Environment::new();
        env.insert_wrapped_fn("call", call);
        assert_eq!(module.with_env(&env)?.run()?, Value::Prim(2.0));
        Ok(())
    }
}
