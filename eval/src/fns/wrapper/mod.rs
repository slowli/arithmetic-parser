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
pub const fn wrap<T, F>(function: F) -> FnWrapper<T, F> {
    FnWrapper::new(function)
}

/// Wrapper of a function containing information about its arguments.
///
/// Using `FnWrapper` allows to define [native functions](NativeFn) with minimum boilerplate
/// and with increased type safety. `FnWrapper`s can be constructed explcitly or indirectly
/// via [`Environment::insert_wrapped_fn()`], [`Value::wrapped_fn()`], or [`wrap()`].
///
/// Arguments of a wrapped function must implement [`TryFromValue`] trait for the applicable
/// grammar, and the output type must implement [`IntoEvalResult`]. If arguments and/or output
/// have non-`'static` lifetime, use the [`wrap_fn`] macro. If you need [`CallContext`] (e.g.,
/// to call functions provided as an argument), use the [`wrap_fn_with_context`] macro.
///
/// [`Environment::insert_wrapped_fn()`]: crate::Environment::insert_wrapped_fn()
/// [`wrap_fn`]: crate::wrap_fn
/// [`wrap_fn_with_context`]: crate::wrap_fn_with_context
///
/// # Examples
///
/// ## Basic function
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// use arithmetic_eval::{fns, Environment, Value, VariableMap};
///
/// # fn main() -> anyhow::Result<()> {
/// let max = fns::wrap(|x: f32, y: f32| if x > y { x } else { y });
///
/// let program = "max(1, 3) == 3 && max(-1, -3) == -1";
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = Environment::new()
///     .insert_native_fn("max", max)
///     .compile_module("test_max", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
///
/// ## Fallible function with complex args
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns::FnWrapper, Environment, Value, VariableMap};
/// fn zip_arrays(xs: Vec<f32>, ys: Vec<f32>) -> Result<Vec<(f32, f32)>, String> {
///     if xs.len() == ys.len() {
///         Ok(xs.into_iter().zip(ys).map(|(x, y)| (x, y)).collect())
///     } else {
///         Err("Arrays must have the same size".to_owned())
///     }
/// }
///
/// # fn main() -> anyhow::Result<()> {
/// let program = "(1, 2, 3).zip((4, 5, 6)) == ((1, 4), (2, 5), (3, 6))";
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_wrapped_fn("zip", zip_arrays)
///     .compile_module("test_zip", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
pub struct FnWrapper<T, F> {
    function: F,
    _arg_types: PhantomData<T>,
}

impl<T, F> fmt::Debug for FnWrapper<T, F>
where
    F: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FnWrapper")
            .field("function", &self.function)
            .finish()
    }
}

impl<T, F: Clone> Clone for FnWrapper<T, F> {
    fn clone(&self) -> Self {
        Self {
            function: self.function.clone(),
            _arg_types: PhantomData,
        }
    }
}

impl<T, F: Copy> Copy for FnWrapper<T, F> {}

// Ideally, we would want to constrain `T` and `F`, but this would make it impossible to declare
// the constructor as `const fn`; see https://github.com/rust-lang/rust/issues/57563.
impl<T, F> FnWrapper<T, F> {
    /// Creates a new wrapper.
    ///
    /// Note that the created wrapper is not guaranteed to be usable as [`NativeFn`]. For this
    /// to be the case, `function` needs to be a function or an [`Fn`] closure,
    /// and the `T` type argument needs to be a tuple with the function return type
    /// and the argument types (in this order).
    ///
    /// [`NativeFn`]: crate::NativeFn
    pub const fn new(function: F) -> Self {
        Self {
            function,
            _arg_types: PhantomData,
        }
    }
}

macro_rules! arity_fn {
    ($arity:tt => $($arg_name:ident : $t:ident),*) => {
        impl<Num, F, Ret, $($t,)*> NativeFn<Num> for FnWrapper<(Ret, $($t,)*), F>
        where
            F: Fn($($t,)*) -> Ret,
            $($t: for<'val> TryFromValue<'val, Num>,)*
            Ret: for<'val> IntoEvalResult<'val, Num>,
        {
            #[allow(clippy::shadow_unrelated)] // makes it easier to write macro
            #[allow(unused_variables, unused_mut)] // `args_iter` is unused for 0-ary functions
            fn evaluate<'a>(
                &self,
                args: Vec<SpannedValue<'a, Num>>,
                context: &mut CallContext<'_, 'a, Num>,
            ) -> EvalResult<'a, Num> {
                context.check_args_count(&args, $arity)?;
                let mut args_iter = args.into_iter().enumerate();

                $(
                    let (index, $arg_name) = args_iter.next().unwrap();
                    let span = $arg_name.with_no_extra();
                    let $arg_name = $t::try_from_value($arg_name.extra).map_err(|mut err| {
                        err.set_arg_index(index);
                        context
                            .call_site_error(ErrorKind::Wrapper(err))
                            .with_span(&span, AuxErrorInfo::InvalidArg)
                    })?;
                )*

                let output = (self.function)($($arg_name,)*);
                output.into_eval_result().map_err(|err| err.into_spanned(context))
            }
        }
    };
}

arity_fn!(0 =>);
arity_fn!(1 => x0: T);
arity_fn!(2 => x0: T, x1: U);
arity_fn!(3 => x0: T, x1: U, x2: V);
arity_fn!(4 => x0: T, x1: U, x2: V, x3: W);
arity_fn!(5 => x0: T, x1: U, x2: V, x3: W, x4: X);
arity_fn!(6 => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y);
arity_fn!(7 => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y, x6: Z);
arity_fn!(8 => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y, x6: Z, x7: A);
arity_fn!(9 => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y, x6: Z, x7: A, x8: B);
arity_fn!(10 => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y, x6: Z, x7: A, x8: B, x9: C);

/// Unary function wrapper.
pub type Unary<T> = FnWrapper<(T, T), fn(T) -> T>;

/// Binary function wrapper.
pub type Binary<T> = FnWrapper<(T, T, T), fn(T, T) -> T>;

/// Ternary function wrapper.
pub type Ternary<T> = FnWrapper<(T, T, T, T), fn(T, T, T) -> T>;

/// Quaternary function wrapper.
pub type Quaternary<T> = FnWrapper<(T, T, T, T, T), fn(T, T, T, T) -> T>;

/// An alternative for [`wrap`] function which works for arguments / return results with
/// non-`'static` lifetime.
///
/// The macro must be called with 2 arguments (in this order):
///
/// - Function arity (from 0 to 10 inclusive)
/// - Function or closure with the specified number of arguments. Using a function is recommended;
///   using a closure may lead to hard-to-debug type inference errors.
///
/// As with `wrap`, all function arguments must implement [`TryFromValue`] and the return result
/// must implement [`IntoEvalResult`]. Unlike `wrap`, the arguments / return result do not
/// need to have a `'static` lifetime; examples include [`Value`]s, [`Function`]s
/// and [`EvalResult`]s. Lifetimes of all arguments and the return result must match.
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{wrap_fn, Function, Environment, Value, VariableMap};
/// fn is_function<T>(value: Value<'_, T>) -> bool {
///     value.is_function()
/// }
///
/// # fn main() -> anyhow::Result<()> {
/// let program = "is_function(is_function) && !is_function(1)";
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("is_function", wrap_fn!(1, is_function))
///     .compile_module("test", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
///
/// Usage of lifetimes:
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{
/// #     wrap_fn, CallContext, Function, Environment, Prelude, Value, VariableMap,
/// # };
/// # use core::iter::FromIterator;
/// // Note that both `Value`s have the same lifetime due to elision.
/// fn take_if<T>(value: Value<'_, T>, condition: bool) -> Value<'_, T> {
///     if condition { value } else { Value::void() }
/// }
///
/// # fn main() -> anyhow::Result<()> {
/// let program = "(1, 2).take_if(true) == (1, 2) && (3, 4).take_if(false) != (3, 4)";
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::from_iter(Prelude.iter())
///     .insert_native_fn("take_if", wrap_fn!(2, take_if))
///     .compile_module("test_take_if", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! wrap_fn {
    (0, $function:expr) => { $crate::wrap_fn!(@arg 0 =>; $function) };
    (1, $function:expr) => { $crate::wrap_fn!(@arg 1 => x0; $function) };
    (2, $function:expr) => { $crate::wrap_fn!(@arg 2 => x0, x1; $function) };
    (3, $function:expr) => { $crate::wrap_fn!(@arg 3 => x0, x1, x2; $function) };
    (4, $function:expr) => { $crate::wrap_fn!(@arg 4 => x0, x1, x2, x3; $function) };
    (5, $function:expr) => { $crate::wrap_fn!(@arg 5 => x0, x1, x2, x3, x4; $function) };
    (6, $function:expr) => { $crate::wrap_fn!(@arg 6 => x0, x1, x2, x3, x4, x5; $function) };
    (7, $function:expr) => { $crate::wrap_fn!(@arg 7 => x0, x1, x2, x3, x4, x5, x6; $function) };
    (8, $function:expr) => {
        $crate::wrap_fn!(@arg 8 => x0, x1, x2, x3, x4, x5, x6, x7; $function)
    };
    (9, $function:expr) => {
        $crate::wrap_fn!(@arg 9 => x0, x1, x2, x3, x4, x5, x6, x7, x8; $function)
    };
    (10, $function:expr) => {
        $crate::wrap_fn!(@arg 10 => x0, x1, x2, x3, x4, x5, x6, x7, x8, x9; $function)
    };

    ($($ctx:ident,)? @arg $arity:expr => $($arg_name:ident),*; $function:expr) => {{
        let function = $function;
        $crate::fns::enforce_closure_type(move |args, context| {
            context.check_args_count(&args, $arity)?;
            let mut args_iter = args.into_iter().enumerate();

            $(
                let (index, $arg_name) = args_iter.next().unwrap();
                let span = $arg_name.with_no_extra();
                let $arg_name = $crate::fns::TryFromValue::try_from_value($arg_name.extra)
                    .map_err(|mut err| {
                        err.set_arg_index(index);
                        context
                            .call_site_error($crate::error::ErrorKind::Wrapper(err))
                            .with_span(&span, $crate::error::AuxErrorInfo::InvalidArg)
                    })?;
            )+

            // We need `$ctx` just as a marker that the function receives a context.
            let output = function($({ let $ctx = (); context },)? $($arg_name,)+);
            $crate::fns::IntoEvalResult::into_eval_result(output)
                .map_err(|err| err.into_spanned(context))
        })
    }}
}

/// Analogue of [`wrap_fn`](crate::wrap_fn) macro that injects the [`CallContext`]
/// as the first argument. This can be used to call functions within the implementation.
///
/// As with `wrap_fn`, this macro must be called with 2 args: the arity of the function
/// (**excluding** `CallContext`), and then the function / closure itself.
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{
/// #     wrap_fn_with_context, CallContext, Function, Environment, Value, Error, VariableMap,
/// # };
/// fn map_array<'a>(
///     context: &mut CallContext<'_, 'a, f32>,
///     array: Vec<Value<'a, f32>>,
///     map_fn: Function<'a, f32>,
/// ) -> Result<Vec<Value<'a, f32>>, Error<'a>> {
///     array
///         .into_iter()
///         .map(|value| {
///             let arg = context.apply_call_span(value);
///             map_fn.evaluate(vec![arg], context)
///         })
///         .collect()
/// }
///
/// # fn main() -> anyhow::Result<()> {
/// let program = "(1, 2, 3).map(|x| x + 3) == (4, 5, 6)";
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert_native_fn("map", wrap_fn_with_context!(2, map_array))
///     .compile_module("test_map", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! wrap_fn_with_context {
    (0, $function:expr) => { $crate::wrap_fn!(_ctx, @arg 0 =>; $function) };
    (1, $function:expr) => { $crate::wrap_fn!(_ctx, @arg 1 => x0; $function) };
    (2, $function:expr) => { $crate::wrap_fn!(_ctx, @arg 2 => x0, x1; $function) };
    (3, $function:expr) => { $crate::wrap_fn!(_ctx, @arg 3 => x0, x1, x2; $function) };
    (4, $function:expr) => { $crate::wrap_fn!(_ctx, @arg 4 => x0, x1, x2, x3; $function) };
    (5, $function:expr) => { $crate::wrap_fn!(_ctx, @arg 5 => x0, x1, x2, x3, x4; $function) };
    (6, $function:expr) => {
        $crate::wrap_fn!(_ctx, @arg 6 => x0, x1, x2, x3, x4, x5; $function)
    };
    (7, $function:expr) => {
        $crate::wrap_fn!(_ctx, @arg 7 => x0, x1, x2, x3, x4, x5, x6; $function)
    };
    (8, $function:expr) => {
        $crate::wrap_fn!(_ctx, @arg 8 => x0, x1, x2, x3, x4, x5, x6, x7; $function)
    };
    (9, $function:expr) => {
        $crate::wrap_fn!(_ctx, @arg 9 => x0, x1, x2, x3, x4, x5, x6, x7, x8; $function)
    };
    (10, $function:expr) => {
        $crate::wrap_fn!(_ctx, @arg 10 => x0, x1, x2, x3, x4, x5, x6, x7, x8, x9; $function)
    };
}

#[doc(hidden)] // necessary for `wrap_fn` macro
pub fn enforce_closure_type<T, A, F>(function: F) -> F
where
    F: for<'a> Fn(Vec<SpannedValue<'a, T>>, &mut CallContext<'_, 'a, A>) -> EvalResult<'a, T>,
{
    function
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        alloc::{format, ToOwned},
        Environment, ExecutableModule, Prelude, Value, WildcardId,
    };

    use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
    use assert_matches::assert_matches;

    #[test]
    fn functions_with_primitive_args() {
        let unary_fn = Unary::new(|x: f32| x + 3.0);
        let binary_fn = Binary::new(f32::min);
        let ternary_fn = Ternary::new(|x: f32, y, z| if x > 0.0 { y } else { z });

        let program = r#"
            unary_fn(2) == 5 && binary_fn(1, -3) == -3 &&
                ternary_fn(1, 2, 3) == 2 && ternary_fn(-1, 2, 3) == 3
        "#;
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();

        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("unary_fn", Value::native_fn(unary_fn))
            .with_import("binary_fn", Value::native_fn(binary_fn))
            .with_import("ternary_fn", Value::native_fn(ternary_fn))
            .build();
        assert_eq!(module.run().unwrap(), Value::Bool(true));
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
    fn functions_with_composite_args() {
        let program = r#"
            (1, 5, -3, 2, 1).array_min_max() == (-3, 5) &&
                total_sum(((1, 2), (3, 4)), ((5, 6, 7), 8)) == 36
        "#;
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();

        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("array_min_max", Value::wrapped_fn(array_min_max))
            .with_import("total_sum", Value::wrapped_fn(overly_convoluted_fn))
            .build();
        assert_eq!(module.run().unwrap(), Value::Bool(true));
    }

    fn sum_arrays(xs: Vec<f32>, ys: Vec<f32>) -> Result<Vec<f32>, String> {
        if xs.len() == ys.len() {
            Ok(xs.into_iter().zip(ys).map(|(x, y)| x + y).collect())
        } else {
            Err("Summed arrays must have the same size".to_owned())
        }
    }

    #[test]
    fn fallible_function() {
        let program = "(1, 2, 3).sum_arrays((4, 5, 6)) == (5, 7, 9)";
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();
        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("sum_arrays", Value::wrapped_fn(sum_arrays))
            .build();
        assert_eq!(module.run().unwrap(), Value::Bool(true));
    }

    #[test]
    fn fallible_function_with_bogus_program() {
        let program = "(1, 2, 3).sum_arrays((4, 5))";
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();

        let err = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("sum_arrays", Value::wrapped_fn(sum_arrays))
            .build()
            .run()
            .unwrap_err();
        assert!(err
            .source()
            .kind()
            .to_short_string()
            .contains("Summed arrays must have the same size"));
    }

    #[test]
    fn function_with_bool_return_value() {
        let contains = wrap(|(a, b): (f32, f32), x: f32| (a..=b).contains(&x));

        let program = "(-1, 2).contains(0) && !(1, 3).contains(0)";
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();
        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("contains", Value::native_fn(contains))
            .build();
        assert_eq!(module.run().unwrap(), Value::Bool(true));
    }

    #[test]
    fn function_with_void_return_value() {
        let mut env = Environment::new();
        env.insert_wrapped_fn("assert_eq", |expected: f32, actual: f32| {
            if (expected - actual).abs() < f32::EPSILON {
                Ok(())
            } else {
                Err(format!(
                    "Assertion failed: expected {}, got {}",
                    expected, actual
                ))
            }
        });

        let program = "assert_eq(3, 1 + 2)";
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();
        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_imports_from(&env)
            .build();
        assert!(module.run().unwrap().is_void());

        let bogus_program = "assert_eq(3, 1 - 2)";
        let bogus_block = Untyped::<F32Grammar>::parse_statements(bogus_program).unwrap();
        let err = ExecutableModule::builder(WildcardId, &bogus_block)
            .unwrap()
            .with_imports_from(&env)
            .build()
            .run()
            .unwrap_err();
        assert_matches!(
            err.source().kind(),
            ErrorKind::NativeCall(ref msg) if msg.contains("Assertion failed")
        );
    }

    #[test]
    fn function_with_bool_argument() {
        let program = "flip_sign(-1, true) == 1 && flip_sign(-1, false) == -1";
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();

        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_imports_from(&Prelude)
            .with_import(
                "flip_sign",
                Value::wrapped_fn(|val: f32, flag: bool| if flag { -val } else { val }),
            )
            .build();
        assert_eq!(module.run().unwrap(), Value::Bool(true));
    }

    #[test]
    fn error_reporting_with_destructuring() {
        let program = "((true, 1), (2, 3)).destructure()";
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();

        let err = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_imports_from(&Prelude)
            .with_import(
                "destructure",
                Value::wrapped_fn(|values: Vec<(bool, f32)>| {
                    values
                        .into_iter()
                        .map(|(flag, x)| if flag { x } else { 0.0 })
                        .sum::<f32>()
                }),
            )
            .build()
            .run()
            .unwrap_err();

        let err_message = err.source().kind().to_short_string();
        assert!(err_message.contains("Cannot convert number to bool"));
        assert!(err_message.contains("location: arg0[1].0"));
    }
}
