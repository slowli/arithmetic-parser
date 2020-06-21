use arithmetic_parser::{create_span_ref, Grammar};

use core::{fmt, marker::PhantomData};

use crate::{
    AuxErrorInfo, CallContext, EvalError, EvalResult, Function, NativeFn, Number, SpannedEvalError,
    SpannedValue, Value,
};

/// Wraps a function enriching it with the information about its arguments.
/// This is a slightly shorter way to create wrappers compared to calling [`FnWrapper::new()`].
///
/// [`FnWrapper::new()`]: struct.FnWrapper.html#method.new
pub const fn wrap<T, F>(function: F) -> FnWrapper<T, F> {
    FnWrapper::new(function)
}

/// Wrapper of a function containing information about its arguments.
///
/// Using `FnWrapper` allows to define [native functions] with minimum boilerplate
/// and with increased type safety.
///
/// Arguments of a wrapped function should implement [`TryFromValue`] trait for the applicable
/// grammar, and the output type should implement [`TryIntoValue`]. If the [`CallContext`] is
/// necessary for execution (for example, if one of args is a [`Function`], which is called
/// during execution), then it can be specified as the first argument
/// as shown [below](#usage-of-context).
///
/// [native functions]: ../trait.NativeFn.html
/// [`TryFromValue`]: trait.TryFromValue.html
/// [`TryIntoValue`]: trait.TryIntoValue.html
/// [`CallContext`]: ../struct.CallContext.html
/// [`Function`]: ../enum.Function.html
///
/// # Examples
///
/// ## Basic function
///
/// ```
/// use arithmetic_parser::{grammars::F32Grammar, GrammarExt, Span};
/// use arithmetic_eval::{fns::FnWrapper, Interpreter, Value};
///
/// let mut interpreter = Interpreter::new();
/// let max = FnWrapper::new(|x: f32, y: f32| if x > y { x } else { y });
/// interpreter.insert_native_fn("max", max);
///
/// let program = "max(1, 3) == 3 && max(-1, -3) == -1";
/// let program = F32Grammar::parse_statements(Span::new(program)).unwrap();
/// let ret = interpreter.evaluate(&program).unwrap();
/// assert_eq!(ret, Value::Bool(true));
/// ```
///
/// ## Fallible function with complex args
///
/// ```
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt, Span};
/// # use arithmetic_eval::{fns::FnWrapper, Interpreter, Value};
/// fn sum_arrays(xs: Vec<f32>, ys: Vec<f32>) -> Result<Vec<f32>, String> {
///     if xs.len() == ys.len() {
///         Ok(xs.into_iter().zip(ys).map(|(x, y)| x + y).collect())
///     } else {
///         Err("Arrays must have the same size".to_owned())
///     }
/// }
///
/// let mut interpreter = Interpreter::new();
/// interpreter.insert_native_fn("sum_arrays", FnWrapper::new(sum_arrays));
///
/// let program = "(1, 2, 3).sum_arrays((4, 5, 6)) == (5, 7, 9)";
/// let program = F32Grammar::parse_statements(Span::new(program)).unwrap();
/// let ret = interpreter.evaluate(&program).unwrap();
/// assert_eq!(ret, Value::Bool(true));
/// ```
///
/// ## Usage of context
///
/// ```
/// # use arithmetic_parser::{grammars::F32Grammar, Grammar, GrammarExt, Span};
/// # use arithmetic_eval::{
/// #     fns::FnWrapper, CallContext, Function, Interpreter, Value, SpannedEvalError,
/// # };
/// fn map_array<'a, G: Grammar<Lit = f32>>(
///     context: &mut CallContext<'_, 'a>,
///     array: Vec<Value<'a, G>>,
///     map_fn: Function<'a, G>,
/// ) -> Result<Vec<Value<'a, G>>, SpannedEvalError<'a>> {
///     array
///         .into_iter()
///         .map(|value| {
///             let arg = context.apply_call_span(value);
///             map_fn.evaluate(vec![arg], context)
///         })
///         .collect()
/// }
///
/// let mut interpreter = Interpreter::new();
/// interpreter.insert_native_fn("map", FnWrapper::new(map_array));
///
/// let program = "(1, 2, 3).map(|x| x + 3) == (4, 5, 6)";
/// let program = F32Grammar::parse_statements(Span::new(program)).unwrap();
/// let ret = interpreter.evaluate(&program).unwrap();
/// assert_eq!(ret, Value::Bool(true));
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
    /// to be the case, `function` needs to be a function or an `Fn` closure,
    /// and the `T` type argument needs to be a tuple with the function return type
    /// and the argument types (in this order).
    ///
    /// [`NativeFn`]: ../trait.NativeFn.html
    pub const fn new(function: F) -> Self {
        Self {
            function,
            _arg_types: PhantomData,
        }
    }
}

/// Fallible conversion from `Value` to a function argument.
///
/// This trait is implemented for base value types (such as [`Number`]s, [`Function`]s, [`Value`]s),
/// and for two container types: vectors and tuples.
///
/// [`Number`]: ../trait.Number.html
/// [`Function`]: ../enum.Function.html
/// [`Value`]: ../enum.Value.html
pub trait TryFromValue<'a, G: Grammar>: Sized {
    /// Attempts to convert `value` to a type supported by the function.
    fn try_from_value(value: Value<'a, G>) -> Result<Self, String>;
}

impl<'a, T: Number, G: Grammar<Lit = T>> TryFromValue<'a, G> for T {
    fn try_from_value(value: Value<'a, G>) -> Result<Self, String> {
        match value {
            Value::Number(number) => Ok(number),
            _ => Err("Expected number".to_owned()),
        }
    }
}

impl<'a, U, G: Grammar> TryFromValue<'a, G> for Vec<U>
where
    U: TryFromValue<'a, G>,
{
    fn try_from_value(value: Value<'a, G>) -> Result<Self, String> {
        match value {
            Value::Tuple(values) => values.into_iter().map(U::try_from_value).collect(),
            _ => Err("Expected tuple".to_owned()),
        }
    }
}

impl<'a, G: Grammar> TryFromValue<'a, G> for Value<'a, G> {
    fn try_from_value(value: Value<'a, G>) -> Result<Self, String> {
        Ok(value)
    }
}

impl<'a, G: Grammar> TryFromValue<'a, G> for Function<'a, G> {
    fn try_from_value(value: Value<'a, G>) -> Result<Self, String> {
        match value {
            Value::Function(function) => Ok(function),
            _ => Err("Expected function".to_owned()),
        }
    }
}

macro_rules! try_from_value_for_tuple {
    ($size:expr => $($var:ident : $ty:ident),+) => {
        impl<'a, G: Grammar, $($ty,)+> TryFromValue<'a, G> for ($($ty,)+)
        where
            $($ty: TryFromValue<'a, G>,)+
        {
            fn try_from_value(value: Value<'a, G>) -> Result<Self, String> {
                const MSG: &str = concat!("Expected ", $size, "-element tuple");

                match value {
                    Value::Tuple(values) if values.len() == $size => {
                        let mut values_iter = values.into_iter();
                        $(
                            let $var = $ty::try_from_value(values_iter.next().unwrap())?;
                        )+
                        Ok(($($var,)+))
                    }
                    _ => Err(MSG.to_owned()),
                }
            }
        }
    };
}

try_from_value_for_tuple!(1 => x0: T);
try_from_value_for_tuple!(2 => x0: T, x1: U);
try_from_value_for_tuple!(3 => x0: T, x1: U, x2: V);
try_from_value_for_tuple!(4 => x0: T, x1: U, x2: V, x3: W);
try_from_value_for_tuple!(5 => x0: T, x1: U, x2: V, x3: W, x4: X);
try_from_value_for_tuple!(6 => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y);
try_from_value_for_tuple!(7 => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y, x6: Z);
try_from_value_for_tuple!(8 => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y, x6: Z, x7: A);
try_from_value_for_tuple!(9 => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y, x6: Z, x7: A, x8: B);
try_from_value_for_tuple!(10 => x0: T, x1: U, x2: V, x3: W, x4: X, x5: Y, x6: Z, x7: A, x8: B, x9: C);

/// Generic error output encompassing all error types supported by [wrapped functions].
///
/// [wrapped functions]: struct.FnWrapper.html
#[derive(Debug)]
pub enum ErrorOutput<'a> {
    /// Error together with the defined span(s).
    Spanned(SpannedEvalError<'a>),
    /// Error message. The error span will be defined as the call span of the native function.
    Message(String),
}

impl<'a> ErrorOutput<'a> {
    fn into_spanned(self, context: &CallContext<'_, 'a>) -> SpannedEvalError<'a> {
        match self {
            Self::Spanned(error) => error,
            Self::Message(message) => context.call_site_error(EvalError::NativeCall(message)),
        }
    }
}

/// Converts type into `Value` or an error. This is used to convert the return type
/// of [wrapped functions] to the result expected by [`NativeFn`].
///
/// Unlike with `TryInto` trait from the standard library, the erroneous result here does not
/// mean that the conversion *itself* is impossible. Rather, it means that the function evaluation
/// has failed for the provided args.
///
///
/// This trait is implemented for base value types (such as [`Number`]s, [`Function`]s, [`Value`]s),
/// for two container types: vectors and tuples, and for `Result`s with the error type
/// convertible to [`ErrorOutput`].
///
/// [`Number`]: ../trait.Number.html
/// [`Function`]: ../enum.Function.html
/// [`Value`]: ../enum.Value.html
/// [wrapped functions]: struct.FnWrapper.html
/// [`NativeFn`]: ../trait.NativeFn.html
/// [`ErrorOutput`]: enum.ErrorOutput.html
pub trait TryIntoValue<'a, G: Grammar> {
    /// Performs the conversion.
    fn try_into_value(self) -> Result<Value<'a, G>, ErrorOutput<'a>>;
}

impl<'a, G: Grammar, U> TryIntoValue<'a, G> for Result<U, String>
where
    U: TryIntoValue<'a, G>,
{
    fn try_into_value(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
        self.map_err(ErrorOutput::Message)
            .and_then(U::try_into_value)
    }
}

impl<'a, G: Grammar, U> TryIntoValue<'a, G> for Result<U, SpannedEvalError<'a>>
where
    U: TryIntoValue<'a, G>,
{
    fn try_into_value(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
        self.map_err(ErrorOutput::Spanned)
            .and_then(U::try_into_value)
    }
}

impl<'a, G: Grammar> TryIntoValue<'a, G> for Value<'a, G> {
    fn try_into_value(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
        Ok(self)
    }
}

impl<'a, G: Grammar> TryIntoValue<'a, G> for Function<'a, G> {
    fn try_into_value(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
        Ok(Value::Function(self))
    }
}

impl<'a, T: Number, G: Grammar<Lit = T>> TryIntoValue<'a, G> for T {
    fn try_into_value(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
        Ok(Value::Number(self))
    }
}

impl<'a, U, G: Grammar> TryIntoValue<'a, G> for Vec<U>
where
    U: TryIntoValue<'a, G>,
{
    fn try_into_value(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
        let values = self
            .into_iter()
            .map(U::try_into_value)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Value::Tuple(values))
    }
}

macro_rules! into_value_for_tuple {
    ($($i:tt : $ty:ident),+) => {
        impl<'a, G: Grammar, $($ty,)+> TryIntoValue<'a, G> for ($($ty,)+)
        where
            $($ty: TryIntoValue<'a, G>,)+
        {
            fn try_into_value(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
                Ok(Value::Tuple(vec![$(self.$i.try_into_value()?,)+]))
            }
        }
    };
}

into_value_for_tuple!(0: T);
into_value_for_tuple!(0: T, 1: U);
into_value_for_tuple!(0: T, 1: U, 2: V);
into_value_for_tuple!(0: T, 1: U, 2: V, 3: W);
into_value_for_tuple!(0: T, 1: U, 2: V, 3: W, 4: X);
into_value_for_tuple!(0: T, 1: U, 2: V, 3: W, 4: X, 5: Y);
into_value_for_tuple!(0: T, 1: U, 2: V, 3: W, 4: X, 5: Y, 6: Z);
into_value_for_tuple!(0: T, 1: U, 2: V, 3: W, 4: X, 5: Y, 6: Z, 7: A);
into_value_for_tuple!(0: T, 1: U, 2: V, 3: W, 4: X, 5: Y, 6: Z, 7: A, 8: B);
into_value_for_tuple!(0: T, 1: U, 2: V, 3: W, 4: X, 5: Y, 6: Z, 7: A, 8: B, 9: C);

/// Marker for use with [wrapper functions] accepting context as the first argument.
///
/// [wrapper functions]: struct.FnWrapper.html
#[derive(Debug)]
pub struct WithContext<T>(T);

macro_rules! arity_fn {
    ($arity:tt => $($arg_name:ident : $t:ident),+) => {
        impl<'a, G, F, Ret, $($t,)+> NativeFn<'a, G> for FnWrapper<(Ret, $($t,)+), F>
        where
            G: Grammar,
            F: Fn($($t,)+) -> Ret,
            $($t: TryFromValue<'a, G>,)+
            Ret: TryIntoValue<'a, G>,
        {
            fn evaluate(
                &self,
                args: Vec<SpannedValue<'a, G>>,
                context: &mut CallContext<'_, 'a>,
            ) -> EvalResult<'a, G> {
                context.check_args_count(&args, $arity)?;
                let mut args_iter = args.into_iter();

                $(
                    let $arg_name = args_iter.next().unwrap();
                    let span = create_span_ref(&$arg_name, ());
                    let $arg_name = $t::try_from_value($arg_name.extra)
                        .map_err(|error_msg| {
                            context
                                .call_site_error(EvalError::native(error_msg))
                                .with_span(&span, AuxErrorInfo::InvalidArg)
                        })?;
                )+

                let output = (self.function)($($arg_name,)+);
                output.try_into_value().map_err(|err| err.into_spanned(context))
            }
        }

        impl<'a, G, F, Ret, $($t,)+> NativeFn<'a, G> for FnWrapper<WithContext<(Ret, $($t,)+)>, F>
        where
            G: Grammar,
            F: Fn(&mut CallContext<'_, 'a>, $($t,)+) -> Ret,
            $($t: TryFromValue<'a, G>,)+
            Ret: TryIntoValue<'a, G>,
        {
            fn evaluate(
                &self,
                args: Vec<SpannedValue<'a, G>>,
                context: &mut CallContext<'_, 'a>,
            ) -> EvalResult<'a, G> {
                context.check_args_count(&args, $arity)?;
                let mut args_iter = args.into_iter();

                $(
                    let $arg_name = args_iter.next().unwrap();
                    let span = create_span_ref(&$arg_name, ());
                    let $arg_name = $t::try_from_value($arg_name.extra)
                        .map_err(|error_msg| {
                            context
                                .call_site_error(EvalError::native(error_msg))
                                .with_span(&span, AuxErrorInfo::InvalidArg)
                        })?;
                )+

                let output = (self.function)(context, $($arg_name,)+);
                output.try_into_value().map_err(|err| err.into_spanned(context))
            }
        }
    };
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Interpreter;
    use arithmetic_parser::{grammars::F32Grammar, GrammarExt, Span};

    use core::f32;

    #[test]
    fn functions_with_primitive_args() {
        let unary_fn = Unary::new(|x: f32| x + 3.0);
        let binary_fn = Binary::new(f32::min);
        let ternary_fn = Ternary::new(|x: f32, y, z| if x > 0.0 { y } else { z });

        let mut interpreter = Interpreter::new();
        interpreter
            .insert_native_fn("unary_fn", unary_fn)
            .insert_native_fn("binary_fn", binary_fn)
            .insert_native_fn("ternary_fn", ternary_fn);

        let program = r#"
            unary_fn(2) == 5 && binary_fn(1, -3) == -3 &&
                ternary_fn(1, 2, 3) == 2 && ternary_fn(-1, 2, 3) == 3
        "#;
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let ret = interpreter.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Bool(true));
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
        let mut interpreter = Interpreter::new();
        interpreter
            .insert_native_fn("array_min_max", FnWrapper::new(array_min_max))
            .insert_native_fn("total_sum", FnWrapper::new(overly_convoluted_fn));

        let program = r#"
            (1, 5, -3, 2, 1).array_min_max() == (-3, 5) &&
                total_sum(((1, 2), (3, 4)), ((5, 6, 7), 8)) == 36
        "#;
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let ret = interpreter.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Bool(true));
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
        let mut interpreter = Interpreter::new();
        interpreter.insert_native_fn("sum_arrays", FnWrapper::new(sum_arrays));

        let program = "(1, 2, 3).sum_arrays((4, 5, 6)) == (5, 7, 9)";
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let ret = interpreter.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Bool(true));

        let program = "(1, 2, 3).sum_arrays((4, 5))";
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let err = interpreter.evaluate(&block).unwrap_err();
        assert!(err
            .source()
            .to_short_string()
            .contains("Summed arrays must have the same size"));
    }
}
