use arithmetic_parser::{create_span_ref, Grammar};

use core::{fmt, marker::PhantomData};

use crate::{
    AuxErrorInfo, CallContext, EvalError, EvalResult, Function, NativeFn, Number, SpannedEvalError,
    SpannedValue, Value, ValueType,
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
/// grammar, and the output type should implement [`IntoEvalResult`]. If the [`CallContext`] is
/// necessary for execution (for example, if one of args is a [`Function`], which is called
/// during execution), then it can be specified as the first argument
/// as shown [below](#usage-of-context).
///
/// [native functions]: ../trait.NativeFn.html
/// [`TryFromValue`]: trait.TryFromValue.html
/// [`IntoEvalResult`]: trait.IntoEvalResult.html
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
/// fn zip_arrays(xs: Vec<f32>, ys: Vec<f32>) -> Result<Vec<(f32, f32)>, String> {
///     if xs.len() == ys.len() {
///         Ok(xs.into_iter().zip(ys).map(|(x, y)| (x, y)).collect())
///     } else {
///         Err("Arrays must have the same size".to_owned())
///     }
/// }
///
/// let mut interpreter = Interpreter::new();
/// interpreter.insert_native_fn("zip", FnWrapper::new(zip_arrays));
///
/// let program = "(1, 2, 3).zip((4, 5, 6)) == ((1, 4), (2, 5), (3, 6))";
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

/// Error raised when a value cannot be converted to the expected type when using
/// [`FnWrapper`].
///
/// [`FnWrapper`]: struct.FnWrapper.html
#[derive(Debug, Clone)]
pub struct FromValueError {
    kind: FromValueErrorKind,
    arg_index: usize,
    location: Vec<FromValueErrorLocation>,
}

impl FromValueError {
    pub(crate) fn invalid_type<T>(expected: ValueType, actual_value: &Value<'_, T>) -> Self
    where
        T: Grammar,
    {
        Self {
            kind: FromValueErrorKind::InvalidType {
                expected,
                actual: actual_value.value_type(),
            },
            arg_index: 0,
            location: vec![],
        }
    }

    fn add_location(mut self, location: FromValueErrorLocation) -> Self {
        self.location.push(location);
        self
    }

    fn set_arg_index(&mut self, index: usize) {
        self.arg_index = index;
        self.location.reverse();
    }

    /// Returns the error kind.
    pub fn kind(&self) -> &FromValueErrorKind {
        &self.kind
    }

    /// Returns the zero-based index of the argument where the error has occurred.
    pub fn arg_index(&self) -> usize {
        self.arg_index
    }

    /// Returns the error location, starting from the outermost one.
    pub fn location(&self) -> &[FromValueErrorLocation] {
        &self.location
    }
}

impl fmt::Display for FromValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}. Error location: arg{}",
            self.kind, self.arg_index
        )?;
        for location_element in &self.location {
            match location_element {
                FromValueErrorLocation::Tuple { index, .. } => write!(formatter, ".{}", index)?,
                FromValueErrorLocation::Array { index, .. } => write!(formatter, "[{}]", index)?,
            }
        }
        Ok(())
    }
}

/// Error kinds for [`FromValueError`].
///
/// [`FromValueError`]: struct.FromValueError.html
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum FromValueErrorKind {
    /// Mismatch between expected and actual value type.
    InvalidType {
        /// Expected value type.
        expected: ValueType,
        /// Actual value type.
        actual: ValueType,
    },
}

impl fmt::Display for FromValueErrorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidType { expected, actual } => {
                write!(formatter, "Cannot convert {} to {}", actual, expected)
            }
        }
    }
}

/// Element of the [`FromValueError`] location.
///
/// Note that the distinction between tuples and arrays is determined by the [`FnWrapper`].
/// If the corresponding type in the wrapper is defined as a tuple, then
/// a [`Tuple`](#variant.Tuple) element will be added to the location; otherwise,
/// an [`Array`](#variant.Array) will be added.
///
/// [`FromValueError`]: struct.FromValueError.html
/// [`FnWrapper`]: struct.FnWrapper.html
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum FromValueErrorLocation {
    /// Location within a tuple.
    Tuple {
        /// Tuple size.
        size: usize,
        /// Zero-based index of the erroneous element.
        index: usize,
    },
    /// Location within an array.
    Array {
        /// Factual array size.
        size: usize,
        /// Zero-based index of the erroneous element.
        index: usize,
    },
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
    fn try_from_value(value: Value<'a, G>) -> Result<Self, FromValueError>;
}

impl<'a, T: Number, G: Grammar<Lit = T>> TryFromValue<'a, G> for T {
    fn try_from_value(value: Value<'a, G>) -> Result<Self, FromValueError> {
        match value {
            Value::Number(number) => Ok(number),
            _ => Err(FromValueError::invalid_type(ValueType::Number, &value)),
        }
    }
}

impl<'a, G: Grammar> TryFromValue<'a, G> for bool {
    fn try_from_value(value: Value<'a, G>) -> Result<Self, FromValueError> {
        match value {
            Value::Bool(flag) => Ok(flag),
            _ => Err(FromValueError::invalid_type(ValueType::Bool, &value)),
        }
    }
}

impl<'a, U, G: Grammar> TryFromValue<'a, G> for Vec<U>
where
    U: TryFromValue<'a, G>,
{
    fn try_from_value(value: Value<'a, G>) -> Result<Self, FromValueError> {
        match value {
            Value::Tuple(values) => {
                let tuple_len = values.len();
                let mut collected = Vec::with_capacity(tuple_len);

                for (index, element) in values.into_iter().enumerate() {
                    let converted = U::try_from_value(element).map_err(|err| {
                        err.add_location(FromValueErrorLocation::Array {
                            size: tuple_len,
                            index,
                        })
                    })?;
                    collected.push(converted);
                }
                Ok(collected)
            }
            _ => Err(FromValueError::invalid_type(ValueType::Array, &value)),
        }
    }
}

impl<'a, G: Grammar> TryFromValue<'a, G> for Value<'a, G> {
    fn try_from_value(value: Value<'a, G>) -> Result<Self, FromValueError> {
        Ok(value)
    }
}

impl<'a, G: Grammar> TryFromValue<'a, G> for Function<'a, G> {
    fn try_from_value(value: Value<'a, G>) -> Result<Self, FromValueError> {
        match value {
            Value::Function(function) => Ok(function),
            _ => Err(FromValueError::invalid_type(ValueType::Function, &value)),
        }
    }
}

macro_rules! try_from_value_for_tuple {
    ($size:expr => $($var:ident : $ty:ident),+) => {
        impl<'a, G: Grammar, $($ty,)+> TryFromValue<'a, G> for ($($ty,)+)
        where
            $($ty: TryFromValue<'a, G>,)+
        {
            fn try_from_value(value: Value<'a, G>) -> Result<Self, FromValueError> {
                const EXPECTED_TYPE: ValueType = ValueType::Tuple($size);

                match value {
                    Value::Tuple(values) if values.len() == $size => {
                        let mut values_iter = values.into_iter().enumerate();
                        $(
                            let (index, $var) = values_iter.next().unwrap();
                            let $var = $ty::try_from_value($var).map_err(|err| {
                                err.add_location(FromValueErrorLocation::Tuple {
                                    size: $size,
                                    index,
                                })
                            })?;
                        )+
                        Ok(($($var,)+))
                    }
                    _ => Err(FromValueError::invalid_type(EXPECTED_TYPE, &value)),
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
pub trait IntoEvalResult<'a, G: Grammar> {
    /// Performs the conversion.
    fn into_eval_result(self) -> Result<Value<'a, G>, ErrorOutput<'a>>;
}

impl<'a, G: Grammar, U> IntoEvalResult<'a, G> for Result<U, String>
where
    U: IntoEvalResult<'a, G>,
{
    fn into_eval_result(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
        self.map_err(ErrorOutput::Message)
            .and_then(U::into_eval_result)
    }
}

impl<'a, G: Grammar, U> IntoEvalResult<'a, G> for Result<U, SpannedEvalError<'a>>
where
    U: IntoEvalResult<'a, G>,
{
    fn into_eval_result(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
        self.map_err(ErrorOutput::Spanned)
            .and_then(U::into_eval_result)
    }
}

impl<'a, G: Grammar> IntoEvalResult<'a, G> for Value<'a, G> {
    fn into_eval_result(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
        Ok(self)
    }
}

impl<'a, G: Grammar> IntoEvalResult<'a, G> for Function<'a, G> {
    fn into_eval_result(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
        Ok(Value::Function(self))
    }
}

impl<'a, T: Number, G: Grammar<Lit = T>> IntoEvalResult<'a, G> for T {
    fn into_eval_result(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
        Ok(Value::Number(self))
    }
}

impl<'a, G: Grammar> IntoEvalResult<'a, G> for () {
    fn into_eval_result(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
        Ok(Value::void())
    }
}

impl<'a, G: Grammar> IntoEvalResult<'a, G> for bool {
    fn into_eval_result(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
        Ok(Value::Bool(self))
    }
}

impl<'a, U, G: Grammar> IntoEvalResult<'a, G> for Vec<U>
where
    U: IntoEvalResult<'a, G>,
{
    fn into_eval_result(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
        let values = self
            .into_iter()
            .map(U::into_eval_result)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Value::Tuple(values))
    }
}

macro_rules! into_value_for_tuple {
    ($($i:tt : $ty:ident),+) => {
        impl<'a, G: Grammar, $($ty,)+> IntoEvalResult<'a, G> for ($($ty,)+)
        where
            $($ty: IntoEvalResult<'a, G>,)+
        {
            fn into_eval_result(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
                Ok(Value::Tuple(vec![$(self.$i.into_eval_result()?,)+]))
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
            Ret: IntoEvalResult<'a, G>,
        {
            fn evaluate(
                &self,
                args: Vec<SpannedValue<'a, G>>,
                context: &mut CallContext<'_, 'a>,
            ) -> EvalResult<'a, G> {
                context.check_args_count(&args, $arity)?;
                let mut args_iter = args.into_iter().enumerate();

                $(
                    let (index, $arg_name) = args_iter.next().unwrap();
                    let span = create_span_ref(&$arg_name, ());
                    let $arg_name = $t::try_from_value($arg_name.extra).map_err(|mut err| {
                        err.set_arg_index(index);
                        context
                            .call_site_error(EvalError::Wrapper(err))
                            .with_span(&span, AuxErrorInfo::InvalidArg)
                    })?;
                )+

                let output = (self.function)($($arg_name,)+);
                output.into_eval_result().map_err(|err| err.into_spanned(context))
            }
        }

        impl<'a, G, F, Ret, $($t,)+> NativeFn<'a, G> for FnWrapper<WithContext<(Ret, $($t,)+)>, F>
        where
            G: Grammar,
            F: Fn(&mut CallContext<'_, 'a>, $($t,)+) -> Ret,
            $($t: TryFromValue<'a, G>,)+
            Ret: IntoEvalResult<'a, G>,
        {
            fn evaluate(
                &self,
                args: Vec<SpannedValue<'a, G>>,
                context: &mut CallContext<'_, 'a>,
            ) -> EvalResult<'a, G> {
                context.check_args_count(&args, $arity)?;
                let mut args_iter = args.into_iter().enumerate();

                $(
                    let (index, $arg_name) = args_iter.next().unwrap();
                    let span = create_span_ref(&$arg_name, ());
                    let $arg_name = $t::try_from_value($arg_name.extra).map_err(|mut err| {
                        err.set_arg_index(index);
                        context
                            .call_site_error(EvalError::Wrapper(err))
                            .with_span(&span, AuxErrorInfo::InvalidArg)
                    })?;
                )+

                let output = (self.function)(context, $($arg_name,)+);
                output.into_eval_result().map_err(|err| err.into_spanned(context))
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
    use assert_matches::assert_matches;
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

    #[test]
    fn function_with_bool_return_value() {
        let mut interpreter = Interpreter::new();
        interpreter.insert_native_fn(
            "contains",
            wrap(|(a, b): (f32, f32), x: f32| (a..=b).contains(&x)),
        );

        let program = "(-1, 2).contains(0) && !(1, 3).contains(0)";
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let ret = interpreter.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Bool(true));
    }

    #[test]
    fn function_with_void_return_value() {
        let mut interpreter = Interpreter::new();
        interpreter.insert_native_fn(
            "assert_eq",
            wrap(|expected: f32, actual: f32| {
                if (expected - actual).abs() < f32::EPSILON {
                    Ok(())
                } else {
                    Err(format!(
                        "Assertion failed: expected {}, got {}",
                        expected, actual
                    ))
                }
            }),
        );

        let program = "assert_eq(3, 1 + 2)";
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let ret = interpreter.evaluate(&block).unwrap();
        assert!(ret.is_void());

        let program = "assert_eq(3, 1 - 2)";
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let err = interpreter.evaluate(&block).unwrap_err();
        assert_matches!(
            err.source(),
            EvalError::NativeCall(ref msg) if msg.contains("Assertion failed")
        );
    }

    #[test]
    fn function_with_bool_argument() {
        let mut interpreter = Interpreter::new();
        interpreter
            .insert_var("true", Value::Bool(true))
            .insert_var("false", Value::Bool(false));
        interpreter.insert_native_fn(
            "flip_sign",
            wrap(|val: f32, flag: bool| if flag { -val } else { val }),
        );

        let program = "flip_sign(-1, true) == 1 && flip_sign(-1, false) == -1";
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let ret = interpreter.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Bool(true));
    }

    #[test]
    fn error_reporting_with_destructuring() {
        let mut interpreter = Interpreter::new();
        interpreter
            .insert_var("true", Value::Bool(true))
            .insert_var("false", Value::Bool(false));
        interpreter.insert_native_fn(
            "destructure",
            wrap(|values: Vec<(bool, f32)>| {
                values
                    .into_iter()
                    .map(|(flag, x)| if flag { x } else { 0.0 })
                    .sum::<f32>()
            }),
        );

        let program = "((true, 1), (2, 3)).destructure()";
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let err_message = interpreter
            .evaluate(&block)
            .unwrap_err()
            .source()
            .to_short_string();
        assert!(err_message.contains("Cannot convert number to bool"));
        assert!(err_message.contains("location: arg0[1].0"));
    }
}
