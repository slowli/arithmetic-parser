use num_traits::{One, Zero};

use core::{cmp, fmt, marker::PhantomData};

use crate::{
    alloc::{vec, String, Vec},
    error::AuxErrorInfo,
    CallContext, Error, ErrorKind, EvalResult, Function, NativeFn, Number, SpannedValue, Value,
    ValueType,
};
use arithmetic_parser::Grammar;

/// Wraps a function enriching it with the information about its arguments.
/// This is a slightly shorter way to create wrappers compared to calling [`FnWrapper::new()`].
///
/// See [`FnWrapper`] for more details on function requirements.
///
/// [`FnWrapper::new()`]: struct.FnWrapper.html#method.new
/// [`FnWrapper`]: struct.FnWrapper.html
pub const fn wrap<T, F>(function: F) -> FnWrapper<T, F> {
    FnWrapper::new(function)
}

/// Wrapper of a function containing information about its arguments.
///
/// Using `FnWrapper` allows to define [native functions] with minimum boilerplate
/// and with increased type safety. `FnWrapper`s can be constructed explcitly or indirectly
/// via [`Environment::insert_wrapped_fn()`], [`Value::wrapped_fn()`], or [`wrap()`].
///
/// Arguments of a wrapped function must implement [`TryFromValue`] trait for the applicable
/// grammar, and the output type must implement [`IntoEvalResult`]. If arguments and/or output
/// have non-`'static` lifetime, use the [`wrap_fn`] macro. If you need [`CallContext`] (e.g.,
/// to call functions provided as an argument), use the [`wrap_fn_with_context`] macro.
///
/// [native functions]: ../trait.NativeFn.html
/// [`Environment::insert_wrapped_fn()`]: ../struct.Environment.html#method.insert_wrapped_fn
/// [`Value::wrapped_fn()`]: ../enum.Value.html#method.wrapped_fn
/// [`wrap()`]: fn.wrap.html
/// [`TryFromValue`]: trait.TryFromValue.html
/// [`IntoEvalResult`]: trait.IntoEvalResult.html
/// [`CallContext`]: ../struct.CallContext.html
/// [`wrap_fn`]: ../macro.wrap_fn.html
/// [`wrap_fn_with_context`]: ../macro.wrap_fn_with_context.html
///
/// # Examples
///
/// ## Basic function
///
/// ```
/// use arithmetic_parser::{grammars::F32Grammar, GrammarExt};
/// use arithmetic_eval::{fns, Environment, Value, VariableMap};
///
/// # fn main() -> anyhow::Result<()> {
/// let max = fns::wrap(|x: f32, y: f32| if x > y { x } else { y });
///
/// let program = "max(1, 3) == 3 && max(-1, -3) == -1";
/// let program = F32Grammar::parse_statements(program)?;
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
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt};
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
/// let program = F32Grammar::parse_statements(program)?;
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

    #[doc(hidden)] // necessary for `wrap_fn` macro
    pub fn set_arg_index(&mut self, index: usize) {
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

#[cfg(feature = "std")]
impl std::error::Error for FromValueError {}

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

macro_rules! try_from_value_for_tuple {
    ($size:expr => $($var:ident : $ty:ident),+) => {
        impl<'a, G: Grammar, $($ty,)+> TryFromValue<'a, G> for ($($ty,)+)
        where
            $($ty: TryFromValue<'a, G>,)+
        {
            #[allow(clippy::shadow_unrelated)] // makes it easier to write macro
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
#[non_exhaustive]
pub enum ErrorOutput<'a> {
    /// Error together with the defined span(s).
    Spanned(Error<'a>),
    /// Error message. The error span will be defined as the call span of the native function.
    Message(String),
}

impl<'a> ErrorOutput<'a> {
    #[doc(hidden)] // necessary for `wrap_fn` macro
    pub fn into_spanned(self, context: &CallContext<'_, 'a>) -> Error<'a> {
        match self {
            Self::Spanned(err) => err,
            Self::Message(message) => context.call_site_error(ErrorKind::native(message)),
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

impl<'a, G: Grammar, U> IntoEvalResult<'a, G> for Result<U, Error<'a>>
where
    U: IntoEvalResult<'a, G>,
{
    fn into_eval_result(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
        self.map_err(ErrorOutput::Spanned)
            .and_then(U::into_eval_result)
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

impl<'a, G> IntoEvalResult<'a, G> for cmp::Ordering
where
    G: Grammar,
    G::Lit: Number,
{
    fn into_eval_result(self) -> Result<Value<'a, G>, ErrorOutput<'a>> {
        Ok(Value::Number(match self {
            Self::Less => -<G::Lit as One>::one(),
            Self::Equal => <G::Lit as Zero>::zero(),
            Self::Greater => <G::Lit as One>::one(),
        }))
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

macro_rules! arity_fn {
    ($arity:tt => $($arg_name:ident : $t:ident),+) => {
        impl<G, F, Ret, $($t,)+> NativeFn<G> for FnWrapper<(Ret, $($t,)+), F>
        where
            G: Grammar,
            F: Fn($($t,)+) -> Ret,
            $($t: for<'val> TryFromValue<'val, G>,)+
            Ret: for<'val> IntoEvalResult<'val, G>,
        {
            #[allow(clippy::shadow_unrelated)] // makes it easier to write macro
            fn evaluate<'a>(
                &self,
                args: Vec<SpannedValue<'a, G>>,
                context: &mut CallContext<'_, 'a>,
            ) -> EvalResult<'a, G> {
                context.check_args_count(&args, $arity)?;
                let mut args_iter = args.into_iter().enumerate();

                $(
                    let (index, $arg_name) = args_iter.next().unwrap();
                    let span = $arg_name.with_no_extra();
                    let $arg_name = $t::try_from_value($arg_name.extra).map_err(|mut err| {
                        err.set_arg_index(index);
                        let enriched_span = context.enrich_call_site_span(&span);
                        context
                            .call_site_error(ErrorKind::Wrapper(err))
                            .with_span(enriched_span, AuxErrorInfo::InvalidArg)
                    })?;
                )+

                let output = (self.function)($($arg_name,)+);
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
/// [`wrap`]: fns/fn.wrap.html
/// [`TryFromValue`]: fns/trait.TryFromValue.html
/// [`IntoEvalResult`]: fns/trait.IntoEvalResult.html
/// [`Value`]: enum.Value.html
/// [`Function`]: enum.Function.html
/// [`EvalResult`]: error/type.EvalResult.html
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::{grammars::F32Grammar, Grammar, GrammarExt};
/// # use arithmetic_eval::{wrap_fn, Function, Environment, Value, VariableMap};
/// fn is_function<G: Grammar>(value: Value<'_, G>) -> bool {
///     value.is_function()
/// }
///
/// # fn main() -> anyhow::Result<()> {
/// let program = "is_function(is_function) && !is_function(1)";
/// let program = F32Grammar::parse_statements(program)?;
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
/// # use arithmetic_parser::{grammars::F32Grammar, Grammar, GrammarExt};
/// # use arithmetic_eval::{
/// #     wrap_fn, CallContext, Function, Environment, Prelude, Value, VariableMap,
/// # };
/// # use core::iter::FromIterator;
/// // Note that both `Value`s have the same lifetime due to elision.
/// fn take_if<G: Grammar>(value: Value<'_, G>, condition: bool) -> Value<'_, G> {
///     if condition { value } else { Value::void() }
/// }
///
/// # fn main() -> anyhow::Result<()> {
/// let program = "(1, 2).take_if(true) == (1, 2) && (3, 4).take_if(false) != (3, 4)";
/// let program = F32Grammar::parse_statements(program)?;
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
                        let enriched_span = context.enrich_call_site_span(&span);
                        context
                            .call_site_error($crate::error::ErrorKind::Wrapper(err))
                            .with_span(enriched_span, $crate::error::AuxErrorInfo::InvalidArg)
                    })?;
            )+

            // We need `$ctx` just as a marker that the function receives a context.
            let output = function($({ let $ctx = (); context },)? $($arg_name,)+);
            $crate::fns::IntoEvalResult::into_eval_result(output)
                .map_err(|err| err.into_spanned(context))
        })
    }}
}

/// Analogue of [`wrap_fn`] macro that injects the [`CallContext`] as the first argument.
/// This can be used to call functions within the implementation.
///
/// As with `wrap_fn`, this macro must be called with 2 args: the arity of the function
/// (**excluding** `CallContext`), and then the function / closure itself.
///
/// [`wrap_fn`]: macro.wrap_fn.html
/// [`CallContext`]: struct.CallContext.html
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::{grammars::F32Grammar, Grammar, GrammarExt};
/// # use arithmetic_eval::{
/// #     wrap_fn_with_context, CallContext, Function, Environment, Value, Error, VariableMap,
/// # };
/// fn map_array<'a, G: Grammar<Lit = f32>>(
///     context: &mut CallContext<'_, 'a>,
///     array: Vec<Value<'a, G>>,
///     map_fn: Function<'a, G>,
/// ) -> Result<Vec<Value<'a, G>>, Error<'a>> {
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
/// let program = F32Grammar::parse_statements(program)?;
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
pub fn enforce_closure_type<G: Grammar, F>(function: F) -> F
where
    F: for<'a> Fn(Vec<SpannedValue<'a, G>>, &mut CallContext<'_, 'a>) -> EvalResult<'a, G>,
{
    function
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Environment, ExecutableModule, Prelude, WildcardId};

    use arithmetic_parser::{grammars::F32Grammar, GrammarExt};
    use assert_matches::assert_matches;
    use core::f32;

    #[test]
    fn functions_with_primitive_args() {
        let unary_fn = Unary::new(|x: f32| x + 3.0);
        let binary_fn = Binary::new(f32::min);
        let ternary_fn = Ternary::new(|x: f32, y, z| if x > 0.0 { y } else { z });

        let program = r#"
            unary_fn(2) == 5 && binary_fn(1, -3) == -3 &&
                ternary_fn(1, 2, 3) == 2 && ternary_fn(-1, 2, 3) == 3
        "#;
        let block = F32Grammar::parse_statements(program).unwrap();

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
        let block = F32Grammar::parse_statements(program).unwrap();

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
        let block = F32Grammar::parse_statements(program).unwrap();
        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("sum_arrays", Value::wrapped_fn(sum_arrays))
            .build();
        assert_eq!(module.run().unwrap(), Value::Bool(true));
    }

    #[test]
    fn fallible_function_with_bogus_program() {
        let program = "(1, 2, 3).sum_arrays((4, 5))";
        let block = F32Grammar::parse_statements(program).unwrap();

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
        let block = F32Grammar::parse_statements(program).unwrap();
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
        let block = F32Grammar::parse_statements(program).unwrap();
        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_imports_from(&env)
            .build();
        assert!(module.run().unwrap().is_void());

        let bogus_program = "assert_eq(3, 1 - 2)";
        let bogus_block = F32Grammar::parse_statements(bogus_program).unwrap();
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
        let block = F32Grammar::parse_statements(program).unwrap();

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
        let block = F32Grammar::parse_statements(program).unwrap();

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
