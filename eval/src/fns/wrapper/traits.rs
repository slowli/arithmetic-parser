//! Traits used in function wrapper.

use core::{cmp, fmt};

use crate::{
    alloc::{vec, String, Vec},
    CallContext, Error, ErrorKind, Function, Number, Object, Tuple, Value, ValueType,
};

/// Error raised when a value cannot be converted to the expected type when using
/// [`FnWrapper`](crate::fns::FnWrapper).
#[derive(Debug, Clone)]
pub struct FromValueError {
    kind: FromValueErrorKind,
    arg_index: usize,
    location: Vec<FromValueErrorLocation>,
}

impl FromValueError {
    pub(crate) fn invalid_type<T>(expected: ValueType, actual_value: &Value<T>) -> Self {
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
                FromValueErrorLocation::Tuple { index, .. } => write!(formatter, ".{index}")?,
                FromValueErrorLocation::Array { index, .. } => write!(formatter, "[{index}]")?,
            }
        }
        Ok(())
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FromValueError {}

/// Error kinds for [`FromValueError`].
#[derive(Debug, Clone, PartialEq, Eq)]
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
                write!(formatter, "Cannot convert {actual} to {expected}")
            }
        }
    }
}

/// Element of the [`FromValueError`] location.
///
/// Note that the distinction between tuples and arrays is determined by the [`FnWrapper`].
/// If the corresponding type in the wrapper is defined as a tuple, then
/// a [`Tuple`](FromValueErrorLocation::Tuple) element will be added to the location; otherwise,
/// an [`Array`](FromValueErrorLocation::Array) will be added.
///
/// [`FnWrapper`]: crate::fns::FnWrapper
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
pub trait TryFromValue<T>: Sized {
    /// Attempts to convert `value` to a type supported by the function.
    fn try_from_value(value: Value<T>) -> Result<Self, FromValueError>;
}

impl<T: Number> TryFromValue<T> for T {
    fn try_from_value(value: Value<T>) -> Result<Self, FromValueError> {
        match value {
            Value::Prim(number) => Ok(number),
            _ => Err(FromValueError::invalid_type(ValueType::Prim, &value)),
        }
    }
}

impl<T> TryFromValue<T> for bool {
    fn try_from_value(value: Value<T>) -> Result<Self, FromValueError> {
        match value {
            Value::Bool(flag) => Ok(flag),
            _ => Err(FromValueError::invalid_type(ValueType::Bool, &value)),
        }
    }
}

impl<T> TryFromValue<T> for Value<T> {
    fn try_from_value(value: Value<T>) -> Result<Self, FromValueError> {
        Ok(value)
    }
}

impl<T> TryFromValue<T> for Function<T> {
    fn try_from_value(value: Value<T>) -> Result<Self, FromValueError> {
        match value {
            Value::Function(function) => Ok(function),
            _ => Err(FromValueError::invalid_type(ValueType::Function, &value)),
        }
    }
}

impl<T> TryFromValue<T> for Tuple<T> {
    fn try_from_value(value: Value<T>) -> Result<Self, FromValueError> {
        match value {
            Value::Tuple(tuple) => Ok(tuple),
            _ => Err(FromValueError::invalid_type(ValueType::Array, &value)),
        }
    }
}

impl<T> TryFromValue<T> for Object<T> {
    fn try_from_value(value: Value<T>) -> Result<Self, FromValueError> {
        match value {
            Value::Object(object) => Ok(object),
            _ => Err(FromValueError::invalid_type(ValueType::Object, &value)),
        }
    }
}

impl<U, T> TryFromValue<T> for Vec<U>
where
    U: TryFromValue<T>,
{
    fn try_from_value(value: Value<T>) -> Result<Self, FromValueError> {
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
        impl<Num, $($ty,)+> TryFromValue<Num> for ($($ty,)+)
        where
            $($ty: TryFromValue<Num>,)+
        {
            #[allow(clippy::shadow_unrelated)] // makes it easier to write macro
            fn try_from_value(value: Value<Num>) -> Result<Self, FromValueError> {
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

/// Generic error output encompassing all error types supported by
/// [wrapped functions](crate::fns::FnWrapper).
#[derive(Debug)]
#[non_exhaustive]
pub enum ErrorOutput {
    /// Error together with the defined span(s).
    Spanned(Error),
    /// Error message. The error span will be defined as the call span of the native function.
    Message(String),
}

impl ErrorOutput {
    #[doc(hidden)] // necessary for `wrap_fn` macro
    pub fn into_spanned<A>(self, context: &CallContext<'_, A>) -> Error {
        match self {
            Self::Spanned(err) => err,
            Self::Message(message) => context.call_site_error(ErrorKind::native(message)),
        }
    }
}

/// Converts type into `Value` or an error. This is used to convert the return type
/// of [wrapped functions](crate::fns::FnWrapper) to the result expected by
/// [`NativeFn`](crate::NativeFn).
///
/// Unlike with `TryInto` trait from the standard library, the erroneous result here does not
/// mean that the conversion *itself* is impossible. Rather, it means that the function evaluation
/// has failed for the provided args.
///
///
/// This trait is implemented for base value types (such as [`Number`]s, [`Function`]s, [`Value`]s),
/// for two container types: vectors and tuples, and for `Result`s with the error type
/// convertible to [`ErrorOutput`].
pub trait IntoEvalResult<T> {
    /// Performs the conversion.
    fn into_eval_result(self) -> Result<Value<T>, ErrorOutput>;
}

impl<T, U> IntoEvalResult<T> for Result<U, String>
where
    U: IntoEvalResult<T>,
{
    fn into_eval_result(self) -> Result<Value<T>, ErrorOutput> {
        self.map_err(ErrorOutput::Message)
            .and_then(U::into_eval_result)
    }
}

impl<T, U> IntoEvalResult<T> for Result<U, Error>
where
    U: IntoEvalResult<T>,
{
    fn into_eval_result(self) -> Result<Value<T>, ErrorOutput> {
        self.map_err(ErrorOutput::Spanned)
            .and_then(U::into_eval_result)
    }
}

impl<T: Number> IntoEvalResult<T> for T {
    fn into_eval_result(self) -> Result<Value<T>, ErrorOutput> {
        Ok(Value::Prim(self))
    }
}

impl<T> IntoEvalResult<T> for () {
    fn into_eval_result(self) -> Result<Value<T>, ErrorOutput> {
        Ok(Value::void())
    }
}

impl<T> IntoEvalResult<T> for bool {
    fn into_eval_result(self) -> Result<Value<T>, ErrorOutput> {
        Ok(Value::Bool(self))
    }
}

impl<T> IntoEvalResult<T> for cmp::Ordering {
    fn into_eval_result(self) -> Result<Value<T>, ErrorOutput> {
        Ok(Value::opaque_ref(self))
    }
}

impl<T> IntoEvalResult<T> for Value<T> {
    fn into_eval_result(self) -> Result<Value<T>, ErrorOutput> {
        Ok(self)
    }
}

impl<T> IntoEvalResult<T> for Function<T> {
    fn into_eval_result(self) -> Result<Value<T>, ErrorOutput> {
        Ok(Value::Function(self))
    }
}

impl<T> IntoEvalResult<T> for Tuple<T> {
    fn into_eval_result(self) -> Result<Value<T>, ErrorOutput> {
        Ok(Value::Tuple(self))
    }
}

impl<T> IntoEvalResult<T> for Object<T> {
    fn into_eval_result(self) -> Result<Value<T>, ErrorOutput> {
        Ok(Value::Object(self))
    }
}

impl<U, T> IntoEvalResult<T> for Vec<U>
where
    U: IntoEvalResult<T>,
{
    fn into_eval_result(self) -> Result<Value<T>, ErrorOutput> {
        let values = self
            .into_iter()
            .map(U::into_eval_result)
            .collect::<Result<Tuple<_>, _>>()?;
        Ok(Value::Tuple(values))
    }
}

macro_rules! into_value_for_tuple {
    ($($i:tt : $ty:ident),+) => {
        impl<Num, $($ty,)+> IntoEvalResult<Num> for ($($ty,)+)
        where
            $($ty: IntoEvalResult<Num>,)+
        {
            fn into_eval_result(self) -> Result<Value<Num>, ErrorOutput> {
                Ok(Value::from(vec![$(self.$i.into_eval_result()?,)+]))
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
