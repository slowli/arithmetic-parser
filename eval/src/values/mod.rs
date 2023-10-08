//! Values used by the interpreter.

// TODO: consider removing lifetimes from `Value` (i.e., strip code spans immediately)

use core::{
    any::{type_name, Any},
    fmt,
};

use crate::{
    alloc::{Arc, Vec},
    fns,
};
use arithmetic_parser::Location;

mod function;
mod object;
mod ops;
mod tuple;

pub use self::{
    function::{CallContext, Function, InterpretedFn, NativeFn},
    object::Object,
    tuple::Tuple,
};

/// Possible high-level types of [`Value`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ValueType {
    /// Primitive type other than `Bool`ean.
    Prim,
    /// Boolean value.
    Bool,
    /// Function value.
    Function,
    /// Tuple of a specific size.
    Tuple(usize),
    /// Object.
    Object,
    /// Array (a tuple of arbitrary size).
    ///
    /// This variant is never returned from [`Value::value_type()`]; at the same time, it is
    /// used for error reporting etc.
    Array,
    /// Opaque reference to a value.
    Ref,
}

impl fmt::Display for ValueType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prim => formatter.write_str("primitive value"),
            Self::Bool => formatter.write_str("boolean value"),
            Self::Function => formatter.write_str("function"),
            Self::Tuple(1) => formatter.write_str("tuple with 1 element"),
            Self::Object => formatter.write_str("object"),
            Self::Tuple(size) => write!(formatter, "tuple with {size} elements"),
            Self::Array => formatter.write_str("array"),
            Self::Ref => formatter.write_str("reference"),
        }
    }
}

/// Opaque reference to a native value.
///
/// The references cannot be created by interpreted code, but can be used as function args
/// or return values of native functions. References are [`Arc`]'d, thus can easily be cloned.
///
/// References are comparable among each other:
///
/// - If the wrapped value implements [`PartialEq`], this implementation will be used
///   for comparison.
/// - If `PartialEq` is not implemented, the comparison is by the `Arc` pointer.
#[derive(Clone)]
pub struct OpaqueRef {
    value: Arc<dyn Any>,
    type_name: &'static str,
    dyn_eq: fn(&dyn Any, &dyn Any) -> bool,
    dyn_fmt: fn(&dyn Any, &mut fmt::Formatter<'_>) -> fmt::Result,
}

impl OpaqueRef {
    /// Creates a reference to `value` that implements equality comparison.
    ///
    /// Prefer using this method if the wrapped type implements [`PartialEq`].
    #[allow(clippy::missing_panics_doc)] // false positive; `unwrap()`s never panic
    pub fn new<T>(value: T) -> Self
    where
        T: Any + fmt::Debug + PartialEq,
    {
        Self {
            value: Arc::new(value),
            type_name: type_name::<T>(),

            dyn_eq: |this, other| {
                let this_cast = this.downcast_ref::<T>().unwrap();
                other
                    .downcast_ref::<T>()
                    .map_or(false, |other_cast| other_cast == this_cast)
            },
            dyn_fmt: |this, formatter| {
                let this_cast = this.downcast_ref::<T>().unwrap();
                fmt::Debug::fmt(this_cast, formatter)
            },
        }
    }

    /// Creates a reference to `value` with the identity comparison: values are considered
    /// equal iff they point to the same data.
    ///
    /// Prefer [`Self::new()`] when possible.
    #[allow(clippy::missing_panics_doc)] // false positive; `unwrap()`s never panic
    pub fn with_identity_eq<T: Any>(value: T) -> Self {
        Self {
            value: Arc::new(value),
            type_name: type_name::<T>(),

            dyn_eq: |this, other| {
                let this_data = (this as *const dyn Any).cast::<()>();
                let other_data = (other as *const dyn Any).cast::<()>();
                this_data == other_data
            },
            dyn_fmt: |this, formatter| fmt::Debug::fmt(&this.type_id(), formatter),
        }
    }

    /// Tries to downcast this reference to a specific type.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.value.downcast_ref()
    }
}

impl PartialEq for OpaqueRef {
    fn eq(&self, other: &Self) -> bool {
        (self.dyn_eq)(self.value.as_ref(), other.value.as_ref())
    }
}

impl fmt::Debug for OpaqueRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("OpaqueRef")
            .field(&self.value.as_ref())
            .finish()
    }
}

impl fmt::Display for OpaqueRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}::", self.type_name)?;
        (self.dyn_fmt)(self.value.as_ref(), formatter)
    }
}

/// Values produced by expressions during their interpretation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Value<T> {
    /// Primitive value, such as a number. This does not include Boolean values,
    /// which are a separate variant.
    ///
    /// Literals must necessarily map to primitive values, but there may be some primitive values
    /// not representable as literals.
    Prim(T),
    /// Boolean value.
    Bool(bool),
    /// Function.
    Function(Function<T>),
    /// Tuple of zero or more values.
    Tuple(Tuple<T>),
    /// Object with zero or more named fields.
    Object(Object<T>),
    /// Opaque reference to a native value.
    Ref(OpaqueRef),
}

/// Value together with a span that has produced it.
pub type SpannedValue<T> = Location<Value<T>>;

impl<T> Value<T> {
    /// Creates a value for a native function.
    pub fn native_fn(function: impl NativeFn<T> + 'static) -> Self {
        Self::Function(Function::Native(Arc::new(function)))
    }

    /// Creates a [wrapped function](fns::FnWrapper).
    ///
    /// Calling this method is equivalent to [`wrap`](fns::wrap)ping a function and calling
    /// [`Self::native_fn()`] on it. Thanks to type inference magic, the Rust compiler
    /// will usually be able to extract the `Args` type param from the function definition,
    /// provided that type of function arguments and its return type are defined explicitly
    /// or can be unequivocally inferred from the declaration.
    pub fn wrapped_fn<const CTX: bool, Args, F>(fn_to_wrap: F) -> Self
    where
        fns::FnWrapper<Args, F, CTX>: NativeFn<T> + 'static,
    {
        let wrapped = fns::wrap::<CTX, Args, _>(fn_to_wrap);
        Self::native_fn(wrapped)
    }

    /// Creates a value for an interpreted function.
    pub(crate) fn interpreted_fn(function: InterpretedFn<T>) -> Self {
        Self::Function(Function::Interpreted(Arc::new(function)))
    }

    /// Creates a void value (an empty tuple).
    pub const fn void() -> Self {
        Self::Tuple(Tuple::void())
    }

    /// Creates a reference to a native variable.
    pub fn opaque_ref(value: impl Any + fmt::Debug + PartialEq) -> Self {
        Self::Ref(OpaqueRef::new(value))
    }

    /// Returns the type of this value.
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::Prim(_) => ValueType::Prim,
            Self::Bool(_) => ValueType::Bool,
            Self::Function(_) => ValueType::Function,
            Self::Tuple(elements) => ValueType::Tuple(elements.len()),
            Self::Object(_) => ValueType::Object,
            Self::Ref(_) => ValueType::Ref,
        }
    }

    /// Checks if this value is void (an empty tuple).
    pub fn is_void(&self) -> bool {
        matches!(self, Self::Tuple(tuple) if tuple.is_empty())
    }

    /// Checks if this value is a function.
    pub fn is_function(&self) -> bool {
        matches!(self, Self::Function(_))
    }

    pub(crate) fn as_object(&self) -> Option<&Object<T>> {
        match self {
            Self::Object(object) => Some(object),
            _ => None,
        }
    }
}

impl<T> From<Vec<Self>> for Value<T> {
    fn from(elements: Vec<Self>) -> Self {
        Self::Tuple(Tuple::from(elements))
    }
}

impl<T: Clone> From<&Value<T>> for Value<T> {
    fn from(reference: &Value<T>) -> Self {
        reference.clone()
    }
}

impl<T: PartialEq> PartialEq for Value<T> {
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Prim(this), Self::Prim(other)) => this == other,
            (Self::Bool(this), Self::Bool(other)) => this == other,
            (Self::Tuple(this), Self::Tuple(other)) => this == other,
            (Self::Object(this), Self::Object(other)) => this == other,
            (Self::Function(this), Self::Function(other)) => this == other,
            (Self::Ref(this), Self::Ref(other)) => this == other,
            _ => false,
        }
    }
}

impl<T: fmt::Display> fmt::Display for Value<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prim(value) => fmt::Display::fmt(value, formatter),
            Self::Bool(true) => formatter.write_str("true"),
            Self::Bool(false) => formatter.write_str("false"),
            Self::Ref(opaque_ref) => fmt::Display::fmt(opaque_ref, formatter),
            Self::Function(function) => fmt::Display::fmt(function, formatter),
            Self::Object(object) => fmt::Display::fmt(object, formatter),
            Self::Tuple(tuple) => fmt::Display::fmt(tuple, formatter),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use core::cmp::Ordering;

    #[test]
    fn opaque_ref_equality() {
        let value = Value::<f32>::opaque_ref(Ordering::Less);
        let same_value = Value::<f32>::opaque_ref(Ordering::Less);
        assert_eq!(value, same_value);
        assert_eq!(value, value.clone());
        let other_value = Value::<f32>::opaque_ref(Ordering::Greater);
        assert_ne!(value, other_value);
    }

    #[test]
    fn opaque_ref_formatting() {
        let value = OpaqueRef::new(Ordering::Less);
        assert_eq!(value.to_string(), "core::cmp::Ordering::Less");
    }
}
