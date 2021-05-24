//! Values used by the interpreter.

use hashbrown::HashMap;

use core::{
    any::{type_name, Any},
    fmt,
};

use crate::{
    alloc::{vec, Rc, String, Vec},
    fns,
};
use arithmetic_parser::{MaybeSpanned, StripCode};

mod env;
mod function;
mod ops;
mod variable_map;

pub use self::{
    env::Environment,
    function::{CallContext, Function, InterpretedFn, NativeFn},
    variable_map::{Assertions, Comparisons, Prelude, VariableMap},
};

/// Possible high-level types of [`Value`]s.
#[derive(Debug, Clone, Copy, PartialEq)]
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
            Self::Tuple(size) => write!(formatter, "tuple with {} elements", size),
            Self::Array => formatter.write_str("array"),
            Self::Ref => formatter.write_str("reference"),
        }
    }
}

/// Opaque reference to a native value.
///
/// The references cannot be created by interpreted code, but can be used as function args
/// or return values of native functions. References are [`Rc`]'d, thus can easily be cloned.
///
/// References are comparable among each other:
///
/// - If the wrapped value implements [`PartialEq`], this implementation will be used
///   for comparison.
/// - If `PartialEq` is not implemented, the comparison is by the `Rc` pointer.
pub struct OpaqueRef {
    value: Rc<dyn Any>,
    type_name: &'static str,
    dyn_eq: fn(&dyn Any, &dyn Any) -> bool,
    dyn_fmt: fn(&dyn Any, &mut fmt::Formatter<'_>) -> fmt::Result,
}

#[allow(renamed_and_removed_lints, clippy::unknown_clippy_lints)]
// ^ `missing_panics_doc` is newer than MSRV, and `clippy::unknown_clippy_lints` is removed
// since Rust 1.51.
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
            value: Rc::new(value),
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
    pub fn with_identity_eq<T>(value: T) -> Self
    where
        T: Any + fmt::Debug,
    {
        Self {
            value: Rc::new(value),
            type_name: type_name::<T>(),

            dyn_eq: |this, other| {
                let this_data = (this as *const dyn Any).cast::<()>();
                let other_data = (other as *const dyn Any).cast::<()>();
                this_data == other_data
            },
            dyn_fmt: |this, formatter| {
                let this_cast = this.downcast_ref::<T>().unwrap();
                fmt::Debug::fmt(this_cast, formatter)
            },
        }
    }

    /// Tries to downcast this reference to a specific type.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.value.downcast_ref()
    }
}

impl Clone for OpaqueRef {
    fn clone(&self) -> Self {
        Self {
            value: Rc::clone(&self.value),
            type_name: self.type_name,
            dyn_eq: self.dyn_eq,
            dyn_fmt: self.dyn_fmt,
        }
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
#[derive(Debug)]
#[non_exhaustive]
pub enum Value<'a, T> {
    /// Primitive value, such as a number. This does not include Boolean values,
    /// which are a separate variant.
    ///
    /// Literals must necessarily map to primitive values, but there may be some primitive values
    /// not representable as literals.
    Prim(T),
    /// Boolean value.
    Bool(bool),
    /// Function.
    Function(Function<'a, T>),
    /// Tuple of zero or more values.
    Tuple(Vec<Value<'a, T>>),
    /// Object with zero or more named fields.
    Object(HashMap<String, Value<'a, T>>),
    /// Opaque reference to a native value.
    Ref(OpaqueRef),
}

/// Value together with a span that has produced it.
pub type SpannedValue<'a, T> = MaybeSpanned<'a, Value<'a, T>>;

impl<'a, T> Value<'a, T> {
    /// Creates a value for a native function.
    pub fn native_fn(function: impl NativeFn<T> + 'static) -> Self {
        Self::Function(Function::Native(Rc::new(function)))
    }

    /// Creates a [wrapped function](fns::FnWrapper).
    ///
    /// Calling this method is equivalent to [`wrap`](fns::wrap)ping a function and calling
    /// [`Self::native_fn()`] on it. Thanks to type inference magic, the Rust compiler
    /// will usually be able to extract the `Args` type param from the function definition,
    /// provided that type of function arguments and its return type are defined explicitly
    /// or can be unequivocally inferred from the declaration.
    pub fn wrapped_fn<Args, F>(fn_to_wrap: F) -> Self
    where
        fns::FnWrapper<Args, F>: NativeFn<T> + 'static,
    {
        let wrapped = fns::wrap::<Args, _>(fn_to_wrap);
        Self::native_fn(wrapped)
    }

    /// Creates a value for an interpreted function.
    pub(crate) fn interpreted_fn(function: InterpretedFn<'a, T>) -> Self {
        Self::Function(Function::Interpreted(Rc::new(function)))
    }

    /// Creates a void value (an empty tuple).
    pub fn void() -> Self {
        Self::Tuple(vec![])
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
}

impl<T: Clone> Clone for Value<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Prim(lit) => Self::Prim(lit.clone()),
            Self::Bool(bool) => Self::Bool(*bool),
            Self::Function(function) => Self::Function(function.clone()),
            Self::Tuple(tuple) => Self::Tuple(tuple.clone()),
            Self::Object(fields) => Self::Object(fields.clone()),
            Self::Ref(reference) => Self::Ref(reference.clone()),
        }
    }
}

impl<T: 'static + Clone> StripCode for Value<'_, T> {
    type Stripped = Value<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        match self {
            Self::Prim(lit) => Value::Prim(lit),
            Self::Bool(bool) => Value::Bool(bool),
            Self::Function(function) => Value::Function(function.strip_code()),
            Self::Tuple(tuple) => {
                Value::Tuple(tuple.into_iter().map(StripCode::strip_code).collect())
            }
            Self::Object(fields) => Value::Object(
                fields
                    .into_iter()
                    .map(|(name, value)| (name, value.strip_code()))
                    .collect(),
            ),
            Self::Ref(reference) => Value::Ref(reference),
        }
    }
}

impl<'a, T: Clone> From<&Value<'a, T>> for Value<'a, T> {
    fn from(reference: &Value<'a, T>) -> Self {
        reference.clone()
    }
}

impl<T: PartialEq> PartialEq for Value<'_, T> {
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Prim(this), Self::Prim(other)) => this == other,
            (Self::Bool(this), Self::Bool(other)) => this == other,
            (Self::Tuple(this), Self::Tuple(other)) => this == other,
            (Self::Object(this), Self::Object(other)) => this == other,
            (Self::Function(this), Self::Function(other)) => this.is_same_function(other),
            (Self::Ref(this), Self::Ref(other)) => this == other,
            _ => false,
        }
    }
}

impl<T: fmt::Display> fmt::Display for Value<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prim(value) => fmt::Display::fmt(value, formatter),
            Self::Bool(true) => formatter.write_str("true"),
            Self::Bool(false) => formatter.write_str("false"),
            Self::Ref(opaque_ref) => fmt::Display::fmt(opaque_ref, formatter),
            Self::Function(_) => formatter.write_str("[function]"),
            Self::Object(fields) => {
                formatter.write_str("#{ ")?;
                for (name, value) in fields.iter() {
                    write!(formatter, "{} = {}; ", name, value)?;
                }
                formatter.write_str("}")
            }
            Self::Tuple(elements) => {
                formatter.write_str("(")?;
                for (i, element) in elements.iter().enumerate() {
                    fmt::Display::fmt(element, formatter)?;
                    if i + 1 < elements.len() {
                        formatter.write_str(", ")?;
                    }
                }
                formatter.write_str(")")
            }
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
