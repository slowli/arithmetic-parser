//! `Object` and tightly related types.

use hashbrown::HashMap;

use core::{
    fmt,
    iter::{self, FromIterator},
    ops,
};

use crate::{
    alloc::Rc, error::AuxErrorInfo, CallContext, ErrorKind, EvalResult, Function, SpannedValue,
    Value, ValueType,
};
use arithmetic_parser::StripCode;

/// Object with zero or more named fields.
///
/// An object functions similarly to a [`HashMap`] with [`String`] keys and [`Value`]
/// values. It allows iteration over name-value pairs, random access by field name,
/// inserting / removing fields etc.
///
/// # Examples
///
/// ```
/// # use arithmetic_eval::{fns, Object, Value, ValueType};
/// let mut obj = Object::<u32>::default();
/// obj.insert("field", Value::Prim(0));
/// obj.insert("other_field", Value::Bool(false));
/// assert_eq!(obj.len(), 2);
/// assert_eq!(obj["field"].value_type(), ValueType::Prim);
/// assert!(obj.iter().all(|(_, val)| !val.is_void()));
///
/// // `Object` implements `FromIterator` / `Extend`.
/// let fields = vec![
///     ("third", Value::Prim(3)),
///     ("fourth", Value::native_fn(fns::Assert)),
/// ];
/// let mut other_obj: Object<u32> = fields.into_iter().collect();
/// other_obj.extend(obj);
/// assert_eq!(other_obj.len(), 4);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Object<'a, T> {
    fields: HashMap<String, Value<'a, T>>,
    prototype: Option<Prototype<'a, T>>,
}

impl<'a, T> From<Object<'a, T>> for Value<'a, T> {
    fn from(object: Object<'a, T>) -> Self {
        Self::Object(object)
    }
}

impl<T> Default for Object<'_, T> {
    fn default() -> Self {
        Self {
            fields: HashMap::new(),
            prototype: None,
        }
    }
}

impl<T: fmt::Display> fmt::Display for Object<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("#{ ")?;
        for (i, (name, value)) in self.iter().enumerate() {
            write!(formatter, "{}: {}", name, value)?;
            if i + 1 < self.len() {
                write!(formatter, ", ")?;
            } else {
                write!(formatter, " ")?;
            }
        }
        formatter.write_str("}")
    }
}

impl<'a, T> Object<'a, T> {
    /// Returns the number of fields in this object.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Checks if this object is empty (has no fields).
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Iterates over name-value pairs for all fields defined in this object.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value<'a, T>)> + '_ {
        self.fields
            .iter()
            .map(|(name, value)| (name.as_str(), value))
    }

    /// Iterates over field names.
    pub fn field_names(&self) -> impl Iterator<Item = &str> + '_ {
        self.fields.keys().map(String::as_str)
    }

    /// Returns the value of a field with the specified name, or `None` if this object
    /// does not contain such a field.
    pub fn get(&self, field_name: &str) -> Option<&Value<'a, T>> {
        self.fields.get(field_name)
    }

    /// Checks whether this object has a field with the specified name.
    pub fn contains_field(&self, field_name: &str) -> bool {
        self.fields.contains_key(field_name)
    }

    /// Inserts a field into this object.
    pub fn insert(
        &mut self,
        field_name: impl Into<String>,
        value: Value<'a, T>,
    ) -> Option<Value<'a, T>> {
        self.fields.insert(field_name.into(), value)
    }

    /// Removes and returns the specified field from this object.
    pub fn remove(&mut self, field_name: &str) -> Option<Value<'a, T>> {
        self.fields.remove(field_name)
    }

    /// Returns the prototype of this object, or `None` if it does not exist.
    pub fn prototype(&self) -> Option<&Prototype<'a, T>> {
        self.prototype.as_ref()
    }

    /// Sets the object prototype.
    pub fn set_prototype(&mut self, prototype: Prototype<'a, T>) {
        self.prototype = Some(prototype);
    }
}

impl<T: 'static + Clone> StripCode for Object<'_, T> {
    type Stripped = Object<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        Object {
            fields: self
                .fields
                .into_iter()
                .map(|(name, value)| (name, value.strip_code()))
                .collect(),
            prototype: self.prototype.map(StripCode::strip_code),
        }
    }
}

impl<'a, T> ops::Index<&str> for Object<'a, T> {
    type Output = Value<'a, T>;

    fn index(&self, index: &str) -> &Self::Output {
        &self.fields[index]
    }
}

impl<'a, T> IntoIterator for Object<'a, T> {
    type Item = (String, Value<'a, T>);
    /// Iterator type should be considered an implementation detail.
    type IntoIter = hashbrown::hash_map::IntoIter<String, Value<'a, T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.fields.into_iter()
    }
}

impl<'r, 'a, T> IntoIterator for &'r Object<'a, T> {
    type Item = (&'r str, &'r Value<'a, T>);
    /// Iterator type should be considered an implementation detail.
    type IntoIter = Iter<'r, 'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.fields
            .iter()
            .map(|(name, value)| (name.as_str(), value))
    }
}

pub type Iter<'r, 'a, T> = iter::Map<
    hashbrown::hash_map::Iter<'r, String, Value<'a, T>>,
    fn((&'r String, &'r Value<'a, T>)) -> (&'r str, &'r Value<'a, T>),
>;

impl<'a, T, S, V> FromIterator<(S, V)> for Object<'a, T>
where
    S: Into<String>,
    V: Into<Value<'a, T>>,
{
    fn from_iter<I: IntoIterator<Item = (S, V)>>(iter: I) -> Self {
        Self {
            fields: iter
                .into_iter()
                .map(|(name, value)| (name.into(), value.into()))
                .collect(),
            prototype: None,
        }
    }
}

impl<'a, T, S, V> Extend<(S, V)> for Object<'a, T>
where
    S: Into<String>,
    V: Into<Value<'a, T>>,
{
    fn extend<I: IntoIterator<Item = (S, V)>>(&mut self, iter: I) {
        let new_fields = iter
            .into_iter()
            .map(|(name, value)| (name.into(), value.into()));
        self.fields.extend(new_fields);
    }
}

/// Prototype of an [`Object`] or a [`Tuple`](crate::Tuple).
///
/// A prototype is quite similar to an [`Object`]; it is a collection of named [`Value`]s.
/// Prototype fields are used for method lookup similar to JavaScript; if an value has a prototype,
/// its method is looked up as the prototype field with the corresponding name, which should
/// be a function. The method call is translated to calling this function with the first argument
/// being the method receiver.
///
/// For example, if method `len()` is called on a value, then field `len` is obtained
/// from the value prototype and is called with the only argument being the value on which
/// the method is called.
///
/// Non-functional fields may make sense in the prototype as well; they can be viewed as
/// static members of the prototype. All fields can be accessed in the script code identically
/// to object fields.
///
/// A prototype can be converted to a [`Function`], and prototypes defined in the script code
/// *are* callable. Such a function associates the prototype with the provided value
/// (an object or a tuple). For values without an associated prototype, method resolution
/// is performed using [`StandardPrototypes`], which can be set for an [`Environment`]
/// or an [`ExecutableModule`].
///
/// Prototypes can be defined both in the host code, and in scripts via [`CreatePrototype`].
///
/// [`CreatePrototype`]: crate::fns::CreatePrototype
/// [`Environment`]: crate::Environment
/// [`ExecutableModule`]: crate::ExecutableModule
///
/// # Examples
///
/// Defining a prototype in the host code.
///
/// ```
/// # use arithmetic_eval::{fns, Environment, Object, Prototype, Value, env::VariableMap};
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # fn main() -> anyhow::Result<()> {
/// let mut proto = Object::default();
/// proto.insert("len", Value::native_fn(fns::Len)); // returns number of fields
/// proto.insert("EMPTY", Object::default().into());
/// let proto = Prototype::from(proto);
///
/// // Let's associate an object with this prototype in the host code.
/// let mut object = Object::<f32>::default();
/// object.insert("x", Value::Prim(3.0));
/// object.insert("y", Value::Prim(-4.0));
/// object.set_prototype(proto.clone());
///
/// let mut env = Environment::new();
/// env.insert("Object", proto.into()).insert("pt", object.into());
/// let program = r#"
///     pt.len() == 2 &&
///     Object(#{ foo: 1 }).len() == 1 &&
///     (Object.len)(Object.EMPTY) == 0
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let ret = env.compile_module("proto", &program)?.run()?;
/// assert_eq!(ret, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Prototype<'a, T> {
    inner: Rc<Object<'a, T>>,
}

impl<'a, T> From<Object<'a, T>> for Prototype<'a, T> {
    fn from(object: Object<'a, T>) -> Self {
        Self {
            inner: Rc::new(object),
        }
    }
}

impl<T> Clone for Prototype<'_, T> {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl<T> PartialEq for Prototype<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<'a, T> From<Prototype<'a, T>> for Function<'a, T> {
    fn from(prototype: Prototype<'a, T>) -> Self {
        Self::Prototype(prototype)
    }
}

impl<'a, T> From<Prototype<'a, T>> for Value<'a, T> {
    fn from(prototype: Prototype<'a, T>) -> Self {
        Self::Function(Function::from(prototype))
    }
}

impl<'a, T> Prototype<'a, T> {
    pub(crate) fn as_object(&self) -> &Object<'a, T> {
        &self.inner
    }

    pub(crate) fn evaluate(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 1)?;
        let mut arg = args.pop().unwrap();

        match &mut arg.extra {
            Value::Object(object) => {
                object.set_prototype(self.clone());
            }
            Value::Tuple(tuple) => {
                tuple.set_prototype(self.clone());
            }
            _ => {
                let err = ErrorKind::native("Function argument must be an object or tuple");
                return Err(ctx
                    .call_site_error(err)
                    .with_span(&arg, AuxErrorInfo::InvalidArg));
            }
        }
        Ok(arg.extra)
    }
}

impl<T: 'static + Clone> StripCode for Prototype<'_, T> {
    type Stripped = Prototype<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        let inner = Rc::try_unwrap(self.inner).unwrap_or_else(|rc| (*rc).clone());
        Prototype {
            inner: Rc::new(inner.strip_code()),
        }
    }
}

/// [`Prototype`]s for standard types, such as [`Object`] and [`Tuple`](crate::Tuple).
///
/// Unlike user-defined prototypes, standard prototypes are not directly accessible
/// from the script code and cannot be extended / redefined there; this is only possible
/// from the host.
///
/// # Merging
///
/// [`AddAssign`](core::ops::AddAssign) implementation merges two sets of prototypes.
/// Each prototype (as an [`Object`]) is [`Extend`]ed with the new values, replacing existing
/// values if necessary. Merging is used in [`Environment::insert_prototypes()`].
///
/// [`Environment::insert_prototypes()`]: crate::Environment::insert_prototypes()
///
/// # Examples
///
/// ```
/// # use arithmetic_eval::{
/// #     fns, Environment, Object, StandardPrototypes, Value, env::{Prelude, VariableMap},
/// # };
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # fn main() -> anyhow::Result<()> {
/// let mut num_proto = Object::default();
/// num_proto.insert("abs", Value::wrapped_fn(f32::abs));
/// num_proto.insert("sin", Value::wrapped_fn(f32::sin));
/// let mut array_proto = Object::default();
/// array_proto.insert("len", Value::native_fn(fns::Len));
///
/// let prototypes = StandardPrototypes::new()
///     .with_primitive_proto(num_proto)
///     .with_array_proto(array_proto);
/// let mut env = Environment::new();
/// env.insert_prototypes(prototypes)
///     .insert_prototypes(Prelude.prototypes());
/// // ^ also insert "standard" prototypes
///
/// let program = r#"
///     array = (1, -2, 3).map(|x| x.abs());
///     array.len() == 3 && array.1 > 0
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let ret = env.compile_module("proto", &program)?.run()?;
/// assert_eq!(ret, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct StandardPrototypes<T> {
    object_proto: Option<Object<'static, T>>,
    array_proto: Option<Object<'static, T>>,
    function_proto: Option<Object<'static, T>>,
    prim_proto: Option<Object<'static, T>>,
    bool_proto: Option<Object<'static, T>>,
}

impl<T> Default for StandardPrototypes<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> StandardPrototypes<T> {
    /// Creates an empty instance.
    pub const fn new() -> Self {
        Self {
            object_proto: None,
            array_proto: None,
            function_proto: None,
            prim_proto: None,
            bool_proto: None,
        }
    }

    /// Sets the object prototype.
    pub fn with_object_proto(mut self, proto: Object<'static, T>) -> Self {
        self.object_proto = Some(proto);
        self
    }

    /// Sets the array / tuple prototype.
    pub fn with_array_proto(mut self, proto: Object<'static, T>) -> Self {
        self.array_proto = Some(proto);
        self
    }

    /// Sets the function prototype.
    pub fn with_function_proto(mut self, proto: Object<'static, T>) -> Self {
        self.function_proto = Some(proto);
        self
    }

    /// Sets the primitive value prototype.
    pub fn with_primitive_proto(mut self, proto: Object<'static, T>) -> Self {
        self.prim_proto = Some(proto);
        self
    }

    /// Sets the Boolean value prototype.
    pub fn with_bool_proto(mut self, proto: Object<'static, T>) -> Self {
        self.bool_proto = Some(proto);
        self
    }

    pub(crate) fn get(&self, value_type: ValueType) -> Option<&Object<'static, T>> {
        match value_type {
            ValueType::Object => self.object_proto.as_ref(),
            ValueType::Tuple(_) | ValueType::Array => self.array_proto.as_ref(),
            ValueType::Function => self.function_proto.as_ref(),
            ValueType::Prim => self.prim_proto.as_ref(),
            ValueType::Bool => self.bool_proto.as_ref(),
            _ => None,
        }
    }

    fn merge_proto(
        old_proto: &mut Option<Object<'static, T>>,
        new_proto: Option<Object<'static, T>>,
    ) {
        if let Some(old_proto) = old_proto {
            if let Some(new_proto) = new_proto {
                old_proto.extend(new_proto);
            }
        } else {
            *old_proto = new_proto;
        }
    }
}

impl<T> ops::AddAssign for StandardPrototypes<T> {
    fn add_assign(&mut self, rhs: Self) {
        Self::merge_proto(&mut self.object_proto, rhs.object_proto);
        Self::merge_proto(&mut self.array_proto, rhs.array_proto);
        Self::merge_proto(&mut self.function_proto, rhs.function_proto);
        Self::merge_proto(&mut self.prim_proto, rhs.prim_proto);
        Self::merge_proto(&mut self.bool_proto, rhs.bool_proto);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fns;

    #[test]
    fn object_to_string() {
        let mut obj = Object::<f32>::default();
        let obj_string = obj.to_string();
        assert_eq!(obj_string, "#{ }");

        obj.insert("x", Value::Prim(3.0));
        let obj_string = obj.to_string();
        assert_eq!(obj_string, "#{ x: 3 }");

        obj.insert("y", Value::Prim(4.0));
        let obj_string = obj.to_string();
        assert!(
            obj_string == "#{ x: 3, y: 4 }" || obj_string == "#{ y: 4, x: 3 }",
            "Unexpected obj_string: {}",
            obj_string
        );
    }

    #[test]
    fn merging_prototypes() {
        let array_proto = vec![("fold", Value::native_fn(fns::Fold))]
            .into_iter()
            .collect();
        let mut prototypes = StandardPrototypes::<f32>::new().with_array_proto(array_proto);

        let new_array_proto = vec![("len", Value::native_fn(fns::Len))]
            .into_iter()
            .collect();
        let object_proto = vec![("len", Value::native_fn(fns::Len))]
            .into_iter()
            .collect();
        let new_prototypes = StandardPrototypes::<f32>::new()
            .with_array_proto(new_array_proto)
            .with_object_proto(object_proto);

        prototypes += new_prototypes;
        assert!(prototypes.object_proto.unwrap()["len"].is_function());
        let array_proto = prototypes.array_proto.unwrap();
        assert!(array_proto["len"].is_function());
        assert!(array_proto["fold"].is_function());
    }
}
