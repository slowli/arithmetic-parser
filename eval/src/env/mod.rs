//! [`Environment`] and other types related to [`Value`] collections.

use hashbrown::{hash_map, HashMap};

use core::{iter, ops};

mod variable_map;
pub use self::variable_map::{Assertions, Comparisons, Prelude};

use crate::{
    alloc::{Rc, String, ToOwned},
    arith::{OrdArithmetic, StdArithmetic},
    exec::Operations,
    fns,
    values::StandardPrototypes,
    NativeFn, Prototype, PrototypeField, Value, ValueType,
};

/// Environment containing named `Value`s.
///
/// Note that the environment implements the [`Index`](ops::Index) trait, which allows to eloquently
/// access or modify environment. Similarly, [`IntoIterator`] / [`Extend`] traits
/// allow to construct environments.
///
/// # Examples
///
/// ```
/// use arithmetic_eval::{env::{Comparisons, Prelude}, Environment, Value};
///
/// // Load environment from the standard containers.
/// let mut env = Environment::<f64>::new();
/// env.extend(Prelude::vars().chain(Comparisons::vars()));
/// // Add a custom variable for a good measure.
/// env.insert("x", Value::Prim(1.0));
///
/// assert_eq!(env["true"], Value::Bool(true));
/// assert_eq!(env["x"], Value::Prim(1.0));
/// for (name, value) in &env {
///     println!("{} -> {:?}", name, value);
/// }
///
/// // It's possible to base an environment on other env, as well.
/// let mut other_env = Environment::new();
/// other_env.extend(
///     env.into_iter().filter(|(_, val)| val.is_function()),
/// );
/// assert!(other_env.get("x").is_none());
/// ```
///
/// Extending [`Prototype`](crate::Prototype)s for standard types:
///
/// ```
/// # use arithmetic_eval::{
/// #     fns, Environment, ExecutableModule, PrototypeField, Value, env::Prelude,
/// # };
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # fn main() -> anyhow::Result<()> {
/// let prototypes = vec![
///     (PrototypeField::prim("abs"), Value::wrapped_fn(f32::abs)),
///     (PrototypeField::prim("sin"), Value::wrapped_fn(f32::sin)),
///     (PrototypeField::array("len"), Value::native_fn(fns::Len)),
/// ];
/// let mut env = Environment::new();
/// env.extend(Prelude::prototypes().chain(prototypes));
/// // ^ also insert "standard" prototypes
///
/// let program = r#"
///     array = (1, -2, 3).map(|x| x.abs());
///     array.len() == 3 && array.1 > 0
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_proto", &program)?;
/// assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Environment<'a, T> {
    variables: HashMap<String, Value<'a, T>>,
    arithmetic: Rc<dyn OrdArithmetic<T>>,
    prototypes: StandardPrototypes<T>,
}

impl<T> Default for Environment<'_, T>
where
    StdArithmetic: OrdArithmetic<T>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Clone for Environment<'_, T> {
    fn clone(&self) -> Self {
        Self {
            variables: self.variables.clone(),
            arithmetic: Rc::clone(&self.arithmetic),
            prototypes: self.prototypes.clone(),
        }
    }
}

/// Compares environments by variables and prototypes; arithmetics are ignored.
impl<T: PartialEq> PartialEq for Environment<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.variables == other.variables && self.prototypes == other.prototypes
    }
}

impl<'a, T> Environment<'a, T>
where
    StdArithmetic: OrdArithmetic<T>,
{
    /// Creates a new environment.
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            arithmetic: Rc::new(StdArithmetic),
            prototypes: StandardPrototypes::new(),
        }
    }
}

impl<'a, T> Environment<'a, T> {
    /// Creates an environment with the specified arithmetic.
    pub fn with_arithmetic<A>(arithmetic: A) -> Self
    where
        A: OrdArithmetic<T> + 'static,
    {
        Self {
            variables: HashMap::new(),
            arithmetic: Rc::new(arithmetic),
            prototypes: StandardPrototypes::new(),
        }
    }

    pub(crate) fn operations(&self) -> Operations<'_, T> {
        Operations::new(&*self.arithmetic, Some(&self.prototypes))
    }

    /// Gets a variable by name.
    pub fn get(&self, name: &str) -> Option<&Value<'a, T>> {
        self.variables.get(name)
    }

    /// Checks if this environment contains a variable with the specified name.
    pub fn contains(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Iterates over variables.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value<'a, T>)> + '_ {
        self.variables
            .iter()
            .map(|(name, value)| (name.as_str(), value))
    }

    /// Returns a [`Prototype`] for a standard type, or `None` if it is undefined.
    pub fn prototype(&self, ty: ValueType) -> Option<&Prototype<'static, T>> {
        self.prototypes.get(ty)
    }

    /// Inserts a variable with the specified name.
    pub fn insert(&mut self, name: &str, value: Value<'a, T>) -> &mut Self {
        self.variables.insert(name.to_owned(), value);
        self
    }

    /// Inserts a native function with the specified name.
    pub fn insert_native_fn(
        &mut self,
        name: &str,
        native_fn: impl NativeFn<T> + 'static,
    ) -> &mut Self {
        self.insert(name, Value::native_fn(native_fn))
    }

    /// Inserts a [wrapped function](fns::FnWrapper) with the specified name.
    ///
    /// Calling this method is equivalent to [`wrap`](fns::wrap)ping a function and calling
    /// [`insert_native_fn()`](Self::insert_native_fn) on it. Thanks to type inference magic,
    /// the Rust compiler will usually be able to extract the `Args` type param
    /// from the function definition, provided that type of function arguments and its return type
    /// are defined explicitly or can be unequivocally inferred from the declaration.
    pub fn insert_wrapped_fn<Args, F>(&mut self, name: &str, fn_to_wrap: F) -> &mut Self
    where
        fns::FnWrapper<Args, F>: NativeFn<T> + 'static,
    {
        let wrapped = fns::wrap::<Args, _>(fn_to_wrap);
        self.insert(name, Value::native_fn(wrapped))
    }
}

impl<T: Clone> Environment<'_, T> {
    /// Inserts a field into a [`Prototype`](crate::Prototype) for one of the standard value types.
    ///
    /// Use the [`Extend`] implementation to modify prototypes more efficiently in case of batch
    /// changes.
    pub fn insert_prototype(
        &mut self,
        field: PrototypeField,
        value: Value<'static, T>,
    ) -> &mut Self {
        self.prototypes.insert(field, value);
        self
    }

    /// Inserts or updates prototypes for all standard types (if they are defined)
    /// as conventionally named variables. If any of such variables is already defined,
    /// it is replaced.
    ///
    /// The naming convention is as follows:
    ///
    /// - `Array`: array / [`Tuple`](crate::Tuple) prototype
    /// - `Object`: [`Object`](crate::Object) prototype
    /// - `Function`: [`Function`](crate::Function) prototype
    /// - `Bool`: Boolean value prototype
    ///
    /// Other prototypes (notably, the prototype for primitive values) are not exposed.
    pub fn insert_prototypes_as_vars(&mut self) -> &mut Self {
        const TYPES_WITH_NAMES: &[(ValueType, &str)] = &[
            (ValueType::Array, "Array"),
            (ValueType::Object, "Object"),
            (ValueType::Function, "Function"),
            (ValueType::Bool, "Bool"),
        ];

        for &(ty, name) in TYPES_WITH_NAMES {
            if let Some(proto) = self.prototype(ty).cloned() {
                self.insert(name, proto.into());
            }
        }
        self
    }
}

impl<'a, T> ops::Index<&str> for Environment<'a, T> {
    type Output = Value<'a, T>;

    fn index(&self, index: &str) -> &Self::Output {
        self.get(index)
            .unwrap_or_else(|| panic!("Variable `{}` is not defined", index))
    }
}

impl<'a, T> IntoIterator for Environment<'a, T> {
    type Item = (String, Value<'a, T>);
    type IntoIter = IntoIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.variables.into_iter(),
        }
    }
}

/// Result of converting `Environment` into an iterator.
#[derive(Debug)]
pub struct IntoIter<'a, T> {
    inner: hash_map::IntoIter<String, Value<'a, T>>,
}

impl<'a, T> Iterator for IntoIter<'a, T> {
    type Item = (String, Value<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> ExactSizeIterator for IntoIter<'_, T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'r, 'a, T> IntoIterator for &'r Environment<'a, T> {
    type Item = (&'r str, &'r Value<'a, T>);
    type IntoIter = Iter<'r, 'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            inner: self
                .variables
                .iter()
                .map(|(name, value)| (name.as_str(), value)),
        }
    }
}

type MapFn<'r, 'a, T> = fn((&'r String, &'r Value<'a, T>)) -> (&'r str, &'r Value<'a, T>);

/// Iterator over references of the `Environment` entries.
#[derive(Debug)]
pub struct Iter<'r, 'a, T> {
    inner: iter::Map<hash_map::Iter<'r, String, Value<'a, T>>, MapFn<'r, 'a, T>>,
}

impl<'r, 'a, T> Iterator for Iter<'r, 'a, T> {
    type Item = (&'r str, &'r Value<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> ExactSizeIterator for Iter<'_, '_, T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, T, S, V> Extend<(S, V)> for Environment<'a, T>
where
    S: Into<String>,
    V: Into<Value<'a, T>>,
{
    fn extend<I: IntoIterator<Item = (S, V)>>(&mut self, iter: I) {
        let variables = iter
            .into_iter()
            .map(|(var_name, value)| (var_name.into(), value.into()));
        self.variables.extend(variables);
    }
}

impl<T: Clone, V> Extend<(PrototypeField, V)> for Environment<'_, T>
where
    V: Into<Value<'static, T>>,
{
    fn extend<I: IntoIterator<Item = (PrototypeField, V)>>(&mut self, iter: I) {
        let prototype_fields = iter.into_iter().map(|(field, value)| (field, value.into()));
        self.prototypes.extend(prototype_fields);
    }
}
