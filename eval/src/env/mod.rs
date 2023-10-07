//! [`Environment`] and other types related to [`Value`] collections.

use core::{iter, ops};

mod variable_map;
pub use self::variable_map::{Assertions, Comparisons, Prelude};

use crate::{
    alloc::{hash_map, Arc, HashMap, String, ToOwned},
    arith::{OrdArithmetic, StdArithmetic},
    exec::Operations,
    fns, NativeFn, Value,
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
/// env.extend(Prelude::iter().chain(Comparisons::iter()));
/// // Add a custom variable for a good measure.
/// env.insert("x", Value::Prim(1.0));
///
/// assert_eq!(env["true"], Value::Bool(true));
/// assert_eq!(env["x"], Value::Prim(1.0));
/// for (name, value) in &env {
///     println!("{name} -> {value:?}");
/// }
///
/// // It's possible to base an environment on other env, as well.
/// let mut other_env = Environment::new();
/// other_env.extend(
///     env.into_iter().filter(|(_, val)| val.is_function()),
/// );
/// assert!(other_env.get("x").is_none());
/// ```
#[derive(Debug, Clone)]
pub struct Environment<T> {
    variables: HashMap<String, Value<T>>,
    arithmetic: Arc<dyn OrdArithmetic<T>>,
}

impl<T> Default for Environment<T>
where
    StdArithmetic: OrdArithmetic<T>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Compares environments by variables; arithmetics are ignored.
impl<T: PartialEq> PartialEq for Environment<T> {
    fn eq(&self, other: &Self) -> bool {
        self.variables == other.variables
    }
}

impl<T> Environment<T>
where
    StdArithmetic: OrdArithmetic<T>,
{
    /// Creates a new environment.
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            arithmetic: Arc::new(StdArithmetic),
        }
    }
}

impl<T> Environment<T> {
    /// Creates an environment with the specified arithmetic.
    pub fn with_arithmetic<A>(arithmetic: A) -> Self
    where
        A: OrdArithmetic<T> + 'static,
    {
        Self {
            variables: HashMap::new(),
            arithmetic: Arc::new(arithmetic),
        }
    }

    pub(crate) fn operations(&self) -> Operations<'_, T> {
        Operations::from(&*self.arithmetic)
    }

    /// Gets a variable by name.
    pub fn get(&self, name: &str) -> Option<&Value<T>> {
        self.variables.get(name)
    }

    /// Checks if this environment contains a variable with the specified name.
    pub fn contains(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Iterates over variables.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value<T>)> + '_ {
        self.variables
            .iter()
            .map(|(name, value)| (name.as_str(), value))
    }

    /// Inserts a variable with the specified name.
    pub fn insert(&mut self, name: &str, value: Value<T>) -> &mut Self {
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

impl<T> ops::Index<&str> for Environment<T> {
    type Output = Value<T>;

    fn index(&self, index: &str) -> &Self::Output {
        self.get(index)
            .unwrap_or_else(|| panic!("Variable `{index}` is not defined"))
    }
}

impl<T> IntoIterator for Environment<T> {
    type Item = (String, Value<T>);
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.variables.into_iter(),
        }
    }
}

/// Result of converting `Environment` into an iterator.
#[derive(Debug)]
pub struct IntoIter<T> {
    inner: hash_map::IntoIter<String, Value<T>>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = (String, Value<T>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'r, T> IntoIterator for &'r Environment<T> {
    type Item = (&'r str, &'r Value<T>);
    type IntoIter = Iter<'r, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            inner: self
                .variables
                .iter()
                .map(|(name, value)| (name.as_str(), value)),
        }
    }
}

type MapFn<'r, T> = fn((&'r String, &'r Value<T>)) -> (&'r str, &'r Value<T>);

/// Iterator over references of the `Environment` entries.
#[derive(Debug)]
pub struct Iter<'r, T> {
    inner: iter::Map<hash_map::Iter<'r, String, Value<T>>, MapFn<'r, T>>,
}

impl<'r, T> Iterator for Iter<'r, T> {
    type Item = (&'r str, &'r Value<T>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T, S, V> Extend<(S, V)> for Environment<T>
where
    S: Into<String>,
    V: Into<Value<T>>,
{
    fn extend<I: IntoIterator<Item = (S, V)>>(&mut self, iter: I) {
        let variables = iter
            .into_iter()
            .map(|(var_name, value)| (var_name.into(), value.into()));
        self.variables.extend(variables);
    }
}
