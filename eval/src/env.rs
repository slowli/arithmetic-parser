//! Environment containing named `Value`s.

use hashbrown::{hash_map, HashMap};

use core::{
    iter::{self, FromIterator},
    ops,
};

use crate::{fns, NativeFn, Value};
use arithmetic_parser::Grammar;

/// Environment containing named `Value`s.
///
/// Note that the environment implements the [`Index`] trait, which allows to eloquently
/// access or modify environment. Similarly, [`IntoIterator`] / [`FromIterator`] / [`Extend`] traits
/// allow to construct environments.
///
/// [`Index`]: https://doc.rust-lang.org/std/ops/trait.Index.html
/// [`IntoIterator`]: https://doc.rust-lang.org/std/iter/trait.IntoIterator.html
/// [`FromIterator`]: https://doc.rust-lang.org/std/iter/trait.FromIterator.html
/// [`Extend`]: https://doc.rust-lang.org/std/iter/trait.Extend.html
///
/// # Examples
///
/// ```
/// use arithmetic_eval::{Environment, Comparisons, Prelude, Value};
/// use arithmetic_parser::grammars::F64Grammar;
///
/// // Load environment from the standard containers.
/// let mut env: Environment<'_, F64Grammar> = Prelude.iter()
///     .chain(Comparisons.iter())
///     // Add a custom variable for a good measure.
///     .chain(vec![("x", Value::Number(1.0))])
///     .collect();
///
/// assert_eq!(env["true"], Value::Bool(true));
/// assert_eq!(env["x"], Value::Number(1.0));
/// for (name, value) in &env {
///     println!("{} -> {:?}", name, value);
/// }
///
/// // It's possible to base an environment on other env, as well.
/// let other_env: Environment<'_, _> = env.into_iter()
///     .filter(|(_, val)| val.is_function())
///     .collect();
/// assert!(other_env.get("x").is_none());
/// ```
#[derive(Debug)]
pub struct Environment<'a, T: Grammar> {
    variables: HashMap<String, Value<'a, T>>,
}

impl<T: Grammar> Default for Environment<'_, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Grammar> Clone for Environment<'_, T> {
    fn clone(&self) -> Self {
        Self {
            variables: self.variables.clone(),
        }
    }
}

impl<T: Grammar> PartialEq for Environment<'_, T>
where
    T::Lit: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.variables == other.variables
    }
}

impl<'a, T: Grammar> Environment<'a, T> {
    /// Creates a new environment.
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    /// Gets a variable by name.
    pub fn get(&self, name: &str) -> Option<&Value<'a, T>> {
        self.variables.get(name)
    }

    /// Iterates over variables.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value<'a, T>)> + '_ {
        self.variables
            .iter()
            .map(|(name, value)| (name.as_str(), value))
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

    /// Inserts a [wrapped function] with the specified name.
    ///
    /// Calling this method is equivalent to [`wrap`]ping a function and calling
    /// [`insert_native_fn()`](#method.insert_native_fn) on it. Thanks to type inference magic,
    /// the Rust compiler will usually be able to extract the `Args` type param
    /// from the function definition, provided that type of function arguments and its return type
    /// are defined explicitly or can be unequivocally inferred from the declaration.
    ///
    /// [wrapped function]: fns/struct.FnWrapper.html
    /// [`wrap`]: fns/fn.wrap.html
    pub fn insert_wrapped_fn<Args, F>(&mut self, name: &str, fn_to_wrap: F) -> &mut Self
    where
        fns::FnWrapper<Args, F>: NativeFn<T> + 'static,
    {
        let wrapped = fns::wrap::<Args, _>(fn_to_wrap);
        self.insert(name, Value::native_fn(wrapped))
    }
}

impl<'a, T: Grammar> ops::Index<&str> for Environment<'a, T> {
    type Output = Value<'a, T>;

    fn index(&self, index: &str) -> &Self::Output {
        self.get(index)
            .unwrap_or_else(|| panic!("Variable `{}` is not defined", index))
    }
}

impl<'a, T: Grammar> IntoIterator for Environment<'a, T> {
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
pub struct IntoIter<'a, T: Grammar> {
    inner: hash_map::IntoIter<String, Value<'a, T>>,
}

impl<'a, T: Grammar> Iterator for IntoIter<'a, T> {
    type Item = (String, Value<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T: Grammar> ExactSizeIterator for IntoIter<'_, T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'r, 'a, T: Grammar> IntoIterator for &'r Environment<'a, T> {
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
pub struct Iter<'r, 'a, T: Grammar> {
    inner: iter::Map<hash_map::Iter<'r, String, Value<'a, T>>, MapFn<'r, 'a, T>>,
}

impl<'r, 'a, T: Grammar> Iterator for Iter<'r, 'a, T> {
    type Item = (&'r str, &'r Value<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T: Grammar> ExactSizeIterator for Iter<'_, '_, T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, T, S, V> FromIterator<(S, V)> for Environment<'a, T>
where
    T: Grammar,
    S: Into<String>,
    V: Into<Value<'a, T>>,
{
    fn from_iter<I: IntoIterator<Item = (S, V)>>(iter: I) -> Self {
        let variables = iter
            .into_iter()
            .map(|(var_name, value)| (var_name.into(), value.into()));
        Self {
            variables: HashMap::from_iter(variables),
        }
    }
}

impl<'a, T, S, V> Extend<(S, V)> for Environment<'a, T>
where
    T: Grammar,
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
