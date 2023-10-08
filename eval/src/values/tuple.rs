//! `Tuple` and tightly related types.

use core::{fmt, iter::FromIterator, ops};

use crate::{
    alloc::{vec, Vec},
    Value,
};

/// Tuple of zero or more values.
///
/// A tuple is similar to a [`Vec`] with [`Value`] elements and can be converted from / to it.
/// It is possible to iterate over elements, index them, etc.
///
/// # Examples
///
/// ```
/// # use arithmetic_eval::{Tuple, Value};
/// let mut tuple = Tuple::<u32>::default();
/// tuple.push(Value::Prim(3));
/// tuple.push(Value::Prim(5));
/// assert_eq!(tuple.len(), 2);
/// assert_eq!(tuple[1], Value::Prim(5));
/// assert!(tuple.iter().all(|val| !val.is_void()));
///
/// // `Tuple` implements `FromIterator` / `Extend`.
/// let mut other_tuple: Tuple<u32> = (0..=2).map(Value::Prim).collect();
/// other_tuple.extend(tuple);
/// assert_eq!(other_tuple.len(), 5);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Tuple<T> {
    elements: Vec<Value<T>>,
}

impl<T> Default for Tuple<T> {
    fn default() -> Self {
        Self::void()
    }
}

impl<T> From<Tuple<T>> for Value<T> {
    fn from(tuple: Tuple<T>) -> Self {
        Self::Tuple(tuple)
    }
}

impl<T> From<Vec<Value<T>>> for Tuple<T> {
    fn from(elements: Vec<Value<T>>) -> Self {
        Self { elements }
    }
}

impl<T> From<Tuple<T>> for Vec<Value<T>> {
    fn from(tuple: Tuple<T>) -> Self {
        tuple.elements
    }
}

impl<T: fmt::Display> fmt::Display for Tuple<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "(")?;
        for (i, element) in self.iter().enumerate() {
            fmt::Display::fmt(element, formatter)?;
            if i + 1 < self.len() {
                formatter.write_str(", ")?;
            } else if self.len() == 1 {
                formatter.write_str(",")?; // terminal ',' to distinguish 1-element tuples
            }
        }
        write!(formatter, ")")
    }
}

impl<T> Tuple<T> {
    /// Creates a new empty tuple (aka a void value).
    pub const fn void() -> Self {
        Self {
            elements: Vec::new(),
        }
    }

    /// Returns the number of elements in this tuple.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Checks if this tuple is empty (has no elements).
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Iterates over the elements in this tuple.
    pub fn iter(&self) -> impl Iterator<Item = &Value<T>> + '_ {
        self.elements.iter()
    }

    /// Pushes a value to the end of this tuple.
    pub fn push(&mut self, value: impl Into<Value<T>>) {
        self.elements.push(value.into());
    }
}

impl<T> ops::Index<usize> for Tuple<T> {
    type Output = Value<T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl<T> IntoIterator for Tuple<T> {
    type Item = Value<T>;
    /// Iterator type should be considered an implementation detail.
    type IntoIter = vec::IntoIter<Value<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

impl<'r, T> IntoIterator for &'r Tuple<T> {
    type Item = &'r Value<T>;
    /// Iterator type should be considered an implementation detail.
    type IntoIter = core::slice::Iter<'r, Value<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.iter()
    }
}

impl<T, V> FromIterator<V> for Tuple<T>
where
    V: Into<Value<T>>,
{
    fn from_iter<I: IntoIterator<Item = V>>(iter: I) -> Self {
        Self {
            elements: iter.into_iter().map(Into::into).collect(),
        }
    }
}

impl<T, V> Extend<V> for Tuple<T>
where
    V: Into<Value<T>>,
{
    fn extend<I: IntoIterator<Item = V>>(&mut self, iter: I) {
        let new_elements = iter.into_iter().map(Into::into);
        self.elements.extend(new_elements);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuple_to_string() {
        let mut tuple = Tuple::<f32>::default();
        assert_eq!(tuple.to_string(), "()");

        tuple.push(Value::Prim(3.0));
        assert_eq!(tuple.to_string(), "(3,)");

        tuple.push(Value::Prim(4.0));
        assert_eq!(tuple.to_string(), "(3, 4)");
    }
}
