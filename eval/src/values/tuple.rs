//! `Tuple` and tightly related types.

use core::{fmt, iter::FromIterator, ops};

use crate::{
    alloc::{vec, Vec},
    Prototype, Value,
};
use arithmetic_parser::StripCode;

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
pub struct Tuple<'a, T> {
    elements: Vec<Value<'a, T>>,
    prototype: Option<Prototype<'a, T>>,
}

impl<'a, T> Default for Tuple<'a, T> {
    fn default() -> Self {
        Self::void()
    }
}

impl<'a, T> From<Tuple<'a, T>> for Value<'a, T> {
    fn from(tuple: Tuple<'a, T>) -> Self {
        Self::Tuple(tuple)
    }
}

impl<'a, T> From<Vec<Value<'a, T>>> for Tuple<'a, T> {
    fn from(elements: Vec<Value<'a, T>>) -> Self {
        Self {
            elements,
            prototype: None,
        }
    }
}

impl<'a, T> From<Tuple<'a, T>> for Vec<Value<'a, T>> {
    fn from(tuple: Tuple<'a, T>) -> Self {
        tuple.elements
    }
}

impl<T: fmt::Display> fmt::Display for Tuple<'_, T> {
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

impl<'a, T> Tuple<'a, T> {
    /// Creates a new empty tuple (aka a void value).
    pub const fn void() -> Self {
        Self {
            elements: Vec::new(),
            prototype: None,
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
    pub fn iter(&self) -> impl Iterator<Item = &Value<'a, T>> + '_ {
        self.elements.iter()
    }

    /// Pushes a value to the end of this tuple.
    pub fn push(&mut self, value: impl Into<Value<'a, T>>) {
        self.elements.push(value.into());
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

impl<T: 'static + Clone> StripCode for Tuple<'_, T> {
    type Stripped = Tuple<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        Tuple {
            elements: self.elements.into_iter().map(Value::strip_code).collect(),
            prototype: self.prototype.map(StripCode::strip_code),
        }
    }
}

impl<'a, T> ops::Index<usize> for Tuple<'a, T> {
    type Output = Value<'a, T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl<'a, T> IntoIterator for Tuple<'a, T> {
    type Item = Value<'a, T>;
    /// Iterator type should be considered an implementation detail.
    type IntoIter = vec::IntoIter<Value<'a, T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

impl<'r, 'a, T> IntoIterator for &'r Tuple<'a, T> {
    type Item = &'r Value<'a, T>;
    /// Iterator type should be considered an implementation detail.
    type IntoIter = core::slice::Iter<'r, Value<'a, T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.iter()
    }
}

impl<'a, T, V> FromIterator<V> for Tuple<'a, T>
where
    V: Into<Value<'a, T>>,
{
    fn from_iter<I: IntoIterator<Item = V>>(iter: I) -> Self {
        Self {
            elements: iter.into_iter().map(Into::into).collect(),
            prototype: None,
        }
    }
}

impl<'a, T, V> Extend<V> for Tuple<'a, T>
where
    V: Into<Value<'a, T>>,
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
