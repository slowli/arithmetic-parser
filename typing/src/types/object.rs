//! Object types.

use std::{collections::HashMap, fmt, iter::FromIterator};

use crate::{PrimitiveType, Type};

/// Object type: named fields with heterogeneous types.
#[derive(Debug, Clone, PartialEq)]
pub struct Object<Prim: PrimitiveType> {
    fields: HashMap<String, Type<Prim>>,
}

impl<Prim: PrimitiveType> Default for Object<Prim> {
    fn default() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }
}

impl<Prim, S, V> FromIterator<(S, V)> for Object<Prim>
where
    Prim: PrimitiveType,
    S: Into<String>,
    V: Into<Type<Prim>>,
{
    fn from_iter<T: IntoIterator<Item = (S, V)>>(iter: T) -> Self {
        Self {
            fields: iter
                .into_iter()
                .map(|(name, ty)| (name.into(), ty.into()))
                .collect(),
        }
    }
}

impl<Prim: PrimitiveType> fmt::Display for Object<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("{")?;
        for (i, (name, ty)) in self.fields.iter().enumerate() {
            write!(formatter, " {}: {}", name, ty)?;
            if i + 1 < self.fields.len() {
                formatter.write_str(",")?;
            }
        }
        formatter.write_str(" }")
    }
}

impl<Prim: PrimitiveType> Object<Prim> {
    /// Creates an empty object.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns type of a field with the specified `name`.
    pub fn field(&self, name: &str) -> Option<&Type<Prim>> {
        self.fields.get(name)
    }

    /// Iterates over fields in this object.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Type<Prim>)> + '_ {
        self.fields.iter().map(|(name, ty)| (name.as_str(), ty))
    }

    /// Iterates over field names in this object.
    pub fn field_names(&self) -> impl Iterator<Item = &str> + '_ {
        self.fields.keys().map(String::as_str)
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut Type<Prim>)> + '_ {
        self.fields.iter_mut().map(|(name, ty)| (name.as_str(), ty))
    }

    pub(crate) fn is_concrete(&self) -> bool {
        self.fields.values().all(Type::is_concrete)
    }
}
