//! Object types.

use std::{
    collections::{HashMap, HashSet},
    fmt, ops,
};

use crate::{
    arith::Substitutions,
    error::{ErrorKind, OpErrors},
    DynConstraints, PrimitiveType, Type,
};

/// Object type: a collection of named fields with heterogeneous types.
///
/// # Notation
///
/// Object types are denoted using a brace notation such as `{ x: Num, y: [(Num, 'T)] }`.
/// Here, `x` and `y` are field names, and `Num` / `[(Num, 'T)]` are types of the corresponding
/// object fields.
///
/// # As constraint
///
/// Object types are *exact*; their extensions cannot be unified with the original types.
/// For example, if a function argument is `{ x: Num, y: Num }`,
/// the function cannot be called with an arg of type `{ x: Num, y: Num, z: Num }`:
///
/// ```
/// # use arithmetic_parser::grammars::{Parse, F32Grammar};
/// # use arithmetic_typing::{error::ErrorKind, Annotated, TypeEnvironment};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let code = "
///     sum_coords = |pt: { x: Num, y: Num }| pt.x + pt.y;
///     sum_coords(#{ x: 3, y: 4 }); // OK
///     sum_coords(#{ x: 3, y: 4, z: 5 }); // fails
/// ";
/// let ast = Annotated::<F32Grammar>::parse_statements(code)?;
/// let err = TypeEnvironment::new().process_statements(&ast).unwrap_err();
/// # assert_eq!(err.len(), 1);
/// let err = err.iter().next().unwrap();
/// assert_matches!(err.kind(), ErrorKind::FieldsMismatch { .. });
/// # Ok(())
/// # }
/// ```
///
/// To bridge this gap, objects can be used as a constraint on types, similarly to [`Constraint`]s.
/// As a constraint, an object specifies *necessary* fields, which can be arbitrarily extended.
///
/// The type inference algorithm uses object constraints, not concrete object types whenever
/// possible:
///
/// ```
/// # use arithmetic_parser::grammars::{Parse, F32Grammar};
/// # use arithmetic_typing::{error::ErrorKind, Annotated, TypeEnvironment};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let code = "
///     sum_coords = |pt| pt.x + pt.y;
///     sum_coords(#{ x: 3, y: 4 }); // OK
///     sum_coords(#{ x: 3, y: 4, z: 5 }); // also OK
/// ";
/// let ast = Annotated::<F32Grammar>::parse_statements(code)?;
/// let mut env = TypeEnvironment::new();
/// env.process_statements(&ast)?;
/// assert_eq!(
///     env["sum_coords"].to_string(),
///     "for<'T: { x: 'U, y: 'U }, 'U: Ops> ('T) -> 'U"
/// );
/// # Ok(())
/// # }
/// ```
///
/// Note that the object constraint in this case refers to another type param, which is
/// constrained on its own!
///
/// [`Constraint`]: crate::arith::Constraint
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

impl<Prim, S, V, const N: usize> From<[(S, V); N]> for Object<Prim>
where
    Prim: PrimitiveType,
    S: Into<String>,
    V: Into<Type<Prim>>,
{
    fn from(entries: [(S, V); N]) -> Self {
        Self {
            fields: entries
                .into_iter()
                .map(|(name, ty)| (name.into(), ty.into()))
                .collect(),
        }
    }
}

impl<Prim: PrimitiveType> fmt::Display for Object<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut sorted_fields: Vec<_> = self.fields.iter().collect();
        sorted_fields.sort_unstable_by_key(|(name, _)| *name);

        formatter.write_str("{")?;
        for (i, (name, ty)) in sorted_fields.into_iter().enumerate() {
            write!(formatter, " {name}: {ty}")?;
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

    pub(crate) fn from_map(fields: HashMap<String, Type<Prim>>) -> Self {
        Self { fields }
    }

    /// Returns type of a field with the specified `name`.
    pub fn get(&self, name: &str) -> Option<&Type<Prim>> {
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

    /// Converts this object into a corresponding dynamic constraint.
    pub fn into_dyn(self) -> Type<Prim> {
        Type::Dyn(DynConstraints::from(self))
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut Type<Prim>)> + '_ {
        self.fields.iter_mut().map(|(name, ty)| (name.as_str(), ty))
    }

    pub(crate) fn is_concrete(&self) -> bool {
        self.fields.values().all(Type::is_concrete)
    }

    pub(crate) fn extend_from(
        &mut self,
        other: Self,
        substitutions: &mut Substitutions<Prim>,
        mut errors: OpErrors<'_, Prim>,
    ) {
        for (field_name, ty) in other.fields {
            if let Some(this_field) = self.fields.get(&field_name) {
                substitutions.unify(this_field, &ty, errors.join_path(field_name.as_str()));
            } else {
                self.fields.insert(field_name, ty);
            }
        }
    }

    pub(crate) fn apply_as_constraint(
        &self,
        ty: &Type<Prim>,
        substitutions: &mut Substitutions<Prim>,
        mut errors: OpErrors<'_, Prim>,
    ) {
        let resolved_ty = if let Type::Var(var) = ty {
            debug_assert!(var.is_free());
            substitutions.insert_obj_constraint(var.index(), self, errors.by_ref());
            substitutions.fast_resolve(ty)
        } else {
            ty
        };

        match resolved_ty {
            Type::Object(rhs) => {
                self.constraint_object(&rhs.clone(), substitutions, errors);
            }
            Type::Dyn(constraints) => {
                if let Some(object) = constraints.inner.object.clone() {
                    self.constraint_object(&object, substitutions, errors);
                } else {
                    errors.push(ErrorKind::CannotAccessFields);
                }
            }
            Type::Any | Type::Var(_) => { /* OK */ }
            _ => errors.push(ErrorKind::CannotAccessFields),
        }
    }

    /// Places an object constraint encoded in `lhs` on a (concrete) object in `rhs`.
    fn constraint_object(
        &self,
        rhs: &Object<Prim>,
        substitutions: &mut Substitutions<Prim>,
        mut errors: OpErrors<'_, Prim>,
    ) {
        let mut missing_fields = HashSet::new();
        for (field_name, lhs_ty) in self.iter() {
            if let Some(rhs_ty) = rhs.get(field_name) {
                substitutions.unify(lhs_ty, rhs_ty, errors.join_path(field_name));
            } else {
                missing_fields.insert(field_name.to_owned());
            }
        }

        if !missing_fields.is_empty() {
            errors.push(ErrorKind::MissingFields {
                fields: missing_fields,
                available_fields: rhs.field_names().map(String::from).collect(),
            });
        }
    }
}

impl<Prim: PrimitiveType> ops::Index<&str> for Object<Prim> {
    type Output = Type<Prim>;

    fn index(&self, field_name: &str) -> &Self::Output {
        self.get(field_name).unwrap_or_else(|| {
            panic!("Object type does not contain field `{field_name}`");
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::Num;

    use assert_matches::assert_matches;

    fn get_err(errors: OpErrors<'_, Num>) -> ErrorKind<Num> {
        let mut errors = errors.into_vec();
        assert_eq!(errors.len(), 1, "{errors:?}");
        errors.pop().unwrap()
    }

    #[test]
    fn placing_obj_constraint() {
        let lhs: Object<Num> = Object::from([("x", Type::NUM)]);
        let mut substitutions = Substitutions::default();
        let mut errors = OpErrors::new();
        lhs.constraint_object(&lhs, &mut substitutions, errors.by_ref());
        assert!(errors.into_vec().is_empty());

        let var_rhs = Object::from([("x", Type::free_var(0))]);
        let mut errors = OpErrors::new();
        lhs.constraint_object(&var_rhs, &mut substitutions, errors.by_ref());
        assert!(errors.into_vec().is_empty());
        assert_eq!(*substitutions.fast_resolve(&Type::free_var(0)), Type::NUM);

        // Extra fields in RHS are fine.
        let extra_rhs = Object::from([("x", Type::free_var(1)), ("y", Type::BOOL)]);
        let mut errors = OpErrors::new();
        lhs.constraint_object(&extra_rhs, &mut substitutions, errors.by_ref());
        assert!(errors.into_vec().is_empty());
        assert_eq!(*substitutions.fast_resolve(&Type::free_var(1)), Type::NUM);

        let missing_field_rhs = Object::from([("y", Type::free_var(2))]);
        let mut errors = OpErrors::new();
        lhs.constraint_object(&missing_field_rhs, &mut substitutions, errors.by_ref());
        assert_matches!(
            get_err(errors),
            ErrorKind::MissingFields { fields, available_fields }
                if fields.len() == 1 && fields.contains("x") &&
                available_fields.len() == 1 && available_fields.contains("y")
        );

        let incompatible_field_rhs = Object::from([("x", Type::BOOL)]);
        let mut errors = OpErrors::new();
        lhs.constraint_object(&incompatible_field_rhs, &mut substitutions, errors.by_ref());
        assert_matches!(
            get_err(errors),
            ErrorKind::TypeMismatch(lhs, rhs) if lhs == Type::NUM && rhs == Type::BOOL
        );
    }
}
