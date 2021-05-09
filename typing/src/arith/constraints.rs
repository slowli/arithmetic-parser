//! `TypeConstraints` and implementations.

use std::{collections::HashMap, fmt};

use crate::{
    error::{ErrorKind, OpErrors},
    PrimitiveType, Slice, Substitutions, Type,
};

/// Container for constraints that can be placed on type variables.
///
/// Constraints can be placed on [function](crate::FnType) type variables, and can be applied
/// to types in [`TypeArithmetic`] impls. For example, [`NumArithmetic`] places
/// a [linearity constraint](NumConstraints::Lin) on types involved in arithmetic ops.
///
/// The constraint mechanism is similar to trait constraints in Rust, but is much more limited:
///
/// - Constraints cannot be parametric (cf. parameters in traits, such `AsRef<_>`
///   or `Iterator<Item = _>`).
/// - Constraints are applied to types in separation; it is impossible to create a constraint
///   involving several type variables.
/// - Constraints cannot contradict each other.
///
/// # Implementation rules
///
/// Usually, this trait should be implemented with something akin to [`bitflags`].
///
/// [`bitflags`]: https://docs.rs/bitflags/
///
/// - [`Default`] must return a container with no restrictions.
/// - [`BitOrAssign`](ops::BitOrAssign) must perform the union of the provided constraints.
/// - [`Display`](fmt::Display) must display constraints in the form `Foo + Bar + Quux`,
///   where `Foo`, `Bar` and `Quux` are *primitive* constraints (i.e., ones not reduced
///   to a combination of other constraints). The primitive constraints must be represented
///   as identifiers (i.e., consist of alphanumeric chars and start with an alphabetic char
///   or `_`).
/// - [`FromStr`] must parse primitive constraints.
///
/// [`TypeArithmetic`]: crate::arith::TypeArithmetic
/// [`NumArithmetic`]: crate::arith::NumArithmetic
pub trait Constraint<Prim: PrimitiveType>: fmt::Display + Send + Sync + 'static {
    /// Applies these constraints to the provided `ty`pe. Returns an error if the type
    /// contradicts the constraints.
    ///
    /// A typical implementation will use `substitutions` to
    /// [place constraints on type vars](Substitutions::insert_constraints()), e.g.,
    /// by recursively traversing and resolving the provided type.
    fn apply(
        &self,
        ty: &Type<Prim>,
        substitutions: &mut Substitutions<Prim>,
        errors: OpErrors<'_, Prim>,
    );

    /// Clones this constraint into a `Box`.
    ///
    /// This method should be implemented by implementing [`Clone`] and boxing its output.
    fn clone_boxed(&self) -> Box<dyn Constraint<Prim>>;
}

impl<Prim: PrimitiveType> fmt::Debug for dyn Constraint<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter
            .debug_tuple("dyn Constraint")
            .field(&self.to_string())
            .finish()
    }
}

impl<Prim: PrimitiveType> Clone for Box<dyn Constraint<Prim>> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

impl<Prim: PrimitiveType> Constraint<Prim> for Box<dyn Constraint<Prim>> {
    fn apply(
        &self,
        ty: &Type<Prim>,
        substitutions: &mut Substitutions<Prim>,
        errors: OpErrors<'_, Prim>,
    ) {
        (**self).apply(ty, substitutions, errors);
    }

    fn clone_boxed(&self) -> Box<dyn Constraint<Prim>> {
        (**self).clone_boxed()
    }
}

/// Numeric constraints. In particular, this is [`TypeConstraints`] associated
/// with the [`Num`](crate::Num) literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumConstraints {
    /// Type can be subject to unary `-` and can participate in `T op Num` / `Num op T` operations.
    ///
    /// Defined recursively as linear primitive types and tuples consisting of `Lin` types.
    Lin,
    /// Type can participate in binary arithmetic ops (`T op T`).
    ///
    /// Defined as a subset of `Lin` types without dynamically sized slices and
    /// any types containing dynamically sized slices.
    Ops,
}

impl fmt::Display for NumConstraints {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Lin => "Lin",
            Self::Ops => "Ops",
        })
    }
}

/// Primitive type which supports a notion of *linearity*. Linear types are types that
/// can be used in arithmetic ops.
pub trait LinearType: PrimitiveType {
    /// Returns `true` iff this type is linear.
    fn is_linear(&self) -> bool;
}

impl<Prim: LinearType> Constraint<Prim> for NumConstraints {
    // TODO: extract common logic for it to be reusable?
    fn apply(
        &self,
        ty: &Type<Prim>,
        substitutions: &mut Substitutions<Prim>,
        mut errors: OpErrors<'_, Prim>,
    ) {
        let resolved_ty = if let Type::Var(var) = ty {
            debug_assert!(var.is_free());
            substitutions.insert_constraint(var.index(), self);
            substitutions.fast_resolve(ty)
        } else {
            ty
        };

        match resolved_ty {
            // `Var`s are taken care of previously. `Any` satisfies any constraints.
            Type::Any(_) | Type::Var(_) => {}
            Type::Prim(lit) if lit.is_linear() => {}

            Type::Function(_) | Type::Prim(_) => {
                errors.push(ErrorKind::failed_constraint(resolved_ty.clone(), *self));
            }

            Type::Tuple(tuple) => {
                let tuple = tuple.clone();

                if *self == Self::Ops {
                    let middle_len = tuple.parts().1.map(Slice::len);
                    if let Some(middle_len) = middle_len {
                        if let Err(err) = substitutions.apply_static_len(middle_len) {
                            errors.push(err);
                        }
                    }
                }

                for (i, element) in tuple.element_types() {
                    self.apply(element, substitutions, errors.with_location(i));
                }
            }
        }
    }

    fn clone_boxed(&self) -> Box<dyn Constraint<Prim>> {
        Box::new(*self)
    }
}

/// Set of [`Constraint`]s.
#[derive(Debug, Clone)]
pub struct ConstraintSet<Prim: PrimitiveType> {
    inner: HashMap<String, Box<dyn Constraint<Prim>>>,
}

impl<Prim: PrimitiveType> Default for ConstraintSet<Prim> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Prim: PrimitiveType> PartialEq for ConstraintSet<Prim> {
    fn eq(&self, other: &Self) -> bool {
        // TODO: is key ordering stable?
        self.inner.keys().eq(other.inner.keys())
    }
}

impl<Prim: PrimitiveType> fmt::Display for ConstraintSet<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let len = self.inner.len();
        for (i, constraint) in self.inner.values().enumerate() {
            fmt::Display::fmt(constraint, formatter)?;
            if i + 1 < len {
                formatter.write_str(" + ")?;
            }
        }
        Ok(())
    }
}

impl<Prim: PrimitiveType> ConstraintSet<Prim> {
    /// Creates an empty set.
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Creates a set with one constraint.
    pub fn just(constraint: impl Constraint<Prim>) -> Self {
        let mut this = Self::new();
        this.insert(constraint);
        this
    }

    /// Checks if this constraint set is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Inserts a constraint into this set.
    pub fn insert(&mut self, constraint: impl Constraint<Prim>) {
        self.inner
            .insert(constraint.to_string(), Box::new(constraint));
    }

    pub(crate) fn get_by_name(&self, name: &str) -> Option<&dyn Constraint<Prim>> {
        self.inner.get(name).map(AsRef::as_ref)
    }

    /// Applies all constraints from this set.
    pub(crate) fn apply_all(
        &self,
        ty: &Type<Prim>,
        substitutions: &mut Substitutions<Prim>,
        mut errors: OpErrors<'_, Prim>,
    ) {
        for constraint in self.inner.values() {
            constraint.apply(ty, substitutions, errors.by_ref());
        }
    }
}
