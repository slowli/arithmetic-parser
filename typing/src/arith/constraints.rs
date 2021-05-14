//! `TypeConstraints` and implementations.

use std::{collections::HashMap, fmt, marker::PhantomData};

use crate::{
    error::{ErrorKind, OpErrors},
    Object, PrimitiveType, Slice, Substitutions, Type,
};

/// Constraint that can be placed on [`Type`]s.
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
/// - [`Display`](fmt::Display) must display constraint as an identifier (e.g., `Lin`).
///   The string presentation of a constraint must be unique within a [`PrimitiveType`];
///   it is used to identify constraints in a [`ConstraintSet`].
///
/// [`TypeArithmetic`]: crate::arith::TypeArithmetic
/// [`NumArithmetic`]: crate::arith::NumArithmetic
pub trait Constraint<Prim: PrimitiveType>: fmt::Display + Send + Sync + 'static {
    /// Applies these constraints to the provided `ty`pe. Returns an error if the type
    /// contradicts the constraints.
    ///
    /// A typical implementation will use `substitutions` to
    /// [place constraints on type vars](Substitutions::insert_constraint()).
    ///
    /// # Tips
    ///
    /// - You can use [`StructConstraint`] for typical use cases, which involve recursively
    ///   traversing `ty`.
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

/// Helper to define *structural* [`Constraint`]s, i.e., constraints recursively checking
/// the provided type.
///
/// The following logic is used to check whether a type satisfies the constraint:
///
/// - Primitive types satisfy the constraint iff the predicate provided in [`Self::new()`]
///   returns `true`.
/// - [`Type::Any`] always satisfies the constraint.
/// - Functional types never satisfy the constraint.
/// - A compound type (i.e., a tuple) satisfies the constraint iff all its items satisfy
///   the constraint.
/// - If [`Self::deny_dyn_slices()`] is set, tuple types need to have static length.
///
/// # Examples
///
/// Defining a constraint type using `StructConstraint`:
///
/// ```
/// # use arithmetic_typing::{
/// #     arith::{Constraint, StructConstraint}, error::OpErrors, PrimitiveType, Substitutions,
/// #     Type,
/// # };
/// # use std::fmt;
///
/// /// Constraint for hashable types.
/// #[derive(Clone, Copy)]
/// struct Hashed;
///
/// impl fmt::Display for Hashed {
///     fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
///         formatter.write_str("Hash")
///     }
/// }
///
/// impl<Prim: PrimitiveType> Constraint<Prim> for Hashed {
///     fn apply(
///         &self,
///         ty: &Type<Prim>,
///         substitutions: &mut Substitutions<Prim>,
///         errors: OpErrors<'_, Prim>,
///     ) {
///         // We can hash everything except for functions (and thus,
///         // types containing functions).
///         StructConstraint::new(*self, |_| true)
///             .apply(ty, substitutions, errors);
///     }
///
///     fn clone_boxed(&self) -> Box<dyn Constraint<Prim>> {
///         Box::new(*self)
///     }
/// }
/// ```
#[derive(Debug)]
pub struct StructConstraint<Prim: PrimitiveType, C, F> {
    constraint: C,
    predicate: F,
    deny_dyn_slices: bool,
    _prim: PhantomData<Prim>,
}

impl<Prim, C, F> StructConstraint<Prim, C, F>
where
    Prim: PrimitiveType,
    C: Constraint<Prim> + Clone,
    F: Fn(&Prim) -> bool,
{
    /// Creates a new helper.
    pub fn new(constraint: C, predicate: F) -> Self {
        Self {
            constraint,
            predicate,
            deny_dyn_slices: false,
            _prim: PhantomData,
        }
    }

    /// Marks that dynamically sized slices should fail the constraint check.
    pub fn deny_dyn_slices(mut self) -> Self {
        self.deny_dyn_slices = true;
        self
    }

    /// Applies the enclosed constraint structurally.
    #[allow(clippy::missing_panics_doc)] // triggered by `debug_assert`
    pub fn apply(
        &self,
        ty: &Type<Prim>,
        substitutions: &mut Substitutions<Prim>,
        mut errors: OpErrors<'_, Prim>,
    ) {
        let resolved_ty = if let Type::Var(var) = ty {
            debug_assert!(var.is_free());
            substitutions.insert_constraint(var.index(), &self.constraint, errors.by_ref());
            substitutions.fast_resolve(ty)
        } else {
            ty
        };

        match resolved_ty {
            // `Var`s are taken care of previously. `Any` satisfies any constraints.
            Type::Any(_) | Type::Var(_) => {}

            Type::Prim(lit) if (self.predicate)(lit) => {}

            Type::Prim(_) | Type::Function(_) => {
                errors.push(ErrorKind::failed_constraint(
                    resolved_ty.clone(),
                    self.constraint.clone(),
                ));
            }

            Type::Tuple(tuple) => {
                let tuple = tuple.clone();

                if self.deny_dyn_slices {
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

            Type::Object(obj) => {
                let obj = obj.clone();
                for (name, element) in obj.iter() {
                    self.apply(element, substitutions, errors.with_location(name));
                }
            }
        }
    }
}

/// Numeric [`Constraint`]s. In particular, they are applicable to the [`Num`](crate::Num) literal.
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
    fn apply(
        &self,
        ty: &Type<Prim>,
        substitutions: &mut Substitutions<Prim>,
        errors: OpErrors<'_, Prim>,
    ) {
        let helper = match self {
            Self::Lin => StructConstraint::new(*self, LinearType::is_linear),
            Self::Ops => StructConstraint::new(*self, LinearType::is_linear).deny_dyn_slices(),
        };
        helper.apply(ty, substitutions, errors);
    }

    fn clone_boxed(&self) -> Box<dyn Constraint<Prim>> {
        Box::new(*self)
    }
}

/// Set of [`Constraint`]s.
///
/// [`Display`](fmt::Display)ed as `Foo + Bar + Quux`, where `Foo`, `Bar` and `Quux` are
/// constraints in the set.
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
        if self.inner.len() == other.inner.len() {
            self.inner.keys().all(|key| other.inner.contains_key(key))
        } else {
            false
        }
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

/// Extended [`ConstraintSet`] that additionally supports object constraints.
#[derive(Debug, Clone, PartialEq)]
pub struct CompleteConstraints<Prim: PrimitiveType> {
    pub(crate) simple: ConstraintSet<Prim>,
    /// Object constraint. Stored as `Type` for convenience.
    pub(crate) object: Option<Type<Prim>>,
}

impl<Prim: PrimitiveType> Default for CompleteConstraints<Prim> {
    fn default() -> Self {
        Self {
            simple: ConstraintSet::new(),
            object: None,
        }
    }
}

impl<Prim: PrimitiveType> From<ConstraintSet<Prim>> for CompleteConstraints<Prim> {
    fn from(constraints: ConstraintSet<Prim>) -> Self {
        Self {
            simple: constraints,
            object: None,
        }
    }
}

impl<Prim: PrimitiveType> fmt::Display for CompleteConstraints<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.object, self.simple.is_empty()) {
            (Some(object), false) => write!(formatter, "{} + {}", object, self.simple),
            (Some(object), true) => fmt::Display::fmt(object, formatter),
            (None, _) => fmt::Display::fmt(&self.simple, formatter),
        }
    }
}

impl<Prim: PrimitiveType> CompleteConstraints<Prim> {
    /// Checks if this constraint set is empty.
    pub fn is_empty(&self) -> bool {
        self.object.is_none() && self.simple.is_empty()
    }

    /// Inserts a constraint into this set.
    pub(crate) fn insert(
        &mut self,
        constraint: impl Constraint<Prim>,
        substitutions: &mut Substitutions<Prim>,
        errors: OpErrors<'_, Prim>,
    ) {
        self.simple.insert(constraint);
        self.check_object_consistency(substitutions, errors);
    }

    /// Extends these constraints from `other`.
    pub(crate) fn extend(
        &mut self,
        other: Self,
        substitutions: &mut Substitutions<Prim>,
        errors: OpErrors<'_, Prim>,
    ) {
        self.simple.inner.extend(other.simple.inner);
        self.check_object_consistency(substitutions, errors);
    }

    /// Applies all constraints from this set.
    pub(crate) fn apply_all(
        &self,
        ty: &Type<Prim>,
        substitutions: &mut Substitutions<Prim>,
        mut errors: OpErrors<'_, Prim>,
    ) {
        self.simple.apply_all(ty, substitutions, errors.by_ref());
        if let Some(Type::Object(lhs)) = &self.object {
            lhs.apply_as_constraint(ty, substitutions, errors);
        }
    }

    /// Inserts an object constraint into this set.
    pub(crate) fn insert_obj_constraint(
        &mut self,
        object: Object<Prim>,
        substitutions: &mut Substitutions<Prim>,
        mut errors: OpErrors<'_, Prim>,
    ) {
        if let Some(Type::Object(existing_object)) = &mut self.object {
            existing_object.extend_from(object, substitutions, errors.by_ref());
        } else {
            self.object = Some(Type::Object(object));
        }
        self.check_object_consistency(substitutions, errors);
    }

    fn check_object_consistency(
        &self,
        substitutions: &mut Substitutions<Prim>,
        errors: OpErrors<'_, Prim>,
    ) {
        if let Some(object) = &self.object {
            self.simple.apply_all(object, substitutions, errors);
        }
    }
}
