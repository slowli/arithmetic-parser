//! `TypeConstraints` and implementations.

use std::{collections::HashMap, fmt, marker::PhantomData};

use crate::{
    arith::Substitutions,
    error::{ErrorKind, OpErrors},
    visit::{self, Visit},
    Function, Object, PrimitiveType, Slice, Tuple, Type, TypeVar,
};

/// Constraint that can be placed on [`Type`]s.
///
/// Constraints can be placed on [`Function`] type variables, and can be applied
/// to types in [`TypeArithmetic`] impls. For example, [`NumArithmetic`] places
/// the [`Linearity`] constraint on types involved in arithmetic ops.
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
    /// Returns a [`Visit`]or that will be applied to constrained [`Type`]s. The visitor
    /// may use `substitutions` to resolve types and `errors` to record constraint errors.
    ///
    /// # Tips
    ///
    /// - You can use [`StructConstraint`] for typical use cases, which involve recursively
    ///   traversing `ty`.
    fn visitor<'r>(
        &self,
        substitutions: &'r mut Substitutions<Prim>,
        errors: OpErrors<'r, Prim>,
    ) -> Box<dyn Visit<Prim> + 'r>;

    /// Clones this constraint into a `Box`.
    ///
    /// This method should be implemented by implementing [`Clone`] and boxing its output.
    fn clone_boxed(&self) -> Box<dyn Constraint<Prim>>;
}

impl<Prim: PrimitiveType> fmt::Debug for dyn Constraint<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
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

/// Marker trait for object-safe constraints, i.e., constraints that can be included
/// into a [`DynConstraints`](crate::DynConstraints).
///
/// Object safety is similar to this notion in Rust. For a constraint `C` to be object-safe,
/// it should be the case that `dyn C` (the untagged union of all types implementing `C`)
/// implements `C`. As an example, this is the case for [`Linearity`], but is not the case
/// for [`Ops`]. Indeed, [`Ops`] requires the type to be addable to itself,
/// which would be impossible for `dyn Ops`.
pub trait ObjectSafeConstraint<Prim: PrimitiveType>: Constraint<Prim> {}

/// Helper to define *structural* [`Constraint`]s, i.e., constraints recursively checking
/// the provided type.
///
/// The following logic is used to check whether a type satisfies the constraint:
///
/// - Primitive types satisfy the constraint iff the predicate provided in [`Self::new()`]
///   returns `true`.
/// - [`Type::Any`] always satisfies the constraint.
/// - [`Type::Dyn`] types satisfy the constraint iff the [`Constraint`] wrapped by this helper
///   is present among [`DynConstraints`](crate::DynConstraints). Thus,
///   if the wrapped constraint is not [object-safe](ObjectSafeConstraint), it will not be satisfied
///   by any `Dyn` type.
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
/// #     arith::{Constraint, StructConstraint, Substitutions}, error::OpErrors, visit::Visit,
/// #     PrimitiveType, Type,
/// # };
/// # use std::fmt;
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
///     fn visitor<'r>(
///         &self,
///         substitutions: &'r mut Substitutions<Prim>,
///         errors: OpErrors<'r, Prim>,
///     ) -> Box<dyn Visit<Prim> + 'r> {
///         // We can hash everything except for functions (and thus,
///         // types containing functions).
///         StructConstraint::new(*self, |_| true)
///             .visitor(substitutions, errors)
///     }
///
///     fn clone_boxed(&self) -> Box<dyn Constraint<Prim>> {
///         Box::new(*self)
///     }
/// }
/// ```
#[derive(Debug)]
pub struct StructConstraint<Prim, C, F> {
    constraint: C,
    predicate: F,
    deny_dyn_slices: bool,
    _prim: PhantomData<Prim>,
}

impl<Prim, C, F> StructConstraint<Prim, C, F>
where
    Prim: PrimitiveType,
    C: Constraint<Prim> + Clone,
    F: Fn(&Prim) -> bool + 'static,
{
    /// Creates a new helper. `predicate` determines whether a particular primitive type
    /// should satisfy the `constraint`.
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

    /// Returns a [`Visit`]or that can be used for [`Constraint::visitor()`] implementations.
    pub fn visitor<'r>(
        self,
        substitutions: &'r mut Substitutions<Prim>,
        errors: OpErrors<'r, Prim>,
    ) -> Box<dyn Visit<Prim> + 'r> {
        Box::new(StructConstraintVisitor {
            inner: self,
            substitutions,
            errors,
        })
    }
}

#[derive(Debug)]
struct StructConstraintVisitor<'r, Prim: PrimitiveType, C, F> {
    inner: StructConstraint<Prim, C, F>,
    substitutions: &'r mut Substitutions<Prim>,
    errors: OpErrors<'r, Prim>,
}

impl<'r, Prim, C, F> Visit<Prim> for StructConstraintVisitor<'r, Prim, C, F>
where
    Prim: PrimitiveType,
    C: Constraint<Prim> + Clone,
    F: Fn(&Prim) -> bool + 'static,
{
    fn visit_type(&mut self, ty: &Type<Prim>) {
        match ty {
            Type::Dyn(constraints) => {
                if !constraints.inner.simple.contains(&self.inner.constraint) {
                    self.errors.push(ErrorKind::failed_constraint(
                        ty.clone(),
                        self.inner.constraint.clone(),
                    ));
                }
            }
            _ => visit::visit_type(self, ty),
        }
    }

    fn visit_var(&mut self, var: TypeVar) {
        debug_assert!(var.is_free());
        self.substitutions.insert_constraint(
            var.index(),
            &self.inner.constraint,
            self.errors.by_ref(),
        );

        let resolved = self.substitutions.fast_resolve(&Type::Var(var)).clone();
        if let Type::Var(_) = resolved {
            // Avoid infinite recursion.
        } else {
            visit::visit_type(self, &resolved);
        }
    }

    fn visit_primitive(&mut self, primitive: &Prim) {
        if !(self.inner.predicate)(primitive) {
            self.errors.push(ErrorKind::failed_constraint(
                Type::Prim(primitive.clone()),
                self.inner.constraint.clone(),
            ));
        }
    }

    fn visit_tuple(&mut self, tuple: &Tuple<Prim>) {
        if self.inner.deny_dyn_slices {
            let middle_len = tuple.parts().1.map(Slice::len);
            if let Some(middle_len) = middle_len {
                if let Err(err) = self.substitutions.apply_static_len(middle_len) {
                    self.errors.push(err);
                }
            }
        }

        for (i, element) in tuple.element_types() {
            self.errors.push_location(i);
            self.visit_type(element);
            self.errors.pop_location();
        }
    }

    fn visit_object(&mut self, obj: &Object<Prim>) {
        for (name, element) in obj.iter() {
            self.errors.push_location(name);
            self.visit_type(element);
            self.errors.pop_location();
        }
    }

    fn visit_function(&mut self, function: &Function<Prim>) {
        self.errors.push(ErrorKind::failed_constraint(
            function.clone().into(),
            self.inner.constraint.clone(),
        ));
    }
}

/// [`Constraint`] for numeric types that can be subject to unary `-` and can participate
/// in `T op Num` / `Num op T` operations.
///
/// Defined recursively as [linear](LinearType) primitive types and tuples / objects consisting
/// of linear types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Linearity;

impl fmt::Display for Linearity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Lin")
    }
}

impl<Prim: LinearType> Constraint<Prim> for Linearity {
    fn visitor<'r>(
        &self,
        substitutions: &'r mut Substitutions<Prim>,
        errors: OpErrors<'r, Prim>,
    ) -> Box<dyn Visit<Prim> + 'r> {
        StructConstraint::new(*self, LinearType::is_linear).visitor(substitutions, errors)
    }

    fn clone_boxed(&self) -> Box<dyn Constraint<Prim>> {
        Box::new(*self)
    }
}

impl<Prim: LinearType> ObjectSafeConstraint<Prim> for Linearity {}

/// Primitive type which supports a notion of *linearity*. Linear types are types that
/// can be used in arithmetic ops.
pub trait LinearType: PrimitiveType {
    /// Returns `true` iff this type is linear.
    fn is_linear(&self) -> bool;
}

/// [`Constraint`] for numeric types that can participate in binary arithmetic ops (`T op T`).
///
/// Defined as a subset of `Lin` types without dynamically sized slices and
/// any types containing dynamically sized slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ops;

impl fmt::Display for Ops {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Ops")
    }
}

impl<Prim: LinearType> Constraint<Prim> for Ops {
    fn visitor<'r>(
        &self,
        substitutions: &'r mut Substitutions<Prim>,
        errors: OpErrors<'r, Prim>,
    ) -> Box<dyn Visit<Prim> + 'r> {
        StructConstraint::new(*self, LinearType::is_linear)
            .deny_dyn_slices()
            .visitor(substitutions, errors)
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
    inner: HashMap<String, (Box<dyn Constraint<Prim>>, bool)>,
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
        for (i, (constraint, _)) in self.inner.values().enumerate() {
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

    fn contains(&self, constraint: &impl Constraint<Prim>) -> bool {
        self.inner.contains_key(&constraint.to_string())
    }

    /// Inserts a constraint into this set.
    pub fn insert(&mut self, constraint: impl Constraint<Prim>) {
        self.inner
            .insert(constraint.to_string(), (Box::new(constraint), false));
    }

    /// Inserts an object-safe constraint into this set.
    pub fn insert_object_safe(&mut self, constraint: impl ObjectSafeConstraint<Prim>) {
        self.inner
            .insert(constraint.to_string(), (Box::new(constraint), true));
    }

    /// Inserts a boxed constraint into this set.
    pub(crate) fn insert_boxed(&mut self, constraint: Box<dyn Constraint<Prim>>) {
        self.inner
            .insert(constraint.to_string(), (constraint, false));
    }

    /// Returns the link to constraint and an indicator whether it is object-safe.
    pub(crate) fn get_by_name(&self, name: &str) -> Option<(&dyn Constraint<Prim>, bool)> {
        self.inner
            .get(name)
            .map(|(constraint, is_object_safe)| (constraint.as_ref(), *is_object_safe))
    }

    /// Applies all constraints from this set.
    pub(crate) fn apply_all(
        &self,
        ty: &Type<Prim>,
        substitutions: &mut Substitutions<Prim>,
        mut errors: OpErrors<'_, Prim>,
    ) {
        for (constraint, _) in self.inner.values() {
            constraint
                .visitor(substitutions, errors.by_ref())
                .visit_type(ty);
        }
    }

    /// Applies all constraints from this set to an object.
    pub(crate) fn apply_all_to_object(
        &self,
        object: &Object<Prim>,
        substitutions: &mut Substitutions<Prim>,
        mut errors: OpErrors<'_, Prim>,
    ) {
        for (constraint, _) in self.inner.values() {
            constraint
                .visitor(substitutions, errors.by_ref())
                .visit_object(object);
        }
    }
}

/// Extended [`ConstraintSet`] that additionally supports object constraints.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompleteConstraints<Prim: PrimitiveType> {
    pub simple: ConstraintSet<Prim>,
    /// Object constraint. Stored as `Type` for convenience.
    pub object: Option<Object<Prim>>,
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

impl<Prim: PrimitiveType> From<Object<Prim>> for CompleteConstraints<Prim> {
    fn from(object: Object<Prim>) -> Self {
        Self {
            simple: ConstraintSet::default(),
            object: Some(object),
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
    pub fn insert(
        &mut self,
        constraint: impl Constraint<Prim>,
        substitutions: &mut Substitutions<Prim>,
        errors: OpErrors<'_, Prim>,
    ) {
        self.simple.insert(constraint);
        self.check_object_consistency(substitutions, errors);
    }

    /// Applies all constraints from this set.
    pub fn apply_all(
        &self,
        ty: &Type<Prim>,
        substitutions: &mut Substitutions<Prim>,
        mut errors: OpErrors<'_, Prim>,
    ) {
        self.simple.apply_all(ty, substitutions, errors.by_ref());
        if let Some(lhs) = &self.object {
            lhs.apply_as_constraint(ty, substitutions, errors);
        }
    }

    /// Maps the object constraint if present.
    pub fn map_object(self, map: impl FnOnce(&mut Object<Prim>)) -> Self {
        Self {
            simple: self.simple,
            object: self.object.map(|mut object| {
                map(&mut object);
                object
            }),
        }
    }

    /// Inserts an object constraint into this set.
    pub fn insert_obj_constraint(
        &mut self,
        object: Object<Prim>,
        substitutions: &mut Substitutions<Prim>,
        mut errors: OpErrors<'_, Prim>,
    ) {
        if let Some(existing_object) = &mut self.object {
            existing_object.extend_from(object, substitutions, errors.by_ref());
        } else {
            self.object = Some(object);
        }
        self.check_object_consistency(substitutions, errors);
    }

    fn check_object_consistency(
        &self,
        substitutions: &mut Substitutions<Prim>,
        errors: OpErrors<'_, Prim>,
    ) {
        if let Some(object) = &self.object {
            self.simple
                .apply_all_to_object(object, substitutions, errors);
        }
    }
}
