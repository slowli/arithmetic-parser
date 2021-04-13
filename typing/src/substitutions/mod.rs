//! Substitutions type and dependencies.

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    ops, ptr,
};

use crate::{
    arith::TypeConstraints,
    error::{ErrorLocation, OpTypeErrors, TupleLenMismatchContext, TypeErrorKind},
    visit::{self, Visit, VisitMut},
    FnType, PrimitiveType, Tuple, TupleLen, TypeVar, UnknownLen, ValueType,
};

mod fns;
use self::fns::{MonoTypeTransformer, ParamMapping};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy)]
enum LenErrorKind {
    UnresolvedParam,
    Mismatch,
    Dynamic(TupleLen),
}

/// Set of equations and constraints on type variables.
#[derive(Debug, Clone)]
pub struct Substitutions<Prim: PrimitiveType> {
    /// Number of type variables.
    type_var_count: usize,
    /// Type variable equations, encoded as `type_var[key] = value`.
    eqs: HashMap<usize, ValueType<Prim>>,
    /// Constraints on type variables.
    constraints: HashMap<usize, Prim::Constraints>,
    /// Number of length variables.
    len_var_count: usize,
    /// Length variable equations.
    length_eqs: HashMap<usize, TupleLen>,
    /// Lengths that have static restriction.
    static_lengths: HashSet<usize>,
}

impl<Prim: PrimitiveType> Default for Substitutions<Prim> {
    fn default() -> Self {
        Self {
            type_var_count: 0,
            eqs: HashMap::new(),
            constraints: HashMap::new(),
            len_var_count: 0,
            length_eqs: HashMap::new(),
            static_lengths: HashSet::new(),
        }
    }
}

impl<Prim: PrimitiveType> Substitutions<Prim> {
    /// Inserts `constraints` for a type var with the specified index and all vars
    /// it is equivalent to.
    pub fn insert_constraints(&mut self, var_idx: usize, constraints: &Prim::Constraints) {
        for idx in self.equivalent_vars(var_idx) {
            let current_constraints = self.constraints.entry(idx).or_default();
            *current_constraints |= constraints;
        }
    }

    /// Returns type var indexes that are equivalent to the provided `var_idx`,
    /// including `var_idx` itself.
    fn equivalent_vars(&self, var_idx: usize) -> Vec<usize> {
        let ty = ValueType::free_var(var_idx);
        let mut ty = &ty;
        let mut equivalent_vars = vec![];

        while let ValueType::Var(var) = ty {
            debug_assert!(var.is_free());
            equivalent_vars.push(var.index());
            if let Some(resolved) = self.eqs.get(&var.index()) {
                ty = resolved;
            } else {
                break;
            }
        }
        equivalent_vars
    }

    /// Marks `len` as static, i.e., not containing [`UnknownLen::Dynamic`] components.
    #[allow(clippy::missing_panics_doc)]
    pub fn apply_static_len(&mut self, len: TupleLen) -> Result<(), TypeErrorKind<Prim>> {
        let resolved = self.resolve_len(len);
        self.apply_static_len_inner(resolved)
            .map_err(|err| match err {
                LenErrorKind::UnresolvedParam => TypeErrorKind::UnresolvedParam,
                LenErrorKind::Dynamic(len) => TypeErrorKind::DynamicLen(len),
                LenErrorKind::Mismatch => unreachable!(),
            })
    }

    // Assumes that `len` is resolved.
    fn apply_static_len_inner(&mut self, len: TupleLen) -> Result<(), LenErrorKind> {
        match len.components().0 {
            None => Ok(()),
            Some(UnknownLen::Some) => Err(LenErrorKind::UnresolvedParam),
            Some(UnknownLen::Dynamic) => Err(LenErrorKind::Dynamic(len)),
            Some(UnknownLen::Var(var)) => {
                if var.is_free() {
                    self.static_lengths.insert(var.index());
                    Ok(())
                } else {
                    Err(LenErrorKind::UnresolvedParam)
                }
            }
        }
    }

    /// Resolves the type by following established equality links between type variables.
    pub fn fast_resolve<'a>(&'a self, mut ty: &'a ValueType<Prim>) -> &'a ValueType<Prim> {
        while let ValueType::Var(var) = ty {
            if !var.is_free() {
                // Bound variables cannot be resolved further.
                break;
            }

            if let Some(resolved) = self.eqs.get(&var.index()) {
                ty = resolved;
            } else {
                break;
            }
        }
        ty
    }

    /// Returns a visitor that resolves the type using equality relations in these `Substitutions`.
    pub fn resolver(&self) -> impl VisitMut<Prim> + '_ {
        TypeResolver {
            substitutions: self,
        }
    }

    fn resolve_len(&self, len: TupleLen) -> TupleLen {
        let mut resolved = len;
        while let (Some(UnknownLen::Var(var)), exact) = resolved.components() {
            if !var.is_free() {
                break;
            }

            if let Some(eq_rhs) = self.length_eqs.get(&var.index()) {
                resolved = eq_rhs.to_owned() + exact;
            } else {
                break;
            }
        }
        resolved
    }

    /// Creates and returns a new type variable.
    pub fn new_type_var(&mut self) -> ValueType<Prim> {
        let new_type = ValueType::free_var(self.type_var_count);
        self.type_var_count += 1;
        new_type
    }

    pub(crate) fn assign_new_type(
        &mut self,
        ty: &mut ValueType<Prim>,
        errors: OpTypeErrors<'_, Prim>,
    ) {
        let mut assigner = TypeAssigner {
            substitutions: self,
            errors,
        };
        assigner.visit_type_mut(ty);
    }

    pub(crate) fn assign_new_len(&mut self, len: &mut TupleLen) {
        if let Some(target_len @ UnknownLen::Some) = len.components_mut().0 {
            *target_len = UnknownLen::free_var(self.len_var_count);
            self.len_var_count += 1;
        }
    }

    /// Unifies types in `lhs` and `rhs`.
    ///
    /// - LHS corresponds to the lvalue in assignments and to called function signature in fn calls.
    /// - RHS corresponds to the rvalue in assignments and to the type of the called function.
    ///
    /// # Errors
    ///
    /// Returns an error if unification is impossible.
    pub fn unify(
        &mut self,
        lhs: &ValueType<Prim>,
        rhs: &ValueType<Prim>,
        mut errors: OpTypeErrors<'_, Prim>,
    ) {
        let resolved_lhs = self.fast_resolve(lhs).to_owned();
        let resolved_rhs = self.fast_resolve(rhs).to_owned();

        // **NB.** LHS and RHS should never switch sides; the side is important for
        // accuracy of error reporting, and for some cases of type inference (e.g.,
        // instantiation of parametric functions).
        match (&resolved_lhs, &resolved_rhs) {
            // Variables should be assigned *before* the equality check and dealing with `Any`
            // to account for `Var <- Any` assignment.
            (ValueType::Var(var), ty) => {
                if var.is_free() {
                    self.unify_var(var.index(), ty, true, errors);
                } else {
                    errors.push(TypeErrorKind::UnresolvedParam);
                }
            }
            (ty, ValueType::Var(var)) => {
                if var.is_free() {
                    self.unify_var(var.index(), ty, false, errors);
                } else {
                    errors.push(TypeErrorKind::UnresolvedParam);
                }
            }

            (ValueType::Any(constraints), ty) | (ty, ValueType::Any(constraints)) => {
                constraints.apply(ty, self, errors);
            }

            // This takes care of `Any` types because they are equal to anything.
            (ty, other_ty) if ty == other_ty => {
                // We already know that types are equal.
            }

            (ValueType::Tuple(lhs_tuple), ValueType::Tuple(rhs_tuple)) => self.unify_tuples(
                lhs_tuple,
                rhs_tuple,
                TupleLenMismatchContext::Assignment,
                errors,
            ),

            (ValueType::Function(lhs_fn), ValueType::Function(rhs_fn)) => {
                self.unify_fn_types(lhs_fn, rhs_fn, errors)
            }

            (ty, other_ty) => {
                let mut resolver = self.resolver();
                let mut ty = ty.to_owned();
                resolver.visit_type_mut(&mut ty);
                let mut other_ty = other_ty.to_owned();
                resolver.visit_type_mut(&mut other_ty);
                errors.push(TypeErrorKind::TypeMismatch(ty, other_ty));
            }
        }
    }

    fn unify_tuples(
        &mut self,
        lhs: &Tuple<Prim>,
        rhs: &Tuple<Prim>,
        context: TupleLenMismatchContext,
        mut errors: OpTypeErrors<'_, Prim>,
    ) {
        let resolved_len = self.unify_lengths(lhs.len(), rhs.len(), context);
        let resolved_len = match resolved_len {
            Ok(len) => len,
            Err(err) => {
                self.unify_tuples_after_error(lhs, rhs, &err, context, errors.by_ref());
                errors.push(err);
                return;
            }
        };

        if let (None, exact) = resolved_len.components() {
            self.unify_tuple_elements(lhs.iter(exact).zip(rhs.iter(exact)), context, errors);
        } else {
            // FIXME: is this always applicable?
            // FIXME: this leads to incorrect error locations!
            self.unify_tuple_elements(lhs.equal_elements_dyn(rhs), context, errors);
        }
    }

    #[inline]
    fn unify_tuple_elements<'it>(
        &mut self,
        pairs: impl Iterator<Item = (&'it ValueType<Prim>, &'it ValueType<Prim>)>,
        context: TupleLenMismatchContext,
        mut errors: OpTypeErrors<'_, Prim>,
    ) {
        for (i, (lhs_elem, rhs_elem)) in pairs.enumerate() {
            let location = match context {
                TupleLenMismatchContext::Assignment => ErrorLocation::TupleElement(i),
                TupleLenMismatchContext::FnArgs => ErrorLocation::FnArg(i),
            };
            self.unify(lhs_elem, rhs_elem, errors.with_location(location));
        }
    }

    /// Tries to unify tuple elements after an error has occurred when unifying their lengths.
    fn unify_tuples_after_error(
        &mut self,
        lhs: &Tuple<Prim>,
        rhs: &Tuple<Prim>,
        err: &TypeErrorKind<Prim>,
        context: TupleLenMismatchContext,
        errors: OpTypeErrors<'_, Prim>,
    ) {
        let (lhs_len, rhs_len) = match err {
            TypeErrorKind::TupleLenMismatch {
                lhs: lhs_len,
                rhs: rhs_len,
                ..
            } => (*lhs_len, *rhs_len),
            _ => return,
        };
        let (lhs_var, lhs_exact) = lhs_len.components();
        let (rhs_var, rhs_exact) = rhs_len.components();

        match (lhs_var, rhs_var) {
            (None, None) => {
                // We've attempted to unify tuples with different known lengths.
                // Iterate over common elements and unify them.
                debug_assert_ne!(lhs_exact, rhs_exact);
                self.unify_tuple_elements(
                    lhs.iter(lhs_exact).zip(rhs.iter(rhs_exact)),
                    context,
                    errors,
                );
            }

            (None, Some(UnknownLen::Dynamic)) => {
                // We've attempted to unify static LHS w/ a dynamic RHS
                // e.g., `(x, y) = filter(...)`.
                self.unify_tuple_elements(
                    lhs.iter(lhs_exact).zip(rhs.iter(rhs_exact)),
                    context,
                    errors,
                );
            }

            _ => { /* Do nothing. */ }
        }
    }

    /// Returns the resolved length that `lhs` and `rhs` are equal to.
    fn unify_lengths(
        &mut self,
        lhs: TupleLen,
        rhs: TupleLen,
        context: TupleLenMismatchContext,
    ) -> Result<TupleLen, TypeErrorKind<Prim>> {
        let resolved_lhs = self.resolve_len(lhs);
        let resolved_rhs = self.resolve_len(rhs);

        self.unify_lengths_inner(resolved_lhs, resolved_rhs)
            .map_err(|err| match err {
                LenErrorKind::UnresolvedParam => TypeErrorKind::UnresolvedParam,
                LenErrorKind::Mismatch => TypeErrorKind::TupleLenMismatch {
                    lhs: resolved_lhs,
                    rhs: resolved_rhs,
                    context,
                },
                LenErrorKind::Dynamic(len) => TypeErrorKind::DynamicLen(len),
            })
    }

    fn unify_lengths_inner(
        &mut self,
        resolved_lhs: TupleLen,
        resolved_rhs: TupleLen,
    ) -> Result<TupleLen, LenErrorKind> {
        let (lhs_var, lhs_exact) = resolved_lhs.components();
        let (rhs_var, rhs_exact) = resolved_rhs.components();

        // First, consider a case when at least one of resolved lengths is exact.
        let (lhs_var, rhs_var) = match (lhs_var, rhs_var) {
            (Some(lhs_var), Some(rhs_var)) => (lhs_var, rhs_var),

            (Some(lhs_var), None) if rhs_exact >= lhs_exact => {
                return self
                    .unify_simple_length(lhs_var, TupleLen::from(rhs_exact - lhs_exact), true)
                    .map(|len| len + lhs_exact);
            }
            (None, Some(rhs_var)) if lhs_exact >= rhs_exact => {
                return self
                    .unify_simple_length(rhs_var, TupleLen::from(lhs_exact - rhs_exact), false)
                    .map(|len| len + rhs_exact);
            }

            (None, None) if lhs_exact == rhs_exact => return Ok(TupleLen::from(lhs_exact)),

            _ => return Err(LenErrorKind::Mismatch),
        };

        match lhs_exact.cmp(&rhs_exact) {
            Ordering::Equal => self.unify_simple_length(lhs_var, TupleLen::from(rhs_var), true),
            Ordering::Greater => {
                let reduced = lhs_var + (lhs_exact - rhs_exact);
                self.unify_simple_length(rhs_var, reduced, false)
                    .map(|len| len + rhs_exact)
            }
            Ordering::Less => {
                let reduced = rhs_var + (rhs_exact - lhs_exact);
                self.unify_simple_length(lhs_var, reduced, true)
                    .map(|len| len + lhs_exact)
            }
        }
    }

    fn unify_simple_length(
        &mut self,
        simple_len: UnknownLen,
        source: TupleLen,
        is_lhs: bool,
    ) -> Result<TupleLen, LenErrorKind> {
        match simple_len {
            UnknownLen::Var(var) if var.is_free() => self.unify_var_length(var.index(), source),
            UnknownLen::Dynamic => self.unify_dyn_length(source, is_lhs),
            _ => Err(LenErrorKind::UnresolvedParam),
        }
    }

    #[inline]
    fn unify_var_length(
        &mut self,
        var_idx: usize,
        source: TupleLen,
    ) -> Result<TupleLen, LenErrorKind> {
        // Check that the source is valid.
        match source.components() {
            (Some(UnknownLen::Some), _) => Err(LenErrorKind::UnresolvedParam),
            (Some(UnknownLen::Var(var)), _) if !var.is_free() => Err(LenErrorKind::UnresolvedParam),

            // Special case is uniting a var with self.
            (Some(UnknownLen::Var(var)), offset) if var.index() == var_idx => {
                if offset == 0 {
                    Ok(source)
                } else {
                    Err(LenErrorKind::Mismatch)
                }
            }

            _ => {
                if self.static_lengths.contains(&var_idx) {
                    self.apply_static_len_inner(source)?;
                }
                self.length_eqs.insert(var_idx, source);
                Ok(source)
            }
        }
    }

    #[inline]
    fn unify_dyn_length(
        &mut self,
        source: TupleLen,
        is_lhs: bool,
    ) -> Result<TupleLen, LenErrorKind> {
        if is_lhs {
            Ok(source) // assignment to dyn length always succeeds
        } else {
            let source_var_idx = match source.components() {
                (Some(UnknownLen::Var(var)), 0) if var.is_free() => var.index(),
                (Some(UnknownLen::Dynamic), 0) => return Ok(source),
                _ => return Err(LenErrorKind::Mismatch),
            };
            self.unify_var_length(source_var_idx, UnknownLen::Dynamic.into())
        }
    }

    fn unify_fn_types(
        &mut self,
        lhs: &FnType<Prim>,
        rhs: &FnType<Prim>,
        mut errors: OpTypeErrors<'_, Prim>,
    ) {
        if lhs.is_parametric() {
            errors.push(TypeErrorKind::UnsupportedParam);
            return;
        }

        let instantiated_lhs = self.instantiate_function(lhs);
        let instantiated_rhs = self.instantiate_function(rhs);

        // Swapping args is intentional. To see why, consider a function
        // `fn(T, U) -> V` called as `fn(A, B) -> C` (`T`, ... `C` are types).
        // In this case, the first arg of actual type `A` will be assigned to type `T`
        // (i.e., `T` is LHS and `A` is RHS); same with `U` and `B`. In contrast,
        // after function execution the return value of type `V` will be assigned
        // to type `C`. (I.e., unification of return values is not swapped.)
        self.unify_tuples(
            &instantiated_rhs.args,
            &instantiated_lhs.args,
            TupleLenMismatchContext::FnArgs,
            errors.by_ref(),
        );

        self.unify(
            &instantiated_lhs.return_type,
            &instantiated_rhs.return_type,
            errors.with_location(ErrorLocation::FnReturnType),
        );
    }

    /// Instantiates a functional type by replacing all type arguments with new type vars.
    fn instantiate_function(&mut self, fn_type: &FnType<Prim>) -> FnType<Prim> {
        if !fn_type.is_parametric() {
            // Fast path: just clone the function type.
            return fn_type.clone();
        }
        let fn_params = fn_type.params.as_ref().expect("fn with params");

        // Map type vars in the function into newly created type vars.
        let mapping = ParamMapping {
            types: fn_params
                .type_params
                .iter()
                .enumerate()
                .map(|(i, (var_idx, _))| (*var_idx, self.type_var_count + i))
                .collect(),
            lengths: fn_params
                .len_params
                .iter()
                .enumerate()
                .map(|(i, (var_idx, _))| (*var_idx, self.len_var_count + i))
                .collect(),
        };
        self.type_var_count += fn_params.type_params.len();
        self.len_var_count += fn_params.len_params.len();

        let mut instantiated_fn_type = fn_type.clone();
        MonoTypeTransformer::transform(&mapping, &mut instantiated_fn_type);

        // Copy constraints on the newly generated const and type vars from the function definition.
        for (original_idx, is_static) in &fn_params.len_params {
            if *is_static {
                let new_idx = mapping.lengths[original_idx];
                self.static_lengths.insert(new_idx);
            }
        }
        for (original_idx, constraints) in &fn_params.type_params {
            let new_idx = mapping.types[original_idx];
            self.constraints.insert(new_idx, constraints.to_owned());
        }

        instantiated_fn_type
    }

    /// Unifies a type variable with the specified index and the specified type.
    fn unify_var(
        &mut self,
        var_idx: usize,
        ty: &ValueType<Prim>,
        is_lhs: bool,
        mut errors: OpTypeErrors<'_, Prim>,
    ) {
        if let ValueType::Var(var) = ty {
            if !var.is_free() {
                errors.push(TypeErrorKind::UnresolvedParam);
                return;
            } else if var.index() == var_idx {
                return;
            }
        }
        let needs_equation = if let ValueType::Any(constraints) = ty {
            self.constraints.insert(var_idx, constraints.to_owned());
            is_lhs
        } else {
            true
        };

        // variables should be resolved in `unify`.
        debug_assert!(!self.eqs.contains_key(&var_idx));
        debug_assert!(if let ValueType::Var(var) = ty {
            !self.eqs.contains_key(&var.index())
        } else {
            true
        });

        let mut checker = OccurrenceChecker {
            substitutions: self,
            var_idx,
            is_recursive: false,
        };
        checker.visit_type(ty);

        if checker.is_recursive {
            let mut resolved_ty = ty.to_owned();
            self.resolver().visit_type_mut(&mut resolved_ty);
            TypeSanitizer { fixed_idx: var_idx }.visit_type_mut(&mut resolved_ty);
            errors.push(TypeErrorKind::RecursiveType(resolved_ty));
        } else {
            if let Some(constraints) = self.constraints.get(&var_idx).cloned() {
                constraints.apply(ty, self, errors);
            }
            if needs_equation {
                let mut ty = ty.to_owned();
                if !is_lhs {
                    // We need to swap `any` types / lengths with new vars so that this type
                    // can be specified further.
                    TypeSpecifier::new(self).visit_type_mut(&mut ty);
                }
                self.eqs.insert(var_idx, ty);
            }
        }
    }
}

/// Checks if a type variable with the specified index is present in `ty`. This method
/// is used to check that types are not recursive.
#[derive(Debug)]
struct OccurrenceChecker<'a, Prim: PrimitiveType> {
    substitutions: &'a Substitutions<Prim>,
    var_idx: usize,
    is_recursive: bool,
}

impl<'a, Prim: PrimitiveType> Visit<'a, Prim> for OccurrenceChecker<'a, Prim> {
    fn visit_type(&mut self, ty: &'a ValueType<Prim>) {
        if self.is_recursive {
            // Skip recursion; we already have our answer at this point.
        } else {
            visit::visit_type(self, ty);
        }
    }

    fn visit_var(&mut self, var: TypeVar) {
        if var.index() == self.var_idx {
            self.is_recursive = true;
        } else if let Some(ty) = self.substitutions.eqs.get(&var.index()) {
            self.visit_type(ty);
        }
    }
}

/// Removes excessive information about type vars. This method is used when types are
/// provided to `TypeError`.
#[derive(Debug)]
struct TypeSanitizer {
    fixed_idx: usize,
}

impl<Prim: PrimitiveType> VisitMut<Prim> for TypeSanitizer {
    fn visit_type_mut(&mut self, ty: &mut ValueType<Prim>) {
        match ty {
            ValueType::Var(var) if var.index() == self.fixed_idx => {
                *ty = ValueType::param(0);
            }
            _ => visit::visit_type_mut(self, ty),
        }
    }
}

/// Replaces `ValueType::Some` and `TupleLen::Some` with new variables.
#[derive(Debug)]
struct TypeAssigner<'a, Prim: PrimitiveType> {
    substitutions: &'a mut Substitutions<Prim>,
    errors: OpTypeErrors<'a, Prim>,
}

impl<Prim: PrimitiveType> VisitMut<Prim> for TypeAssigner<'_, Prim> {
    fn visit_type_mut(&mut self, ty: &mut ValueType<Prim>) {
        match ty {
            ValueType::Some => *ty = self.substitutions.new_type_var(),
            ValueType::Function(function) if function.is_parametric() => {
                // Can occur, for example, with function declarations:
                //
                // ```
                // identity: ('T) -> 'T = |x| x;
                // ```
                //
                // We don't handle such cases yet, because unifying functions with type params
                // is quite difficult.
                self.errors.push(TypeErrorKind::UnsupportedParam);
                *ty = self.substitutions.new_type_var();
            }
            _ => visit::visit_type_mut(self, ty),
        }
    }

    fn visit_tuple_mut(&mut self, tuple: &mut Tuple<Prim>) {
        let (start, middle, end) = tuple.parts_mut();

        for (i, ty) in start.iter_mut().enumerate() {
            self.errors.push_location(ErrorLocation::TupleElement(i));
            self.visit_type_mut(ty);
            self.errors.pop_location();
        }

        if let Some(middle) = middle {
            self.errors.push_location(ErrorLocation::TupleMiddle);
            self.visit_middle_len_mut(middle.len_mut());
            self.visit_type_mut(middle.element_mut());
            self.errors.pop_location();
        }

        for (i, ty) in end.iter_mut().enumerate() {
            self.errors.push_location(ErrorLocation::TupleEnd(i));
            self.visit_type_mut(ty);
            self.errors.pop_location();
        }
    }

    fn visit_middle_len_mut(&mut self, len: &mut TupleLen) {
        self.substitutions.assign_new_len(len);
    }
}

/// Visitor that performs type resolution based on `Substitutions`.
#[derive(Debug, Clone, Copy)]
struct TypeResolver<'a, Prim: PrimitiveType> {
    substitutions: &'a Substitutions<Prim>,
}

impl<Prim: PrimitiveType> VisitMut<Prim> for TypeResolver<'_, Prim> {
    fn visit_type_mut(&mut self, ty: &mut ValueType<Prim>) {
        let fast_resolved = self.substitutions.fast_resolve(ty);
        if !ptr::eq(ty, fast_resolved) {
            *ty = fast_resolved.to_owned();
        }
        visit::visit_type_mut(self, ty);
    }

    fn visit_middle_len_mut(&mut self, len: &mut TupleLen) {
        *len = self.substitutions.resolve_len(*len);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Variance {
    Co,
    Contra,
}

impl ops::Not for Variance {
    type Output = Self;

    fn not(self) -> Self {
        match self {
            Self::Co => Self::Contra,
            Self::Contra => Self::Co,
        }
    }
}

/// Visitor that swaps `any` types / lengths with new vars, but only if they are in a covariant
/// position (return types, args of arg functions, etc.).
#[derive(Debug)]
struct TypeSpecifier<'a, Prim: PrimitiveType> {
    substitutions: &'a mut Substitutions<Prim>,
    variance: Variance,
}

impl<'a, Prim: PrimitiveType> TypeSpecifier<'a, Prim> {
    fn new(substitutions: &'a mut Substitutions<Prim>) -> Self {
        Self {
            substitutions,
            variance: Variance::Co,
        }
    }
}

impl<Prim: PrimitiveType> VisitMut<Prim> for TypeSpecifier<'_, Prim> {
    fn visit_type_mut(&mut self, ty: &mut ValueType<Prim>) {
        if let ValueType::Any(constraints) = ty {
            if self.variance == Variance::Co {
                let var_idx = self.substitutions.type_var_count;
                self.substitutions
                    .constraints
                    .insert(var_idx, constraints.to_owned());
                *ty = ValueType::free_var(var_idx);
                self.substitutions.type_var_count += 1;
            }
        } else {
            visit::visit_type_mut(self, ty);
        }
    }

    fn visit_middle_len_mut(&mut self, len: &mut TupleLen) {
        if self.variance != Variance::Co {
            return;
        }
        if let (Some(var_len @ UnknownLen::Dynamic), _) = len.components_mut() {
            let var_idx = self.substitutions.len_var_count;
            self.substitutions.len_var_count += 1;
            *var_len = UnknownLen::free_var(var_idx);
        }
    }

    fn visit_function_mut(&mut self, function: &mut FnType<Prim>) {
        // Since the visiting order doesn't matter, we visit the return type (which preserves
        // variance) first.
        self.visit_type_mut(&mut function.return_type);

        let old_variance = self.variance;
        self.variance = !self.variance;
        self.visit_tuple_mut(&mut function.args);
        self.variance = old_variance;
    }
}
