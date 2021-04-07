//! Substitutions type and dependencies.

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    ptr,
};

use crate::{
    arith::TypeConstraints,
    visit::{self, Visit, VisitMut},
    FnType, LengthKind, PrimitiveType, Tuple, TupleLen, TupleLenMismatchContext, TypeErrorKind,
    TypeVar, UnknownLen, ValueType,
};

mod fns;
use self::fns::{MonoTypeTransformer, ParamMapping};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy)]
enum LenErrorKind {
    UnresolvedParam,
    Mismatch,
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
    /// Length variable known to be dynamic.
    dyn_lengths: HashSet<usize>,
}

impl<Prim: PrimitiveType> Default for Substitutions<Prim> {
    fn default() -> Self {
        Self {
            type_var_count: 0,
            eqs: HashMap::new(),
            constraints: HashMap::new(),
            len_var_count: 0,
            length_eqs: HashMap::new(),
            dyn_lengths: HashSet::new(),
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

    pub(crate) fn assign_new_type(
        &mut self,
        ty: &mut ValueType<Prim>,
    ) -> Result<(), TypeErrorKind<Prim>> {
        let mut assigner = TypeAssigner {
            substitutions: self,
            outcome: Ok(()),
        };
        assigner.visit_type_mut(ty);
        assigner.outcome
    }

    pub(crate) fn assign_new_len(&mut self, len: &mut TupleLen) {
        let target_len = match len.components_mut().0 {
            Some(target_len) => target_len,
            None => return,
        };
        let is_dynamic = match target_len {
            UnknownLen::Some => false,
            UnknownLen::Dynamic => true,
            _ => return,
        };

        if is_dynamic {
            self.dyn_lengths.insert(self.len_var_count);
        }
        *target_len = UnknownLen::free_var(self.len_var_count);
        self.len_var_count += 1;
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
    ) -> Result<(), TypeErrorKind<Prim>> {
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
                    self.unify_var(var.index(), ty, true)
                } else {
                    Err(TypeErrorKind::UnresolvedParam)
                }
            }
            (ty, ValueType::Var(var)) => {
                if var.is_free() {
                    self.unify_var(var.index(), ty, false)
                } else {
                    Err(TypeErrorKind::UnresolvedParam)
                }
            }

            // This takes care of `Any` types because they are equal to anything.
            (ty, other_ty) if ty == other_ty => {
                // We already know that types are equal.
                Ok(())
            }

            (ValueType::Tuple(lhs_tuple), ValueType::Tuple(rhs_tuple)) => {
                self.unify_tuples(lhs_tuple, rhs_tuple, TupleLenMismatchContext::Assignment)
            }

            (ValueType::Function(lhs_fn), ValueType::Function(rhs_fn)) => {
                self.unify_fn_types(lhs_fn, rhs_fn)
            }

            (ty, other_ty) => {
                let mut resolver = self.resolver();
                let mut ty = ty.to_owned();
                resolver.visit_type_mut(&mut ty);
                let mut other_ty = other_ty.to_owned();
                resolver.visit_type_mut(&mut other_ty);
                Err(TypeErrorKind::TypeMismatch(ty, other_ty))
            }
        }
    }

    fn unify_tuples(
        &mut self,
        lhs: &Tuple<Prim>,
        rhs: &Tuple<Prim>,
        context: TupleLenMismatchContext,
    ) -> Result<(), TypeErrorKind<Prim>> {
        let resolved_len = self.unify_lengths(lhs.len(), rhs.len(), context)?;

        if let (None, exact) = resolved_len.components() {
            for (lhs_elem, rhs_elem) in lhs.equal_elements_static(rhs, exact) {
                self.unify(lhs_elem, rhs_elem)?;
            }
        } else {
            // FIXME: is this always applicable?
            for (lhs_elem, rhs_elem) in lhs.equal_elements_dyn(rhs) {
                self.unify(lhs_elem, rhs_elem)?;
            }
        }

        Ok(())
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
        let var_idx = match simple_len {
            UnknownLen::Var(var) if var.is_free() => var.index(),
            _ => return Err(LenErrorKind::UnresolvedParam),
        };
        let source_var_idx = match source.components().0 {
            None => None,
            Some(UnknownLen::Var(var)) if var.is_free() => Some(var.index()),
            Some(_) => return Err(LenErrorKind::UnresolvedParam),
        };

        let (is_dyn, is_source_dyn) = match source_var_idx {
            // Same length variable.
            Some(idx) if idx == var_idx => {
                return if source.components().1 == 0 {
                    Ok(source) // lengths are already unified
                } else {
                    // x = x + C, C > 0: cannot be unified
                    Err(LenErrorKind::Mismatch)
                };
            }
            // Different length variables.
            Some(source_var_idx) => {
                let is_dyn = self.dyn_lengths.contains(&var_idx);
                let is_source_dyn = self.dyn_lengths.contains(&source_var_idx);
                (is_dyn, is_source_dyn)
            }
            // No length variable.
            None => {
                let is_dyn = self.dyn_lengths.contains(&var_idx);
                (is_dyn, false)
            }
        };

        let (is_lhs_dyn, is_rhs_dyn) = if is_lhs {
            (is_dyn, is_source_dyn)
        } else {
            (is_source_dyn, is_dyn)
        };

        match (is_lhs_dyn, is_rhs_dyn) {
            // Dynamic vars are different, hence cannot be unified.
            (true, true) => Err(LenErrorKind::Mismatch),
            // LHS is dynamic, RHS is not.
            (true, false) => {
                // Check additionally that LHS does not have a non-zero exact part.
                let is_valid = is_lhs || source.components().1 == 0;
                if is_valid {
                    Ok(source) // no new equation required
                } else {
                    Err(LenErrorKind::Mismatch)
                }
            }
            // LHS is not dynamic; RHS may or may not be.
            (false, _) => {
                if is_dyn {
                    Err(LenErrorKind::Mismatch)
                } else {
                    self.length_eqs.insert(var_idx, source);
                    Ok(source)
                }
            }
        }
    }

    fn unify_fn_types(
        &mut self,
        lhs: &FnType<Prim>,
        rhs: &FnType<Prim>,
    ) -> Result<(), TypeErrorKind<Prim>> {
        if lhs.is_parametric() {
            return Err(TypeErrorKind::UnsupportedParam);
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
        )?;

        self.unify(&instantiated_lhs.return_type, &instantiated_rhs.return_type)
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
        for (original_idx, kind) in &fn_params.len_params {
            if *kind == LengthKind::Dynamic {
                let new_idx = mapping.lengths[original_idx];
                self.dyn_lengths.insert(new_idx);
            }
        }
        for (original_idx, constraints) in &fn_params.type_params {
            let new_idx = mapping.types[original_idx];
            self.constraints.insert(new_idx, constraints.to_owned());
        }

        instantiated_fn_type
    }

    /// Unifies a type variable with the specified index and the specified type.
    ///
    /// # Errors
    ///
    /// Returns an error if the unification is impossible.
    fn unify_var(
        &mut self,
        var_idx: usize,
        ty: &ValueType<Prim>,
        is_lhs: bool,
    ) -> Result<(), TypeErrorKind<Prim>> {
        if let ValueType::Var(var) = ty {
            if !var.is_free() {
                return Err(TypeErrorKind::UnresolvedParam);
            } else if var.index() == var_idx {
                return Ok(());
            }
        }
        if matches!(ty, ValueType::Any) && !is_lhs {
            // `Any` <- `Var` assignment; we shouldn't create a type var equation.
            return Ok(());
        }

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
            Err(TypeErrorKind::RecursiveType(resolved_ty))
        } else {
            if let Some(constraints) = self.constraints.get(&var_idx).cloned() {
                constraints.apply(ty, self)?;
            }
            self.eqs.insert(var_idx, ty.clone());
            Ok(())
        }
    }

    /// Returns the return type of the function.
    pub(crate) fn unify_fn_call(
        &mut self,
        definition: &ValueType<Prim>,
        arg_types: Vec<ValueType<Prim>>,
    ) -> Result<ValueType<Prim>, TypeErrorKind<Prim>> {
        let mut return_type = ValueType::Some;
        self.assign_new_type(&mut return_type)?;

        let called_fn_type = FnType::new(arg_types.into(), return_type.clone());
        self.unify(&called_fn_type.into(), definition)
            .map(|()| return_type)
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
    outcome: Result<(), TypeErrorKind<Prim>>,
}

impl<Prim: PrimitiveType> VisitMut<Prim> for TypeAssigner<'_, Prim> {
    fn visit_type_mut(&mut self, ty: &mut ValueType<Prim>) {
        if self.outcome.is_err() {
            return;
        }

        match ty {
            ValueType::Some => {
                *ty = ValueType::free_var(self.substitutions.type_var_count);
                self.substitutions.type_var_count += 1;
            }
            _ => visit::visit_type_mut(self, ty),
        }
    }

    fn visit_middle_len_mut(&mut self, len: &mut TupleLen) {
        self.substitutions.assign_new_len(len);
    }

    fn visit_function_mut(&mut self, function: &mut FnType<Prim>) {
        if function.is_parametric() {
            // Can occur, for example, with function declarations:
            //
            // ```
            // identity: ('T) -> 'T = |x| x;
            // ```
            //
            // We don't handle such cases yet, because unifying functions with type params
            // is quite difficult.
            self.outcome = Err(TypeErrorKind::UnsupportedParam);
        } else {
            visit::visit_function_mut(self, function);
        }
    }
}

/// Mutable visitor that performs type resolution based on `Substitutions`.
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
