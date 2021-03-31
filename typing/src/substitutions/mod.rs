//! Substitutions type and dependencies.

use std::{
    collections::{HashMap, HashSet},
    ptr,
};

use crate::{
    arith::TypeConstraints,
    visit::{self, Visit, VisitMut},
    FnType, PrimitiveType, Tuple, TupleLen, TupleLenMismatchContext, TypeErrorKind, ValueType,
};

mod fns;
use self::fns::{MonoTypeTransformer, ParamMapping};

#[cfg(test)]
mod tests;

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
    const_var_count: usize,
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
            const_var_count: 0,
            length_eqs: HashMap::new(),
            dyn_lengths: HashSet::new(),
        }
    }
}

impl<Prim: PrimitiveType> Substitutions<Prim> {
    pub(crate) fn type_constraints(&self) -> &HashMap<usize, Prim::Constraints> {
        &self.constraints
    }

    pub(crate) fn dyn_lengths(&self) -> &HashSet<usize> {
        &self.dyn_lengths
    }

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
        let ty = ValueType::Var(var_idx);
        let mut ty = &ty;
        let mut equivalent_vars = vec![];

        while let ValueType::Var(idx) = ty {
            equivalent_vars.push(*idx);
            if let Some(resolved) = self.eqs.get(idx) {
                ty = resolved;
            } else {
                break;
            }
        }
        equivalent_vars
    }

    /// Resolves the type by following established equality links between type variables.
    pub fn fast_resolve<'a>(&'a self, mut ty: &'a ValueType<Prim>) -> &'a ValueType<Prim> {
        while let ValueType::Var(idx) = ty {
            if let Some(resolved) = self.eqs.get(idx) {
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

    fn resolve_len<'a>(&'a self, mut len: &'a TupleLen) -> TupleLen {
        while let TupleLen::Var(idx) = len {
            let resolved = self.length_eqs.get(idx);
            if let Some(resolved) = resolved {
                len = resolved;
            } else {
                break;
            }
        }

        if let TupleLen::Compound(compound_len) = len {
            let (var, exact) = compound_len.components();
            self.resolve_len(var) + exact
        } else {
            len.to_owned()
        }
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

    pub(crate) fn assign_new_length(&mut self, length: &mut TupleLen) {
        let is_dynamic = match length {
            TupleLen::Some => false,
            TupleLen::Dynamic => true,
            _ => return,
        };

        if is_dynamic {
            self.dyn_lengths.insert(self.const_var_count);
        }
        *length = TupleLen::Var(self.const_var_count);
        self.const_var_count += 1;
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
        use self::ValueType::{Function, Param, Tuple, Var};

        let resolved_lhs = self.fast_resolve(lhs).to_owned();
        let resolved_rhs = self.fast_resolve(rhs).to_owned();

        // **NB.** LHS and RHS should never switch sides; the side is important for
        // accuracy of error reporting, and for some cases of type inference (e.g.,
        // instantiation of parametric functions).
        match (&resolved_lhs, &resolved_rhs) {
            (ty, other_ty) if ty == other_ty => {
                // We already know that types are equal.
                Ok(())
            }

            (Var(idx), ty) | (ty, Var(idx)) => self.unify_var(*idx, ty),

            (Param(_), _) | (_, Param(_)) => Err(TypeErrorKind::UnresolvedParam),

            (Tuple(lhs_tuple), Tuple(rhs_tuple)) => {
                self.unify_tuples(lhs_tuple, rhs_tuple, TupleLenMismatchContext::Assignment)
            }

            (Function(lhs_fn), Function(rhs_fn)) => self.unify_fn_types(lhs_fn, rhs_fn),

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
        let resolved_len = self.unify_lengths(&lhs.len(), &rhs.len(), context)?;

        if let TupleLen::Exact(len) = resolved_len {
            for (lhs_elem, rhs_elem) in lhs.equal_elements_static(rhs, len) {
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
        lhs: &TupleLen,
        rhs: &TupleLen,
        context: TupleLenMismatchContext,
    ) -> Result<TupleLen, TypeErrorKind<Prim>> {
        let resolved_lhs = self.resolve_len(lhs);
        let resolved_rhs = self.resolve_len(rhs);

        match (&resolved_lhs, &resolved_rhs) {
            (TupleLen::Param(_), _) | (_, TupleLen::Param(_)) => {
                Err(TypeErrorKind::UnresolvedParam)
            }

            (TupleLen::Var(x), TupleLen::Var(y)) if x == y => {
                // Lengths are already unified.
                Ok(resolved_lhs)
            }

            // Different dyn lengths cannot be unified.
            (TupleLen::Var(x), TupleLen::Var(y))
                if self.dyn_lengths.contains(x) && self.dyn_lengths.contains(y) =>
            {
                Err(TypeErrorKind::TupleLenMismatch {
                    lhs: resolved_lhs,
                    rhs: resolved_rhs,
                    context,
                })
            }

            (TupleLen::Exact(x), TupleLen::Exact(y)) if x == y => {
                // Lengths are known to be the same.
                Ok(resolved_lhs)
            }

            (TupleLen::Compound(_), _) => {
                self.unify_compound_length(resolved_lhs, resolved_rhs, true, context)
            }
            (_, TupleLen::Compound(_)) => {
                self.unify_compound_length(resolved_lhs, resolved_rhs, false, context)
            }

            // Dynamic length can be unified at LHS position with anything other than other
            // dynamic lengths, which we've checked previously.
            (TupleLen::Var(x), _) if self.dyn_lengths.contains(&x) => Ok(resolved_rhs),

            (TupleLen::Var(x), other) if !self.dyn_lengths.contains(x) => {
                self.length_eqs.insert(*x, other.to_owned());
                Ok(resolved_rhs)
            }
            (other, TupleLen::Var(x)) if !self.dyn_lengths.contains(x) => {
                self.length_eqs.insert(*x, other.to_owned());
                Ok(resolved_lhs)
            }

            _ => Err(TypeErrorKind::TupleLenMismatch {
                lhs: resolved_lhs,
                rhs: resolved_rhs,
                context,
            }),
        }
    }

    fn unify_compound_length(
        &mut self,
        resolved_lhs: TupleLen,
        resolved_rhs: TupleLen,
        is_lhs: bool,
        context: TupleLenMismatchContext,
    ) -> Result<TupleLen, TypeErrorKind<Prim>> {
        let (compound_len, other_len) = if is_lhs {
            (&resolved_lhs, &resolved_rhs)
        } else {
            (&resolved_rhs, &resolved_lhs)
        };
        let compound_len = match compound_len {
            TupleLen::Compound(compound) => compound,
            _ => unreachable!(),
        };

        let (var, exact) = compound_len.components();

        match other_len {
            TupleLen::Exact(other_exact) if *other_exact >= exact => {
                if is_lhs {
                    self.unify_lengths(var, &TupleLen::Exact(other_exact - exact), context)?;
                } else {
                    self.unify_lengths(&TupleLen::Exact(other_exact - exact), var, context)?;
                }
                return Ok(TupleLen::Exact(*other_exact));
            }

            TupleLen::Compound(other_compound_len) => {
                let (other_var, other_exact) = other_compound_len.components();
                if exact == other_exact {
                    return if is_lhs {
                        self.unify_lengths(var, other_var, context)
                    } else {
                        self.unify_lengths(other_var, var, context)
                    };
                }
            }

            _ => { /* Do nothing. */ }
        }

        Err(TypeErrorKind::TupleLenMismatch {
            lhs: resolved_lhs,
            rhs: resolved_rhs,
            context,
        })
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

        // Map type vars in the function into newly created type vars.
        let mapping = ParamMapping {
            types: fn_type
                .type_params
                .iter()
                .enumerate()
                .map(|(i, (var_idx, _))| (*var_idx, self.type_var_count + i))
                .collect(),
            lengths: fn_type
                .len_params
                .iter()
                .enumerate()
                .map(|(i, (var_idx, _))| (*var_idx, self.const_var_count + i))
                .collect(),
        };
        self.type_var_count += fn_type.type_params.len();
        self.const_var_count += fn_type.len_params.len();

        let mut instantiated_fn_type = fn_type.clone();
        MonoTypeTransformer::new(&mapping).visit_function_mut(&mut instantiated_fn_type);

        // Copy constraints on the newly generated const and type vars from the function definition.
        for (original_idx, description) in &fn_type.len_params {
            if description.is_dynamic {
                let new_idx = mapping.lengths[original_idx];
                self.dyn_lengths.insert(new_idx);
            }
        }
        for (original_idx, description) in &fn_type.type_params {
            let new_idx = mapping.types[original_idx];
            self.constraints
                .insert(new_idx, description.constraints.to_owned());
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
    ) -> Result<(), TypeErrorKind<Prim>> {
        // variables should be resolved in `unify`.
        debug_assert!(!self.eqs.contains_key(&var_idx));
        debug_assert!(if let ValueType::Var(idx) = ty {
            !self.eqs.contains_key(idx)
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

    fn visit_var(&mut self, index: usize) {
        if index == self.var_idx {
            self.is_recursive = true;
        } else if let Some(ty) = self.substitutions.eqs.get(&index) {
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
            ValueType::Var(idx) if *idx == self.fixed_idx => {
                *ty = ValueType::Param(0);
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
                *ty = ValueType::Var(self.substitutions.type_var_count);
                self.substitutions.type_var_count += 1;
            }
            _ => visit::visit_type_mut(self, ty),
        }
    }

    fn visit_middle_len_mut(&mut self, len: &mut TupleLen) {
        self.substitutions.assign_new_length(len);
    }

    fn visit_function_mut(&mut self, function: &mut FnType<Prim>) {
        if function.is_parametric() {
            // Can occur, for example, with function declarations:
            //
            // ```
            // identity: fn<T>(T) -> T = |x| x;
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
        *len = self.substitutions.resolve_len(len);
    }
}
