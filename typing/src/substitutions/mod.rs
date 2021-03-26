//! Substitutions type and dependencies.

use std::collections::{HashMap, HashSet};

use crate::{
    arith::TypeConstraints, FnArgs, FnType, PrimitiveType, Tuple, TupleLength, TypeErrorKind,
    ValueType,
};

mod fns;
use self::fns::{ParamMapping, SubstitutionContext};

#[cfg(test)]
mod tests;

/// Set of equations and constraints on type variables.
#[derive(Debug, Clone)]
pub struct Substitutions<Prim: PrimitiveType> {
    /// Number of type variables.
    type_var_count: usize,
    /// Type variable equations, encoded as `type_var[key] = value`.
    eqs: HashMap<usize, ValueType<Prim>>,
    constraints: HashMap<usize, Prim::Constraints>,
    /// Number of length variables.
    const_var_count: usize,
    /// Length variable equations.
    length_eqs: HashMap<usize, TupleLength>,
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
    pub fn insert_constraint(&mut self, var_idx: usize, constraints: &Prim::Constraints) {
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

    /// Resolves the type using equality relations in these `Substitutions`.
    ///
    /// Compared to `fast_resolve`, this method will also recursively resolve tuples
    /// and function types.
    pub fn resolve(&self, ty: &ValueType<Prim>) -> ValueType<Prim> {
        let ty = self.fast_resolve(ty);
        match ty {
            ValueType::Tuple(tuple) => {
                let mut mapped_tuple = tuple.map_types(|ty| self.resolve(ty));
                if let Some(len) = mapped_tuple.middle_len_mut() {
                    *len = self.resolve_len(len);
                }
                ValueType::Tuple(mapped_tuple)
            }

            ValueType::Function(fn_type) => {
                let resolved_fn_type = fn_type.map_types(|ty| self.resolve(ty));
                ValueType::Function(Box::new(resolved_fn_type))
            }

            _ => ty.clone(),
        }
    }

    fn resolve_len<'a>(&'a self, mut len: &'a TupleLength) -> TupleLength {
        while let TupleLength::Var(idx) = len {
            let resolved = self.length_eqs.get(idx);
            if let Some(resolved) = resolved {
                len = resolved;
            } else {
                break;
            }
        }

        if let TupleLength::Compound(compound_len) = len {
            compound_len.map_items(|item| self.resolve_len(item))
        } else {
            len.to_owned()
        }
    }

    pub(crate) fn assign_new_type(
        &mut self,
        ty: &mut ValueType<Prim>,
    ) -> Result<(), TypeErrorKind<Prim>> {
        match ty {
            ValueType::Prim(_) | ValueType::Var(_) => {
                // Do nothing.
            }

            // Checked previously when considering a function.
            ValueType::Param(_) => unreachable!(),

            ValueType::Some => {
                *ty = ValueType::Var(self.type_var_count);
                self.type_var_count += 1;
            }

            ValueType::Function(fn_type) => {
                if fn_type.is_parametric() {
                    // Can occur, for example, with function declarations:
                    //
                    // ```
                    // identity: fn<T>(T) -> T = |x| x;
                    // ```
                    //
                    // We don't handle such cases yet, because unifying functions with type params
                    // is quite difficult.
                    return Err(TypeErrorKind::UnsupportedParam);
                }
                for referenced_ty in fn_type.arg_and_return_types_mut() {
                    self.assign_new_type(referenced_ty)?;
                }
            }

            ValueType::Tuple(tuple) => {
                let mut empty_length = TupleLength::Exact(0);
                let middle_len = tuple.middle_len_mut().unwrap_or(&mut empty_length);
                self.assign_new_length(middle_len);

                for element in tuple.element_types_mut() {
                    self.assign_new_type(element)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn assign_new_length(&mut self, length: &mut TupleLength) {
        if let TupleLength::Some { is_dynamic } = length {
            if *is_dynamic {
                self.dyn_lengths.insert(self.const_var_count);
            }
            *length = TupleLength::Var(self.const_var_count);
            self.const_var_count += 1;
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

            (Param(_), _) | (_, Param(_)) => {
                unreachable!("Type params must be transformed into vars before unification")
            }

            (Tuple(lhs_tuple), Tuple(rhs_tuple)) => self.unify_tuples(lhs_tuple, rhs_tuple),

            (Function(lhs_fn), Function(rhs_fn)) => self.unify_fn_types(lhs_fn, rhs_fn),

            (ty, other_ty) => Err(TypeErrorKind::IncompatibleTypes(
                self.resolve(ty),
                self.resolve(other_ty),
            )),
        }
    }

    fn unify_tuples(
        &mut self,
        lhs: &Tuple<Prim>,
        rhs: &Tuple<Prim>,
    ) -> Result<(), TypeErrorKind<Prim>> {
        dbg!(lhs, rhs);
        let resolved_len = self.unify_lengths(&lhs.len(), &rhs.len())?;
        dbg!(&resolved_len);

        if let TupleLength::Exact(len) = resolved_len {
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
        lhs: &TupleLength,
        rhs: &TupleLength,
    ) -> Result<TupleLength, TypeErrorKind<Prim>> {
        let resolved_lhs = self.resolve_len(lhs);
        let resolved_rhs = self.resolve_len(rhs);

        match (&resolved_lhs, &resolved_rhs) {
            (TupleLength::Var(x), TupleLength::Var(y)) if x == y => {
                // Lengths are already unified.
                Ok(resolved_lhs)
            }

            // Different dyn lengths cannot be unified.
            (TupleLength::Var(x), TupleLength::Var(y))
                if self.dyn_lengths.contains(x) && self.dyn_lengths.contains(y) =>
            {
                Err(TypeErrorKind::IncompatibleLengths(
                    resolved_lhs,
                    resolved_rhs,
                ))
            }

            (TupleLength::Exact(x), TupleLength::Exact(y)) if x == y => {
                // Lengths are known to be the same.
                Ok(resolved_lhs)
            }

            (TupleLength::Compound(_), _) | (_, TupleLength::Compound(_)) => {
                todo!()
            }

            // Dynamic length can be unified at LHS position with anything other than other
            // dynamic lengths, which we've checked previously.
            (TupleLength::Var(x), _) if self.dyn_lengths.contains(&x) => Ok(resolved_rhs),

            (TupleLength::Var(x), other) if !self.dyn_lengths.contains(x) => {
                self.length_eqs.insert(*x, other.to_owned());
                Ok(resolved_rhs)
            }
            (other, TupleLength::Var(x)) if !self.dyn_lengths.contains(x) => {
                self.length_eqs.insert(*x, other.to_owned());
                Ok(resolved_lhs)
            }

            _ => Err(TypeErrorKind::IncompatibleLengths(
                resolved_lhs,
                resolved_rhs,
            )),
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

        // FIXME: Check if the argument number matches. If not, we can error immediately.

        let instantiated_lhs = self.instantiate_function(lhs);
        let instantiated_rhs = self.instantiate_function(rhs);

        let (FnArgs::List(lhs_list), FnArgs::List(rhs_list)) =
            (&instantiated_lhs.args, &instantiated_rhs.args);
        // Swapping args is intentional. To see why, consider a function
        // `fn(T, U) -> V` called as `fn(A, B) -> C` (`T`, ... `C` are types).
        // In this case, the first arg of actual type `A` will be assigned to type `T`
        // (i.e., `T` is LHS and `A` is RHS); same with `U` and `B`. In contrast,
        // after function execution the return value of type `V` will be assigned
        // to type `C`. (I.e., unification of return values is not swapped.)
        self.unify_tuples(rhs_list, lhs_list)?;

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
            constants: fn_type
                .len_params
                .iter()
                .enumerate()
                .map(|(i, (var_idx, _))| (*var_idx, self.const_var_count + i))
                .collect(),
        };
        self.type_var_count += fn_type.type_params.len();
        self.const_var_count += fn_type.len_params.len();

        let instantiated_fn_type =
            fn_type.substitute_type_vars(&mapping, SubstitutionContext::ParamsToVars);

        // Copy constraints on the newly generated const and type vars from the function definition.
        for (original_idx, description) in &fn_type.len_params {
            if description.is_dynamic {
                let new_idx = mapping.constants[original_idx];
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

    /// Checks if a type variable with the specified index is present in `ty`. This method
    /// is used to check that types are not recursive.
    fn check_occurrence(&self, var_idx: usize, ty: &ValueType<Prim>) -> bool {
        match ty {
            ValueType::Var(i) if *i == var_idx => true,

            ValueType::Var(i) => self
                .eqs
                .get(i)
                .map_or(false, |subst| self.check_occurrence(var_idx, subst)),

            ValueType::Tuple(tuple) => tuple
                .element_types()
                .any(|element| self.check_occurrence(var_idx, element)),

            ValueType::Function(fn_type) => fn_type
                .arg_and_return_types()
                .any(|ty| self.check_occurrence(var_idx, ty)),

            _ => false,
        }
    }

    /// Removes excessive information about type vars. This method is used when types are
    /// provided to `TypeError`.
    pub(crate) fn sanitize_type(&self, fixed_idx: usize, ty: &ValueType<Prim>) -> ValueType<Prim> {
        match self.resolve(ty) {
            ValueType::Var(i) if i == fixed_idx => ValueType::Param(0),

            ValueType::Tuple(tuple) => {
                ValueType::Tuple(tuple.map_types(|element| self.sanitize_type(fixed_idx, element)))
            }

            ValueType::Function(fn_type) => {
                let sanitized_fn_type = fn_type.map_types(|ty| self.sanitize_type(fixed_idx, ty));
                ValueType::Function(Box::new(sanitized_fn_type))
            }

            simple_ty => simple_ty,
        }
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

        if self.check_occurrence(var_idx, ty) {
            Err(TypeErrorKind::RecursiveType(
                self.sanitize_type(var_idx, ty),
            ))
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

        let called_fn_type = FnType::new(FnArgs::List(arg_types.into()), return_type.clone());
        self.unify(&called_fn_type.into(), definition)
            .map(|()| return_type)
    }
}
