//! Substitutions type and dependencies.

use std::collections::{HashMap, HashSet};

use crate::{FnArgs, FnType, SubstitutionContext, TypeError, ValueType};
use arithmetic_parser::{grammars::Grammar, Spanned, SpannedExpr};

/// Set of equations and constraints on type variables.
#[derive(Debug, Clone, Default)]
pub(crate) struct Substitutions {
    /// Number of type variables.
    type_var_count: usize,
    /// Type variable equations, encoded as `type_var[key] = value`.
    eqs: HashMap<usize, ValueType>,
    /// Set of type variables known to be linear.
    lin: HashSet<usize>,
}

impl Substitutions {
    pub fn linear_types(&self) -> &HashSet<usize> {
        &self.lin
    }

    /// Resolves the type by following established equality links between type variables.
    fn fast_resolve<'a>(&'a self, mut ty: &'a ValueType) -> &'a ValueType {
        while let ValueType::TypeVar(idx) = ty {
            let resolved = self.eqs.get(&idx);
            if let Some(resolved) = resolved {
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
    pub fn resolve<'a>(&'a self, ty: &'a ValueType) -> ValueType {
        let ty = self.fast_resolve(ty);
        match ty {
            ValueType::Tuple(fragments) => {
                ValueType::Tuple(fragments.iter().map(|frag| self.resolve(frag)).collect())
            }

            ValueType::Function(fn_type) => {
                let resolved_fn_type = fn_type.map_types(|ty| self.resolve(ty));
                ValueType::Function(Box::new(resolved_fn_type))
            }

            _ => ty.clone(),
        }
    }

    pub fn assign_new_type(&mut self, ty: &mut ValueType) {
        if let ValueType::Any = ty {
            *ty = ValueType::TypeVar(self.type_var_count);
            self.type_var_count += 1;
        } else if let ValueType::Tuple(fragments) = ty {
            for fragment in fragments {
                self.assign_new_type(fragment);
            }
        }
        // FIXME: What about other types?
    }

    /// Unifies types in `lhs` and `rhs`.
    ///
    /// # Errors
    ///
    /// Returns an error if unification is impossible.
    pub fn unify(&mut self, lhs: &ValueType, rhs: &ValueType) -> Result<(), TypeError> {
        use self::ValueType::*;

        match (lhs, rhs) {
            (lhs, rhs) if lhs == rhs => {
                // We already know that types are equal.
                Ok(())
            }
            (TypeVar(idx), rhs) => self.unify_var(*idx, rhs),
            (lhs, TypeVar(idx)) => self.unify_var(*idx, lhs),

            (TypeParam(_), _) | (_, TypeParam(_)) => {
                unreachable!("Type params must be transformed into vars before unification")
            }

            (Tuple(types_l), Tuple(types_r)) => {
                if types_l.len() != types_r.len() {
                    return Err(TypeError::TupleLenMismatch(types_l.len(), types_r.len()));
                }
                for (type_l, type_r) in types_l.iter().zip(types_r) {
                    self.unify(type_l, type_r)?;
                }
                Ok(())
            }

            (Function(fn_l), Function(fn_r)) => self.unify_fn_types(fn_l, fn_r),

            // FIXME: resolve types
            (x, y) => Err(TypeError::IncompatibleTypes(x.clone(), y.clone())),
        }
    }

    fn unify_fn_types(&mut self, lhs: &FnType, rhs: &FnType) -> Result<(), TypeError> {
        // First, check if the argument number matches. If not, we can error immediately.
        if let (FnArgs::List(lhs_list), FnArgs::List(rhs_list)) = (&lhs.args, &rhs.args) {
            if lhs_list.len() != rhs_list.len() {
                return Err(TypeError::ArgLenMismatch {
                    expected: lhs_list.len(),
                    actual: rhs_list.len(),
                });
            }
        }

        let instantiated_lhs = self.instantiate_function(lhs)?;
        let instantiated_rhs = self.instantiate_function(rhs)?;

        if let (FnArgs::List(lhs_list), FnArgs::List(rhs_list)) =
            (&instantiated_lhs.args, &instantiated_rhs.args)
        {
            for (lhs_arg, rhs_arg) in lhs_list.iter().zip(rhs_list) {
                self.unify(lhs_arg, rhs_arg)?;
            }
        }

        self.unify(&instantiated_lhs.return_type, &instantiated_rhs.return_type)
    }

    /// Instantiates a functional type by replacing all type arguments with new type vars.
    fn instantiate_function(&mut self, fn_type: &FnType) -> Result<FnType, TypeError> {
        if fn_type.type_params.is_empty() {
            // Fast path: just clone the function type.
            return Ok(fn_type.clone());
        }

        // Map type vars in the function into newly created type vars.
        let mapping: HashMap<_, _> = fn_type
            .type_params
            .iter()
            .enumerate()
            .map(|(i, (var_idx, _))| (*var_idx, self.type_var_count + i))
            .collect();

        let instantiated_fn_type =
            fn_type.substitute_type_vars(&mapping, SubstitutionContext::ParamsToVars);

        // Copy constraints on the newly generated type vars from the function definition.
        for (&original_idx, &new_idx) in &mapping {
            if fn_type.is_linear(original_idx) {
                self.mark_as_linear(&ValueType::TypeVar(new_idx))?;
            }
        }

        self.type_var_count += fn_type.type_params.len();
        Ok(instantiated_fn_type)
    }

    /// Checks if a type variable with the specified index is present in `ty`. This method
    /// is used to check that types are not recursive.
    fn check_occurrence(&self, var_idx: usize, ty: &ValueType) -> bool {
        match ty {
            ValueType::TypeVar(i) if *i == var_idx => true,

            ValueType::TypeVar(i) => {
                if let Some(subst) = self.eqs.get(i) {
                    self.check_occurrence(var_idx, subst)
                } else {
                    // `ty` points to a different type variable
                    false
                }
            }

            ValueType::Tuple(elements) => elements
                .iter()
                .any(|element| self.check_occurrence(var_idx, element)),

            ValueType::Function(fn_type) => fn_type
                .arg_and_return_types()
                .any(|ty| self.check_occurrence(var_idx, ty)),

            _ => false,
        }
    }

    /// Removes excessive information about type vars. This method is used when types are
    /// provided to `TypeError`.
    pub fn sanitize_type(&self, fixed_idx: Option<usize>, ty: &ValueType) -> ValueType {
        match self.resolve(ty) {
            ValueType::TypeVar(i) if Some(i) == fixed_idx => ValueType::TypeVar(0),
            ValueType::TypeVar(_) => ValueType::Any,

            ValueType::Tuple(elements) => ValueType::Tuple(
                elements
                    .iter()
                    .map(|element| self.sanitize_type(fixed_idx, element))
                    .collect(),
            ),

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
    fn unify_var(&mut self, var_idx: usize, ty: &ValueType) -> Result<(), TypeError> {
        if let Some(subst) = self.eqs.get(&var_idx).cloned() {
            return self.unify(&subst, ty);
        }
        if let ValueType::TypeVar(idx) = ty {
            if let Some(subst) = self.eqs.get(&idx).cloned() {
                return self.unify(&ValueType::TypeVar(var_idx), &subst);
            }
        }

        if self.check_occurrence(var_idx, ty) {
            Err(TypeError::RecursiveType(
                self.sanitize_type(Some(var_idx), ty),
            ))
        } else {
            if self.lin.contains(&var_idx) {
                self.mark_as_linear(ty)?;
            }
            self.eqs.insert(var_idx, ty.clone());
            Ok(())
        }
    }

    /// Recursively marks `ty` as a linear type.
    pub fn mark_as_linear(&mut self, ty: &ValueType) -> Result<(), TypeError> {
        match ty {
            ValueType::TypeVar(idx) => {
                self.lin.insert(*idx);
                if let Some(subst) = self.eqs.get(idx).cloned() {
                    self.mark_as_linear(&subst)
                } else {
                    Ok(())
                }
            }
            ValueType::Any | ValueType::TypeParam(_) => unreachable!(),

            ValueType::Bool | ValueType::Function(_) => {
                let reported_ty = self.sanitize_type(None, ty);
                Err(TypeError::NonLinearType(reported_ty))
            }
            ValueType::Number => {
                // This type is linear by definition.
                Ok(())
            }

            ValueType::Tuple(ref fragments) => {
                for fragment in fragments {
                    let reported_ty = self.sanitize_type(None, ty);
                    self.mark_as_linear(fragment)
                        .map_err(|_| TypeError::NonLinearType(reported_ty))?;
                }
                Ok(())
            }
        }
    }

    pub fn unify_spanned_expr<'a, T: Grammar>(
        &mut self,
        expr: &ValueType,
        expr_span: &SpannedExpr<'a, T>,
        expected: ValueType,
    ) -> Result<(), Spanned<'a, TypeError>> {
        self.unify(&expr, &expected)
            .map_err(|e| expr_span.copy_with_extra(e))
    }

    /// Returns the return type of the function.
    pub fn unify_fn_call(
        &mut self,
        definition: &ValueType,
        arg_types: Vec<ValueType>,
    ) -> Result<ValueType, TypeError> {
        let mut return_type = ValueType::Any;
        self.assign_new_type(&mut return_type);

        let called_fn_type = FnType::new(arg_types, return_type.clone());
        self.unify(definition, &ValueType::Function(Box::new(called_fn_type)))
            .map(|()| return_type)
    }
}
