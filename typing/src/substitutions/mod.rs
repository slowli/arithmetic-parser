//! Substitutions type and dependencies.

use std::collections::{HashMap, HashSet};

use crate::{FnArgs, FnType, TupleLength, TypeError, ValueType};
use arithmetic_parser::{grammars::Grammar, Spanned, SpannedExpr};

mod fns;
use self::fns::{ParamMapping, SubstitutionContext};

/// Set of equations and constraints on type variables.
#[derive(Debug, Clone, Default)]
pub(crate) struct Substitutions {
    /// Number of type variables.
    type_var_count: usize,
    /// Type variable equations, encoded as `type_var[key] = value`.
    eqs: HashMap<usize, ValueType>,
    /// Set of type variables known to be linear.
    lin: HashSet<usize>,
    /// Number of length variables.
    const_var_count: usize,
    /// Length variable equations.
    length_eqs: HashMap<usize, TupleLength>,
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

            ValueType::Slice { element, length } => ValueType::Slice {
                element: Box::new(self.resolve(element)),
                length: self.resolve_len(*length),
            },

            _ => ty.clone(),
        }
    }

    fn resolve_len(&self, mut len: TupleLength) -> TupleLength {
        while let TupleLength::Var(idx) = len {
            let resolved = self.length_eqs.get(&idx).copied();
            if let Some(resolved) = resolved {
                len = resolved;
            } else {
                break;
            }
        }
        len
    }

    pub fn assign_new_type(&mut self, ty: &mut ValueType) {
        match ty {
            ValueType::Number
            | ValueType::Bool
            | ValueType::TypeVar(_)
            | ValueType::TypeParam(_) => {
                // Do nothing.
            }

            ValueType::Any => {
                *ty = ValueType::TypeVar(self.type_var_count);
                self.type_var_count += 1;
            }

            ValueType::Function(fn_type) => {
                for referenced_ty in fn_type.arg_and_return_types_mut() {
                    self.assign_new_type(referenced_ty);
                }
            }
            ValueType::Tuple(elements) => {
                for element in elements {
                    self.assign_new_type(element);
                }
            }
            ValueType::Slice { element, length } => {
                if matches!(length, TupleLength::Any) {
                    *length = TupleLength::Var(self.const_var_count);
                    self.const_var_count += 1;
                }
                self.assign_new_type(element);
            }
        }
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
                    return Err(TypeError::IncompatibleLengths(
                        TupleLength::Exact(types_l.len()),
                        TupleLength::Exact(types_r.len()),
                    ));
                }
                for (type_l, type_r) in types_l.iter().zip(types_r) {
                    self.unify(type_l, type_r)?;
                }
                Ok(())
            }

            (Slice { element, length }, Tuple(tuple_elements))
            | (Tuple(tuple_elements), Slice { element, length }) => {
                self.unify_lengths(*length, TupleLength::Exact(tuple_elements.len()))?;
                for tuple_element in tuple_elements {
                    self.unify(tuple_element, element)?;
                }
                Ok(())
            }

            (
                Slice { element, length },
                Slice {
                    element: rhs_element,
                    length: rhs_length,
                },
            ) => {
                self.unify_lengths(*length, *rhs_length)?;
                self.unify(element, rhs_element)
            }

            (Function(fn_l), Function(fn_r)) => self.unify_fn_types(fn_l, fn_r),

            // FIXME: resolve types
            (x, y) => Err(TypeError::IncompatibleTypes(x.clone(), y.clone())),
        }
    }

    fn unify_lengths(&mut self, lhs: TupleLength, rhs: TupleLength) -> Result<(), TypeError> {
        let resolved_lhs = self.resolve_len(lhs);
        let resolved_rhs = self.resolve_len(rhs);

        match (resolved_lhs, resolved_rhs) {
            (TupleLength::Var(x), TupleLength::Var(y)) if x == y => {
                // Lengths are already unified.
                Ok(())
            }

            (TupleLength::Var(x), other) | (other, TupleLength::Var(x)) => {
                self.length_eqs.insert(x, other);
                Ok(())
            }

            _ => Err(TypeError::IncompatibleLengths(resolved_lhs, resolved_rhs)),
        }
    }

    fn unify_fn_types(&mut self, lhs: &FnType, rhs: &FnType) -> Result<(), TypeError> {
        // Check if both functions are parametric.
        // Can occur, for example, with function declarations:
        //
        // ```
        // identity: fn<T>(T) -> T = |x| x;
        // ```
        //
        // We don't handle such cases yet, because unifying functions with type params
        // is quite difficult.
        if lhs.is_parametric() && rhs.is_parametric() {
            return Err(TypeError::UnsupportedParam);
        }

        // Check if the argument number matches. If not, we can error immediately.
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
        if fn_type.type_params.is_empty() && fn_type.const_params.is_empty() {
            // Fast path: just clone the function type.
            return Ok(fn_type.clone());
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
                .const_params
                .iter()
                .enumerate()
                .map(|(i, (var_idx, _))| (*var_idx, self.const_var_count + i))
                .collect(),
        };

        let instantiated_fn_type =
            fn_type.substitute_type_vars(&mapping, SubstitutionContext::ParamsToVars);

        // Copy constraints on the newly generated type vars from the function definition.
        for (&original_idx, &new_idx) in &mapping.types {
            if fn_type.is_linear(original_idx) {
                self.mark_as_linear(&ValueType::TypeVar(new_idx))?;
            }
        }

        self.type_var_count += fn_type.type_params.len();
        self.const_var_count += fn_type.const_params.len();
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

            ValueType::Slice { element, .. } => self.check_occurrence(var_idx, element),

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

            ValueType::Tuple(elements) => {
                for element in elements {
                    self.mark_as_linear(element).map_err(|_| {
                        let reported_ty = self.sanitize_type(None, ty);
                        TypeError::NonLinearType(reported_ty)
                    })?;
                }
                Ok(())
            }
            ValueType::Slice { element, .. } => self.mark_as_linear(element).map_err(|_| {
                let reported_ty = self.sanitize_type(None, ty);
                TypeError::NonLinearType(reported_ty)
            }),
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

    /// Handles a binary operation.
    ///
    /// Binary ops fall into 3 cases: `Num op T == T`, `T op Num == T`, or `T op T == T`.
    /// We assume `T op T` by default, only falling into two other cases if one of operands
    /// is known to be a number and the other is not a number.
    pub fn unify_binary_op(
        &mut self,
        lhs_ty: &ValueType,
        rhs_ty: &ValueType,
    ) -> Result<ValueType, TypeError> {
        self.mark_as_linear(lhs_ty)?;
        self.mark_as_linear(rhs_ty)?;

        // Try to determine the case right away.
        let resolved_lhs_ty = self.fast_resolve(lhs_ty);
        let resolved_rhs_ty = self.fast_resolve(rhs_ty);

        match (resolved_lhs_ty.is_number(), resolved_rhs_ty.is_number()) {
            (Some(true), Some(false)) => Ok(resolved_rhs_ty.to_owned()),
            (Some(false), Some(true)) => Ok(resolved_lhs_ty.to_owned()),
            _ => {
                self.unify(lhs_ty, rhs_ty)?;
                Ok(lhs_ty.to_owned())
            }
        }
    }
}
