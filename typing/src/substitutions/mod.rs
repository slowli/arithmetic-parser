//! Substitutions type and dependencies.

use std::collections::{HashMap, HashSet};

use crate::{FnArgs, FnType, TupleLength, TypeErrorKind, ValueType};
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
        while let ValueType::Var(idx) = ty {
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

    pub fn assign_new_type(&mut self, ty: &mut ValueType) -> Result<(), TypeErrorKind> {
        match ty {
            ValueType::Number | ValueType::Bool | ValueType::Var(_) => {
                // Do nothing.
            }

            // Checked previously when considering a function.
            ValueType::Param(_) => unreachable!(),

            ValueType::Any => {
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
            ValueType::Tuple(elements) => {
                for element in elements {
                    self.assign_new_type(element)?;
                }
            }
            ValueType::Slice { element, length } => {
                if matches!(length, TupleLength::Any) {
                    *length = TupleLength::Var(self.const_var_count);
                    self.const_var_count += 1;
                }
                self.assign_new_type(element)?;
            }
        }
        Ok(())
    }

    /// Unifies types in `lhs` and `rhs`.
    ///
    /// - LHS corresponds to the lvalue in assignments and to called function signature in fn calls.
    /// - RHS corresponds to the rvalue in assignments and to the type of the called function.
    ///
    /// # Errors
    ///
    /// Returns an error if unification is impossible.
    pub fn unify(&mut self, lhs: &ValueType, rhs: &ValueType) -> Result<(), TypeErrorKind> {
        use self::ValueType::{Function, Param, Slice, Tuple, Var};

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

            (Tuple(lhs_types), Tuple(rhs_types)) => {
                if lhs_types.len() != rhs_types.len() {
                    return Err(TypeErrorKind::IncompatibleLengths(
                        TupleLength::Exact(lhs_types.len()),
                        TupleLength::Exact(rhs_types.len()),
                    ));
                }
                for (lhs_type, rhs_type) in lhs_types.iter().zip(rhs_types) {
                    self.unify(lhs_type, rhs_type)?;
                }
                Ok(())
            }

            (Slice { element, length }, Tuple(tuple_elements)) => {
                self.unify_lengths(*length, TupleLength::Exact(tuple_elements.len()))?;
                for tuple_element in tuple_elements {
                    self.unify(element, tuple_element)?;
                }
                Ok(())
            }
            (Tuple(tuple_elements), Slice { element, length }) => {
                self.unify_lengths(TupleLength::Exact(tuple_elements.len()), *length)?;
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

            (Function(lhs_fn), Function(rhs_fn)) => self.unify_fn_types(lhs_fn, rhs_fn),

            (ty, other_ty) => Err(TypeErrorKind::IncompatibleTypes(
                self.sanitize_type(None, ty),
                self.sanitize_type(None, other_ty),
            )),
        }
    }

    fn unify_lengths(&mut self, lhs: TupleLength, rhs: TupleLength) -> Result<(), TypeErrorKind> {
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

            (TupleLength::Dynamic, _) => {
                // Any length can be interpreted as a dynamic length, but not the other way around.
                Ok(())
            }

            _ => Err(TypeErrorKind::IncompatibleLengths(
                resolved_lhs,
                resolved_rhs,
            )),
        }
    }

    fn unify_fn_types(&mut self, lhs: &FnType, rhs: &FnType) -> Result<(), TypeErrorKind> {
        if lhs.is_parametric() {
            return Err(TypeErrorKind::UnsupportedParam);
        }

        // Check if the argument number matches. If not, we can error immediately.
        if let (FnArgs::List(lhs_list), FnArgs::List(rhs_list)) = (&lhs.args, &rhs.args) {
            if lhs_list.len() != rhs_list.len() {
                return Err(TypeErrorKind::ArgLenMismatch {
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
                // Swapping args is intentional. To see why, consider a function
                // `fn(T, U) -> V` called as `fn(A, B) -> C` (`T`, ... `C` are types).
                // In this case, the first arg of actual type `A` will be assigned to type `T`
                // (i.e., `T` is LHS and `A` is RHS); same with `U` and `B`. In contrast,
                // after function execution the return value of type `V` will be assigned
                // to type `C`. (I.e., unification of return values is not swapped.)
                self.unify(rhs_arg, lhs_arg)?;
            }
        }

        self.unify(&instantiated_lhs.return_type, &instantiated_rhs.return_type)
    }

    /// Instantiates a functional type by replacing all type arguments with new type vars.
    fn instantiate_function(&mut self, fn_type: &FnType) -> Result<FnType, TypeErrorKind> {
        if !fn_type.is_parametric() {
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
                self.mark_as_linear(&ValueType::Var(new_idx))?;
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
            ValueType::Var(i) if *i == var_idx => true,

            ValueType::Var(i) => self
                .eqs
                .get(i)
                .map_or(false, |subst| self.check_occurrence(var_idx, subst)),

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
            ValueType::Var(i) if Some(i) == fixed_idx => ValueType::Var(0),
            ValueType::Var(_) => ValueType::Any,

            ValueType::Tuple(elements) => ValueType::Tuple(
                elements
                    .iter()
                    .map(|element| self.sanitize_type(fixed_idx, element))
                    .collect(),
            ),

            ValueType::Slice { element, length } => ValueType::Slice {
                element: Box::new(self.sanitize_type(fixed_idx, &element)),
                length: match length {
                    TupleLength::Var(_) => TupleLength::Any,
                    _ => length,
                },
            },

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
    fn unify_var(&mut self, var_idx: usize, ty: &ValueType) -> Result<(), TypeErrorKind> {
        // variables should be resolved in `unify`.
        debug_assert!(!self.eqs.contains_key(&var_idx));
        debug_assert!(if let ValueType::Var(idx) = ty {
            !self.eqs.contains_key(idx)
        } else {
            true
        });

        if self.check_occurrence(var_idx, ty) {
            Err(TypeErrorKind::RecursiveType(
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
    #[allow(clippy::map_err_ignore)] // ignoring the original error is intentional
    pub fn mark_as_linear(&mut self, ty: &ValueType) -> Result<(), TypeErrorKind> {
        match ty {
            ValueType::Var(idx) => {
                self.lin.insert(*idx);
                self.eqs
                    .get(idx)
                    .cloned()
                    .map_or(Ok(()), |subst| self.mark_as_linear(&subst))
            }

            ValueType::Any | ValueType::Param(_) => unreachable!(),

            ValueType::Bool | ValueType::Function(_) => {
                let reported_ty = self.sanitize_type(None, ty);
                Err(TypeErrorKind::NonLinearType(reported_ty))
            }
            ValueType::Number => {
                // This type is linear by definition.
                Ok(())
            }

            ValueType::Tuple(elements) => {
                for element in elements {
                    self.mark_as_linear(element).map_err(|_| {
                        let reported_ty = self.sanitize_type(None, ty);
                        TypeErrorKind::NonLinearType(reported_ty)
                    })?;
                }
                Ok(())
            }
            ValueType::Slice { element, .. } => self.mark_as_linear(element).map_err(|_| {
                let reported_ty = self.sanitize_type(None, ty);
                TypeErrorKind::NonLinearType(reported_ty)
            }),
        }
    }

    pub fn unify_spanned_expr<'a, T: Grammar>(
        &mut self,
        expr: &ValueType,
        expr_span: &SpannedExpr<'a, T>,
        expected: &ValueType,
    ) -> Result<(), Spanned<'a, TypeErrorKind>> {
        // FIXME: should switch LHS / RHS.
        self.unify(&expr, expected)
            .map_err(|e| expr_span.copy_with_extra(e))
    }

    /// Returns the return type of the function.
    pub fn unify_fn_call(
        &mut self,
        definition: &ValueType,
        arg_types: Vec<ValueType>,
    ) -> Result<ValueType, TypeErrorKind> {
        let mut return_type = ValueType::Any;
        self.assign_new_type(&mut return_type)?;

        let called_fn_type = FnType::new(arg_types, return_type.clone());
        self.unify(&ValueType::Function(Box::new(called_fn_type)), definition)
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
    ) -> Result<ValueType, TypeErrorKind> {
        self.mark_as_linear(lhs_ty)?;
        self.mark_as_linear(rhs_ty)?;

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
