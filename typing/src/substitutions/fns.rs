//! Functional type substitutions.

use std::collections::HashMap;

use crate::{
    types::{ParamConstraints, ParamQuantifier},
    visit::{self, VisitMut},
    FnType, PrimitiveType, Substitutions, TupleLen, Type, UnknownLen,
};

impl<Prim: PrimitiveType> FnType<Prim> {
    /// Performs final transformations on this function, bounding all of its type vars
    /// to the function or its child functions.
    pub(crate) fn finalize(&mut self, substitutions: &Substitutions<Prim>) {
        // 1. Replace `Var`s with `Param`s.
        let mut transformer = PolyTypeTransformer::default();
        transformer.visit_function_mut(self);
        let mapping = transformer.mapping;

        // 2. Extract constraints on type params and lengths.
        let type_params = mapping
            .types
            .into_iter()
            .filter_map(|(var_idx, param_idx)| {
                let constraints = substitutions.constraints.get(&var_idx);
                constraints
                    .filter(|constraints| **constraints != Prim::Constraints::default())
                    .cloned()
                    .map(|constraints| (param_idx, constraints))
            })
            .collect();

        let static_lengths = mapping
            .lengths
            .into_iter()
            .filter_map(|(var_idx, param_idx)| {
                if substitutions.static_lengths.contains(&var_idx) {
                    Some(param_idx)
                } else {
                    None
                }
            })
            .collect();

        // 3. Set constraints for the function.
        ParamQuantifier::set_params(
            self,
            ParamConstraints {
                type_params,
                static_lengths,
            },
        );
    }
}

#[derive(Debug, Default)]
pub(super) struct ParamMapping {
    pub types: HashMap<usize, usize>,
    pub lengths: HashMap<usize, usize>,
}

/// Replaces `Var`s with `Param`s and creates the corresponding `mapping`.
#[derive(Debug, Default)]
struct PolyTypeTransformer {
    mapping: ParamMapping,
}

impl<Prim: PrimitiveType> VisitMut<Prim> for PolyTypeTransformer {
    fn visit_type_mut(&mut self, ty: &mut Type<Prim>) {
        match ty {
            Type::Var(var) if var.is_free() => {
                let type_count = self.mapping.types.len();
                let param_idx = *self.mapping.types.entry(var.index()).or_insert(type_count);
                *ty = Type::param(param_idx);
            }
            _ => visit::visit_type_mut(self, ty),
        }
    }

    fn visit_middle_len_mut(&mut self, len: &mut TupleLen) {
        let target_len = match len.components_mut() {
            (Some(var), _) => var,
            _ => return,
        };
        if let UnknownLen::Var(var) = target_len {
            debug_assert!(var.is_free());
            let len_count = self.mapping.lengths.len();
            let param_idx = *self.mapping.lengths.entry(var.index()).or_insert(len_count);
            *target_len = UnknownLen::param(param_idx);
        }
    }
}

/// Makes functional types monomorphic by replacing type / length params with vars.
#[derive(Debug)]
pub(super) struct MonoTypeTransformer<'a> {
    mapping: &'a ParamMapping,
}

impl<'a> MonoTypeTransformer<'a> {
    pub fn transform<Prim: PrimitiveType>(mapping: &'a ParamMapping, function: &mut FnType<Prim>) {
        function.params = None;
        Self { mapping }.visit_function_mut(function);
    }
}

impl<Prim: PrimitiveType> VisitMut<Prim> for MonoTypeTransformer<'_> {
    fn visit_type_mut(&mut self, ty: &mut Type<Prim>) {
        match ty {
            Type::Var(var) if !var.is_free() => {
                if let Some(mapped_idx) = self.mapping.types.get(&var.index()) {
                    *ty = Type::free_var(*mapped_idx);
                }
            }
            _ => visit::visit_type_mut(self, ty),
        }
    }

    fn visit_middle_len_mut(&mut self, len: &mut TupleLen) {
        let target_len = match len.components_mut() {
            (Some(var), _) => var,
            _ => return,
        };

        if let UnknownLen::Var(var) = target_len {
            if !var.is_free() {
                if let Some(mapped_len) = self.mapping.lengths.get(&var.index()) {
                    *target_len = UnknownLen::free_var(*mapped_len);
                }
            }
        }
    }
}
