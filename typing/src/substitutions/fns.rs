//! Functional type substitutions.

use std::collections::{HashMap, HashSet};

use crate::{
    types::{ParamConstraints, ParamQuantifier},
    visit::{self, VisitMut},
    FnType, PrimitiveType, Slice, Substitutions, TupleLen, UnknownLen, ValueType,
};

impl<Prim: PrimitiveType> FnType<Prim> {
    /// Performs final transformations on this type, transforming all of its type vars
    /// into type params.
    pub(crate) fn finalize(&mut self, substitutions: &Substitutions<Prim>) {
        // 1. Replace `Var`s with `Param`s.
        let mut transformer = PolyTypeTransformer::default();
        transformer.visit_function_mut(self);
        let mapping = transformer.mapping;
        let vararg_lengths = transformer.vararg_lengths;

        let type_params = mapping
            .types
            .into_iter()
            .filter_map(|(var_idx, param_idx)| {
                let constraints = substitutions.constraints.get(&var_idx).cloned();
                constraints.map(|constraints| (param_idx, constraints))
            })
            .collect();

        // `vararg_lengths` are dynamic within function context, but must be set to non-dynamic
        // for the function definition.
        let dyn_lengths = mapping
            .lengths
            .into_iter()
            .filter_map(|(var_idx, param_idx)| {
                if substitutions.dyn_lengths.contains(&var_idx)
                    && !vararg_lengths.contains(&var_idx)
                {
                    Some(param_idx)
                } else {
                    None
                }
            })
            .collect();

        ParamQuantifier::set_params(
            self,
            ParamConstraints {
                type_params,
                dyn_lengths,
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
    vararg_lengths: HashSet<usize>,
}

impl<Prim: PrimitiveType> VisitMut<Prim> for PolyTypeTransformer {
    fn visit_type_mut(&mut self, ty: &mut ValueType<Prim>) {
        match ty {
            ValueType::Var(idx) => {
                let type_count = self.mapping.types.len();
                let param_idx = *self.mapping.types.entry(*idx).or_insert(type_count);
                *ty = ValueType::Param(param_idx);
            }
            _ => visit::visit_type_mut(self, ty),
        }
    }

    fn visit_middle_len_mut(&mut self, len: &mut TupleLen) {
        let target_len = match len.components_mut() {
            (Some(var), _) => var,
            _ => return,
        };
        if let UnknownLen::Var(idx) = target_len {
            let len_count = self.mapping.lengths.len();
            let param_idx = *self.mapping.lengths.entry(*idx).or_insert(len_count);
            *target_len = UnknownLen::Param(param_idx);
        }
    }

    fn visit_function_mut(&mut self, function: &mut FnType<Prim>) {
        let (_, vararg, _) = function.args.parts();
        let vararg_len = vararg
            .map(Slice::len)
            .and_then(|len| match len.components() {
                (Some(UnknownLen::Var(idx)), _) => Some(idx),
                _ => None,
            });
        if let Some(vararg_len) = vararg_len {
            self.vararg_lengths.insert(vararg_len);
        }

        visit::visit_function_mut(self, function);
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
    fn visit_type_mut(&mut self, ty: &mut ValueType<Prim>) {
        match ty {
            ValueType::Param(idx) => {
                *ty = self
                    .mapping
                    .types
                    .get(idx)
                    .copied()
                    .map_or(ValueType::Param(*idx), ValueType::Var)
            }
            _ => visit::visit_type_mut(self, ty),
        }
    }

    fn visit_middle_len_mut(&mut self, len: &mut TupleLen) {
        let target_len = match len.components_mut() {
            (Some(var), _) => var,
            _ => return,
        };

        if let UnknownLen::Param(idx) = target_len {
            *target_len = self
                .mapping
                .lengths
                .get(idx)
                .copied()
                .map_or(UnknownLen::Param(*idx), UnknownLen::Var);
        }
    }
}
