//! Functional type substitutions.

use std::{
    collections::{hash_map::Entry, HashMap},
    sync::Arc,
};

use crate::{
    arith::{CompleteConstraints, Substitutions},
    types::{FnParams, ParamConstraints, ParamQuantifier},
    visit::{self, VisitMut},
    Function, Object, PrimitiveType, TupleLen, Type, UnknownLen,
};

impl<Prim: PrimitiveType> Function<Prim> {
    /// Performs final transformations on this function, bounding all of its type vars
    /// to the function or its child functions.
    pub(crate) fn finalize(&mut self, substitutions: &Substitutions<Prim>) {
        // 1. Replace `Var`s with `Param`s.
        let mut transformer = PolyTypeTransformer::new(substitutions);
        transformer.visit_function_mut(self);
        let mapping = transformer.mapping;
        let mut resolved_objects = transformer.resolved_objects;

        // 2. Extract constraints on type params and lengths.
        let type_params = mapping
            .types
            .into_iter()
            .filter_map(|(var_idx, param_idx)| {
                let constraints = substitutions.constraints.get(&var_idx);
                constraints
                    .filter(|constraints| !constraints.is_empty())
                    .cloned()
                    .map(|constraints| {
                        let resolved = constraints.map_object(|object| {
                            if let Some(resolved) = resolved_objects.remove(&var_idx) {
                                *object = resolved;
                            }
                        });
                        (param_idx, resolved)
                    })
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
        ParamQuantifier::fill_params(
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
#[derive(Debug)]
struct PolyTypeTransformer<'a, Prim: PrimitiveType> {
    mapping: ParamMapping,
    resolved_objects: HashMap<usize, Object<Prim>>,
    substitutions: &'a Substitutions<Prim>,
}

impl<'a, Prim: PrimitiveType> PolyTypeTransformer<'a, Prim> {
    fn new(substitutions: &'a Substitutions<Prim>) -> Self {
        Self {
            mapping: ParamMapping::default(),
            resolved_objects: HashMap::new(),
            substitutions,
        }
    }

    fn object_constraint(&self, var_idx: usize) -> Option<&'a Object<Prim>> {
        let constraints = self.substitutions.constraints.get(&var_idx)?;
        constraints.object.as_ref()
    }
}

impl<Prim: PrimitiveType> VisitMut<Prim> for PolyTypeTransformer<'_, Prim> {
    fn visit_type_mut(&mut self, ty: &mut Type<Prim>) {
        match ty {
            Type::Var(var) if var.is_free() => {
                let type_count = self.mapping.types.len();
                let var_idx = var.index();
                let entry = self.mapping.types.entry(var_idx);
                let is_new_var = matches!(entry, Entry::Vacant(_));
                let param_idx = *entry.or_insert(type_count);
                *ty = Type::param(param_idx);

                if is_new_var {
                    // Resolve object constraints only when we're visiting the variable the
                    // first time.
                    if let Some(object) = self.object_constraint(var_idx) {
                        let mut resolved_object = object.clone();
                        self.substitutions
                            .resolver()
                            .visit_object_mut(&mut resolved_object);
                        self.visit_object_mut(&mut resolved_object);
                        self.resolved_objects.insert(var_idx, resolved_object);
                    }
                }
            }
            _ => visit::visit_type_mut(self, ty),
        }
    }

    fn visit_middle_len_mut(&mut self, len: &mut TupleLen) {
        let (Some(target_len), _) = len.components_mut() else { return };
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
    pub fn transform<Prim: PrimitiveType>(
        mapping: &'a ParamMapping,
        function: &mut Function<Prim>,
    ) {
        function.params = None;
        Self { mapping }.visit_function_mut(function);
    }

    pub fn transform_constraints<Prim: PrimitiveType>(
        mapping: &'a ParamMapping,
        constraints: &CompleteConstraints<Prim>,
    ) -> CompleteConstraints<Prim> {
        constraints.clone().map_object(|object| {
            Self { mapping }.visit_object_mut(object);
        })
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
        let (Some(target_len), _) = len.components_mut() else { return };
        if let UnknownLen::Var(var) = target_len {
            if !var.is_free() {
                if let Some(mapped_len) = self.mapping.lengths.get(&var.index()) {
                    *target_len = UnknownLen::free_var(*mapped_len);
                }
            }
        }
    }

    fn visit_function_mut(&mut self, function: &mut Function<Prim>) {
        visit::visit_function_mut(self, function);

        if let Some(params) = function.params.as_deref() {
            // TODO: make this check more precise?
            let needs_modifying = params
                .type_params
                .iter()
                .any(|(_, type_params)| type_params.object.is_some());

            // We need to monomorphize types in the object constraint as well.
            if needs_modifying {
                let mapped_params = params.type_params.iter().map(|(i, constraints)| {
                    let mapped_constraints = constraints
                        .clone()
                        .map_object(|object| self.visit_object_mut(object));
                    (*i, mapped_constraints)
                });
                function.params = Some(Arc::new(FnParams {
                    type_params: mapped_params.collect(),
                    len_params: params.len_params.clone(),
                    constraints: None,
                }));
            }
        }
    }
}
