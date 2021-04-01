//! Functional type substitutions.

use std::collections::{HashMap, HashSet};

use crate::{
    types::TypeParamDescription,
    visit::{self, Visit, VisitMut},
    FnType, LengthKind, PrimitiveType, SimpleTupleLen, Slice, Substitutions, Tuple, TupleLen,
    ValueType,
};

impl<Prim: PrimitiveType> FnType<Prim> {
    /// Performs final transformations on this type, transforming all of its type vars
    /// into type params.
    pub(crate) fn finalize(&mut self, substitutions: &Substitutions<Prim>) {
        let mut tree = FnTypeTree::new(self);
        tree.infer_type_params(&HashSet::new(), &HashSet::new());
        let mapping = tree.create_param_mapping();
        tree.merge_into(self, substitutions);

        PolyTypeTransformer { mapping }.visit_function_mut(self);
    }
}

/// Quantity of type variable mentions.
#[derive(Debug, Clone, Copy, PartialEq)]
enum VarQuantity {
    UniqueVar,
    UniqueFunction,
    Repeated,
}

/// Helper type for inferring type parameters in functions.
///
/// The gist of the problem is that type params placement belongs which function in a tree
/// (which is ultimately rooted in a function not defined within another function) has exclusive
/// mention of a certain type var.
#[derive(Debug)]
struct FnTypeTree {
    children: Vec<FnTypeTree>,
    all_type_vars: HashMap<usize, VarQuantity>,
    all_length_vars: HashMap<usize, VarQuantity>,
    type_params: HashSet<usize>,
    length_params: HashSet<usize>,
}

// TODO: add unit tests covering main methods
impl FnTypeTree {
    fn new<Prim: PrimitiveType>(base: &FnType<Prim>) -> Self {
        #[derive(Debug, Default)]
        struct VarExtractor {
            children: Vec<FnTypeTree>,
            type_vars: HashMap<usize, VarQuantity>,
            length_vars: HashMap<usize, VarQuantity>,
        }

        impl<'ast, P: PrimitiveType> Visit<'ast, P> for VarExtractor {
            fn visit_var(&mut self, index: usize) {
                self.type_vars
                    .entry(index)
                    .and_modify(|qty| *qty = VarQuantity::Repeated)
                    .or_insert(VarQuantity::UniqueVar);
            }

            fn visit_tuple(&mut self, tuple: &'ast Tuple<P>) {
                let (_, middle, _) = tuple.parts();
                let var_len = middle.and_then(|middle| middle.len().components().0);

                if let Some(SimpleTupleLen::Var(idx)) = var_len {
                    self.length_vars
                        .entry(idx)
                        .and_modify(|qty| *qty = VarQuantity::Repeated)
                        .or_insert(VarQuantity::UniqueVar);
                }

                visit::visit_tuple(self, tuple);
            }

            fn visit_function(&mut self, function: &'ast FnType<P>) {
                let child_tree = FnTypeTree::new(function);

                for &type_var in child_tree.all_type_vars.keys() {
                    self.type_vars
                        .entry(type_var)
                        .and_modify(|qty| *qty = VarQuantity::Repeated)
                        .or_insert(VarQuantity::UniqueFunction);
                }
                for &const_var in child_tree.all_length_vars.keys() {
                    self.length_vars
                        .entry(const_var)
                        .and_modify(|qty| *qty = VarQuantity::Repeated)
                        .or_insert(VarQuantity::UniqueFunction);
                }

                self.children.push(child_tree);
            }
        }

        let mut extractor = VarExtractor::default();
        visit::visit_function(&mut extractor, base);

        Self {
            children: extractor.children,
            all_type_vars: extractor.type_vars,
            all_length_vars: extractor.length_vars,
            type_params: HashSet::new(),
            length_params: HashSet::new(),
        }
    }

    /// Recursively infers type params for the function and all child functions.
    fn infer_type_params(
        &mut self,
        parent_types: &HashSet<usize>,
        parent_lengths: &HashSet<usize>,
    ) {
        fn filter_params(
            quantities: &HashMap<usize, VarQuantity>,
            parent_params: &HashSet<usize>,
        ) -> HashSet<usize> {
            quantities
                .iter()
                .filter_map(|(idx, qty)| {
                    if *qty != VarQuantity::UniqueFunction && !parent_params.contains(idx) {
                        Some(*idx)
                    } else {
                        None
                    }
                })
                .collect()
        }

        self.type_params = filter_params(&self.all_type_vars, parent_types);
        self.length_params = filter_params(&self.all_length_vars, parent_lengths);

        let mut parent_and_self_types = parent_types.clone();
        parent_and_self_types.extend(self.type_params.iter().copied());
        let mut parent_and_self_lengths = parent_lengths.clone();
        parent_and_self_lengths.extend(self.length_params.iter().copied());

        for child_tree in &mut self.children {
            child_tree.infer_type_params(&parent_and_self_types, &parent_and_self_lengths);
        }
    }

    fn create_param_mapping(&self) -> ParamMapping {
        fn map_enumerated(values: Vec<usize>) -> HashMap<usize, usize> {
            values
                .into_iter()
                .enumerate()
                .map(|(i, var_idx)| (var_idx, i))
                .collect()
        }

        let mut sorted_type_params: Vec<_> = self.all_type_vars.keys().copied().collect();
        sorted_type_params.sort_unstable();
        let mut sorted_const_params: Vec<_> = self.all_length_vars.keys().copied().collect();
        sorted_const_params.sort_unstable();

        ParamMapping {
            types: map_enumerated(sorted_type_params),
            lengths: map_enumerated(sorted_const_params),
        }
    }

    /// Recursively sets `type_params` for `base`.
    fn merge_into<Prim: PrimitiveType>(
        self,
        base: &mut FnType<Prim>,
        substitutions: &Substitutions<Prim>,
    ) {
        #[derive(Debug)]
        struct Merger<'a, P: PrimitiveType> {
            reversed_children: Vec<FnTypeTree>,
            substitutions: &'a Substitutions<P>,
        }

        impl<P: PrimitiveType> VisitMut<P> for Merger<'_, P> {
            fn visit_function_mut(&mut self, function: &mut FnType<P>) {
                self.reversed_children
                    .pop()
                    .expect("Missing child")
                    .merge_into(function, self.substitutions);
            }
        }

        let constraints = substitutions.type_constraints();
        base.type_params = self
            .type_params
            .into_iter()
            .map(|idx| {
                let type_constraints = constraints.get(&idx).cloned().unwrap_or_default();
                (idx, TypeParamDescription::new(type_constraints))
            })
            .collect();
        base.type_params.sort_unstable_by_key(|(idx, _)| *idx);

        let dynamic_lengths = substitutions.dyn_lengths();

        let (_, vararg, _) = base.args.parts();
        let vararg_length = vararg
            .map(Slice::len)
            .and_then(|len| match len.components() {
                (Some(SimpleTupleLen::Var(idx)), _) => Some(idx),
                _ => None,
            });
        // `vararg_length` is dynamic within function context, but must be set to non-dynamic
        // for the function definition.

        base.len_params = self
            .length_params
            .into_iter()
            .map(|idx| {
                let kind = if vararg_length != Some(idx) && dynamic_lengths.contains(&idx) {
                    LengthKind::Dynamic
                } else {
                    LengthKind::Static
                };
                (idx, kind.into())
            })
            .collect();
        base.len_params.sort_unstable_by_key(|(idx, _)| *idx);

        let mut reversed_children = self.children;
        reversed_children.reverse();
        let mut merger = Merger {
            reversed_children,
            substitutions,
        };
        visit::visit_function_mut(&mut merger, base);
    }
}

#[derive(Debug)]
pub(super) struct ParamMapping {
    pub types: HashMap<usize, usize>,
    pub lengths: HashMap<usize, usize>,
}

/// Makes functional types polymorphic on all free type vars.
#[derive(Debug)]
struct PolyTypeTransformer {
    mapping: ParamMapping,
}

impl<Prim: PrimitiveType> VisitMut<Prim> for PolyTypeTransformer {
    fn visit_type_mut(&mut self, ty: &mut ValueType<Prim>) {
        match ty {
            ValueType::Var(idx) => {
                *ty = ValueType::Param(self.mapping.types[idx]);
            }
            _ => visit::visit_type_mut(self, ty),
        }
    }

    fn visit_middle_len_mut(&mut self, len: &mut TupleLen) {
        let target_len = match len.components_mut() {
            (Some(var), _) => var,
            _ => return,
        };
        if let SimpleTupleLen::Var(idx) = target_len {
            *target_len = SimpleTupleLen::Param(self.mapping.lengths[idx]);
        }
    }

    fn visit_function_mut(&mut self, function: &mut FnType<Prim>) {
        for (idx, _) in &mut function.type_params {
            *idx = self.mapping.types[idx];
        }
        function.type_params.sort_unstable_by_key(|(idx, _)| *idx);

        for (idx, _) in &mut function.len_params {
            *idx = self.mapping.lengths[idx];
        }
        function.len_params.sort_unstable_by_key(|(idx, _)| *idx);

        visit::visit_function_mut(self, function);
    }
}

/// Makes functional types monomorphic by replacing type / length params with vars.
#[derive(Debug)]
pub(super) struct MonoTypeTransformer<'a> {
    mapping: &'a ParamMapping,
}

impl<'a> MonoTypeTransformer<'a> {
    pub fn new(mapping: &'a ParamMapping) -> Self {
        Self { mapping }
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

        if let SimpleTupleLen::Param(idx) = target_len {
            *target_len = self
                .mapping
                .lengths
                .get(idx)
                .copied()
                .map_or(SimpleTupleLen::Param(*idx), SimpleTupleLen::Var);
        }
    }

    fn visit_function_mut(&mut self, function: &mut FnType<Prim>) {
        function
            .type_params
            .retain(|(idx, _)| !self.mapping.types.contains_key(idx));
        function
            .len_params
            .retain(|(idx, _)| !self.mapping.lengths.contains_key(idx));

        visit::visit_function_mut(self, function);
    }
}
