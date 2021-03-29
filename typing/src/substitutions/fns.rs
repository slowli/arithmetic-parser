//! Functional type substitutions.

use std::collections::{HashMap, HashSet};

use crate::types::LenParamDescription;
use crate::{
    types::TypeParamDescription,
    visit::{self, Visit, VisitMut},
    FnType, PrimitiveType, Substitutions, Tuple, TupleLength, ValueType,
};

impl<Prim: PrimitiveType> FnType<Prim> {
    /// Performs final transformations on this type, transforming all of its type vars
    /// into type params.
    pub(crate) fn finalize(&mut self, substitutions: &Substitutions<Prim>) {
        let mut tree = FnTypeTree::new(self);
        tree.infer_type_params(&HashSet::new(), &HashSet::new());
        let mapping = tree.create_param_mapping();
        tree.merge_into(self, substitutions);

        *self = self.substitute_type_vars(&mapping, SubstitutionContext::VarsToParams);
    }

    /// Recursively substitutes type vars in this type according to `type_var_mapping`.
    /// Returns the type with substituted vars.
    pub(super) fn substitute_type_vars(
        &self,
        mapping: &ParamMapping,
        context: SubstitutionContext,
    ) -> Self {
        #[allow(clippy::option_if_let_else)] // false positive
        fn map_params<'a, T: 'static>(
            params: impl Iterator<Item = (usize, T)> + 'a,
            mapping: &'a HashMap<usize, usize>,
            context: SubstitutionContext,
        ) -> impl Iterator<Item = (usize, T)> + 'a {
            params.filter_map(move |(var_idx, description)| {
                if let Some(mapped_idx) = mapping.get(&var_idx) {
                    if context == SubstitutionContext::ParamsToVars {
                        // The params in mapping got instantiated into vars;
                        // we must remove them from the `type_params`.
                        None
                    } else {
                        Some((*mapped_idx, description))
                    }
                } else {
                    Some((var_idx, description))
                }
            })
        }

        let substituted_args = self.args.map(
            |arg| arg.substitute_type_vars(mapping, context),
            |len| len.substitute_vars(mapping, context),
        );
        let return_type = self.return_type.substitute_type_vars(mapping, context);

        let const_params = self.len_params.iter().copied();
        let const_params = map_params(const_params, &mapping.constants, context);
        let type_params = self.type_params.iter().cloned();
        let type_params = map_params(type_params, &mapping.types, context);

        FnType::new(substituted_args, return_type)
            .with_len_params(const_params.collect())
            .with_type_params(type_params.collect())
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
    all_const_vars: HashMap<usize, VarQuantity>,
    type_params: HashSet<usize>,
    const_params: HashSet<usize>,
}

// TODO: add unit tests covering main methods
impl FnTypeTree {
    fn new<Prim: PrimitiveType>(base: &FnType<Prim>) -> Self {
        #[derive(Debug, Default)]
        struct VarExtractor {
            children: Vec<FnTypeTree>,
            type_vars: HashMap<usize, VarQuantity>,
            const_vars: HashMap<usize, VarQuantity>,
        }

        impl<'ast, P: PrimitiveType> Visit<'ast, P> for VarExtractor {
            fn visit_var(&mut self, index: usize) {
                self.type_vars
                    .entry(index)
                    .and_modify(|qty| *qty = VarQuantity::Repeated)
                    .or_insert(VarQuantity::UniqueVar);
            }

            fn visit_tuple(&mut self, tuple: &'ast Tuple<P>) {
                // FIXME: handle compound lengths
                if let TupleLength::Var(idx) = tuple.len() {
                    self.const_vars
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
                for &const_var in child_tree.all_const_vars.keys() {
                    self.const_vars
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
            all_const_vars: extractor.const_vars,
            type_params: HashSet::new(),
            const_params: HashSet::new(),
        }
    }

    /// Recursively infers type params for the function and all child functions.
    fn infer_type_params(&mut self, parent_types: &HashSet<usize>, parent_consts: &HashSet<usize>) {
        self.type_params = self
            .all_type_vars
            .iter()
            .filter_map(|(idx, qty)| {
                if *qty != VarQuantity::UniqueFunction && !parent_types.contains(idx) {
                    Some(*idx)
                } else {
                    None
                }
            })
            .collect();

        self.const_params = self
            .all_const_vars
            .iter()
            .filter_map(|(idx, qty)| {
                if *qty != VarQuantity::UniqueFunction && !parent_consts.contains(idx) {
                    Some(*idx)
                } else {
                    None
                }
            })
            .collect();

        let mut parent_and_self_types = parent_types.clone();
        parent_and_self_types.extend(self.type_params.iter().copied());
        let mut parent_and_self_consts = parent_consts.clone();
        parent_and_self_consts.extend(self.const_params.iter().copied());

        for child_tree in &mut self.children {
            child_tree.infer_type_params(&parent_and_self_types, &parent_and_self_consts);
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
        let mut sorted_const_params: Vec<_> = self.all_const_vars.keys().copied().collect();
        sorted_const_params.sort_unstable();

        ParamMapping {
            types: map_enumerated(sorted_type_params),
            constants: map_enumerated(sorted_const_params),
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
        let vararg_length = base.args.middle_len_mut().and_then(|len| match len {
            TupleLength::Var(idx) => Some(*idx),
            _ => None,
        });
        // `vararg_length` is dynamic within function context, but must be set to non-dynamic
        // for the function definition.

        base.len_params = self
            .const_params
            .into_iter()
            .map(|idx| {
                let is_dynamic = vararg_length != Some(idx) && dynamic_lengths.contains(&idx);
                (idx, LenParamDescription { is_dynamic })
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

impl TupleLength {
    fn substitute_vars(&self, mapping: &ParamMapping, context: SubstitutionContext) -> Self {
        match self {
            Self::Var(idx) if context == SubstitutionContext::VarsToParams => {
                Self::Param(mapping.constants[idx])
            }

            Self::Param(idx) if context == SubstitutionContext::ParamsToVars => mapping
                .constants
                .get(idx)
                .copied()
                .map_or(Self::Param(*idx), Self::Var),

            _ => self.to_owned(),
        }
    }
}

impl<Prim: PrimitiveType> ValueType<Prim> {
    /// Recursively substitutes type vars in this type according to `type_var_mapping`.
    /// Returns the type with substituted vars.
    fn substitute_type_vars(&self, mapping: &ParamMapping, context: SubstitutionContext) -> Self {
        match self {
            Self::Var(idx) if context == SubstitutionContext::VarsToParams => {
                Self::Param(mapping.types[idx])
            }
            Self::Param(idx) if context == SubstitutionContext::ParamsToVars => mapping
                .types
                .get(idx)
                .copied()
                .map_or(Self::Param(*idx), Self::Var),

            Self::Tuple(tuple) => {
                let mapped_tuple = tuple.map(
                    |element| element.substitute_type_vars(mapping, context),
                    |len| len.substitute_vars(mapping, context),
                );
                ValueType::Tuple(mapped_tuple)
            }

            Self::Function(fn_type) => {
                Self::Function(Box::new(fn_type.substitute_type_vars(mapping, context)))
            }

            _ => self.clone(),
        }
    }
}

#[derive(Debug)]
pub(super) struct ParamMapping {
    pub types: HashMap<usize, usize>,
    pub constants: HashMap<usize, usize>,
}

/// Context for mapping `Var`s into `Param`s or back.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum SubstitutionContext {
    /// Mapping `Var`s to `Param`s. This occurs for top-level functions once
    /// their signature is established.
    VarsToParams,
    /// Mapping `Param`s to `Var`s. This occurs when instantiating a function type
    /// with type params.
    ParamsToVars,
}
