//! Functional type substitutions.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{ConstParamDescription, FnArgs, FnType, TupleLength, TypeParamDescription, ValueType};

impl FnType {
    /// Performs final transformations on this type, transforming all of its type vars
    /// into type params.
    pub(crate) fn finalize(&mut self, linear_types: &HashSet<usize>) {
        let mut tree = FnTypeTree::new(self);
        tree.infer_type_params(&HashSet::new(), &HashSet::new(), linear_types);
        let mapping = tree.create_param_mapping();
        tree.merge_into(self);

        *self = self.substitute_type_vars(&mapping, SubstitutionContext::VarsToParams);
    }

    /// Recursively substitutes type vars in this type according to `type_var_mapping`.
    /// Returns the type with substituted vars.
    pub(super) fn substitute_type_vars(
        &self,
        mapping: &ParamMapping,
        context: SubstitutionContext,
    ) -> Self {
        fn map_params<T: Copy>(
            params: &BTreeMap<usize, T>,
            mapping: &HashMap<usize, usize>,
            context: SubstitutionContext,
        ) -> BTreeMap<usize, T> {
            params
                .iter()
                .filter_map(|(var_idx, description)| {
                    if let Some(mapped_idx) = mapping.get(var_idx) {
                        if context == SubstitutionContext::ParamsToVars {
                            // The params in mapping got instantiated into vars; we must remove them
                            // from the `type_params`.
                            None
                        } else {
                            Some((*mapped_idx, *description))
                        }
                    } else {
                        Some((*var_idx, *description))
                    }
                })
                .collect()
        }

        let substituted_args = match &self.args {
            FnArgs::List(args) => FnArgs::List(
                args.iter()
                    .map(|arg| arg.substitute_type_vars(mapping, context))
                    .collect(),
            ),
            FnArgs::Any => FnArgs::Any,
        };

        FnType {
            args: substituted_args,
            return_type: self.return_type.substitute_type_vars(mapping, context),
            type_params: map_params(&self.type_params, &mapping.types, context),
            const_params: map_params(&self.const_params, &mapping.constants, context),
        }
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
    type_params: BTreeMap<usize, TypeParamDescription>,
    const_params: BTreeMap<usize, ConstParamDescription>,
}

impl FnTypeTree {
    fn new(base: &FnType) -> Self {
        fn recurse(
            children: &mut Vec<FnTypeTree>,
            type_vars: &mut HashMap<usize, VarQuantity>,
            const_vars: &mut HashMap<usize, VarQuantity>,
            ty: &ValueType,
        ) {
            match ty {
                ValueType::TypeVar(idx) => {
                    type_vars
                        .entry(*idx)
                        .and_modify(|qty| *qty = VarQuantity::Repeated)
                        .or_insert(VarQuantity::UniqueVar);
                }

                ValueType::Slice { element, length } => {
                    recurse(children, type_vars, const_vars, element);
                    if let TupleLength::Var(idx) = length {
                        const_vars
                            .entry(*idx)
                            .and_modify(|qty| *qty = VarQuantity::Repeated)
                            .or_insert(VarQuantity::UniqueVar);
                    }
                }

                ValueType::Tuple(elements) => {
                    for element in elements {
                        recurse(children, type_vars, const_vars, element);
                    }
                }

                ValueType::Function(fn_type) => {
                    let child_tree = FnTypeTree::new(fn_type);

                    for &type_var in child_tree.all_type_vars.keys() {
                        type_vars
                            .entry(type_var)
                            .and_modify(|qty| *qty = VarQuantity::Repeated)
                            .or_insert(VarQuantity::UniqueFunction);
                    }
                    for &const_var in child_tree.all_const_vars.keys() {
                        const_vars
                            .entry(const_var)
                            .and_modify(|qty| *qty = VarQuantity::Repeated)
                            .or_insert(VarQuantity::UniqueFunction);
                    }

                    children.push(child_tree);
                }

                _ => { /* Do nothing. */ }
            }
        }

        let mut children = vec![];
        let mut all_type_vars = HashMap::new();
        let mut all_const_vars = HashMap::new();
        for ty in base.arg_and_return_types() {
            recurse(&mut children, &mut all_type_vars, &mut all_const_vars, ty);
        }

        Self {
            children,
            all_type_vars,
            all_const_vars,
            type_params: BTreeMap::new(),
            const_params: BTreeMap::new(),
        }
    }

    /// Recursively infers type params for the function and all child functions.
    fn infer_type_params(
        &mut self,
        parent_types: &HashSet<usize>,
        parent_consts: &HashSet<usize>,
        linear_types: &HashSet<usize>,
    ) {
        self.type_params = self
            .all_type_vars
            .iter()
            .filter_map(|(idx, qty)| {
                if *qty != VarQuantity::UniqueFunction && !parent_types.contains(idx) {
                    let description = TypeParamDescription {
                        maybe_non_linear: !linear_types.contains(idx),
                    };
                    Some((*idx, description))
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
                    Some((*idx, ConstParamDescription))
                } else {
                    None
                }
            })
            .collect();

        let parent_and_self_types: HashSet<_> = self.type_params.keys().copied().collect();
        let parent_and_self_consts: HashSet<_> = self.const_params.keys().copied().collect();
        for child_tree in &mut self.children {
            child_tree.infer_type_params(
                &parent_and_self_types,
                &parent_and_self_consts,
                linear_types,
            );
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
    fn merge_into(self, base: &mut FnType) {
        fn recurse(reversed_children: &mut Vec<FnTypeTree>, ty: &mut ValueType) {
            match ty {
                ValueType::Tuple(elements) => {
                    for element in elements {
                        recurse(reversed_children, element);
                    }
                }

                ValueType::Slice { element, .. } => {
                    recurse(reversed_children, element);
                }

                ValueType::Function(fn_type) => {
                    reversed_children
                        .pop()
                        .expect("Missing child")
                        .merge_into(fn_type);
                }

                _ => { /* Do nothing. */ }
            }
        }

        base.type_params = self.type_params;
        base.const_params = self.const_params;
        let mut reversed_children = self.children;
        reversed_children.reverse();
        for ty in base.arg_and_return_types_mut() {
            recurse(&mut reversed_children, ty);
        }
    }
}

impl TupleLength {
    fn substitute_vars(self, mapping: &ParamMapping, context: SubstitutionContext) -> Self {
        match self {
            Self::Var(idx) if context == SubstitutionContext::VarsToParams => {
                Self::Param(mapping.constants[&idx])
            }

            Self::Param(idx) if context == SubstitutionContext::ParamsToVars => {
                if let Some(mapped_idx) = mapping.constants.get(&idx) {
                    Self::Var(*mapped_idx)
                } else {
                    Self::Param(idx)
                }
            }

            _ => self,
        }
    }
}

impl ValueType {
    /// Recursively substitutes type vars in this type according to `type_var_mapping`.
    /// Returns the type with substituted vars.
    fn substitute_type_vars(&self, mapping: &ParamMapping, context: SubstitutionContext) -> Self {
        match self {
            Self::TypeVar(idx) if context == SubstitutionContext::VarsToParams => {
                Self::TypeParam(mapping.types[idx])
            }
            Self::TypeParam(idx) if context == SubstitutionContext::ParamsToVars => {
                if let Some(mapped_idx) = mapping.types.get(idx) {
                    Self::TypeVar(*mapped_idx)
                } else {
                    // This type param is not mapped; it is retained in the resulting function.
                    Self::TypeParam(*idx)
                }
            }

            Self::Slice { element, length } => Self::Slice {
                element: Box::new(element.substitute_type_vars(mapping, context)),
                length: length.substitute_vars(mapping, context),
            },

            Self::Tuple(fragments) => ValueType::Tuple(
                fragments
                    .iter()
                    .map(|element| element.substitute_type_vars(mapping, context))
                    .collect(),
            ),

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

/// Context for mapping `TypeVar`s into `TypeParam`s or back.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum SubstitutionContext {
    /// Mapping `TypeVar`s to `TypeParam`s. This occurs for top-level functions once
    /// their signature is established.
    VarsToParams,
    /// Mapping `TypeParam`s to `TypeVar`s. This occurs when instantiating a function type
    /// with type params.
    ParamsToVars,
}
