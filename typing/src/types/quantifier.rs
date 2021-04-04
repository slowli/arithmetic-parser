//! Type validation.

#![allow(missing_docs)]

use std::{
    collections::{HashMap, HashSet},
    mem,
};

use crate::{
    types::{FnParams, ParamConstraints},
    visit::{self, Visit, VisitMut},
    FnType, LengthKind, PrimitiveType, Tuple, TupleLen, UnknownLen, ValueType,
};

#[derive(Debug, Default)]
struct ParamStats {
    mentioning_fns: HashSet<usize>,
}

#[derive(Debug, Clone, Copy)]
struct FunctionInfo {
    parent: usize,
    depth: usize,
}

#[derive(Debug)]
pub(crate) struct ParamQuantifier {
    type_params: HashMap<usize, ParamStats>,
    len_params: HashMap<usize, ParamStats>,
    functions: Vec<FunctionInfo>,
    current_function_idx: usize,
    current_function_depth: usize,
}

impl ParamQuantifier {
    fn new() -> Self {
        Self {
            type_params: HashMap::new(),
            len_params: HashMap::new(),
            functions: vec![],
            current_function_idx: usize::MAX, // immediately overridden
            current_function_depth: 0,        // immediately overridden
        }
    }

    fn place_param(functions: &[FunctionInfo], stats: ParamStats) -> usize {
        let depths = stats.mentioning_fns.iter().map(|&idx| functions[idx].depth);
        let max_depth = depths
            .clone()
            .max()
            .expect("A param must be mentioned at least once by construction");
        let min_depth = depths
            .min()
            .expect("A param must be mentioned at least once by construction");

        // Group mentions by function depth.
        let mut mentions_by_depth: Vec<HashSet<_>> =
            vec![HashSet::new(); max_depth + 1 - min_depth];
        for idx in stats.mentioning_fns {
            let depth = functions[idx].depth;
            mentions_by_depth[depth - min_depth].insert(idx);
        }

        // Map functions to parents until we have a single function on top.
        let mut depth = max_depth;
        while depth > min_depth {
            let indexes = mentions_by_depth.pop().unwrap();
            let prev_level = mentions_by_depth.last_mut().unwrap();
            prev_level.extend(indexes.into_iter().map(|idx| functions[idx].parent));
            depth -= 1;
        }

        let mut level = mentions_by_depth.pop().unwrap();
        debug_assert!(mentions_by_depth.is_empty());
        while level.len() > 1 {
            level = level.into_iter().map(|idx| functions[idx].parent).collect();
        }
        level.into_iter().next().unwrap()
    }

    fn place_params_of_certain_kind(
        functions: &[FunctionInfo],
        params: HashMap<usize, ParamStats>,
    ) -> HashMap<usize, Vec<usize>> {
        let placements = params
            .into_iter()
            .map(|(idx, stats)| (idx, Self::place_param(functions, stats)));
        let mut params: HashMap<_, Vec<_>> = HashMap::new();
        for (idx, fn_idx) in placements {
            params.entry(fn_idx).or_default().push(idx);
        }

        for function_params in params.values_mut() {
            function_params.sort_unstable();
        }

        params
    }

    fn place_params<Prim>(self, constraints: ParamConstraints<Prim>) -> ParamPlacement<Prim>
    where
        Prim: PrimitiveType,
    {
        let functions = &self.functions;
        ParamPlacement::new(
            Self::place_params_of_certain_kind(functions, self.type_params),
            Self::place_params_of_certain_kind(functions, self.len_params),
            constraints,
        )
    }

    pub fn set_params<Prim: PrimitiveType>(
        function: &mut FnType<Prim>,
        constraints: ParamConstraints<Prim>,
    ) {
        let mut analyzer = Self::new();
        analyzer.visit_function(function);
        let mut placement = analyzer.place_params(constraints);
        placement.visit_function_mut(function);
    }
}

impl<'a, Prim: PrimitiveType> Visit<'a, Prim> for ParamQuantifier {
    fn visit_type(&mut self, ty: &'a ValueType<Prim>) {
        match ty {
            ValueType::Var(var) => {
                assert!(!var.is_free(), "Free vars should be forbidden earlier");
                let stats = self.type_params.entry(var.index()).or_default();
                stats.mentioning_fns.insert(self.current_function_idx);
            }
            _ => visit::visit_type(self, ty),
        }
    }

    fn visit_tuple(&mut self, tuple: &'a Tuple<Prim>) {
        let (_, middle, _) = tuple.parts();
        let middle_len = middle.and_then(|middle| middle.len().components().0);
        let middle_len = if let Some(len) = middle_len {
            len
        } else {
            visit::visit_tuple(self, tuple);
            return;
        };

        if let UnknownLen::Var(var) = middle_len {
            assert!(!var.is_free(), "Free vars should be forbidden earlier");
            let stats = self.len_params.entry(var.index()).or_default();
            stats.mentioning_fns.insert(self.current_function_idx);
        }
        visit::visit_tuple(self, tuple);
    }

    fn visit_function(&mut self, function: &'a FnType<Prim>) {
        let this_function_idx = self.functions.len();
        let old_function_idx = mem::replace(&mut self.current_function_idx, this_function_idx);

        self.functions.push(FunctionInfo {
            parent: old_function_idx,
            depth: self.current_function_depth,
        });
        self.current_function_depth += 1;

        visit::visit_function(self, function);

        self.current_function_idx = old_function_idx;
        self.current_function_depth -= 1;
    }
}

#[derive(Debug)]
struct ParamPlacement<Prim: PrimitiveType> {
    // Grouped by function index.
    type_params: HashMap<usize, Vec<usize>>,
    next_type_param: usize,
    // Grouped by function index.
    len_params: HashMap<usize, Vec<usize>>,
    next_len_param: usize,
    function_count: usize,
    current_function_idx: usize,
    constraints: ParamConstraints<Prim>,
}

impl<Prim: PrimitiveType> ParamPlacement<Prim> {
    fn next_param(params: &HashMap<usize, Vec<usize>>) -> usize {
        params
            .values()
            .flatten()
            .copied()
            .max()
            .map_or(0, |idx| idx + 1)
    }

    fn new(
        type_params: HashMap<usize, Vec<usize>>,
        len_params: HashMap<usize, Vec<usize>>,
        constraints: ParamConstraints<Prim>,
    ) -> Self {
        let next_type_param = Self::next_param(&type_params);
        let next_len_param = Self::next_param(&len_params);
        Self {
            type_params,
            next_type_param,
            len_params,
            next_len_param,
            function_count: 0,
            current_function_idx: usize::MAX,
            constraints,
        }
    }
}

impl<Prim: PrimitiveType> VisitMut<Prim> for ParamPlacement<Prim> {
    fn visit_type_mut(&mut self, ty: &mut ValueType<Prim>) {
        if let ValueType::Some = ty {
            *ty = ValueType::param(self.next_type_param);
            self.type_params
                .entry(self.current_function_idx)
                .or_default()
                .push(self.next_type_param);
            self.next_type_param += 1;
        } else {
            visit::visit_type_mut(self, ty);
        }
    }

    fn visit_middle_len_mut(&mut self, len: &mut TupleLen) {
        let target_len = match len.components_mut().0 {
            Some(target_len) => target_len,
            None => return,
        };
        let is_dynamic = match target_len {
            UnknownLen::Some => false,
            UnknownLen::Dynamic => true,
            _ => return,
        };

        *target_len = UnknownLen::param(self.next_len_param);
        self.len_params
            .entry(self.current_function_idx)
            .or_default()
            .push(self.next_len_param);
        if is_dynamic {
            self.constraints.dyn_lengths.insert(self.next_len_param);
        }
        self.next_len_param += 1;
    }

    // FIXME: what if the params are already present on the `function`?
    fn visit_function_mut(&mut self, function: &mut FnType<Prim>) {
        let this_function_idx = self.function_count;
        let old_function_idx = mem::replace(&mut self.current_function_idx, this_function_idx);
        self.function_count += 1;

        visit::visit_function_mut(self, function);

        let mut params = FnParams::default();
        if let Some(type_params) = self.type_params.remove(&self.current_function_idx) {
            params.type_params = type_params
                .into_iter()
                .map(|idx| {
                    let constraints = self
                        .constraints
                        .type_params
                        .remove(&idx)
                        .unwrap_or_default();
                    (idx, constraints)
                })
                .collect();
        }
        if let Some(len_params) = self.len_params.remove(&self.current_function_idx) {
            params.len_params = len_params
                .into_iter()
                .map(|idx| {
                    let kind = if self.constraints.dyn_lengths.contains(&idx) {
                        LengthKind::Dynamic
                    } else {
                        LengthKind::Static
                    };
                    (idx, kind)
                })
                .collect();
        }
        function.params = Some(params);

        self.current_function_idx = old_function_idx;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Num;

    #[test]
    fn analyzing_map_fn() {
        let map_arg = FnType::builder()
            .with_arg(ValueType::param(0))
            .returning(ValueType::param(1));
        let mut map_fn = <FnType>::builder()
            .with_arg(ValueType::param(0).repeat(UnknownLen::param(0)))
            .with_arg(map_arg)
            .returning(ValueType::param(1).repeat(UnknownLen::param(0)));

        let mut analyzer = ParamQuantifier::new();
        analyzer.visit_function(&map_fn);

        assert_eq!(analyzer.functions.len(), 2);
        assert_eq!(analyzer.functions[0].parent, usize::MAX);
        assert_eq!(analyzer.functions[0].depth, 0);
        assert_eq!(analyzer.functions[1].parent, 0);
        assert_eq!(analyzer.functions[1].depth, 1);

        let mut both_fn_indexes = HashSet::new();
        both_fn_indexes.extend(vec![0_usize, 1]);
        assert_eq!(analyzer.type_params.len(), 2);
        assert_eq!(analyzer.type_params[&0].mentioning_fns, both_fn_indexes);
        assert_eq!(analyzer.type_params[&1].mentioning_fns, both_fn_indexes);

        assert_eq!(analyzer.len_params.len(), 1);
        let mut root_fn_index = HashSet::new();
        root_fn_index.insert(0);
        assert_eq!(analyzer.len_params[&0].mentioning_fns, root_fn_index);

        let mut placement = analyzer.place_params(ParamConstraints::default());
        let expected_type_params: HashMap<_, _> = vec![(0, vec![0, 1])].into_iter().collect();
        assert_eq!(placement.type_params, expected_type_params);
        let expected_len_params: HashMap<_, _> = vec![(0, vec![0])].into_iter().collect();
        assert_eq!(placement.len_params, expected_len_params);

        placement.visit_function_mut(&mut map_fn);
        assert_eq!(
            map_fn.to_string(),
            "fn<len N; T, U>([T; N], fn(T) -> U) -> [U; N]"
        );
    }

    #[test]
    fn transforming_wildcards_into_params() {
        let mut merge_fn = <FnType>::builder()
            .with_arg(ValueType::param(0).repeat(UnknownLen::Some))
            .with_arg(ValueType::param(0).repeat(UnknownLen::Some))
            .returning(ValueType::param(0).repeat(UnknownLen::Dynamic));
        ParamQuantifier::set_params(&mut merge_fn, ParamConstraints::default());
        assert_eq!(
            merge_fn.to_string(),
            "fn<len N, M, L*; T>([T; N], [T; M]) -> [T; L]"
        );
    }

    #[test]
    fn placing_params() {
        #[rustfmt::skip]
        let functions = vec![
            FunctionInfo { parent: usize::MAX, depth: 0 },
            FunctionInfo { parent: 0, depth: 1 },
            FunctionInfo { parent: 1, depth: 2 },
            FunctionInfo { parent: 1, depth: 2 },
            FunctionInfo { parent: 0, depth: 1 },
            FunctionInfo { parent: 4, depth: 2 },
            FunctionInfo { parent: 0, depth: 1 },
        ];
        // Corresponds to this tree:
        //      0
        //  1   4   6
        // 2 3  5

        let type_param_mentions: &[(&[usize], usize)] = &[
            (&[0_usize], 0),
            (&[1], 1),
            (&[2, 4, 5], 0),
            (&[2, 3], 1),
            (&[3], 3),
        ];
        let type_params = type_param_mentions
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, (mentions, _))| {
                (
                    idx,
                    ParamStats {
                        mentioning_fns: mentions.iter().copied().collect(),
                    },
                )
            })
            .collect();

        let analyzer = ParamQuantifier {
            type_params,
            len_params: HashMap::new(),
            functions,
            current_function_idx: 0,
            current_function_depth: 0,
        };
        let placements = analyzer
            .place_params::<Num>(ParamConstraints::default())
            .type_params;

        for (i, (_, expected_placement)) in type_param_mentions.iter().copied().enumerate() {
            assert!(
                placements[&expected_placement].contains(&i),
                "Unexpected placements: {:?}",
                placements
            );
        }
    }
}
