//! Showcases `Visit` and `VisitMut` traits to perform variable quantization.

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    mem,
};

use arithmetic_typing::{
    visit::{self, Visit, VisitMut},
    FnType, LengthKind, Prelude, PrimitiveType, SimpleTupleLen, Tuple, TupleLen, ValueType,
};

/// Function quantifier that allows to transform implicitly quantified types (`ValueType::Some`)
/// and lengths (`TupleLen::Some`, `TupleLen::Dynamic`) into explicit type params of the wrapping
/// function.
#[derive(Debug, Default)]
struct FnQuantifier {
    type_params: BTreeSet<usize>,
    new_type_params: Vec<HashSet<usize>>,
    len_params: BTreeSet<usize>,
    new_len_params: Vec<HashMap<usize, LengthKind>>,
    is_in_function: bool,
}

impl FnQuantifier {
    fn next_value(params: &BTreeSet<usize>) -> usize {
        params
            .iter()
            .rev()
            .next()
            .copied()
            .map(|largest| largest + 1)
            .unwrap_or(0)
    }
}

impl<Prim: PrimitiveType> VisitMut<Prim> for FnQuantifier {
    fn visit_type_mut(&mut self, ty: &mut ValueType<Prim>) {
        match ty {
            ValueType::Some => {
                let next_type_param = Self::next_value(&self.type_params);
                *ty = ValueType::Param(next_type_param);
                self.type_params.insert(next_type_param);
                self.new_type_params
                    .last_mut()
                    .unwrap()
                    .insert(next_type_param);
            }
            other => visit::visit_type_mut(self, other),
        }
    }

    fn visit_middle_len_mut(&mut self, len: &mut TupleLen) {
        let target_len = match len.components_mut().0 {
            Some(len) => len,
            None => return,
        };
        let kind = match target_len {
            SimpleTupleLen::Some => LengthKind::Static,
            SimpleTupleLen::Dynamic => LengthKind::Dynamic,
            _ => return,
        };

        let next_len_param = Self::next_value(&self.len_params);
        *target_len = SimpleTupleLen::Param(next_len_param);
        self.len_params.insert(next_len_param);
        self.new_len_params
            .last_mut()
            .unwrap()
            .insert(next_len_param, kind);
    }

    fn visit_function_mut(&mut self, function: &mut FnType<Prim>) {
        self.len_params
            .extend(function.len_params().map(|(idx, _)| idx));
        self.type_params
            .extend(function.type_params().map(|(idx, _)| idx));
        self.new_len_params.push(HashMap::new());
        self.new_type_params.push(HashSet::new());

        let was_in_function = mem::replace(&mut self.is_in_function, true);
        visit::visit_function_mut(self, function);

        for (idx, kind) in self.new_len_params.pop().unwrap() {
            function.insert_len_param(idx, kind);
        }
        for idx in self.new_type_params.pop().unwrap() {
            function.insert_type_param(idx, Prim::Constraints::default());
        }

        // If we are done with a top-level function, we can reset length / type param
        // indexes: each top-level function can be treated independently.
        self.is_in_function = was_in_function;
        if !was_in_function {
            debug_assert!(self.new_len_params.is_empty());
            debug_assert!(self.new_type_params.is_empty());
            self.len_params.clear();
            self.type_params.clear();
        }
    }
}

pub fn quantify<Prim: PrimitiveType>(ty: &mut ValueType<Prim>) {
    FnQuantifier::default().visit_type_mut(ty);
}

/// `Visit`or that counts mentions of type / length params in function declarations.
#[derive(Debug, Default)]
struct Mentions {
    type_params: HashMap<usize, usize>,
    len_params: HashMap<usize, usize>,
}

impl<'a, Prim: PrimitiveType> Visit<'a, Prim> for Mentions {
    fn visit_param(&mut self, index: usize) {
        *self.type_params.entry(index).or_default() += 1;
    }

    fn visit_tuple(&mut self, tuple: &'a Tuple<Prim>) {
        let (_, middle, _) = tuple.parts();
        let len = middle.and_then(|middle| middle.len().components().0);
        if let Some(SimpleTupleLen::Param(idx)) = len {
            *self.len_params.entry(idx).or_default() += 1;
        }
        visit::visit_tuple(self, tuple);
    }
}

#[derive(Debug, Default)]
struct FnSimplifier {
    removed_types: HashSet<usize>,
    removed_lengths: HashMap<usize, LengthKind>,
}

impl<Prim: PrimitiveType> VisitMut<Prim> for FnSimplifier {
    fn visit_type_mut(&mut self, ty: &mut ValueType<Prim>) {
        match ty {
            ValueType::Param(idx) if self.removed_types.contains(idx) => {
                *ty = ValueType::Some;
            }
            other => visit::visit_type_mut(self, other),
        }
    }

    fn visit_middle_len_mut(&mut self, len: &mut TupleLen) {
        let target_len = match len.components_mut().0 {
            Some(target_len) => target_len,
            None => return,
        };
        if let SimpleTupleLen::Param(idx) = target_len {
            if let Some(kind) = self.removed_lengths.get(&idx) {
                *target_len = match kind {
                    LengthKind::Static => SimpleTupleLen::Some,
                    LengthKind::Dynamic => SimpleTupleLen::Dynamic,
                    _ => unreachable!(),
                }
            }
        }
    }

    fn visit_function_mut(&mut self, function: &mut FnType<Prim>) {
        let mut mentions = Mentions::default();
        mentions.visit_function(function);

        let def = Prim::Constraints::default();
        let mut removed_types: HashSet<_> = function
            .type_params()
            .filter_map(|(idx, constraints)| {
                if mentions.type_params.get(&idx) == Some(&1) && *constraints == def {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        for &idx in &removed_types {
            function.remove_type_param(idx);
        }
        // Store removed types from the parent function; we will restore them later.
        mem::swap(&mut removed_types, &mut self.removed_types);

        let mut removed_lengths: HashMap<_, _> = function
            .len_params()
            .filter_map(|(idx, kind)| {
                if mentions.len_params.get(&idx) == Some(&1) {
                    Some((idx, kind))
                } else {
                    None
                }
            })
            .collect();
        for &idx in removed_lengths.keys() {
            function.remove_len_param(idx);
        }
        mem::swap(&mut removed_lengths, &mut self.removed_lengths);

        visit::visit_function_mut(self, function);

        // Restore stashed types / lengths to remove.
        self.removed_types = removed_types;
        self.removed_lengths = removed_lengths;
    }
}

fn simplify<Prim: PrimitiveType>(ty: &mut ValueType<Prim>) {
    FnSimplifier::default().visit_type_mut(ty);
}

fn roundtrip(simple_str: &str, mut std_type: ValueType) -> anyhow::Result<()> {
    let mut simple_type: ValueType = simple_str.parse()?;
    quantify(&mut simple_type);
    println!("Quantified {} to {}", simple_str, simple_type);

    assert_eq!(simple_type, std_type);

    simplify(&mut std_type);
    println!("Simplified {} to {}", simple_type, std_type);
    assert_eq!(std_type.to_string(), simple_str);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let merge_fn_str = "fn<T>([T; _], [T; _]) -> [T]";
    roundtrip(merge_fn_str, Prelude::Merge.into())?;

    let fold_fn_str = "fn<T, U>([T; _], U, fn(U, T) -> U) -> U";
    roundtrip(fold_fn_str, Prelude::Fold.into())?;

    let tuple_str = "(fn([Num; _]) -> Num, fn<T>(T) -> fn([T; _]) -> [T])";
    let std_tuple_str = "(fn<len N>([Num; N]) -> Num, fn<T>(T) -> fn<len N, M*>([T; N]) -> [T; M])";
    roundtrip(tuple_str, std_tuple_str.parse()?)
}
