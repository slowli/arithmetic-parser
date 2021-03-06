use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
};

mod env;
mod error;
mod substitutions;

pub use self::{env::TypeEnvironment, error::TypeError};

#[derive(Debug, Clone, Copy, PartialEq)]
struct TypeParamDescription {
    maybe_non_linear: bool,
    is_external: bool,
}

/// Quantity of type variable mentions.
#[derive(Debug, Clone, Copy, PartialEq)]
enum TypeVarQuantity {
    UniqueVar,
    UniqueFunction,
    Repeated,
}

/// Functional type.
#[derive(Debug, Clone, PartialEq)]
pub struct FnType {
    /// Type of function arguments.
    args: FnArgs,
    /// Type of the value returned by the function.
    return_type: ValueType,
    /// Indexes of type params associated with this function. The params can be either free
    /// or bound by the parent scope.
    type_params: BTreeMap<usize, TypeParamDescription>,
}

impl fmt::Display for FnType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("fn")?;

        let free_params = self
            .type_params
            .iter()
            .filter(|(_, description)| !description.is_external);
        let free_params_count = free_params.clone().count();

        if free_params_count > 0 {
            formatter.write_str("<")?;
            for (i, (&var_idx, description)) in free_params.enumerate() {
                formatter.write_str(ValueType::type_param(var_idx).as_ref())?;
                if description.maybe_non_linear {
                    formatter.write_str(": ?Lin")?;
                }
                if i + 1 < free_params_count {
                    formatter.write_str(", ")?;
                }
            }
            formatter.write_str(">")?;
        }

        write!(formatter, "({})", self.args)?;
        if !self.return_type.is_void() {
            write!(formatter, " -> {}", self.return_type)?;
        }
        Ok(())
    }
}

impl FnType {
    pub(crate) fn new(args: Vec<ValueType>, return_type: ValueType) -> Self {
        Self {
            args: FnArgs::List(args),
            return_type,
            type_params: BTreeMap::new(), // filled in later
        }
    }

    /// Performs final transformations on this type, transforming all of its type vars
    /// into type params.
    pub(crate) fn finalize(&mut self, linear_types: &HashSet<usize>) {
        let mut tree = FnTypeTree::new(self);
        tree.infer_type_params(&HashSet::new(), linear_types);

        let mut sorted_type_params: Vec<_> = tree.all_type_vars.keys().copied().collect();
        sorted_type_params.sort_unstable();

        tree.merge_into(self);

        let mapping: HashMap<usize, usize> = sorted_type_params
            .into_iter()
            .enumerate()
            .map(|(i, var_idx)| (var_idx, i))
            .collect();

        *self = self.substitute_type_vars(&mapping, SubstitutionContext::VarsToParams);
    }

    /// Checks if a type variable with the specified index is linear.
    pub(crate) fn is_linear(&self, var_idx: usize) -> bool {
        !self.type_params[&var_idx].maybe_non_linear
    }

    /// Recursively substitutes type vars in this type according to `type_var_mapping`.
    /// Returns the type with substituted vars.
    pub(crate) fn substitute_type_vars(
        &self,
        mapping: &HashMap<usize, usize>,
        context: SubstitutionContext,
    ) -> Self {
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

            type_params: self
                .type_params
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
                .collect(),
        }
    }

    pub(crate) fn arg_and_return_types(&self) -> impl Iterator<Item = &ValueType> + '_ {
        let args_slice = match &self.args {
            FnArgs::List(args) => args.as_slice(),
            FnArgs::Any => &[],
        };
        args_slice.iter().chain(Some(&self.return_type))
    }

    fn arg_and_return_types_mut(&mut self) -> impl Iterator<Item = &mut ValueType> + '_ {
        let args_slice = match &mut self.args {
            FnArgs::List(args) => args.as_mut_slice(),
            FnArgs::Any => &mut [],
        };
        args_slice.iter_mut().chain(Some(&mut self.return_type))
    }

    /// Maps argument and return types. The mapping function must not touch type params
    /// of the function.
    pub(crate) fn map_types<F>(&self, mut map_fn: F) -> Self
    where
        F: FnMut(&ValueType) -> ValueType,
    {
        Self {
            args: match &self.args {
                FnArgs::List(args) => FnArgs::List(args.iter().map(&mut map_fn).collect()),
                FnArgs::Any => FnArgs::Any,
            },
            return_type: map_fn(&self.return_type),
            type_params: self.type_params.clone(),
        }
    }
}

/// Helper type for inferring type parameters in functions.
///
/// The gist of the problem is that type params placement belongs which function in a tree
/// (which is ultimately rooted in a function not defined within another function) has exclusive
/// mention of a certain type var.
#[derive(Debug)]
struct FnTypeTree {
    children: Vec<FnTypeTree>,
    all_type_vars: HashMap<usize, TypeVarQuantity>,
    type_params: BTreeMap<usize, TypeParamDescription>,
}

impl FnTypeTree {
    fn new(base: &FnType) -> Self {
        fn recurse(
            children: &mut Vec<FnTypeTree>,
            type_vars: &mut HashMap<usize, TypeVarQuantity>,
            ty: &ValueType,
        ) {
            match ty {
                ValueType::TypeVar(idx) => {
                    type_vars
                        .entry(*idx)
                        .and_modify(|qty| *qty = TypeVarQuantity::Repeated)
                        .or_insert(TypeVarQuantity::UniqueVar);
                }

                ValueType::Tuple(fragments) => {
                    for fragment in fragments {
                        recurse(children, type_vars, fragment);
                    }
                }

                ValueType::Function(fn_type) => {
                    let child_tree = FnTypeTree::new(fn_type);

                    for &type_var in child_tree.all_type_vars.keys() {
                        type_vars
                            .entry(type_var)
                            .and_modify(|qty| *qty = TypeVarQuantity::Repeated)
                            .or_insert(TypeVarQuantity::UniqueFunction);
                    }

                    children.push(child_tree);
                }

                _ => { /* Do nothing. */ }
            }
        }

        let mut children = vec![];
        let mut all_type_vars = HashMap::new();
        for ty in base.arg_and_return_types() {
            recurse(&mut children, &mut all_type_vars, ty);
        }

        Self {
            children,
            all_type_vars,
            type_params: BTreeMap::new(),
        }
    }

    /// Recursively infers type params for the function and all child functions.
    fn infer_type_params(&mut self, parent_vars: &HashSet<usize>, linear_types: &HashSet<usize>) {
        self.type_params = self
            .all_type_vars
            .iter()
            .filter_map(|(idx, qty)| {
                if *qty != TypeVarQuantity::UniqueFunction {
                    let description = TypeParamDescription {
                        maybe_non_linear: !linear_types.contains(idx),
                        is_external: parent_vars.contains(idx),
                    };
                    Some((*idx, description))
                } else {
                    None
                }
            })
            .collect();

        let parent_and_self_vars: HashSet<_> = self.type_params.keys().copied().collect();
        for child_tree in &mut self.children {
            child_tree.infer_type_params(&parent_and_self_vars, linear_types);
        }
    }

    /// Recursively sets `type_params` for `base`.
    fn merge_into(self, base: &mut FnType) {
        fn recurse(reversed_children: &mut Vec<FnTypeTree>, ty: &mut ValueType) {
            match ty {
                ValueType::Tuple(fragments) => {
                    for fragment in fragments {
                        recurse(reversed_children, fragment);
                    }
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
        let mut reversed_children = self.children;
        reversed_children.reverse();
        for ty in base.arg_and_return_types_mut() {
            recurse(&mut reversed_children, ty);
        }
    }
}

/// Type of function arguments.
#[derive(Debug, Clone, PartialEq)]
pub enum FnArgs {
    /// Any arguments are accepted.
    Any,
    /// Lists accepted arguments.
    List(Vec<ValueType>),
}

impl fmt::Display for FnArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FnArgs::Any => formatter.write_str("..."),
            FnArgs::List(args) => {
                for (i, arg) in args.iter().enumerate() {
                    fmt::Display::fmt(arg, formatter)?;
                    if i + 1 < args.len() {
                        formatter.write_str(", ")?;
                    }
                }
                Ok(())
            }
        }
    }
}

// FIXME: `[T]` and `[T; N]` types.
/// Possible value type.
#[derive(Debug, Clone)]
pub enum ValueType {
    /// Any type.
    Any,
    /// Boolean.
    Bool,
    /// Number.
    Number,
    /// Function.
    Function(Box<FnType>),
    /// Tuple.
    Tuple(Vec<ValueType>),
    /// Type variable. In contrast to `TypeParam`s, `TypeVar`s are used exclusively during
    /// inference and cannot occur in standalone function signatures.
    TypeVar(usize),
    /// Type parameter in a function definition.
    TypeParam(usize),
}

impl PartialEq for ValueType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Any, _)
            | (_, Self::Any)
            | (Self::Bool, Self::Bool)
            | (Self::Number, Self::Number) => true,

            (Self::TypeVar(x), Self::TypeVar(y)) => x == y,
            (Self::TypeParam(x), Self::TypeParam(y)) => x == y,
            (Self::Tuple(xs), Self::Tuple(ys)) => xs == ys,

            // FIXME: function equality?
            _ => false,
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Any => formatter.write_str("_"),
            Self::TypeVar(idx) | Self::TypeParam(idx) => {
                formatter.write_str(Self::type_param(*idx).as_ref())
            }

            Self::Bool => formatter.write_str("Bool"),
            Self::Number => formatter.write_str("Num"),
            Self::Function(fn_type) => fmt::Display::fmt(fn_type, formatter),

            Self::Tuple(fragments) => {
                formatter.write_str("(")?;
                for (i, frag) in fragments.iter().enumerate() {
                    fmt::Display::fmt(frag, formatter)?;
                    if i + 1 < fragments.len() {
                        formatter.write_str(", ")?;
                    }
                }
                formatter.write_str(")")
            }
        }
    }
}

impl From<FnType> for ValueType {
    fn from(fn_type: FnType) -> Self {
        Self::Function(Box::new(fn_type))
    }
}

impl ValueType {
    pub(crate) fn type_param(index: usize) -> Cow<'static, str> {
        const PARAM_NAMES: &str = "TUVXYZ";
        PARAM_NAMES
            .get(index..=index)
            .map(Cow::from)
            .unwrap_or_else(|| Cow::from(format!("T{}", index - PARAM_NAMES.len())))
    }

    pub(crate) fn void() -> Self {
        Self::Tuple(Vec::new())
    }

    /// Checks if this type is void (i.e., an empty tuple).
    pub fn is_void(&self) -> bool {
        matches!(self, Self::Tuple(elements) if elements.is_empty())
    }

    /// Recursively substitutes type vars in this type according to `type_var_mapping`.
    /// Returns the type with substituted vars.
    pub(crate) fn substitute_type_vars(
        &self,
        mapping: &HashMap<usize, usize>,
        context: SubstitutionContext,
    ) -> Self {
        match self {
            Self::TypeVar(idx) if context == SubstitutionContext::VarsToParams => {
                ValueType::TypeParam(mapping[idx])
            }
            Self::TypeParam(idx) if context == SubstitutionContext::ParamsToVars => {
                if let Some(mapped_idx) = mapping.get(idx) {
                    ValueType::TypeVar(*mapped_idx)
                } else {
                    // This type param is not mapped; it is retained in the resulting function.
                    ValueType::TypeParam(*idx)
                }
            }

            Self::Tuple(fragments) => ValueType::Tuple(
                fragments
                    .iter()
                    .map(|element| element.substitute_type_vars(mapping, context))
                    .collect(),
            ),

            Self::Function(fn_type) => {
                ValueType::Function(Box::new(fn_type.substitute_type_vars(mapping, context)))
            }

            _ => self.clone(),
        }
    }

    /// Returns `Some(true)` if this type is known to be a number, `Some(false)` if it's known
    /// not to be a number, and `None` if either case is possible.
    pub(crate) fn is_number(&self) -> Option<bool> {
        match self {
            Self::Number => Some(true),
            Self::Tuple(_) | Self::Bool | Self::Function(_) => Some(false),
            _ => None,
        }
    }
}

/// Context for mapping `TypeVar`s into `TypeParam`s or back.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SubstitutionContext {
    /// Mapping `TypeVar`s to `TypeParam`s. This occurs for top-level functions once
    /// their signature is established.
    VarsToParams,
    /// Mapping `TypeParam`s to `TypeVar`s. This occurs when instantiating a function type
    /// with type params.
    ParamsToVars,
}

// FIXME: test Display
