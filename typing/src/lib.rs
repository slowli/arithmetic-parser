use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
};

mod context;
mod error;
mod substitutions;

pub use self::error::TypeError;

#[derive(Debug, Clone, Copy, PartialEq)]
struct TypeParamDescription {
    maybe_non_linear: bool,
    is_external: bool,
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
    pub(crate) fn new(
        args: Vec<ValueType>,
        return_type: ValueType,
        var_count_in_outer_scope: usize,
        linear_types: &HashSet<usize>,
    ) -> Self {
        fn extract_type_params(
            type_params: &mut HashSet<usize>,
            ty: &ValueType,
            include_inner_fns: bool,
        ) {
            match ty {
                ValueType::TypeVar(idx) => {
                    type_params.insert(*idx);
                }

                ValueType::Tuple(fragments) => {
                    for fragment in fragments {
                        extract_type_params(type_params, fragment, include_inner_fns);
                    }
                }

                ValueType::Function(fn_type) if include_inner_fns => {
                    if let FnArgs::List(args) = &fn_type.args {
                        for arg in args {
                            extract_type_params(type_params, arg, include_inner_fns);
                        }
                    }
                    extract_type_params(type_params, &fn_type.return_type, include_inner_fns);
                }

                _ => { /* Do nothing. */ }
            }
        }

        let mut type_params = HashSet::new();
        for ty in args.iter().chain(Some(&return_type)) {
            // We intentionally do not recurse into functions; type vars only present in functions
            // are type params for these functions, not for this function.
            extract_type_params(&mut type_params, ty, false);
        }

        let reduction_mapping: Option<HashMap<usize, usize>> = if type_params
            .iter()
            .all(|&idx| idx >= var_count_in_outer_scope)
        {
            // Function signature has no type var dependencies. Correspondingly, we remap
            // type params for this fn and all inner fns, so that they will be properly printed
            // (param indexes will start from 0 and will not contain gaps).
            let mut all_type_params = HashSet::new();
            for ty in args.iter().chain(Some(&return_type)) {
                extract_type_params(&mut all_type_params, ty, true);
            }

            let mut sorted_type_params: Vec<_> = all_type_params.into_iter().collect();
            sorted_type_params.sort_unstable();
            Some(
                sorted_type_params
                    .into_iter()
                    .enumerate()
                    .map(|(i, var_idx)| (var_idx, i))
                    .collect(),
            )
        } else {
            None
        };

        let type_params = type_params
            .into_iter()
            .map(|var_idx| {
                let description = TypeParamDescription {
                    maybe_non_linear: !linear_types.contains(&var_idx),
                    is_external: var_idx < var_count_in_outer_scope,
                };
                (var_idx, description)
            })
            .collect();

        let this = Self {
            args: FnArgs::List(args),
            return_type,
            type_params,
        };

        // FIXME: Can bugs occur if we remap type params for an inner fn?
        if let Some(mapping) = reduction_mapping {
            this.substitute_type_vars(&mapping, SubstitutionContext::VarsToParams)
        } else {
            this
        }
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
