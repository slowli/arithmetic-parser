//! Functional type (`FnType`) and closely related types.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::Arc,
};

use crate::{
    types::ParamQuantifier, LengthKind, LengthVar, Num, PrimitiveType, Tuple, TupleLen, TypeVar,
    ValueType,
};

#[derive(Debug, Clone)]
pub(crate) struct ParamConstraints<Prim: PrimitiveType> {
    pub type_params: HashMap<usize, Prim::Constraints>,
    pub static_lengths: HashSet<usize>,
}

impl<Prim: PrimitiveType> Default for ParamConstraints<Prim> {
    fn default() -> Self {
        Self {
            type_params: HashMap::new(),
            static_lengths: HashSet::new(),
        }
    }
}

impl<Prim: PrimitiveType> fmt::Display for ParamConstraints<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.static_lengths.is_empty() {
            formatter.write_str("len! ")?;
            for (i, len) in self.static_lengths.iter().enumerate() {
                write!(formatter, "{}", LengthVar::param_str(*len))?;
                if i + 1 < self.static_lengths.len() {
                    formatter.write_str(", ")?;
                }
            }

            if !self.type_params.is_empty() {
                formatter.write_str("; ")?;
            }
        }

        let type_param_count = self.type_params.len();
        for (i, (idx, constraints)) in self.type_params().enumerate() {
            write!(formatter, "'{}: {}", TypeVar::param_str(idx), constraints)?;
            if i + 1 < type_param_count {
                formatter.write_str(", ")?;
            }
        }

        Ok(())
    }
}

impl<Prim: PrimitiveType> ParamConstraints<Prim> {
    fn is_empty(&self) -> bool {
        self.type_params.is_empty() && self.static_lengths.is_empty()
    }

    // Sort params by ascending index to have a consistent `Display` presentation.
    #[cfg(test)]
    fn type_params(&self) -> impl Iterator<Item = (usize, &Prim::Constraints)> + '_ {
        let mut type_params: Vec<_> = self.type_params.iter().map(|(&idx, c)| (idx, c)).collect();
        type_params.sort_unstable_by_key(|(idx, _)| *idx);
        type_params.into_iter()
    }

    #[cfg(not(test))]
    fn type_params(&self) -> impl Iterator<Item = (usize, &Prim::Constraints)> + '_ {
        self.type_params.iter().map(|(&idx, c)| (idx, c))
    }
}

#[derive(Debug)]
pub(crate) struct FnParams<Prim: PrimitiveType> {
    /// Type params associated with this function. Filled in by `FnQuantifier`.
    pub type_params: Vec<(usize, Prim::Constraints)>,
    /// Length params associated with this function. Filled in by `FnQuantifier`.
    pub len_params: Vec<(usize, LengthKind)>,
    /// Constraints for params of this function and child functions.
    pub constraints: Option<ParamConstraints<Prim>>,
}

impl<Prim: PrimitiveType> Default for FnParams<Prim> {
    fn default() -> Self {
        Self {
            type_params: vec![],
            len_params: vec![],
            constraints: None,
        }
    }
}

impl<Prim: PrimitiveType> PartialEq for FnParams<Prim> {
    fn eq(&self, other: &Self) -> bool {
        self.type_params == other.type_params && self.len_params == other.len_params
    }
}

impl<Prim: PrimitiveType> FnParams<Prim> {
    fn is_empty(&self) -> bool {
        self.len_params.is_empty() && self.type_params.is_empty()
    }
}

/// Functional type.
///
/// # Notation
///
/// Functional types are denoted as follows:
///
/// ```text
/// for<len M*; 'T: Lin> (['T; N], 'T) -> ['T; M]
/// ```
///
/// Here:
///
/// - `len M*` and `'T: Lin` are constraints on [length params] and [type params], respectively.
///   Length and/or type params constraints may be empty. Unconstrained type / length params
///   (such as length `N` in the example) do not need to be mentioned.
/// - `Lin` is a [constraint] on the type param.
/// - `*` after `M` denotes that `M` is a [dynamic length] (i.e., cannot be unified with
///   any other length during type inference).
/// - `N`, `M` and `'T` are parameter names. The args and the return type may reference these
///   parameters.
/// - `['T; N]` and `'T` are types of the function arguments.
/// - `['T; M]` is the return type.
///
/// The `for` constraints can only be present on top-level functions, but not in functions
/// mentioned in args / return types of other functions.
///
/// The `-> _` part is mandatory, even if the function returns [`ValueType::void()`].
///
/// A function may accept variable number of arguments of the same type along
/// with other args. (This construction is known as *varargs*.) This is denoted similarly
/// to middles in [`Tuple`]s. For example, `(...[Num; N]) -> Num` denotes a function
/// that accepts any number of `Num` args and returns a `Num` value.
///
/// [length params]: crate::LengthVar
/// [type params]: crate::TypeVar
/// [constraint]: crate::TypeConstraints
/// [dynamic length]: crate::LengthKind::Dynamic
///
/// # Construction
///
/// Functional types can be constructed via [`Self::builder()`] or parsed from a string.
///
/// With [`Self::builder()`], type / length params are *implicit*; they are computed automatically
/// when a function or [`FnWithConstraints`] is supplied to a [`TypeEnvironment`]. Computations
/// include both the function itself, and any child functions.
///
/// [`TypeEnvironment`]: crate::TypeEnvironment
///
/// # Examples
///
/// ```
/// # use arithmetic_typing::{LengthKind, FnType, Slice, ValueType};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let fn_type: FnType = "([Num; N]) -> Num".parse()?;
/// assert_eq!(*fn_type.return_type(), ValueType::NUM);
/// assert_matches!(
///     fn_type.args().parts(),
///     ([ValueType::Tuple(t)], None, [])
///         if t.as_slice().map(Slice::element) == Some(&ValueType::NUM)
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FnType<Prim: PrimitiveType = Num> {
    /// Type of function arguments.
    pub(crate) args: Tuple<Prim>,
    /// Type of the value returned by the function.
    pub(crate) return_type: ValueType<Prim>,
    /// Cache for function params.
    pub(crate) params: Option<Arc<FnParams<Prim>>>,
}

impl<Prim: PrimitiveType> fmt::Display for FnType<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let constraints = self
            .params
            .as_ref()
            .and_then(|params| params.constraints.as_ref());
        if let Some(constraints) = constraints {
            if !constraints.is_empty() {
                write!(formatter, "for<{}> ", constraints)?;
            }
        }

        self.args.format_as_tuple(formatter)?;
        write!(formatter, " -> {}", self.return_type)?;
        Ok(())
    }
}

impl<Prim: PrimitiveType> FnType<Prim> {
    pub(crate) fn new(args: Tuple<Prim>, return_type: ValueType<Prim>) -> Self {
        Self {
            args,
            return_type,
            params: None,
        }
    }

    /// Returns a builder for `FnType`s.
    pub fn builder() -> FnTypeBuilder<Prim> {
        FnTypeBuilder::default()
    }

    /// Gets the argument types of this function.
    pub fn args(&self) -> &Tuple<Prim> {
        &self.args
    }

    /// Gets the return type of this function.
    pub fn return_type(&self) -> &ValueType<Prim> {
        &self.return_type
    }

    pub(crate) fn set_params(&mut self, params: FnParams<Prim>) {
        self.params = Some(Arc::new(params));
    }

    pub(crate) fn is_parametric(&self) -> bool {
        self.params
            .as_ref()
            .map_or(false, |params| !params.is_empty())
    }

    /// Returns `true` iff this type does not contain type / length variables.
    ///
    /// See [`TypeEnvironment`](crate::TypeEnvironment) for caveats of dealing with
    /// non-concrete types.
    pub fn is_concrete(&self) -> bool {
        self.args.is_concrete() && self.return_type.is_concrete()
    }

    /// Marks type params with the specified `indexes` to have `constraints`.
    ///
    /// # Panics
    ///
    /// - Panics if parameters were already computed for the function.
    pub fn with_constraints(
        self,
        indexes: &[usize],
        constraints: &Prim::Constraints,
    ) -> FnWithConstraints<Prim> {
        assert!(
            self.params.is_none(),
            "Cannot attach constraints to a function with computed params: `{}`",
            self
        );

        let type_params = if *constraints == Prim::Constraints::default() {
            HashMap::new()
        } else {
            indexes
                .iter()
                .map(|&idx| (idx, constraints.clone()))
                .collect()
        };

        FnWithConstraints {
            function: self,
            constraints: ParamConstraints {
                type_params,
                static_lengths: HashSet::new(),
            },
        }
    }
}

/// Function together with constraints on type variables contained either in the function itself
/// or any of the child functions.
///
/// Constructed via [`FnType::with_constraints()`].
#[derive(Debug)]
pub struct FnWithConstraints<Prim: PrimitiveType> {
    function: FnType<Prim>,
    constraints: ParamConstraints<Prim>,
}

impl<Prim: PrimitiveType> fmt::Display for FnWithConstraints<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.constraints.is_empty() {
            fmt::Display::fmt(&self.function, formatter)
        } else {
            write!(formatter, "for<{}> {}", self.constraints, self.function)
        }
    }
}

impl<Prim: PrimitiveType> FnWithConstraints<Prim> {
    /// Marks type params with the specified `indexes` to have `constraints`. If some constraints
    /// are already present for some of the types, they are overwritten.
    pub fn with_constraints(mut self, indexes: &[usize], constraints: &Prim::Constraints) -> Self {
        if *constraints != Prim::Constraints::default() {
            let new_constraints = indexes.iter().map(|&idx| (idx, constraints.clone()));
            self.constraints.type_params.extend(new_constraints);
        }
        self
    }
}

impl<Prim: PrimitiveType> From<FnWithConstraints<Prim>> for FnType<Prim> {
    fn from(value: FnWithConstraints<Prim>) -> Self {
        let mut function = value.function;
        ParamQuantifier::set_params(&mut function, value.constraints);
        function
    }
}

impl<Prim: PrimitiveType> From<FnWithConstraints<Prim>> for ValueType<Prim> {
    fn from(value: FnWithConstraints<Prim>) -> Self {
        FnType::from(value).into()
    }
}

/// Builder for functional types.
///
/// **Tip.** You may also use [`FromStr`](core::str::FromStr) implementation to parse
/// functional types.
///
/// # Examples
///
/// Signature for a function summing a slice of numbers:
///
/// ```
/// # use arithmetic_typing::{FnType, UnknownLen, ValueType, TypeEnvironment};
/// # use std::iter;
/// let sum_fn_type = FnType::builder()
///     .with_arg(ValueType::NUM.repeat(UnknownLen::param(0)))
///     .returning(ValueType::NUM);
/// assert_eq!(sum_fn_type.to_string(), "([Num; N]) -> Num");
/// ```
///
/// Signature for a slice mapping function:
///
/// ```
/// # use arithmetic_typing::{arith::LinConstraints, FnType, UnknownLen, ValueType};
/// # use std::iter;
/// // Definition of the mapping arg.
/// let map_fn_arg = <FnType>::builder()
///     .with_arg(ValueType::param(0))
///     .returning(ValueType::param(1));
///
/// let map_fn_type = <FnType>::builder()
///     .with_arg(ValueType::param(0).repeat(UnknownLen::param(0)))
///     .with_arg(map_fn_arg)
///     .returning(ValueType::param(1).repeat(UnknownLen::Dynamic))
///     .with_constraints(&[1], &LinConstraints::LIN);
/// assert_eq!(
///     map_fn_type.to_string(),
///     "for<'U: Lin> (['T; N], ('T) -> 'U) -> ['U]"
/// );
/// ```
///
/// Signature of a function with varargs:
///
/// ```
/// # use arithmetic_typing::{arith::LinConstraints, FnType, UnknownLen, ValueType};
/// # use std::iter;
/// let fn_type = <FnType>::builder()
///     .with_varargs(ValueType::param(0), UnknownLen::param(0))
///     .with_arg(ValueType::BOOL)
///     .returning(ValueType::param(0));
/// assert_eq!(fn_type.to_string(), "(...['T; N], Bool) -> 'T");
/// ```
#[derive(Debug, Clone)]
pub struct FnTypeBuilder<Prim: PrimitiveType = Num> {
    args: Tuple<Prim>,
}

impl<Prim: PrimitiveType> Default for FnTypeBuilder<Prim> {
    fn default() -> Self {
        Self {
            args: Tuple::empty(),
        }
    }
}

impl<Prim: PrimitiveType> FnTypeBuilder<Prim> {
    /// Adds a new argument to the function definition.
    pub fn with_arg(mut self, arg: impl Into<ValueType<Prim>>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Adds or sets varargs in the function definition.
    pub fn with_varargs(
        mut self,
        element: impl Into<ValueType<Prim>>,
        len: impl Into<TupleLen>,
    ) -> Self {
        self.args.set_middle(element.into(), len.into());
        self
    }

    /// Declares the return type of the function and builds it.
    pub fn returning(self, return_type: impl Into<ValueType<Prim>>) -> FnType<Prim> {
        FnType::new(self.args, return_type.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{arith::LinConstraints, UnknownLen};

    #[test]
    fn constraints_display() {
        let constraints: ParamConstraints<Num> = ParamConstraints {
            type_params: vec![(0, LinConstraints::LIN)].into_iter().collect(),
            static_lengths: HashSet::new(),
        };
        assert_eq!(constraints.to_string(), "'T: Lin");

        let constraints: ParamConstraints<Num> = ParamConstraints {
            type_params: vec![(0, LinConstraints::LIN)].into_iter().collect(),
            static_lengths: vec![0].into_iter().collect(),
        };
        assert_eq!(constraints.to_string(), "len! N; 'T: Lin");
    }

    #[test]
    fn fn_with_constraints_display() {
        let sum_fn = <FnType>::builder()
            .with_arg(ValueType::param(0).repeat(UnknownLen::Some))
            .returning(ValueType::param(0))
            .with_constraints(&[0], &LinConstraints::LIN);
        assert_eq!(sum_fn.to_string(), "for<'T: Lin> (['T; _]) -> 'T");
    }

    #[test]
    fn fn_builder_with_quantified_arg() {
        let sum_fn: FnType = FnType::builder()
            .with_arg(ValueType::NUM.repeat(UnknownLen::param(0)))
            .returning(ValueType::NUM)
            .with_constraints(&[], &LinConstraints::LIN)
            .into();
        assert_eq!(sum_fn.to_string(), "([Num; N]) -> Num");

        let complex_fn: FnType = FnType::builder()
            .with_arg(ValueType::NUM)
            .with_arg(sum_fn.clone())
            .returning(ValueType::NUM)
            .with_constraints(&[], &LinConstraints::LIN)
            .into();
        assert_eq!(complex_fn.to_string(), "(Num, ([Num; N]) -> Num) -> Num");

        let other_complex_fn: FnType = FnType::builder()
            .with_varargs(ValueType::NUM, UnknownLen::param(0))
            .with_arg(sum_fn)
            .returning(ValueType::NUM)
            .with_constraints(&[], &LinConstraints::LIN)
            .into();
        assert_eq!(
            other_complex_fn.to_string(),
            "(...[Num; N], ([Num; N]) -> Num) -> Num"
        );
    }
}
