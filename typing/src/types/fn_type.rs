//! Functional type (`FnType`) and closely related types.

use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use crate::{
    types::ParamQuantifier, LengthKind, LengthVar, Num, PrimitiveType, Tuple, TupleLen, TypeVar,
    ValueType,
};

#[derive(Debug, Clone)]
pub(crate) struct ParamConstraints<Prim: PrimitiveType> {
    pub type_params: HashMap<usize, Prim::Constraints>,
    pub dyn_lengths: HashSet<usize>,
}

impl<Prim: PrimitiveType> Default for ParamConstraints<Prim> {
    fn default() -> Self {
        Self {
            type_params: HashMap::new(),
            dyn_lengths: HashSet::new(),
        }
    }
}

impl<Prim: PrimitiveType> fmt::Display for ParamConstraints<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.dyn_lengths.is_empty() {
            formatter.write_str("len ")?;
            for (i, len) in self.dyn_lengths.iter().enumerate() {
                formatter.write_str(LengthVar::param_str(*len).as_ref())?;
                if i + 1 < self.dyn_lengths.len() {
                    formatter.write_str(", ")?;
                }
            }
            formatter.write_str(" = *")?;

            if !self.type_params.is_empty() {
                formatter.write_str("; ")?;
            }
        }

        let type_param_count = self.type_params.len();
        for (&idx, constraints) in &self.type_params {
            write!(formatter, "{}: {}", TypeVar::param_str(idx), constraints)?;
            if idx + 1 < type_param_count {
                formatter.write_str(", ")?;
            }
        }

        Ok(())
    }
}

impl<Prim: PrimitiveType> ParamConstraints<Prim> {
    fn is_empty(&self) -> bool {
        self.type_params.is_empty() && self.dyn_lengths.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FnParams<Prim: PrimitiveType> {
    /// Type params associated with this function. Filled in by `FnQuantifier`.
    pub type_params: Vec<(usize, Prim::Constraints)>,
    /// Length params associated with this function. Filled in by `FnQuantifier`.
    pub len_params: Vec<(usize, LengthKind)>,
}

impl<Prim: PrimitiveType> Default for FnParams<Prim> {
    fn default() -> Self {
        Self {
            type_params: vec![],
            len_params: vec![],
        }
    }
}

impl<Prim: PrimitiveType> fmt::Display for FnParams<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.len_params.is_empty() {
            formatter.write_str("len ")?;
            for (i, (var_idx, kind)) in self.len_params.iter().enumerate() {
                formatter.write_str(LengthVar::param_str(*var_idx).as_ref())?;
                if *kind == LengthKind::Dynamic {
                    formatter.write_str("*")?;
                }
                if i + 1 < self.len_params.len() {
                    formatter.write_str(", ")?;
                }
            }

            if !self.type_params.is_empty() {
                formatter.write_str("; ")?;
            }
        }

        for (i, (var_idx, constraints)) in self.type_params.iter().enumerate() {
            formatter.write_str(TypeVar::param_str(*var_idx).as_ref())?;
            if *constraints != Prim::Constraints::default() {
                write!(formatter, ": {}", constraints)?;
            }
            if i + 1 < self.type_params.len() {
                formatter.write_str(", ")?;
            }
        }
        Ok(())
    }
}

impl<Prim: PrimitiveType> FnParams<Prim> {
    fn is_empty(&self) -> bool {
        self.len_params.is_empty() && self.type_params.is_empty()
    }
}

/// Functional type.
///
/// Functional types can be constructed via [`Self::builder()`] or parsed from a string.
///
/// # Notation
///
/// Functional types are denoted similarly to Rust:
///
/// ```text
/// fn<len N, M*; T: Lin>([T; N], T) -> [T; M]
/// ```
///
/// Here:
///
/// - `len N, M*` and `T: Lin` are length params and type params, respectively.
///   Length and/or type params may be empty.
/// - `N`, `M` and `T` are parameter names. The args and the return type may reference these
///   parameters and/or parameters of the outer function(s), if any.
/// - `Lin` is a [constraint] on the type param.
/// - `*` after `M` denotes that `M` is a [dynamic length] (i.e., cannot be unified with
///   any other length during type inference).
/// - `[T; N]` and `T` are types of the function arguments.
/// - `[T; M]` is the return type.
///
/// If a function returns [`ValueType::void()`], the `-> _` part may be omitted.
///
/// A function may accept variable number of arguments of the same type along
/// with other args. (This construction is known as *varargs*.) This is denoted similarly
/// to middles in [`Tuple`]s. For example, `fn<len N>(...[Num; N]) -> Num` denotes a function
/// that accepts any number of `Num` args and returns a `Num` value.
///
/// [constraint]: crate::TypeConstraints
/// [dynamic length]: crate::LengthKind::Dynamic
///
/// # Examples
///
/// ```
/// # use arithmetic_typing::{LengthKind, FnType, Slice, ValueType};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let fn_type: FnType = "fn<len N>([Num; N]) -> Num".parse()?;
/// assert_eq!(*fn_type.return_type(), ValueType::NUM);
/// assert_eq!(
///     fn_type.len_params().collect::<Vec<_>>(),
///     vec![(0, LengthKind::Static)]
/// );
///
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
    pub(crate) params: Option<FnParams<Prim>>,
}

impl<Prim: PrimitiveType> fmt::Display for FnType<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("fn")?;

        if let Some(params) = &self.params {
            if !params.is_empty() {
                formatter.write_str("<")?;
                fmt::Display::fmt(params, formatter)?;
                formatter.write_str(">")?;
            }
        }

        self.args.format_as_tuple(formatter)?;
        if !self.return_type.is_void() {
            write!(formatter, " -> {}", self.return_type)?;
        }
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

    /// Iterates over type params of this function together with their constraints.
    pub fn type_params(&self) -> impl Iterator<Item = (usize, &Prim::Constraints)> + '_ {
        let type_params = self
            .params
            .as_ref()
            .map_or(&[] as &[_], |params| &params.type_params);
        type_params
            .iter()
            .map(|(idx, constraints)| (*idx, constraints))
    }

    /// Iterates over length params of this function together with their type.
    pub fn len_params(&self) -> impl Iterator<Item = (usize, LengthKind)> + '_ {
        let len_params = self
            .params
            .as_ref()
            .map_or(&[] as &[_], |params| &params.len_params);
        len_params.iter().map(|(idx, kind)| (*idx, *kind))
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

    /// Adds the type params with the specified `indexes` and `constraints`
    /// to the function definition.
    ///
    /// # Panics
    ///
    /// Panics if this function has already undergone var quantification.
    pub fn with_constraints(
        self,
        indexes: &[usize],
        constraints: &Prim::Constraints,
    ) -> FnWithConstraints<Prim> {
        assert!(
            self.params.is_none(),
            "Cannot attach constraints to a quantified function `{}`",
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
                dyn_lengths: HashSet::new(),
            },
        }
    }
}

/// Function together with constraints on type variables contained either in the function itself
/// or any of the child functions.
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
            write!(formatter, "{} where {}", self.function, self.constraints)
        }
    }
}

impl<Prim: PrimitiveType> FnWithConstraints<Prim> {
    /// Adds the type variables with the specified `indexes` and `constraints`
    /// to the function definition.
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
/// The builder does not check well-formedness of the function, such as that all referenced
/// length / type params are declared.
///
/// **Tip.** You may also use [`FromStr`](core::str::FromStr) implementation to parse
/// functional types.
///
/// # Examples
///
/// Signature for a function summing a slice of numbers:
///
/// ```
/// # use arithmetic_typing::{FnType, UnknownLen, ValueType};
/// # use std::iter;
/// let sum_fn_type = FnType::builder()
///     .with_arg(ValueType::NUM.repeat(UnknownLen::Some))
///     .returning(ValueType::NUM);
/// assert_eq!(sum_fn_type.to_string(), "fn([Num; _]) -> Num");
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
///     .with_arg(ValueType::param(0).repeat(UnknownLen::Some))
///     .with_arg(map_fn_arg)
///     .returning(ValueType::param(1).repeat(UnknownLen::Dynamic))
///     .with_constraints(&[1], &LinConstraints::LIN);
/// assert_eq!(
///     map_fn_type.to_string(),
///     "fn([T; _], fn(T) -> U) -> [U] where U: Lin"
/// );
/// ```
///
/// Signature of a function with varargs:
///
/// ```
/// # use arithmetic_typing::{arith::LinConstraints, FnType, UnknownLen, ValueType};
/// # use std::iter;
/// let fn_type = <FnType>::builder()
///     .with_varargs(ValueType::param(0), UnknownLen::Some)
///     .with_arg(ValueType::BOOL)
///     .returning(ValueType::param(0));
/// assert_eq!(
///     fn_type.to_string(),
///     "fn(...[T; _], Bool) -> T"
/// );
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

// FIXME: disallow quantified functions as args / return type?
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
    use crate::arith::LinConstraints;

    #[test]
    fn constraints_display() {
        let constraints: ParamConstraints<Num> = ParamConstraints {
            type_params: vec![(0, LinConstraints::LIN)].into_iter().collect(),
            dyn_lengths: HashSet::new(),
        };
        assert_eq!(constraints.to_string(), "T: Lin");

        let constraints: ParamConstraints<Num> = ParamConstraints {
            type_params: vec![(0, LinConstraints::LIN)].into_iter().collect(),
            dyn_lengths: vec![0].into_iter().collect(),
        };
        assert_eq!(constraints.to_string(), "len N = *; T: Lin");
    }
}
