//! Functional type (`FnType`) and closely related types.

use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use crate::{
    types::type_param, LengthKind, Num, PrimitiveType, Tuple, TupleLen, UnknownLen, ValueType,
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
                formatter.write_str(UnknownLen::const_param(*var_idx).as_ref())?;
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
            formatter.write_str(type_param(*var_idx).as_ref())?;
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
///     .with_len_params(&[0])
///     .with_arg(ValueType::NUM.repeat(UnknownLen::Param(0)))
///     .returning(ValueType::NUM);
/// assert_eq!(
///     sum_fn_type.to_string(),
///     "fn<len N>([Num; N]) -> Num"
/// );
/// ```
///
/// Signature for a slice mapping function:
///
/// ```
/// # use arithmetic_typing::{arith::LinConstraints, FnType, UnknownLen, ValueType};
/// # use std::iter;
/// // Definition of the mapping arg. Note that the definition uses type params,
/// // but does not declare them (they are bound to the parent function).
/// let map_fn_arg = <FnType>::builder()
///     .with_arg(ValueType::Param(0))
///     .returning(ValueType::Param(1));
///
/// let map_fn_type = <FnType>::builder()
///     .with_len_params(&[0])
///     .with_type_params(&[0])
///     .with_constrained_type_params(&[1], LinConstraints::LIN)
///     .with_arg(ValueType::Param(0).repeat(UnknownLen::Param(0)))
///     .with_arg(map_fn_arg)
///     .returning(ValueType::Param(1).repeat(UnknownLen::Param(0)));
/// assert_eq!(
///     map_fn_type.to_string(),
///     "fn<len N; T, U: Lin>([T; N], fn(T) -> U) -> [U; N]"
/// );
/// ```
///
/// Signature of a function with varargs:
///
/// ```
/// # use arithmetic_typing::{arith::LinConstraints, FnType, UnknownLen, ValueType};
/// # use std::iter;
/// let fn_type = <FnType>::builder()
///     .with_len_params(&[0])
///     .with_constrained_type_params(&[0], LinConstraints::LIN)
///     .with_varargs(ValueType::Param(0), UnknownLen::Param(0))
///     .with_arg(ValueType::BOOL)
///     .returning(ValueType::Param(0));
/// assert_eq!(
///     fn_type.to_string(),
///     "fn<len N; T: Lin>(...[T; N], Bool) -> T"
/// );
/// ```
#[derive(Debug, Clone)]
pub struct FnTypeBuilder<Prim: PrimitiveType = Num> {
    args: Tuple<Prim>,
    constraints: ParamConstraints<Prim>,
}

impl<Prim: PrimitiveType> Default for FnTypeBuilder<Prim> {
    fn default() -> Self {
        Self {
            args: Tuple::empty(),
            constraints: ParamConstraints::default(),
        }
    }
}

// FIXME: disallow quantified functions as args / return type?
impl<Prim: PrimitiveType> FnTypeBuilder<Prim> {
    /// Adds the type params with the specified `indexes` and `constraints`
    /// to the function definition.
    pub fn with_constrained_type_params(
        mut self,
        indexes: &[usize],
        constraints: &Prim::Constraints,
    ) -> Self {
        self.constraints
            .type_params
            .extend(indexes.iter().map(|&idx| (idx, constraints.clone())));
        self
    }

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
        // FIXME: return fn + constraints?
        FnType::new(self.args, return_type.into())
    }
}
