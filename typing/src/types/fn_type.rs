//! Functional type (`FnType`) and closely related types.

use std::{collections::HashMap, fmt};

use super::type_param;
use crate::{LengthKind, Num, PrimitiveType, TupleLength, ValueType};

/// Description of a constant parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LenParamDescription {
    pub is_dynamic: bool,
}

impl From<LengthKind> for LenParamDescription {
    fn from(value: LengthKind) -> Self {
        Self {
            is_dynamic: value == LengthKind::Dynamic,
        }
    }
}

/// Description of a type parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TypeParamDescription<C> {
    pub constraints: C,
}

impl<C> TypeParamDescription<C> {
    pub fn new(constraints: C) -> Self {
        Self { constraints }
    }
}

/// Functional type.
///
/// Functional types can be constructed via [`Self::builder()`] or parsed from a string.
///
/// # Examples
///
/// ```
/// # use arithmetic_typing::{LengthKind, FnArgs, FnType, ValueType};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let fn_type: FnType = "fn<len N>([Num; N]) -> Num".parse()?;
/// assert_eq!(*fn_type.return_type(), ValueType::NUM);
/// assert_eq!(
///     fn_type.len_params().collect::<Vec<_>>(),
///     vec![(0, LengthKind::Static)]
/// );
///
/// let args = match fn_type.args() {
///     FnArgs::List(args) => args,
///     _ => unreachable!(),
/// };
/// assert_matches!(
///     args.as_slice(),
///     [ValueType::Slice { element, .. }] if **element == ValueType::NUM
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FnType<Prim: PrimitiveType = Num> {
    /// Type of function arguments.
    pub(crate) args: FnArgs<Prim>,
    /// Type of the value returned by the function.
    pub(crate) return_type: ValueType<Prim>,
    /// Type params associated with this function. The indexes of params should
    /// monotonically increase (necessary for correct display) and must be distinct.
    pub(crate) type_params: Vec<(usize, TypeParamDescription<Prim::Constraints>)>,
    /// Indexes of length params associated with this function. The indexes should
    /// monotonically increase (necessary for correct display) and must be distinct.
    pub(crate) len_params: Vec<(usize, LenParamDescription)>,
}

impl<Prim: PrimitiveType> fmt::Display for FnType<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("fn")?;

        if self.len_params.len() + self.type_params.len() > 0 {
            formatter.write_str("<")?;

            if !self.len_params.is_empty() {
                formatter.write_str("len ")?;
                for (i, (var_idx, description)) in self.len_params.iter().enumerate() {
                    formatter.write_str(TupleLength::const_param(*var_idx).as_ref())?;
                    if description.is_dynamic {
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

            for (i, (var_idx, description)) in self.type_params.iter().enumerate() {
                formatter.write_str(type_param(*var_idx).as_ref())?;
                if description.constraints != Prim::Constraints::default() {
                    write!(formatter, ": {}", description.constraints)?;
                }
                if i + 1 < self.type_params.len() {
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

impl<Prim: PrimitiveType> FnType<Prim> {
    pub(crate) fn new(args: FnArgs<Prim>, return_type: ValueType<Prim>) -> Self {
        Self {
            args,
            return_type,
            type_params: Vec::new(), // filled in by `Self::with_type_params()`
            len_params: Vec::new(),  // filled in by `Self::with_len_params()`
        }
    }

    pub(crate) fn with_len_params(mut self, mut params: Vec<(usize, LenParamDescription)>) -> Self {
        params.sort_unstable_by_key(|(idx, _)| *idx);
        self.len_params = params;
        self
    }

    pub(crate) fn with_type_params(
        mut self,
        mut params: Vec<(usize, TypeParamDescription<Prim::Constraints>)>,
    ) -> Self {
        params.sort_unstable_by_key(|(idx, _)| *idx);
        self.type_params = params;
        self
    }

    /// Returns a builder for `FnType`s.
    pub fn builder() -> FnTypeBuilder<Prim> {
        FnTypeBuilder::default()
    }

    /// Gets the argument types of this function.
    pub fn args(&self) -> &FnArgs<Prim> {
        &self.args
    }

    /// Gets the return type of this function.
    pub fn return_type(&self) -> &ValueType<Prim> {
        &self.return_type
    }

    /// Iterates over type params of this function together with their constraints.
    pub fn type_params(&self) -> impl Iterator<Item = (usize, &Prim::Constraints)> + '_ {
        self.type_params
            .iter()
            .map(|(idx, description)| (*idx, &description.constraints))
    }

    /// Iterates over length params of this function together with their type.
    pub fn len_params(&self) -> impl Iterator<Item = (usize, LengthKind)> + '_ {
        self.len_params.iter().map(|(idx, description)| {
            let length_type = if description.is_dynamic {
                LengthKind::Dynamic
            } else {
                LengthKind::Static
            };
            (*idx, length_type)
        })
    }

    /// Returns `true` iff the function has at least one length or type param.
    pub fn is_parametric(&self) -> bool {
        !self.len_params.is_empty() || !self.type_params.is_empty()
    }

    /// Returns `true` iff this type does not contain type / length variables.
    ///
    /// See [`TypeEnvironment`](crate::TypeEnvironment) for caveats of dealing with
    /// non-concrete types.
    pub fn is_concrete(&self) -> bool {
        self.arg_and_return_types().all(ValueType::is_concrete)
    }

    pub(crate) fn arg_and_return_types(&self) -> impl Iterator<Item = &ValueType<Prim>> + '_ {
        let args_slice = match &self.args {
            FnArgs::List(args) => args.as_slice(),
            FnArgs::Any => &[],
        };
        args_slice.iter().chain(Some(&self.return_type))
    }

    pub(crate) fn arg_and_return_types_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut ValueType<Prim>> + '_ {
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
        F: FnMut(&ValueType<Prim>) -> ValueType<Prim>,
    {
        Self {
            args: match &self.args {
                FnArgs::List(args) => FnArgs::List(args.iter().map(&mut map_fn).collect()),
                FnArgs::Any => FnArgs::Any,
            },
            return_type: map_fn(&self.return_type),
            type_params: self.type_params.clone(),
            len_params: self.len_params.clone(),
        }
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
/// # use arithmetic_typing::{FnType, TupleLength, ValueType};
/// # use std::iter;
/// let sum_fn_type = FnType::builder()
///     .with_len_params(iter::once(0))
///     .with_arg(ValueType::NUM.repeat(TupleLength::Param(0)))
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
/// # use arithmetic_typing::{arith::LinConstraints, FnType, TupleLength, ValueType};
/// # use std::iter;
/// // Definition of the mapping arg. Note that the definition uses type params,
/// // but does not declare them (they are bound to the parent function).
/// let map_fn_arg = <FnType>::builder()
///     .with_arg(ValueType::Param(0))
///     .returning(ValueType::Param(1));
///
/// let map_fn_type = <FnType>::builder()
///     .with_len_params(iter::once(0))
///     .with_type_params(iter::once(0))
///     .with_constrained_type_params(iter::once(1), LinConstraints::LIN)
///     .with_arg(ValueType::Param(0).repeat(TupleLength::Param(0)))
///     .with_arg(map_fn_arg)
///     .returning(ValueType::Param(1).repeat(TupleLength::Param(0)));
/// assert_eq!(
///     map_fn_type.to_string(),
///     "fn<len N; T, U: Lin>([T; N], fn(T) -> U) -> [U; N]"
/// );
/// ```
#[derive(Debug)]
pub struct FnTypeBuilder<Prim: PrimitiveType = Num> {
    args: FnArgs<Prim>,
    type_params: HashMap<usize, TypeParamDescription<Prim::Constraints>>,
    const_params: HashMap<usize, LenParamDescription>,
}

impl<Prim: PrimitiveType> Default for FnTypeBuilder<Prim> {
    fn default() -> Self {
        Self {
            args: FnArgs::List(Vec::new()),
            type_params: HashMap::new(),
            const_params: HashMap::new(),
        }
    }
}

// TODO: support validation similarly to AST conversions.
impl<Prim: PrimitiveType> FnTypeBuilder<Prim> {
    /// Adds the length params with the specified `indexes` to the function definition.
    pub fn with_len_params(mut self, indexes: impl Iterator<Item = usize>) -> Self {
        let static_description = LenParamDescription { is_dynamic: false };
        self.const_params
            .extend(indexes.map(|idx| (idx, static_description)));
        self
    }

    /// Adds the dynamic length params with the specified `indexes` to the function definition.
    pub fn with_dyn_len_params(mut self, indexes: impl Iterator<Item = usize>) -> Self {
        let dyn_description = LenParamDescription { is_dynamic: true };
        self.const_params
            .extend(indexes.map(|idx| (idx, dyn_description)));
        self
    }

    /// Adds the type params with the specified `indexes` to the function definition.
    /// The params are unconstrained.
    pub fn with_type_params(self, indexes: impl Iterator<Item = usize>) -> Self {
        self.with_constrained_type_params(indexes, Prim::Constraints::default())
    }

    /// Adds the type params with the specified `indexes` and `constraints`
    /// to the function definition.
    pub fn with_constrained_type_params(
        mut self,
        indexes: impl Iterator<Item = usize>,
        constraints: Prim::Constraints,
    ) -> Self {
        let description = TypeParamDescription { constraints };
        self.type_params
            .extend(indexes.map(|i| (i, description.clone())));
        self
    }

    /// Adds a new argument to the function definition.
    pub fn with_arg(mut self, arg: impl Into<ValueType<Prim>>) -> Self {
        match &mut self.args {
            FnArgs::List(args) => {
                args.push(arg.into());
            }
            FnArgs::Any => unreachable!(),
        }
        self
    }

    /// Declares the return type of the function and builds it.
    pub fn returning(self, return_type: ValueType<Prim>) -> FnType<Prim> {
        FnType::new(self.args, return_type)
            .with_len_params(self.const_params.into_iter().collect())
            .with_type_params(self.type_params.into_iter().collect())
    }
}

/// Type of function arguments.
#[derive(Debug, Clone, PartialEq)]
pub enum FnArgs<Prim: PrimitiveType> {
    /// Any arguments are accepted.
    // TODO: allow to parse any args
    Any,
    /// Lists accepted arguments.
    List(Vec<ValueType<Prim>>),
}

impl<Prim: PrimitiveType> fmt::Display for FnArgs<Prim> {
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
