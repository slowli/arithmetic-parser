//! Functional type (`FnType`) and closely related types.

use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use super::type_param;
use crate::{LiteralType, Num, TupleLength, ValueType};

/// Description of a type parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TypeParamDescription {
    /// Can this type param be non-linear?
    // TODO: Change linearity to opt-in, rather than opt-out
    pub maybe_non_linear: bool,
}

/// Functional type.
#[derive(Debug, Clone, PartialEq)]
pub struct FnType<Lit = Num> {
    /// Type of function arguments.
    pub(crate) args: FnArgs<Lit>,
    /// Type of the value returned by the function.
    pub(crate) return_type: ValueType<Lit>,
    /// Type params associated with this function. The indexes of params should
    /// monotonically increase (necessary for correct display) and must be distinct.
    pub(crate) type_params: Vec<(usize, TypeParamDescription)>,
    /// Indexes of const params associated with this function. The indexes should
    /// monotonically increase (necessary for correct display) and must be distinct.
    pub(crate) const_params: Vec<usize>,
}

impl<Lit: fmt::Display> fmt::Display for FnType<Lit> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("fn")?;

        if self.const_params.len() + self.type_params.len() > 0 {
            formatter.write_str("<")?;

            if !self.const_params.is_empty() {
                formatter.write_str("const ")?;
                for (i, &var_idx) in self.const_params.iter().enumerate() {
                    formatter.write_str(TupleLength::const_param(var_idx).as_ref())?;
                    if i + 1 < self.const_params.len() {
                        formatter.write_str(", ")?;
                    }
                }

                if !self.type_params.is_empty() {
                    formatter.write_str("; ")?;
                }
            }

            for (i, (var_idx, description)) in self.type_params.iter().enumerate() {
                formatter.write_str(type_param(*var_idx).as_ref())?;
                if description.maybe_non_linear {
                    formatter.write_str(": ?Lin")?;
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

impl<Lit: LiteralType> FnType<Lit> {
    pub(crate) fn new(args: FnArgs<Lit>, return_type: ValueType<Lit>) -> Self {
        Self {
            args,
            return_type,
            type_params: Vec::new(),  // filled in later
            const_params: Vec::new(), // filled in later
        }
    }

    pub(crate) fn with_const_params(mut self, mut params: Vec<usize>) -> Self {
        params.sort_unstable();
        self.const_params = params;
        self
    }

    pub(crate) fn with_type_params(
        mut self,
        mut params: Vec<(usize, TypeParamDescription)>,
    ) -> Self {
        params.sort_unstable_by_key(|(idx, _)| *idx);
        self.type_params = params;
        self
    }

    /// Returns a builder for `FnType`s.
    pub fn builder() -> FnTypeBuilder<Lit> {
        FnTypeBuilder::default()
    }

    /// Returns `true` iff the function has at least one const or type param.
    pub fn is_parametric(&self) -> bool {
        !self.const_params.is_empty() || !self.type_params.is_empty()
    }

    /// Checks if a type variable with the specified index is linear.
    pub(crate) fn linear_type_params(&self) -> impl Iterator<Item = usize> + '_ {
        self.type_params.iter().filter_map(|(idx, description)| {
            if description.maybe_non_linear {
                None
            } else {
                Some(*idx)
            }
        })
    }

    pub(crate) fn arg_and_return_types(&self) -> impl Iterator<Item = &ValueType<Lit>> + '_ {
        let args_slice = match &self.args {
            FnArgs::List(args) => args.as_slice(),
            FnArgs::Any => &[],
        };
        args_slice.iter().chain(Some(&self.return_type))
    }

    pub(crate) fn arg_and_return_types_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut ValueType<Lit>> + '_ {
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
        F: FnMut(&ValueType<Lit>) -> ValueType<Lit>,
    {
        Self {
            args: match &self.args {
                FnArgs::List(args) => FnArgs::List(args.iter().map(&mut map_fn).collect()),
                FnArgs::Any => FnArgs::Any,
            },
            return_type: map_fn(&self.return_type),
            type_params: self.type_params.clone(),
            const_params: self.const_params.clone(),
        }
    }
}

/// Builder for functional types.
///
/// The builder does not check well-formedness of the function, such as that all referenced
/// const / type params are declared.
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
///     .with_const_params(iter::once(0))
///     .with_arg(ValueType::NUM.repeat(TupleLength::Param(0)))
///     .returning(ValueType::NUM);
/// assert_eq!(
///     sum_fn_type.to_string(),
///     "fn<const N>([Num; N]) -> Num"
/// );
/// ```
///
/// Signature for a slice mapping function:
///
/// ```
/// # use arithmetic_typing::{FnType, TupleLength, ValueType};
/// # use std::iter;
/// // Definition of the mapping arg. Note that the definition uses type params,
/// // but does not declare them (they are bound to the parent function).
/// let map_fn_arg = <FnType>::builder()
///     .with_arg(ValueType::Param(0))
///     .returning(ValueType::Param(1));
///
/// let map_fn_type = <FnType>::builder()
///     .with_const_params(iter::once(0))
///     .with_type_params(0..=1, false)
///     .with_arg(ValueType::Param(0).repeat(TupleLength::Param(0)))
///     .with_arg(map_fn_arg)
///     .returning(ValueType::Param(1).repeat(TupleLength::Param(0)));
/// assert_eq!(
///     map_fn_type.to_string(),
///     "fn<const N; T: ?Lin, U: ?Lin>([T; N], fn(T) -> U) -> [U; N]"
/// );
/// ```
#[derive(Debug)]
pub struct FnTypeBuilder<Lit = Num> {
    args: FnArgs<Lit>,
    type_params: HashMap<usize, TypeParamDescription>,
    const_params: HashSet<usize>,
}

impl<Lit> Default for FnTypeBuilder<Lit> {
    fn default() -> Self {
        Self {
            args: FnArgs::List(Vec::new()),
            type_params: HashMap::new(),
            const_params: HashSet::new(),
        }
    }
}

// TODO: support validation similarly to AST conversions.
impl<Lit: LiteralType> FnTypeBuilder<Lit> {
    /// Adds the const params with the specified `indexes` to the function definition.
    pub fn with_const_params(mut self, indexes: impl Iterator<Item = usize>) -> Self {
        self.const_params.extend(indexes);
        self
    }

    /// Adds the type params with the specified `indexes` to the function definition.
    /// `linear` determines if the type params are linear (i.e., can be used as arguments
    /// in binary ops).
    pub fn with_type_params(mut self, indexes: impl Iterator<Item = usize>, linear: bool) -> Self {
        let description = TypeParamDescription {
            maybe_non_linear: !linear,
        };
        self.type_params.extend(indexes.map(|i| (i, description)));
        self
    }

    /// Adds a new argument to the function definition.
    pub fn with_arg(mut self, arg: impl Into<ValueType<Lit>>) -> Self {
        match &mut self.args {
            FnArgs::List(args) => {
                args.push(arg.into());
            }
            FnArgs::Any => unreachable!(),
        }
        self
    }

    /// Declares the return type of the function and builds it.
    pub fn returning(self, return_type: ValueType<Lit>) -> FnType<Lit> {
        FnType::new(self.args, return_type)
            .with_const_params(self.const_params.into_iter().collect())
            .with_type_params(self.type_params.into_iter().collect())
    }
}

/// Type of function arguments.
#[derive(Debug, Clone, PartialEq)]
pub enum FnArgs<Lit> {
    /// Any arguments are accepted.
    // TODO: allow to parse any args
    Any,
    /// Lists accepted arguments.
    List(Vec<ValueType<Lit>>),
}

impl<Lit: fmt::Display> fmt::Display for FnArgs<Lit> {
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
