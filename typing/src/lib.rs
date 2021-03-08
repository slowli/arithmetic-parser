//! Hindley-Milner type inference for arithmetic expressions parsed
//! by the [`arithmetic-parser`] crate.
//!
//! This crate allows to parse type annotations as a part of a [`Grammar`], and to infer
//! and check types for expressions / statements produced by `arithmetic-parser`.
//! Type inference is *partially* compatible with the interpreter from [`arithmetic-eval`];
//! if the inference algorithm succeeds on a certain expression / statement / block,
//! it will execute successfully, but not all successfully executing items pass type inference.
//!
//! # Type system
//!
//! The type system corresponds to types of `Value`s in `arithmetic-eval`:
//!
//! - There are 2 primitive types: Boolean (`Bool`) and number (`Num`).
//! - There is only one container type - a tuple. It can be represented either
//!   in the tuple form, such as `(Num, Bool)`, or as a slice, such as `[Num]` or `[Num; 3]`.
//!   As in Rust, all slice elements must have the same type. Unlike Rust, tuple and slice
//!   forms are equivalent; e.g., `[Num; 3]` and `(Num, Num, Num)` are the same type.
//! - Functions are first-class types. Functions can have type and/or const params.
//! - Type params can be constrained. The only currently supported constraint is type *linearity*.
//!   Linear types are types that support binary arithmetic ops (e.g., `+`): numbers and
//!   tuples consisting of linear elements.
//! - Const params always specify tuple length.
//!
//! # Inference rules
//!
//! FIXME
//!
//! # Examples
//!
//! FIXME
//!
//! [`arithmetic-parser`]: https://crates.io/crates/arithmetic-parser
//! [`Grammar`]: arithmetic_parser::grammars::Grammar
//! [`arithmetic-eval`]: https://crates.io/crates/arithmetic-eval

#![warn(missing_docs, missing_debug_implementations)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::module_name_repetitions,
    clippy::similar_names // too many false positives because of lhs / rhs
)]

use std::{borrow::Cow, collections::BTreeMap, fmt};

mod env;
mod error;
pub mod parser;
mod substitutions;

pub use self::{env::TypeEnvironment, error::TypeError};

/// Description of a type parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TypeParamDescription {
    /// Can this type param be non-linear?
    maybe_non_linear: bool,
}

/// Description of a constant parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ConstParamDescription;

/// Functional type.
#[derive(Debug, Clone, PartialEq)]
pub struct FnType {
    /// Type of function arguments.
    args: FnArgs,
    /// Type of the value returned by the function.
    return_type: ValueType,
    /// Indexes of type params associated with this function.
    type_params: BTreeMap<usize, TypeParamDescription>,
    /// Indexes of const params associated with this function.
    const_params: BTreeMap<usize, ConstParamDescription>,
}

impl fmt::Display for FnType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("fn")?;

        if self.const_params.len() + self.type_params.len() > 0 {
            formatter.write_str("<")?;

            if !self.const_params.is_empty() {
                formatter.write_str("const ")?;
                for (i, (&var_idx, _)) in self.const_params.iter().enumerate() {
                    formatter.write_str(TupleLength::const_param(var_idx).as_ref())?;
                    if i + 1 < self.const_params.len() {
                        formatter.write_str(", ")?;
                    }
                }

                if !self.type_params.is_empty() {
                    formatter.write_str("; ")?;
                }
            }

            for (i, (&var_idx, description)) in self.type_params.iter().enumerate() {
                formatter.write_str(ValueType::type_param(var_idx).as_ref())?;
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

impl FnType {
    pub(crate) fn new(args: Vec<ValueType>, return_type: ValueType) -> Self {
        Self {
            args: FnArgs::List(args),
            return_type,
            type_params: BTreeMap::new(),  // filled in later
            const_params: BTreeMap::new(), // filled in later
        }
    }

    /// Returns `true` iff the function has at least one const or type param.
    pub fn is_parametric(&self) -> bool {
        !self.const_params.is_empty() || !self.type_params.is_empty()
    }

    /// Checks if a type variable with the specified index is linear.
    pub(crate) fn is_linear(&self, var_idx: usize) -> bool {
        !self.type_params[&var_idx].maybe_non_linear
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
            const_params: self.const_params.clone(),
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

/// Tuple length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TupleLength {
    /// Dynamic length that can vary at runtime.
    Dynamic,
    /// Wildcard length.
    Any,
    /// Exact known length.
    Exact(usize),
    /// Length variable. In contrast to `Param`s, `Var`s are used exclusively during
    /// inference and cannot occur in standalone function signatures.
    Var(usize),
    /// Length parameter in a function definition.
    Param(usize),
}

impl fmt::Display for TupleLength {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dynamic => formatter.write_str("*"),
            Self::Any => formatter.write_str("_"),
            Self::Exact(len) => fmt::Display::fmt(len, formatter),
            Self::Var(idx) | Self::Param(idx) => {
                formatter.write_str(Self::const_param(*idx).as_ref())
            }
        }
    }
}

impl TupleLength {
    fn const_param(index: usize) -> Cow<'static, str> {
        const PARAM_NAMES: &str = "NMLKJI";
        PARAM_NAMES.get(index..=index).map_or_else(
            || Cow::from(format!("N{}", index - PARAM_NAMES.len())),
            Cow::from,
        )
    }
}

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
    /// Slice.
    Slice {
        /// Type of slice elements.
        element: Box<ValueType>,
        /// Slice length.
        length: TupleLength,
    },
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

            (Self::TypeVar(x), Self::TypeVar(y)) | (Self::TypeParam(x), Self::TypeParam(y)) => {
                x == y
            }

            (Self::Tuple(xs), Self::Tuple(ys)) => xs == ys,

            (
                Self::Slice { element, length },
                Self::Slice {
                    element: other_element,
                    length: other_length,
                },
            ) => length == other_length && element == other_element,

            (Self::Tuple(xs), Self::Slice { element, length })
            | (Self::Slice { element, length }, Self::Tuple(xs)) => {
                *length == TupleLength::Exact(xs.len()) && xs.iter().all(|x| x == element.as_ref())
            }

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

            Self::Slice {
                element,
                length: TupleLength::Dynamic,
            } => {
                write!(formatter, "[{}]", element)
            }
            Self::Slice {
                element,
                length: TupleLength::Exact(len),
            } => {
                // Format slice as a tuple since its size is statically known.
                formatter.write_str("(")?;
                for i in 0..*len {
                    fmt::Display::fmt(element, formatter)?;
                    if i + 1 < *len {
                        formatter.write_str(", ")?;
                    }
                }
                formatter.write_str(")")
            }
            Self::Slice { element, length } => {
                write!(formatter, "[{}; {}]", element, length)
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
    fn type_param(index: usize) -> Cow<'static, str> {
        const PARAM_NAMES: &str = "TUVXYZ";
        PARAM_NAMES.get(index..=index).map_or_else(
            || Cow::from(format!("T{}", index - PARAM_NAMES.len())),
            Cow::from,
        )
    }

    pub(crate) fn void() -> Self {
        Self::Tuple(Vec::new())
    }

    /// Checks if this type is void (i.e., an empty tuple).
    pub fn is_void(&self) -> bool {
        matches!(self, Self::Tuple(elements) if elements.is_empty())
    }

    /// Returns `Some(true)` if this type is known to be a number, `Some(false)` if it's known
    /// not to be a number, and `None` if either case is possible.
    pub(crate) fn is_number(&self) -> Option<bool> {
        match self {
            Self::Number => Some(true),
            Self::Tuple(_) | Self::Slice { .. } | Self::Bool | Self::Function(_) => Some(false),
            _ => None,
        }
    }
}
