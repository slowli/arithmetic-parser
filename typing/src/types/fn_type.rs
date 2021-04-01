//! Functional type (`FnType`) and closely related types.

#![allow(renamed_and_removed_lints, clippy::unknown_clippy_lints)]
// ^ `missing_panics_doc` is newer than MSRV, and `clippy::unknown_clippy_lints` is removed
// since Rust 1.51.

use std::{collections::HashMap, fmt};

use super::type_param;
use crate::{LengthKind, Num, PrimitiveType, SimpleTupleLen, Tuple, TupleLen, ValueType};

/// Description of a constant parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LenParamDescription {
    pub kind: LengthKind,
}

impl From<LengthKind> for LenParamDescription {
    fn from(kind: LengthKind) -> Self {
        Self { kind }
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
                    formatter.write_str(SimpleTupleLen::const_param(*var_idx).as_ref())?;
                    if description.kind == LengthKind::Dynamic {
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
    pub fn args(&self) -> &Tuple<Prim> {
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
        self.len_params
            .iter()
            .map(|(idx, description)| (*idx, description.kind))
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
        self.args.is_concrete() && self.return_type.is_concrete()
    }

    /// Adds or replaces a type parameter with the specified `index`.
    pub fn insert_type_param(&mut self, index: usize, constraints: Prim::Constraints) {
        let value_to_insert = (index, TypeParamDescription::new(constraints));
        let insert_pos = self
            .type_params
            .binary_search_by_key(&index, |(idx, _)| *idx);
        match insert_pos {
            Ok(pos) => self.type_params[pos] = value_to_insert,
            Err(pos) => self.type_params.insert(pos, value_to_insert),
        }
    }

    /// Removes a type parameter with the specified index. Returns the constraints on the param
    /// if it was present.
    pub fn remove_type_param(&mut self, index: usize) -> Option<Prim::Constraints> {
        let remove_pos = self
            .type_params
            .binary_search_by_key(&index, |(idx, _)| *idx);
        if let Ok(pos) = remove_pos {
            Some(self.type_params.remove(pos).1.constraints)
        } else {
            None
        }
    }

    /// Adds or replaces a length parameter with the specified `index`.
    pub fn insert_len_param(&mut self, index: usize, kind: LengthKind) {
        let value_to_insert = (index, LenParamDescription::from(kind));
        let insert_pos = self
            .len_params
            .binary_search_by_key(&index, |(idx, _)| *idx);
        match insert_pos {
            Ok(pos) => self.len_params[pos] = value_to_insert,
            Err(pos) => self.len_params.insert(pos, value_to_insert),
        }
    }

    /// Removes a length parameter with the specified index. Returns the length type if it was
    /// present.
    pub fn remove_len_param(&mut self, index: usize) -> Option<LengthKind> {
        let remove_pos = self
            .len_params
            .binary_search_by_key(&index, |(idx, _)| *idx);
        if let Ok(pos) = remove_pos {
            Some(self.len_params.remove(pos).1.kind)
        } else {
            None
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
/// # use arithmetic_typing::{FnType, SimpleTupleLen, ValueType};
/// # use std::iter;
/// let sum_fn_type = FnType::builder()
///     .with_len_params(&[0])
///     .with_arg(ValueType::NUM.repeat(SimpleTupleLen::Param(0)))
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
/// # use arithmetic_typing::{arith::LinConstraints, FnType, SimpleTupleLen, ValueType};
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
///     .with_arg(ValueType::Param(0).repeat(SimpleTupleLen::Param(0)))
///     .with_arg(map_fn_arg)
///     .returning(ValueType::Param(1).repeat(SimpleTupleLen::Param(0)));
/// assert_eq!(
///     map_fn_type.to_string(),
///     "fn<len N; T, U: Lin>([T; N], fn(T) -> U) -> [U; N]"
/// );
/// ```
///
/// Signature of a function with varargs:
///
/// ```
/// # use arithmetic_typing::{arith::LinConstraints, FnType, SimpleTupleLen, ValueType};
/// # use std::iter;
/// let fn_type = <FnType>::builder()
///     .with_len_params(&[0])
///     .with_constrained_type_params(&[0], LinConstraints::LIN)
///     .with_varargs(ValueType::Param(0), SimpleTupleLen::Param(0))
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
    type_params: HashMap<usize, TypeParamDescription<Prim::Constraints>>,
    const_params: HashMap<usize, LenParamDescription>,
}

impl<Prim: PrimitiveType> Default for FnTypeBuilder<Prim> {
    fn default() -> Self {
        Self {
            args: Tuple::empty(),
            type_params: HashMap::new(),
            const_params: HashMap::new(),
        }
    }
}

// TODO: support validation similarly to AST conversions.
impl<Prim: PrimitiveType> FnTypeBuilder<Prim> {
    /// Adds the length params with the specified `indexes` to the function definition.
    pub fn with_len_params(mut self, indexes: &[usize]) -> Self {
        let static_description = LengthKind::Static.into();
        self.const_params
            .extend(indexes.iter().map(|&idx| (idx, static_description)));
        self
    }

    /// Adds the dynamic length params with the specified `indexes` to the function definition.
    pub fn with_dyn_len_params(mut self, indexes: &[usize]) -> Self {
        let dyn_description = LengthKind::Dynamic.into();
        self.const_params
            .extend(indexes.iter().map(|&idx| (idx, dyn_description)));
        self
    }

    /// Adds the type params with the specified `indexes` to the function definition.
    /// The params are unconstrained.
    pub fn with_type_params(self, indexes: &[usize]) -> Self {
        self.with_constrained_type_params(indexes, Prim::Constraints::default())
    }

    /// Adds the type params with the specified `indexes` and `constraints`
    /// to the function definition.
    pub fn with_constrained_type_params(
        mut self,
        indexes: &[usize],
        constraints: Prim::Constraints,
    ) -> Self {
        let description = TypeParamDescription { constraints };
        self.type_params
            .extend(indexes.iter().map(|&idx| (idx, description.clone())));
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
        FnType::new(self.args, return_type.into())
            .with_len_params(self.const_params.into_iter().collect())
            .with_type_params(self.type_params.into_iter().collect())
    }
}
