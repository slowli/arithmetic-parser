//! `TypeConstraints` and implementations.

use std::{fmt, ops, str::FromStr};

use crate::{
    arith::OpConstraintSettings,
    error::{SpannedTypeErrors, TypeErrorKind},
    PrimitiveType, Slice, Substitutions, ValueType,
};

/// Container for constraints that can be placed on type variables.
///
/// Constraints can be placed on [function](crate::FnType) type variables, and can be applied
/// to types in [`TypeArithmetic`] impls. For example, [`NumArithmetic`] places
/// a [linearity constraint](NumConstraints::Lin) on types involved in arithmetic ops.
///
/// The constraint mechanism is similar to trait constraints in Rust, but is much more limited:
///
/// - Constraints cannot be parametric (cf. parameters in traits, such `AsRef<_>`
///   or `Iterator<Item = _>`).
/// - Constraints are applied to types in separation; it is impossible to create a constraint
///   involving several type variables.
/// - Constraints cannot contradict each other.
///
/// # Implementation rules
///
/// Usually, this trait should be implemented with something akin to [`bitflags`].
///
/// [`bitflags`]: https://docs.rs/bitflags/
///
/// - [`Default`] must return a container with no restrictions.
/// - [`BitOrAssign`](ops::BitOrAssign) must perform the union of the provided constraints.
/// - [`Display`](fmt::Display) must display constraints in the form `Foo + Bar + Quux`,
///   where `Foo`, `Bar` and `Quux` are *primitive* constraints (i.e., ones not reduced
///   to a combination of other constraints). The primitive constraints must be represented
///   as identifiers (i.e., consist of alphanumeric chars and start with an alphabetic char
///   or `_`).
/// - [`FromStr`] must parse primitive constraints.
///
/// [`TypeArithmetic`]: crate::arith::TypeArithmetic
/// [`NumArithmetic`]: crate::arith::NumArithmetic
pub trait TypeConstraints<Prim>:
    Clone
    + Default
    + PartialEq
    + fmt::Debug
    + fmt::Display
    + FromStr
    + for<'op> ops::BitOrAssign<&'op Self>
    + Send
    + Sync
    + 'static
where
    Prim: PrimitiveType<Constraints = Self>,
{
    /// Applies these constraints to the provided `ty`pe. Returns an error if the type
    /// contradicts the constraints.
    ///
    /// A typical implementation will use `substitutions` to
    /// [place constraints on type vars](Substitutions::insert_constraints()), e.g.,
    /// by recursively traversing and resolving the provided type.
    fn apply(
        &self,
        ty: &ValueType<Prim>,
        substitutions: &mut Substitutions<Prim>,
        errors: SpannedTypeErrors<'_, Prim>,
    );
}

/// Numeric constraints. In particular, this is [`TypeConstraints`] associated
/// with the [`Num`](crate::Num) literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumConstraints {
    /// No constraints.
    None,
    /// Type can be subject to unary `-` and can participate in `T op Num` / `Num op T` operations.
    ///
    /// Defined recursively as linear primitive types and tuples consisting of `Lin` types.
    Lin,
    /// Type can participate in binary arithmetic ops (`T op T`).
    ///
    /// Defined as a subset of `Lin` types without dynamically sized slices and
    /// any types containing dynamically sized slices.
    Ops,
}

impl Default for NumConstraints {
    fn default() -> Self {
        Self::None
    }
}

impl fmt::Display for NumConstraints {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::None => "",
            Self::Lin => "Lin",
            Self::Ops => "Ops",
        })
    }
}

impl FromStr for NumConstraints {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Lin" => Ok(Self::Lin),
            "Ops" => Ok(Self::Ops),
            _ => Err(anyhow::anyhow!("Expected `Lin` or `Ops`")),
        }
    }
}

impl ops::BitOrAssign<&Self> for NumConstraints {
    fn bitor_assign(&mut self, rhs: &Self) {
        *self = match (*self, rhs) {
            (Self::Ops, _) | (_, Self::Ops) => Self::Ops,
            (Self::Lin, _) | (_, Self::Lin) => Self::Lin,
            _ => Self::None,
        };
    }
}

impl NumConstraints {
    /// Default constraint settings for arithmetic ops.
    pub const OP_SETTINGS: OpConstraintSettings<'static, Self> = OpConstraintSettings {
        lin: &Self::Lin,
        ops: &Self::Ops,
    };
}

/// Primitive type which supports a notion of *linearity*. Linear types are types that
/// can be used in arithmetic ops.
pub trait LinearType: PrimitiveType<Constraints = NumConstraints> {
    /// Returns `true` iff this type is linear.
    fn is_linear(&self) -> bool;
}

impl<Prim: LinearType> TypeConstraints<Prim> for NumConstraints {
    // TODO: extract common logic for it to be reusable?
    fn apply(
        &self,
        ty: &ValueType<Prim>,
        substitutions: &mut Substitutions<Prim>,
        mut errors: SpannedTypeErrors<'_, Prim>,
    ) {
        if *self == Self::None {
            // The default constraint: does nothing.
            return;
        }

        let resolved_ty = if let ValueType::Var(var) = ty {
            debug_assert!(var.is_free());
            substitutions.insert_constraints(var.index(), self);
            substitutions.fast_resolve(ty)
        } else {
            ty
        };

        match resolved_ty {
            ValueType::Some => unreachable!(),

            // `Var`s are taken care of previously. `Any` satisfies any constraints.
            ValueType::Any(_) | ValueType::Var(_) => {}
            ValueType::Prim(lit) if lit.is_linear() => {}

            ValueType::Function(_) | ValueType::Prim(_) => {
                errors.push(TypeErrorKind::failed_constraint(
                    ty.to_owned(),
                    self.to_owned(),
                ));
            }

            ValueType::Tuple(tuple) => {
                let tuple = tuple.to_owned();

                if *self == Self::Ops {
                    let middle_len = tuple.parts().1.map(Slice::len);
                    if let Some(middle_len) = middle_len {
                        if let Err(err) = substitutions.apply_static_len(middle_len) {
                            errors.push(err);
                        }
                    }
                }

                for element in tuple.element_types() {
                    self.apply(element, substitutions, errors.by_ref());
                }
            }
        }
    }
}

/// [`TypeConstraints`] implementation with no supported constraints.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct NoConstraints(());

impl fmt::Display for NoConstraints {
    fn fmt(&self, _formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl FromStr for NoConstraints {
    type Err = anyhow::Error;

    fn from_str(_: &str) -> Result<Self, Self::Err> {
        Err(anyhow::anyhow!("Cannot be instantiated"))
    }
}

impl ops::BitOrAssign<&Self> for NoConstraints {
    fn bitor_assign(&mut self, _rhs: &Self) {
        // Do nothing
    }
}

impl<Prim> TypeConstraints<Prim> for NoConstraints
where
    Prim: PrimitiveType<Constraints = Self>,
{
    fn apply(
        &self,
        _ty: &ValueType<Prim>,
        _substitutions: &mut Substitutions<Prim>,
        _errors: SpannedTypeErrors<'_, Prim>,
    ) {
        // Do nothing
    }
}
