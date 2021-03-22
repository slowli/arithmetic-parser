//! `TypeConstraints` and implementations.

use std::{fmt, ops, str::FromStr};

use crate::{PrimitiveType, Substitutions, TypeErrorKind, ValueType};

/// Container for constraints that can be placed on type parameters / variables.
///
/// Constraints can be placed on [function](crate::FnType) type params, and can be applied
/// to types in [`TypeArithmetic`] impls. For example, [`NumArithmetic`] places
/// a [linearity constraint](LinConstraints::LIN) on types involved in arithmetic ops.
///
/// The constraint mechanism is similar to trait constraints in Rust, but is much more limited:
///
/// - Constraints cannot be parametric (cf. parameters in traits, such `AsRef<_>`
///   or `Iterator<Item = _>`).
/// - Constraints are applied to types in separation; it is impossible to create a constraint
///   involving several type params.
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
    /// [place constraints on type vars](Substitutions::insert_constraint()), e.g.,
    /// by recursively traversing and resolving the provided type.
    fn apply(
        &self,
        ty: &ValueType<Prim>,
        substitutions: &mut Substitutions<Prim>,
    ) -> Result<(), TypeErrorKind<Prim>>;
}

/// Linearity constraints. In particular, this is [`TypeConstraints`] associated
/// with the [`Num`](crate::Num) literal.
///
/// There is only one supported constraint: [linearity](Self::LIN). Linear types are types
/// that can be used as arguments of arithmetic ops.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LinConstraints {
    is_linear: bool,
}

impl fmt::Display for LinConstraints {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_linear {
            formatter.write_str("Lin")
        } else {
            Ok(())
        }
    }
}

impl FromStr for LinConstraints {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Lin" => Ok(Self { is_linear: true }),
            _ => Err(anyhow::anyhow!("Expected `Lin`")),
        }
    }
}

impl ops::BitOrAssign<&Self> for LinConstraints {
    #[allow(clippy::suspicious_op_assign_impl)] // "logical or" is intentional
    fn bitor_assign(&mut self, rhs: &Self) {
        self.is_linear = self.is_linear || rhs.is_linear;
    }
}

impl LinConstraints {
    /// Encumbered type is linear; it can be used as an argument in arithmetic ops.
    /// Recursively defined as primitive types for which [`LinearType::is_linear()`]
    /// returns `true`, and tuples in which all elements are linear.
    ///
    /// Displayed as `Lin`.
    pub const LIN: Self = Self { is_linear: true };
}

/// Primitive type which supports a notion of *linearity*. Linear types are types that
/// can be used in arithmetic ops.
pub trait LinearType: PrimitiveType<Constraints = LinConstraints> {
    /// Returns `true` iff this type is linear.
    fn is_linear(&self) -> bool;
}

impl<Prim: LinearType> TypeConstraints<Prim> for LinConstraints {
    // TODO: extract common logic for it to be reusable?
    fn apply(
        &self,
        ty: &ValueType<Prim>,
        substitutions: &mut Substitutions<Prim>,
    ) -> Result<(), TypeErrorKind<Prim>> {
        if !self.is_linear {
            // The default constraint: does nothing.
            return Ok(());
        }

        let resolved_ty = if let ValueType::Var(idx) = ty {
            substitutions.insert_constraint(*idx, self);
            substitutions.fast_resolve(ty)
        } else {
            ty
        };

        match resolved_ty {
            // `Var`s are taken care of previously.
            ValueType::Var(_) => Ok(()),

            ValueType::Prim(lit) if lit.is_linear() => Ok(()),

            ValueType::Some | ValueType::Param(_) => unreachable!(),

            ValueType::Bool | ValueType::Function(_) | ValueType::Prim(_) => Err(
                TypeErrorKind::failed_constraint(ty.to_owned(), self.to_owned()),
            ),

            ValueType::Tuple(elements) => {
                for element in elements.to_owned() {
                    self.apply(&element, substitutions)?;
                }
                Ok(())
            }
            ValueType::Slice { element, .. } => self.apply(&element.to_owned(), substitutions),
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
        // does nothing
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
    ) -> Result<(), TypeErrorKind<Prim>> {
        Ok(())
    }
}
