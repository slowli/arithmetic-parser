//! `TypeArithmetic` and its implementations.

use num_traits::{NumOps, Pow};

use std::ops;

use crate::{
    LinConstraints, LiteralType, MapLiteralType, Num, Substitutions, TypeConstraints,
    TypeErrorKind, TypeResult, ValueType,
};
use arithmetic_parser::{BinaryOp, Spanned, UnaryOp};

/// Arithmetic allowing to customize literal types and how unary and binary operations are handled
/// during type inference.
pub trait TypeArithmetic<Lit: LiteralType> {
    /// Handles a unary operation.
    fn process_unary_op<'a>(
        &self,
        substitutions: &mut Substitutions<Lit>,
        spans: UnaryOpSpans<'a, Lit>,
    ) -> TypeResult<'a, Lit>;

    /// Handles a binary operation.
    fn process_binary_op<'a>(
        &self,
        substitutions: &mut Substitutions<Lit>,
        spans: BinaryOpSpans<'a, Lit>,
    ) -> TypeResult<'a, Lit>;
}

/// Code spans related to a unary operation.
///
/// Used in [`TypeArithmetic::process_unary_op()`].
#[derive(Debug, Clone)]
pub struct UnaryOpSpans<'a, Lit: LiteralType> {
    /// Total span of the operation call.
    pub total: Spanned<'a>,
    /// Spanned unary operation.
    pub op: Spanned<'a, UnaryOp>,
    /// Span of the inner operation.
    pub inner: Spanned<'a, ValueType<Lit>>,
}

/// Code spans related to a binary operation.
///
/// Used in [`TypeArithmetic::process_binary_op()`].
#[derive(Debug, Clone)]
pub struct BinaryOpSpans<'a, Lit: LiteralType> {
    /// Total span of the operation call.
    pub total: Spanned<'a>,
    /// Spanned binary operation.
    pub op: Spanned<'a, BinaryOp>,
    /// Spanned left-hand side.
    pub lhs: Spanned<'a, ValueType<Lit>>,
    /// Spanned right-hand side.
    pub rhs: Spanned<'a, ValueType<Lit>>,
}

/// Simplest [`TypeArithmetic`] implementation that defines unary / binary ops only on
/// the Boolean type. Useful as a building block for more complex arithmetics.
#[derive(Debug, Clone, Copy, Default)]
pub struct BoolArithmetic;

impl<Lit: LiteralType> TypeArithmetic<Lit> for BoolArithmetic {
    /// Processes a unary operation.
    ///
    /// - `!` requires a Boolean input and outputs a Boolean.
    /// - Other operations fail with [`TypeErrorKind::Unsupported`].
    fn process_unary_op<'a>(
        &self,
        substitutions: &mut Substitutions<Lit>,
        spans: UnaryOpSpans<'a, Lit>,
    ) -> TypeResult<'a, Lit> {
        let op = spans.op.extra;
        match op {
            UnaryOp::Not => {
                substitutions
                    .unify(&ValueType::Bool, &spans.inner.extra)
                    .map_err(|err| err.with_span(&spans.inner))?;
                Ok(ValueType::Bool)
            }

            _ => Err(TypeErrorKind::unsupported(op).with_span(&spans.op)),
        }
    }

    /// Processes a binary operation.
    ///
    /// - `==` and `!=` require LHS and RHS to have the same type (no matter which one).
    ///   These ops return `Bool`.
    /// - `&&` and `||` require LHS and RHS to have `Bool` type. These ops return `Bool`.
    /// - Other operations fail with [`TypeErrorKind::Unsupported`].
    fn process_binary_op<'a>(
        &self,
        substitutions: &mut Substitutions<Lit>,
        spans: BinaryOpSpans<'a, Lit>,
    ) -> TypeResult<'a, Lit> {
        let op = spans.op.extra;
        let lhs_ty = &spans.lhs.extra;
        let rhs_ty = &spans.rhs.extra;

        match op {
            BinaryOp::Eq | BinaryOp::NotEq => {
                substitutions.unify(&lhs_ty, &rhs_ty).map_err(|err| {
                    err.into_op_mismatch(spans.lhs.extra, spans.rhs.extra, op)
                        .with_span(&spans.total)
                })?;
                Ok(ValueType::Bool)
            }

            BinaryOp::And | BinaryOp::Or => {
                substitutions
                    .unify(&ValueType::Bool, lhs_ty)
                    .map_err(|err| err.with_span(&spans.lhs))?;
                substitutions
                    .unify(&ValueType::Bool, rhs_ty)
                    .map_err(|err| err.with_span(&spans.rhs))?;
                Ok(ValueType::Bool)
            }

            _ => Err(TypeErrorKind::unsupported(op).with_span(&spans.op)),
        }
    }
}

/// Arithmetic on [`Num`]bers.
///
/// # Type inference for literals
///
/// All literals have a single type, [`Num`].
///
/// # Unary ops
///
/// - Unary minus is follows the equation `-T == T`, where `T` is any linear type.
/// - Unary negation is only defined for `Bool`s.
///
/// # Binary ops
///
/// Binary ops fall into 3 cases: `Num op T == T`, `T op Num == T`, or `T op T == T`,
/// where `T` is any linear type (that is, `Num` or tuple of linear types).
/// `T op T` is assumed by default, only falling into two other cases if one of operands
/// is known to be a number and the other is not a number.
///
/// # Comparisons
///
/// Order comparisons (`>`, `<`, `>=`, `<=`) can be switched on or off. Use
/// [`Self::with_comparisons()`] constructor to switch them on. If switched on, both arguments
/// of the order comparison must be numbers.
#[derive(Debug, Clone)]
pub struct NumArithmetic {
    comparisons_enabled: bool,
}

impl NumArithmetic {
    /// Creates an instance of arithmetic that does not support order comparisons.
    pub const fn without_comparisons() -> Self {
        Self {
            comparisons_enabled: false,
        }
    }

    /// Creates an instance of arithmetic that supports order comparisons.
    pub const fn with_comparisons() -> Self {
        Self {
            comparisons_enabled: true,
        }
    }

    /// Applies [binary ops](#binary-ops) logic to unify the given LHS and RHS types.
    /// Returns the result type of the binary operation.
    ///
    /// This logic can be reused by other [`TypeArithmetic`] implementations.
    pub fn unify_binary_op<Lit: LiteralType>(
        substitutions: &mut Substitutions<Lit>,
        lhs_ty: &ValueType<Lit>,
        rhs_ty: &ValueType<Lit>,
        constraints: &Lit::Constraints,
    ) -> Result<ValueType<Lit>, TypeErrorKind<Lit>> {
        constraints.apply(lhs_ty, substitutions)?;
        constraints.apply(rhs_ty, substitutions)?;

        let resolved_lhs_ty = substitutions.fast_resolve(lhs_ty);
        let resolved_rhs_ty = substitutions.fast_resolve(rhs_ty);

        match (resolved_lhs_ty.is_literal(), resolved_rhs_ty.is_literal()) {
            (Some(true), Some(false)) => Ok(resolved_rhs_ty.to_owned()),
            (Some(false), Some(true)) => Ok(resolved_lhs_ty.to_owned()),
            _ => {
                substitutions.unify(lhs_ty, rhs_ty)?;
                Ok(lhs_ty.to_owned())
            }
        }
    }

    /// FIXME
    pub fn process_unary_op<'a, Lit: LiteralType>(
        substitutions: &mut Substitutions<Lit>,
        spans: UnaryOpSpans<'a, Lit>,
        constraints: &Lit::Constraints,
    ) -> TypeResult<'a, Lit> {
        let op = spans.op.extra;
        let inner_ty = &spans.inner.extra;

        match op {
            UnaryOp::Not => BoolArithmetic.process_unary_op(substitutions, spans),

            UnaryOp::Neg => {
                constraints
                    .apply(inner_ty, substitutions)
                    .map_err(|err| err.with_span(&spans.inner))?;
                Ok(spans.inner.extra)
            }

            _ => Err(TypeErrorKind::unsupported(op).with_span(&spans.op)),
        }
    }

    /// FIXME
    pub fn process_binary_op<'a, Lit: LiteralType>(
        substitutions: &mut Substitutions<Lit>,
        spans: BinaryOpSpans<'a, Lit>,
        comparable_type: Option<Lit>,
        constraints: &Lit::Constraints,
    ) -> TypeResult<'a, Lit> {
        let op = spans.op.extra;
        let lhs_ty = &spans.lhs.extra;
        let rhs_ty = &spans.rhs.extra;

        match op {
            BinaryOp::And | BinaryOp::Or | BinaryOp::Eq | BinaryOp::NotEq => {
                BoolArithmetic.process_binary_op(substitutions, spans)
            }

            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Power => {
                Self::unify_binary_op(substitutions, lhs_ty, rhs_ty, constraints)
                    .map_err(|err| err.with_span(&spans.total))
            }

            BinaryOp::Ge | BinaryOp::Le | BinaryOp::Lt | BinaryOp::Gt => {
                if let Some(ty) = comparable_type {
                    let ty = ValueType::Lit(ty);
                    substitutions
                        .unify(&ty, lhs_ty)
                        .map_err(|err| err.with_span(&spans.lhs))?;
                    substitutions
                        .unify(&ty, rhs_ty)
                        .map_err(|err| err.with_span(&spans.rhs))?;
                    Ok(ValueType::Bool)
                } else {
                    Err(TypeErrorKind::unsupported(op).with_span(&spans.op))
                }
            }

            _ => Err(TypeErrorKind::unsupported(op).with_span(&spans.op)),
        }
    }
}

impl<Val> MapLiteralType<Val> for NumArithmetic
where
    Val: Clone + NumOps + PartialEq + ops::Neg<Output = Val> + Pow<Val, Output = Val>,
{
    type Lit = Num;

    fn type_of_literal(&self, _: &Val) -> Self::Lit {
        Num
    }
}

impl TypeArithmetic<Num> for NumArithmetic {
    fn process_unary_op<'a>(
        &self,
        substitutions: &mut Substitutions<Num>,
        spans: UnaryOpSpans<'a, Num>,
    ) -> TypeResult<'a, Num> {
        Self::process_unary_op(substitutions, spans, &LinConstraints::LIN)
    }

    fn process_binary_op<'a>(
        &self,
        substitutions: &mut Substitutions<Num>,
        spans: BinaryOpSpans<'a, Num>,
    ) -> TypeResult<'a, Num> {
        let comparable_type = if self.comparisons_enabled {
            Some(Num)
        } else {
            None
        };
        Self::process_binary_op(substitutions, spans, comparable_type, &LinConstraints::LIN)
    }
}
