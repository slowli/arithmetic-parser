//! Types allowing to customize various aspects of the type system, such as type constraints
//! and behavior of unary / binary ops.

use num_traits::NumOps;

use crate::{
    error::{ErrorKind, ErrorLocation, OpErrors},
    Num, PrimitiveType, Substitutions, Type,
};
use arithmetic_parser::{BinaryOp, UnaryOp};

mod constraints;

pub use self::constraints::{
    CompleteConstraints, Constraint, ConstraintSet, LinearType, NumConstraints, StructConstraint,
};

/// Maps a literal value from a certain [`Grammar`] to its type. This assumes that all literals
/// are primitive.
///
/// [`Grammar`]: arithmetic_parser::grammars::Grammar
pub trait MapPrimitiveType<Val> {
    /// Types of literals output by this mapper.
    type Prim: PrimitiveType;

    /// Gets the type of the provided literal value.
    fn type_of_literal(&self, lit: &Val) -> Self::Prim;
}

/// Arithmetic allowing to customize primitive types and how unary and binary operations are handled
/// during type inference.
///
/// # Examples
///
/// See crate examples for examples how define custom arithmetics.
pub trait TypeArithmetic<Prim: PrimitiveType> {
    /// Handles a unary operation.
    fn process_unary_op(
        &self,
        substitutions: &mut Substitutions<Prim>,
        context: &UnaryOpContext<Prim>,
        errors: OpErrors<'_, Prim>,
    ) -> Type<Prim>;

    /// Handles a binary operation.
    fn process_binary_op(
        &self,
        substitutions: &mut Substitutions<Prim>,
        context: &BinaryOpContext<Prim>,
        errors: OpErrors<'_, Prim>,
    ) -> Type<Prim>;
}

/// Code spans related to a unary operation.
///
/// Used in [`TypeArithmetic::process_unary_op()`].
#[derive(Debug, Clone)]
pub struct UnaryOpContext<Prim: PrimitiveType> {
    /// Unary operation.
    pub op: UnaryOp,
    /// Operation argument.
    pub arg: Type<Prim>,
}

/// Code spans related to a binary operation.
///
/// Used in [`TypeArithmetic::process_binary_op()`].
#[derive(Debug, Clone)]
pub struct BinaryOpContext<Prim: PrimitiveType> {
    /// Binary operation.
    pub op: BinaryOp,
    /// Spanned left-hand side.
    pub lhs: Type<Prim>,
    /// Spanned right-hand side.
    pub rhs: Type<Prim>,
}

/// [`PrimitiveType`] that has Boolean type as one of its variants.
pub trait WithBoolean: PrimitiveType {
    /// Boolean type.
    const BOOL: Self;
}

/// Simplest [`TypeArithmetic`] implementation that defines unary / binary ops only on
/// the Boolean type. Useful as a building block for more complex arithmetics.
#[derive(Debug, Clone, Copy, Default)]
pub struct BoolArithmetic;

impl<Prim: WithBoolean> TypeArithmetic<Prim> for BoolArithmetic {
    /// Processes a unary operation.
    ///
    /// - `!` requires a Boolean input and outputs a Boolean.
    /// - Other operations fail with [`ErrorKind::UnsupportedFeature`].
    fn process_unary_op<'a>(
        &self,
        substitutions: &mut Substitutions<Prim>,
        context: &UnaryOpContext<Prim>,
        mut errors: OpErrors<'_, Prim>,
    ) -> Type<Prim> {
        let op = context.op;
        if op == UnaryOp::Not {
            substitutions.unify(&Type::BOOL, &context.arg, errors);
            Type::BOOL
        } else {
            let err = ErrorKind::unsupported(op);
            errors.push(err);
            substitutions.new_type_var()
        }
    }

    /// Processes a binary operation.
    ///
    /// - `==` and `!=` require LHS and RHS to have the same type (no matter which one).
    ///   These ops return `Bool`.
    /// - `&&` and `||` require LHS and RHS to have `Bool` type. These ops return `Bool`.
    /// - Other operations fail with [`ErrorKind::UnsupportedFeature`].
    fn process_binary_op(
        &self,
        substitutions: &mut Substitutions<Prim>,
        context: &BinaryOpContext<Prim>,
        mut errors: OpErrors<'_, Prim>,
    ) -> Type<Prim> {
        match context.op {
            BinaryOp::Eq | BinaryOp::NotEq => {
                substitutions.unify(&context.lhs, &context.rhs, errors);
                Type::BOOL
            }

            BinaryOp::And | BinaryOp::Or => {
                substitutions.unify(
                    &Type::BOOL,
                    &context.lhs,
                    errors.with_location(ErrorLocation::Lhs),
                );
                substitutions.unify(
                    &Type::BOOL,
                    &context.rhs,
                    errors.with_location(ErrorLocation::Rhs),
                );
                Type::BOOL
            }

            _ => {
                errors.push(ErrorKind::unsupported(context.op));
                substitutions.new_type_var()
            }
        }
    }
}

/// Settings for constraints placed on arguments of binary arithmetic operations.
#[derive(Debug)]
pub struct OpConstraintSettings<'a, Prim: PrimitiveType> {
    /// Constraint applied to the argument of `T op Num` / `Num op T` ops.
    pub lin: &'a dyn Constraint<Prim>,
    /// Constraint applied to the arguments of in-kind binary arithmetic ops (`T op T`).
    pub ops: &'a dyn Constraint<Prim>,
}

impl<Prim: PrimitiveType> Clone for OpConstraintSettings<'_, Prim> {
    fn clone(&self) -> Self {
        Self {
            lin: self.lin,
            ops: self.ops,
        }
    }
}

impl<Prim: PrimitiveType> Copy for OpConstraintSettings<'_, Prim> {}

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
    ///
    /// # Arguments
    ///
    /// - `settings` are applied to arguments of arithmetic ops.
    pub fn unify_binary_op<Prim: PrimitiveType>(
        substitutions: &mut Substitutions<Prim>,
        context: &BinaryOpContext<Prim>,
        mut errors: OpErrors<'_, Prim>,
        settings: OpConstraintSettings<'_, Prim>,
    ) -> Type<Prim> {
        let lhs_ty = &context.lhs;
        let rhs_ty = &context.rhs;
        let resolved_lhs_ty = substitutions.fast_resolve(lhs_ty);
        let resolved_rhs_ty = substitutions.fast_resolve(rhs_ty);

        match (
            resolved_lhs_ty.is_primitive(),
            resolved_rhs_ty.is_primitive(),
        ) {
            (Some(true), Some(false)) => {
                let resolved_rhs_ty = resolved_rhs_ty.clone();
                settings
                    .lin
                    .visitor(substitutions, errors.with_location(ErrorLocation::Lhs))
                    .visit_type(lhs_ty);
                settings
                    .lin
                    .visitor(substitutions, errors.with_location(ErrorLocation::Rhs))
                    .visit_type(rhs_ty);
                resolved_rhs_ty
            }
            (Some(false), Some(true)) => {
                let resolved_lhs_ty = resolved_lhs_ty.clone();
                settings
                    .lin
                    .visitor(substitutions, errors.with_location(ErrorLocation::Lhs))
                    .visit_type(lhs_ty);
                settings
                    .lin
                    .visitor(substitutions, errors.with_location(ErrorLocation::Rhs))
                    .visit_type(rhs_ty);
                resolved_lhs_ty
            }
            _ => {
                let lhs_is_valid = errors.with_location(ErrorLocation::Lhs).check(|errors| {
                    settings
                        .ops
                        .visitor(substitutions, errors)
                        .visit_type(lhs_ty);
                });
                let rhs_is_valid = errors.with_location(ErrorLocation::Rhs).check(|errors| {
                    settings
                        .ops
                        .visitor(substitutions, errors)
                        .visit_type(rhs_ty);
                });

                if lhs_is_valid && rhs_is_valid {
                    substitutions.unify(lhs_ty, rhs_ty, errors);
                }
                if lhs_is_valid {
                    lhs_ty.clone()
                } else {
                    rhs_ty.clone()
                }
            }
        }
    }

    /// Processes a unary operation according to [the numeric arithmetic rules](#unary-ops).
    /// Returns the result type of the unary operation.
    ///
    /// This logic can be reused by other [`TypeArithmetic`] implementations.
    pub fn process_unary_op<Prim: WithBoolean>(
        substitutions: &mut Substitutions<Prim>,
        context: &UnaryOpContext<Prim>,
        mut errors: OpErrors<'_, Prim>,
        constraints: &impl Constraint<Prim>,
    ) -> Type<Prim> {
        match context.op {
            UnaryOp::Not => BoolArithmetic.process_unary_op(substitutions, context, errors),
            UnaryOp::Neg => {
                constraints
                    .visitor(substitutions, errors)
                    .visit_type(&context.arg);
                context.arg.clone()
            }
            _ => {
                errors.push(ErrorKind::unsupported(context.op));
                substitutions.new_type_var()
            }
        }
    }

    /// Processes a binary operation according to [the numeric arithmetic rules](#binary-ops).
    /// Returns the result type of the unary operation.
    ///
    /// This logic can be reused by other [`TypeArithmetic`] implementations.
    ///
    /// # Arguments
    ///
    /// - If `comparable_type` is set to `Some(_)`, it will be used to unify arguments of
    ///   order comparisons. If `comparable_type` is `None`, order comparisons are not supported.
    /// - `constraints` are applied to arguments of arithmetic ops.
    pub fn process_binary_op<Prim: WithBoolean>(
        substitutions: &mut Substitutions<Prim>,
        context: &BinaryOpContext<Prim>,
        mut errors: OpErrors<'_, Prim>,
        comparable_type: Option<Prim>,
        settings: OpConstraintSettings<'_, Prim>,
    ) -> Type<Prim> {
        match context.op {
            BinaryOp::And | BinaryOp::Or | BinaryOp::Eq | BinaryOp::NotEq => {
                BoolArithmetic.process_binary_op(substitutions, context, errors)
            }

            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Power => {
                Self::unify_binary_op(substitutions, context, errors, settings)
            }

            BinaryOp::Ge | BinaryOp::Le | BinaryOp::Lt | BinaryOp::Gt => {
                if let Some(ty) = comparable_type {
                    let ty = Type::Prim(ty);
                    substitutions.unify(
                        &ty,
                        &context.lhs,
                        errors.with_location(ErrorLocation::Lhs),
                    );
                    substitutions.unify(
                        &ty,
                        &context.rhs,
                        errors.with_location(ErrorLocation::Rhs),
                    );
                } else {
                    let err = ErrorKind::unsupported(context.op);
                    errors.push(err);
                }
                Type::BOOL
            }

            _ => {
                errors.push(ErrorKind::unsupported(context.op));
                substitutions.new_type_var()
            }
        }
    }
}

impl<Val> MapPrimitiveType<Val> for NumArithmetic
where
    Val: Clone + NumOps + PartialEq,
{
    type Prim = Num;

    fn type_of_literal(&self, _: &Val) -> Self::Prim {
        Num::Num
    }
}

impl TypeArithmetic<Num> for NumArithmetic {
    fn process_unary_op<'a>(
        &self,
        substitutions: &mut Substitutions<Num>,
        context: &UnaryOpContext<Num>,
        errors: OpErrors<'_, Num>,
    ) -> Type<Num> {
        Self::process_unary_op(substitutions, context, errors, &NumConstraints::Lin)
    }

    fn process_binary_op<'a>(
        &self,
        substitutions: &mut Substitutions<Num>,
        context: &BinaryOpContext<Num>,
        errors: OpErrors<'_, Num>,
    ) -> Type<Num> {
        const OP_SETTINGS: OpConstraintSettings<'static, Num> = OpConstraintSettings {
            lin: &NumConstraints::Lin,
            ops: &NumConstraints::Ops,
        };

        let comparable_type = if self.comparisons_enabled {
            Some(Num::Num)
        } else {
            None
        };

        Self::process_binary_op(substitutions, context, errors, comparable_type, OP_SETTINGS)
    }
}
