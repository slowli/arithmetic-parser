//! `TypeEnvironment` and related types.

use num_traits::{NumOps, Pow};

use std::{
    collections::HashMap,
    iter::{self, FromIterator},
    mem, ops,
};

use crate::{
    substitutions::Substitutions, FnType, LiteralType, Num, TypeError, TypeErrorKind, ValueType,
};
use arithmetic_parser::{
    grammars::Grammar, BinaryOp, Block, Destructure, Expr, FnDefinition, Lvalue, Spanned,
    SpannedExpr, SpannedLvalue, SpannedStatement, Statement, UnaryOp,
};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod type_annotation_tests;

/// Result of inferring type for a certain expression.
pub type TypeResult<'a, Lit> = Result<ValueType<Lit>, TypeError<'a, Lit>>;

/// Arithmetic allowing to customize literal types and how unary and binary operations are handled
/// during type inference.
pub trait TypeArithmetic<Val> {
    /// Types of literals that this arithmetic operates on.
    type Lit: LiteralType;

    /// Gets the type of the provided literal.
    fn type_of_literal(&self, lit: &Val) -> Self::Lit;

    /// Handles a unary operation.
    fn process_unary_op<'a>(
        &self,
        substitutions: &mut Substitutions<Self::Lit>,
        spans: UnaryOpSpans<'a, Self::Lit>,
    ) -> TypeResult<'a, Self::Lit>;

    /// Handles a binary operation.
    fn process_binary_op<'a>(
        &self,
        substitutions: &mut Substitutions<Self::Lit>,
        spans: BinaryOpSpans<'a, Self::Lit>,
    ) -> TypeResult<'a, Self::Lit>;
}

/// Code spans related to a unary operation.
///
/// Used in [`TypeArithmetic::process_unary_op()`].
#[derive(Debug, Clone)]
pub struct UnaryOpSpans<'a, Lit> {
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
pub struct BinaryOpSpans<'a, Lit> {
    /// Total span of the operation call.
    pub total: Spanned<'a>,
    /// Spanned binary operation.
    pub op: Spanned<'a, BinaryOp>,
    /// Spanned left-hand side.
    pub lhs: Spanned<'a, ValueType<Lit>>,
    /// Spanned right-hand side.
    pub rhs: Spanned<'a, ValueType<Lit>>,
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
/// where `T` is any linear type.
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

    fn unify_binary_op<Lit: LiteralType>(
        substitutions: &mut Substitutions<Lit>,
        lhs_ty: &ValueType<Lit>,
        rhs_ty: &ValueType<Lit>,
    ) -> Result<ValueType<Lit>, TypeErrorKind<Lit>> {
        substitutions.mark_as_linear(lhs_ty)?;
        substitutions.mark_as_linear(rhs_ty)?;

        let resolved_lhs_ty = substitutions.fast_resolve(lhs_ty);
        let resolved_rhs_ty = substitutions.fast_resolve(rhs_ty);

        match (resolved_lhs_ty.is_number(), resolved_rhs_ty.is_number()) {
            (Some(true), Some(false)) => Ok(resolved_rhs_ty.to_owned()),
            (Some(false), Some(true)) => Ok(resolved_lhs_ty.to_owned()),
            _ => {
                substitutions.unify(lhs_ty, rhs_ty)?;
                Ok(lhs_ty.to_owned())
            }
        }
    }
}

impl<Val> TypeArithmetic<Val> for NumArithmetic
where
    Val: Clone + NumOps + PartialEq + ops::Neg<Output = Val> + Pow<Val, Output = Val>,
{
    type Lit = Num;

    fn type_of_literal(&self, _: &Val) -> Self::Lit {
        Num
    }

    fn process_unary_op<'a>(
        &self,
        substitutions: &mut Substitutions<Self::Lit>,
        spans: UnaryOpSpans<'a, Self::Lit>,
    ) -> TypeResult<'a, Self::Lit> {
        let op = spans.op.extra;
        let inner_ty = &spans.inner.extra;

        match op {
            UnaryOp::Not => {
                substitutions
                    .unify(&ValueType::Bool, inner_ty)
                    .map_err(|err| err.with_span(&spans.inner))?;
                Ok(ValueType::Bool)
            }

            UnaryOp::Neg => {
                substitutions
                    .mark_as_linear(inner_ty)
                    .map_err(|err| err.with_span(&spans.inner))?;
                Ok(spans.inner.extra)
            }

            _ => Err(TypeErrorKind::unsupported(op).with_span(&spans.op)),
        }
    }

    fn process_binary_op<'a>(
        &self,
        substitutions: &mut Substitutions<Self::Lit>,
        spans: BinaryOpSpans<'a, Self::Lit>,
    ) -> TypeResult<'a, Self::Lit> {
        let op = spans.op.extra;
        let lhs_ty = &spans.lhs.extra;
        let rhs_ty = &spans.rhs.extra;

        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Power => {
                Self::unify_binary_op(substitutions, lhs_ty, rhs_ty)
                    .map_err(|err| err.with_span(&spans.total))
            }

            BinaryOp::Eq | BinaryOp::NotEq => {
                substitutions.unify(&lhs_ty, &rhs_ty).map_err(|err| {
                    err.into_op_mismatch(substitutions, lhs_ty, rhs_ty, op)
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

            BinaryOp::Ge | BinaryOp::Le | BinaryOp::Lt | BinaryOp::Gt => {
                if self.comparisons_enabled {
                    substitutions
                        .unify(&ValueType::Lit(Num), lhs_ty)
                        .map_err(|err| err.with_span(&spans.lhs))?;
                    substitutions
                        .unify(&ValueType::Lit(Num), rhs_ty)
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

/// Environment containing type information on named variables.
#[derive(Debug, Clone)]
pub struct TypeEnvironment<Lit> {
    variables: HashMap<String, ValueType<Lit>>,
}

impl<Lit: LiteralType> Default for TypeEnvironment<Lit> {
    fn default() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }
}

impl<Lit: LiteralType> TypeEnvironment<Lit> {
    /// Creates an empty environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Gets type of the specified variable.
    pub fn get_type(&self, name: &str) -> Option<&ValueType<Lit>> {
        self.variables.get(name)
    }

    /// Iterates over variables contained in this env.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ValueType<Lit>)> + '_ {
        self.variables.iter().map(|(name, ty)| (name.as_str(), ty))
    }

    /// Sets type of a variable.
    pub fn insert_type(&mut self, name: &str, value_type: ValueType<Lit>) -> &mut Self {
        self.variables.insert(name.to_owned(), value_type);
        self
    }

    /// Processes statements with the default type arithmetic. After processing, the context
    /// will contain type info about newly declared vars.
    ///
    /// This method is a shortcut for calling `process_with_arithmetic` with
    /// [`NumArithmetic::without_comparisons()`].
    pub fn process_statements<'a, T>(
        &mut self,
        block: &Block<'a, T>,
    ) -> Result<ValueType<Lit>, TypeError<'a, Lit>>
    where
        T: Grammar<Type = ValueType<Lit>>,
        NumArithmetic: TypeArithmetic<T::Lit, Lit = Lit>,
    {
        self.process_with_arithmetic(&NumArithmetic::without_comparisons(), block)
    }

    /// Processes statements with a given `arithmetic`. After processing, the context will contain
    /// type info about newly declared vars.
    pub fn process_with_arithmetic<'a, T, A>(
        &mut self,
        arithmetic: &A,
        block: &Block<'a, T>,
    ) -> Result<ValueType<Lit>, TypeError<'a, Lit>>
    where
        T: Grammar<Type = ValueType<Lit>>,
        A: TypeArithmetic<T::Lit, Lit = Lit>,
    {
        TypeProcessor::new(self, arithmetic).process_statements(block)
    }
}

impl<Lit: LiteralType> ops::Index<&str> for TypeEnvironment<Lit> {
    type Output = ValueType<Lit>;

    fn index(&self, name: &str) -> &Self::Output {
        self.get_type(name)
            .unwrap_or_else(|| panic!("Variable `{}` is not defined", name))
    }
}

impl<Lit: LiteralType, S, Ty> FromIterator<(S, Ty)> for TypeEnvironment<Lit>
where
    S: Into<String>,
    Ty: Into<ValueType<Lit>>,
{
    fn from_iter<I: IntoIterator<Item = (S, Ty)>>(iter: I) -> Self {
        Self {
            variables: iter
                .into_iter()
                .map(|(name, ty)| (name.into(), ty.into()))
                .collect(),
        }
    }
}

impl<Lit: LiteralType, S, Ty> Extend<(S, Ty)> for TypeEnvironment<Lit>
where
    S: Into<String>,
    Ty: Into<ValueType<Lit>>,
{
    fn extend<I: IntoIterator<Item = (S, Ty)>>(&mut self, iter: I) {
        self.variables
            .extend(iter.into_iter().map(|(name, ty)| (name.into(), ty.into())))
    }
}

/// Processor for deriving type information.
struct TypeProcessor<'a, L, Lit: LiteralType> {
    root_scope: &'a mut TypeEnvironment<Lit>,
    inner_scopes: Vec<TypeEnvironment<Lit>>,
    arithmetic: &'a dyn TypeArithmetic<L, Lit = Lit>,
    is_in_function: bool,
}

impl<'a, L, Lit: LiteralType> TypeProcessor<'a, L, Lit> {
    fn new(
        env: &'a mut TypeEnvironment<Lit>,
        arithmetic: &'a dyn TypeArithmetic<L, Lit = Lit>,
    ) -> Self {
        Self {
            root_scope: env,
            inner_scopes: vec![],
            arithmetic,
            is_in_function: false,
        }
    }
}

impl<L, Lit: LiteralType> TypeProcessor<'_, L, Lit> {
    fn get_type(&self, name: &str) -> Option<&ValueType<Lit>> {
        self.inner_scopes
            .iter()
            .rev()
            .flat_map(|scope| scope.variables.get(name))
            .next()
            .or_else(|| self.root_scope.get_type(name))
    }

    fn insert_type(&mut self, name: &str, ty: ValueType<Lit>) {
        let scope = self
            .inner_scopes
            .last_mut()
            .unwrap_or(&mut *self.root_scope);
        scope.insert_type(name, ty);
    }

    fn process_expr_inner<'a, T>(
        &mut self,
        substitutions: &mut Substitutions<Lit>,
        expr: &SpannedExpr<'a, T>,
    ) -> TypeResult<'a, Lit>
    where
        T: Grammar<Lit = L, Type = ValueType<Lit>>,
    {
        match &expr.extra {
            Expr::Variable => self.process_var(expr),

            Expr::Literal(lit) => Ok(ValueType::Lit(self.arithmetic.type_of_literal(lit))),

            Expr::Tuple(ref terms) => {
                let term_types: Result<Vec<_>, _> = terms
                    .iter()
                    .map(|term| self.process_expr_inner(substitutions, term))
                    .collect();
                term_types.map(ValueType::Tuple)
            }

            Expr::Function { name, args } => {
                let fn_type = self.process_expr_inner(substitutions, name)?;
                self.process_fn_call(substitutions, expr, &fn_type, args.iter())
            }

            Expr::Method {
                name,
                receiver,
                args,
            } => {
                let fn_type = self.process_var(name)?;
                let all_args = iter::once(receiver.as_ref()).chain(args);
                self.process_fn_call(substitutions, expr, &fn_type, all_args)
            }

            Expr::Block(block) => {
                self.inner_scopes.push(TypeEnvironment::default());
                let result = self.process_block(substitutions, block);
                self.inner_scopes.pop(); // intentionally called even on failure
                result
            }

            Expr::FnDefinition(def) => self
                .process_fn_def(substitutions, def)
                .map(|fn_type| ValueType::Function(Box::new(fn_type))),

            Expr::Unary { op, inner } => self.process_unary_op(substitutions, expr, *op, inner),

            Expr::Binary { lhs, rhs, op } => {
                self.process_binary_op(substitutions, expr, *op, lhs, rhs)
            }

            _ => Err(TypeErrorKind::unsupported(expr.extra.ty()).with_span(expr)),
        }
    }

    #[inline]
    fn process_var<'a, T>(&self, name: &Spanned<'a, T>) -> TypeResult<'a, Lit> {
        self.get_type(name.fragment()).cloned().ok_or_else(|| {
            TypeErrorKind::UndefinedVar((*name.fragment()).to_owned()).with_span(name)
        })
    }

    fn process_block<'a, T>(
        &mut self,
        substitutions: &mut Substitutions<Lit>,
        block: &Block<'a, T>,
    ) -> TypeResult<'a, Lit>
    where
        T: Grammar<Lit = L, Type = ValueType<Lit>>,
    {
        for statement in &block.statements {
            self.process_statement(substitutions, statement)?;
        }
        block.return_value.as_ref().map_or_else(
            || Ok(ValueType::void()),
            |return_value| self.process_expr_inner(substitutions, return_value),
        )
    }

    /// Processes an lvalue type by replacing `Any` types with newly created type vars.
    fn process_lvalue<'a>(
        &mut self,
        substitutions: &mut Substitutions<Lit>,
        lvalue: &SpannedLvalue<'a, ValueType<Lit>>,
    ) -> TypeResult<'a, Lit> {
        match &lvalue.extra {
            Lvalue::Variable { ty } => {
                let mut value_type = ty.as_ref().map_or(ValueType::Any, |ty| ty.extra.clone());
                substitutions
                    .assign_new_type(&mut value_type)
                    .map_err(|err| err.with_span(ty.as_ref().unwrap()))?;
                // `unwrap` is safe: an error can only occur with a type annotation present.

                self.insert_type(lvalue.fragment(), value_type.clone());
                Ok(value_type)
            }

            Lvalue::Tuple(destructure) => {
                let element_types = self.process_destructure(substitutions, destructure)?;
                Ok(ValueType::Tuple(element_types))
            }

            _ => Err(TypeErrorKind::unsupported(lvalue.extra.ty()).with_span(lvalue)),
        }
    }

    #[inline]
    fn process_destructure<'a>(
        &mut self,
        substitutions: &mut Substitutions<Lit>,
        destructure: &Destructure<'a, ValueType<Lit>>,
    ) -> Result<Vec<ValueType<Lit>>, TypeError<'a, Lit>> {
        if let Some(middle) = &destructure.middle {
            // TODO: allow middles with explicitly set type.
            return Err(TypeErrorKind::UnsupportedDestructure.with_span(middle));
        }

        destructure
            .start
            .iter()
            .chain(&destructure.end)
            .map(|element| self.process_lvalue(substitutions, element))
            .collect()
    }

    fn process_fn_call<'it, 'a: 'it, T>(
        &mut self,
        substitutions: &mut Substitutions<Lit>,
        call_expr: &SpannedExpr<'a, T>,
        fn_type: &ValueType<Lit>,
        args: impl Iterator<Item = &'it SpannedExpr<'a, T>>,
    ) -> TypeResult<'a, Lit>
    where
        T: Grammar<Lit = L, Type = ValueType<Lit>>,
    {
        let arg_types: Result<Vec<_>, _> = args
            .map(|arg| self.process_expr_inner(substitutions, arg))
            .collect();
        let arg_types = arg_types?;
        let return_type = substitutions
            .unify_fn_call(fn_type, arg_types)
            .map_err(|err| err.with_span(call_expr))?;

        Ok(return_type)
    }

    #[inline]
    fn process_unary_op<'a, T>(
        &mut self,
        substitutions: &mut Substitutions<Lit>,
        unary_expr: &SpannedExpr<'a, T>,
        op: Spanned<'a, UnaryOp>,
        inner: &SpannedExpr<'a, T>,
    ) -> TypeResult<'a, Lit>
    where
        T: Grammar<Lit = L, Type = ValueType<Lit>>,
    {
        let inner_ty = self.process_expr_inner(substitutions, inner)?;
        let spans = UnaryOpSpans {
            total: unary_expr.with_no_extra(),
            op,
            inner: inner.copy_with_extra(inner_ty),
        };
        self.arithmetic.process_unary_op(substitutions, spans)
    }

    #[inline]
    fn process_binary_op<'a, T>(
        &mut self,
        substitutions: &mut Substitutions<Lit>,
        binary_expr: &SpannedExpr<'a, T>,
        op: Spanned<'a, BinaryOp>,
        lhs: &SpannedExpr<'a, T>,
        rhs: &SpannedExpr<'a, T>,
    ) -> TypeResult<'a, Lit>
    where
        T: Grammar<Lit = L, Type = ValueType<Lit>>,
    {
        let lhs_ty = self.process_expr_inner(substitutions, lhs)?;
        let rhs_ty = self.process_expr_inner(substitutions, rhs)?;
        let spans = BinaryOpSpans {
            total: binary_expr.with_no_extra(),
            op,
            lhs: lhs.copy_with_extra(lhs_ty),
            rhs: rhs.copy_with_extra(rhs_ty),
        };
        self.arithmetic.process_binary_op(substitutions, spans)
    }

    fn process_fn_def<'a, T>(
        &mut self,
        substitutions: &mut Substitutions<Lit>,
        def: &FnDefinition<'a, T>,
    ) -> Result<FnType<Lit>, TypeError<'a, Lit>>
    where
        T: Grammar<Lit = L, Type = ValueType<Lit>>,
    {
        self.inner_scopes.push(TypeEnvironment::default());
        let was_in_function = mem::replace(&mut self.is_in_function, true);
        let result = self.process_fn_def_inner(substitutions, def);
        // Perform basic finalization in any case.
        self.inner_scopes.pop();
        self.is_in_function = was_in_function;

        let (arg_types, return_type) = result?;
        let arg_types = arg_types
            .iter()
            .map(|arg| substitutions.resolve(arg))
            .collect();
        let return_type = substitutions.resolve(&return_type);

        let mut fn_type = FnType::new(arg_types, return_type);
        if !self.is_in_function {
            fn_type.finalize(substitutions.linear_types());
        }

        Ok(fn_type)
    }

    /// Fallible part of fn definition processing.
    #[allow(clippy::type_complexity)] // FIXME
    fn process_fn_def_inner<'a, T>(
        &mut self,
        substitutions: &mut Substitutions<Lit>,
        def: &FnDefinition<'a, T>,
    ) -> Result<(Vec<ValueType<Lit>>, ValueType<Lit>), TypeError<'a, Lit>>
    where
        T: Grammar<Lit = L, Type = ValueType<Lit>>,
    {
        let arg_types = self.process_destructure(substitutions, &def.args.extra)?;
        let return_type = self.process_block(substitutions, &def.body)?;
        Ok((arg_types, return_type))
    }

    fn process_statement<'a, T>(
        &mut self,
        substitutions: &mut Substitutions<Lit>,
        statement: &SpannedStatement<'a, T>,
    ) -> TypeResult<'a, Lit>
    where
        T: Grammar<Lit = L, Type = ValueType<Lit>>,
    {
        match &statement.extra {
            Statement::Expr(expr) => self.process_expr_inner(substitutions, expr),

            Statement::Assignment { lhs, rhs } => {
                let rhs_ty = self.process_expr_inner(substitutions, rhs)?;
                let lhs_ty = self.process_lvalue(substitutions, lhs)?;
                substitutions
                    .unify(&lhs_ty, &rhs_ty)
                    .map(|()| ValueType::void())
                    .map_err(|err| err.with_span(statement))
            }

            _ => Err(TypeErrorKind::unsupported(statement.extra.ty()).with_span(statement)),
        }
    }

    fn process_statements<'a, T>(&mut self, block: &Block<'a, T>) -> TypeResult<'a, Lit>
    where
        T: Grammar<Lit = L, Type = ValueType<Lit>>,
    {
        let mut substitutions = Substitutions::default();

        let result = self.process_block(&mut substitutions, block);

        // We need to resolve vars even if an error occurred.
        debug_assert!(self.inner_scopes.is_empty());
        for var_type in self.root_scope.variables.values_mut() {
            *var_type = substitutions.resolve(var_type);
        }
        result.map(|ty| substitutions.resolve(&ty))
    }
}
