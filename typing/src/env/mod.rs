//! `TypeEnvironment` and related types.

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

/// FIXME
pub trait TypeArithmetic<Lit> {
    /// FIXME
    type LitType: LiteralType;

    /// FIXME
    fn type_of_literal(&self, lit: &Lit) -> Self::LitType;
}

/// FIXME
#[derive(Debug, Clone, Default)]
pub struct NumArithmetic;

impl<Lit> TypeArithmetic<Lit> for NumArithmetic {
    type LitType = Num;

    fn type_of_literal(&self, _: &Lit) -> Self::LitType {
        Num
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

    /// Processes statements. After processing, the context will contain type info
    /// about newly declared vars.
    pub fn process_statements<'a, T>(
        &mut self,
        block: &Block<'a, T>,
    ) -> Result<ValueType<Lit>, TypeError<'a, Lit>>
    where
        T: Grammar<Type = ValueType<Lit>>,
        NumArithmetic: TypeArithmetic<T::Lit, LitType = Lit>,
    {
        TypeProcessor::new(self, &NumArithmetic::default())
            .process_statements(block)
            .map_err(TypeError::new)
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
    arithmetic: &'a dyn TypeArithmetic<L, LitType = Lit>,
    is_in_function: bool,
}

impl<'a, L, Lit: LiteralType> TypeProcessor<'a, L, Lit> {
    fn new(
        env: &'a mut TypeEnvironment<Lit>,
        arithmetic: &'a dyn TypeArithmetic<L, LitType = Lit>,
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
    ) -> Result<ValueType<Lit>, Spanned<'a, TypeErrorKind<Lit>>>
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

            Expr::Unary { op, inner } => self.process_unary_op(substitutions, op, inner),

            Expr::Binary { lhs, rhs, op } => {
                self.process_binary_op(substitutions, expr, op.extra, lhs, rhs)
            }

            _ => {
                let err = TypeErrorKind::unsupported(expr.extra.ty());
                Err(expr.copy_with_extra(err))
            }
        }
    }

    #[inline]
    fn process_var<'a, T>(
        &self,
        name: &Spanned<'a, T>,
    ) -> Result<ValueType<Lit>, Spanned<'a, TypeErrorKind<Lit>>> {
        self.get_type(name.fragment()).cloned().ok_or_else(|| {
            let e = TypeErrorKind::UndefinedVar((*name.fragment()).to_owned());
            name.copy_with_extra(e)
        })
    }

    fn process_block<'a, T>(
        &mut self,
        substitutions: &mut Substitutions<Lit>,
        block: &Block<'a, T>,
    ) -> Result<ValueType<Lit>, Spanned<'a, TypeErrorKind<Lit>>>
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
    ) -> Result<ValueType<Lit>, Spanned<'a, TypeErrorKind<Lit>>> {
        match &lvalue.extra {
            Lvalue::Variable { ty } => {
                let mut value_type = ty.as_ref().map_or(ValueType::Any, |ty| ty.extra.clone());
                substitutions
                    .assign_new_type(&mut value_type)
                    .map_err(|err| ty.as_ref().unwrap().copy_with_extra(err))?;
                // `unwrap` is safe: an error can only occur with a type annotation present.

                self.insert_type(lvalue.fragment(), value_type.clone());
                Ok(value_type)
            }

            Lvalue::Tuple(destructure) => {
                let element_types = self.process_destructure(substitutions, destructure)?;
                Ok(ValueType::Tuple(element_types))
            }

            _ => {
                let err = TypeErrorKind::unsupported(lvalue.extra.ty());
                Err(lvalue.copy_with_extra(err))
            }
        }
    }

    #[inline]
    fn process_destructure<'a>(
        &mut self,
        substitutions: &mut Substitutions<Lit>,
        destructure: &Destructure<'a, ValueType<Lit>>,
    ) -> Result<Vec<ValueType<Lit>>, Spanned<'a, TypeErrorKind<Lit>>> {
        if let Some(middle) = &destructure.middle {
            // TODO: allow middles with explicitly set type.
            let err = middle.copy_with_extra(TypeErrorKind::UnsupportedDestructure);
            return Err(err);
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
    ) -> Result<ValueType<Lit>, Spanned<'a, TypeErrorKind<Lit>>>
    where
        T: Grammar<Lit = L, Type = ValueType<Lit>>,
    {
        let arg_types: Result<Vec<_>, _> = args
            .map(|arg| self.process_expr_inner(substitutions, arg))
            .collect();
        let arg_types = arg_types?;
        let return_type = substitutions
            .unify_fn_call(fn_type, arg_types)
            .map_err(|e| call_expr.copy_with_extra(e))?;

        Ok(return_type)
    }

    #[inline]
    fn process_unary_op<'a, T>(
        &mut self,
        substitutions: &mut Substitutions<Lit>,
        op: &Spanned<'a, UnaryOp>,
        inner: &SpannedExpr<'a, T>,
    ) -> Result<ValueType<Lit>, Spanned<'a, TypeErrorKind<Lit>>>
    where
        T: Grammar<Lit = L, Type = ValueType<Lit>>,
    {
        let inner_type = self.process_expr_inner(substitutions, inner)?;
        match op.extra {
            UnaryOp::Not => {
                substitutions.unify_spanned_expr(&inner_type, inner, &ValueType::Bool)?;
                Ok(ValueType::Bool)
            }

            UnaryOp::Neg => {
                substitutions
                    .mark_as_linear(&inner_type)
                    .map_err(|e| inner.copy_with_extra(e))?;
                Ok(inner_type)
            }

            _ => Err(op.copy_with_extra(TypeErrorKind::unsupported(op.extra))),
        }
    }

    #[inline]
    fn process_binary_op<'a, T>(
        &mut self,
        substitutions: &mut Substitutions<Lit>,
        binary_expr: &SpannedExpr<'a, T>,
        op: BinaryOp,
        lhs: &SpannedExpr<'a, T>,
        rhs: &SpannedExpr<'a, T>,
    ) -> Result<ValueType<Lit>, Spanned<'a, TypeErrorKind<Lit>>>
    where
        T: Grammar<Lit = L, Type = ValueType<Lit>>,
    {
        let lhs_ty = self.process_expr_inner(substitutions, lhs)?;
        let rhs_ty = self.process_expr_inner(substitutions, rhs)?;

        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Power => {
                substitutions
                    .unify_binary_op(&lhs_ty, &rhs_ty)
                    .map_err(|e| binary_expr.copy_with_extra(e))
            }

            BinaryOp::Eq | BinaryOp::NotEq => {
                substitutions.unify(&lhs_ty, &rhs_ty).map_err(|e| {
                    let e = e.into_op_mismatch(substitutions, &lhs_ty, &rhs_ty, op);
                    binary_expr.copy_with_extra(e)
                })?;
                Ok(ValueType::Bool)
            }

            BinaryOp::And | BinaryOp::Or => {
                substitutions.unify_spanned_expr(&lhs_ty, lhs, &ValueType::Bool)?;
                substitutions.unify_spanned_expr(&rhs_ty, rhs, &ValueType::Bool)?;
                Ok(ValueType::Bool)
            }

            // FIXME: optionally support order comparisons
            _ => Err(binary_expr.copy_with_extra(TypeErrorKind::unsupported(op))),
        }
    }

    fn process_fn_def<'a, T>(
        &mut self,
        substitutions: &mut Substitutions<Lit>,
        def: &FnDefinition<'a, T>,
    ) -> Result<FnType<Lit>, Spanned<'a, TypeErrorKind<Lit>>>
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
    ) -> Result<(Vec<ValueType<Lit>>, ValueType<Lit>), Spanned<'a, TypeErrorKind<Lit>>>
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
    ) -> Result<ValueType<Lit>, Spanned<'a, TypeErrorKind<Lit>>>
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
                    .map_err(|e| statement.copy_with_extra(e))
            }

            _ => {
                let err = TypeErrorKind::unsupported(statement.extra.ty());
                Err(statement.copy_with_extra(err))
            }
        }
    }

    fn process_statements<'a, T>(
        &mut self,
        block: &Block<'a, T>,
    ) -> Result<ValueType<Lit>, Spanned<'a, TypeErrorKind<Lit>>>
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
        result
    }
}
