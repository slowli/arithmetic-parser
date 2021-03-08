//! `TypeEnvironment` and related types.

use std::{collections::HashMap, iter, mem, ops};

use crate::{substitutions::Substitutions, FnType, TypeError, ValueType};
use arithmetic_parser::{
    grammars::Grammar, BinaryOp, Block, Destructure, Expr, FnDefinition, Lvalue, Spanned,
    SpannedExpr, SpannedLvalue, SpannedStatement, Statement, UnaryOp,
};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod type_hint_tests;

/// Analogue of `Scope` for type information.
#[derive(Debug, Default)]
struct TypeScope {
    variables: HashMap<String, ValueType>,
}

/// Environment for deriving type information.
#[derive(Debug)]
pub struct TypeEnvironment {
    scopes: Vec<TypeScope>,
    is_in_function: bool,
}

impl Default for TypeEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeEnvironment {
    /// Creates a type context based on the interpreter context.
    pub fn new() -> Self {
        Self {
            scopes: vec![TypeScope::default()],
            is_in_function: false,
        }
    }

    /// Gets type of the specified variable.
    pub fn get_type(&self, name: &str) -> Option<&ValueType> {
        self.scopes
            .iter()
            .rev()
            .flat_map(|scope| scope.variables.get(name))
            .next()
    }

    /// Sets type of a variable.
    pub fn insert_type(&mut self, name: &str, value_type: ValueType) {
        self.scopes
            .last_mut()
            .unwrap()
            .variables
            .insert(name.to_owned(), value_type);
    }

    fn process_expr_inner<'a, T>(
        &mut self,
        substitutions: &mut Substitutions,
        expr: &SpannedExpr<'a, T>,
    ) -> Result<ValueType, Spanned<'a, TypeError>>
    where
        T: Grammar<Type = ValueType>,
    {
        match &expr.extra {
            Expr::Variable => self.process_var(expr),

            Expr::Literal(_) => Ok(ValueType::Number),

            Expr::Tuple(ref terms) => {
                let term_types: Result<Vec<_>, _> = terms
                    .iter()
                    .map(|term| self.process_expr_inner(substitutions, term))
                    .collect();
                term_types.map(ValueType::Tuple)
            }

            Expr::Function { name, args } => {
                let fn_type = self.process_expr_inner(substitutions, name)?;
                self.process_fn_call(substitutions, expr, fn_type, args.iter())
            }

            Expr::Method {
                name,
                receiver,
                args,
            } => {
                let fn_type = self.process_var(name)?;
                let all_args = iter::once(receiver.as_ref()).chain(args);
                self.process_fn_call(substitutions, expr, fn_type, all_args)
            }

            Expr::Block(Block {
                statements,
                return_value,
                ..
            }) => {
                self.scopes.push(TypeScope::default());
                for statement in statements {
                    self.process_statement(substitutions, statement)?;
                }
                let return_type = if let Some(return_value) = return_value {
                    self.process_expr_inner(substitutions, return_value)?
                } else {
                    ValueType::void()
                };

                // TODO: do we need to pop a scope on error?
                self.scopes.pop();
                Ok(return_type)
            }

            Expr::FnDefinition(def) => self
                .process_fn_def(substitutions, def)
                .map(|fn_type| ValueType::Function(Box::new(fn_type))),

            Expr::Unary { op, inner } => self.process_unary_op(substitutions, op, inner),

            Expr::Binary { lhs, rhs, op } => {
                self.process_binary_op(substitutions, expr, op.extra, lhs, rhs)
            }

            _ => {
                let err = TypeError::unsupported(expr.extra.ty());
                Err(expr.copy_with_extra(err))
            }
        }
    }

    #[inline]
    fn process_var<'a, T>(
        &self,
        name: &Spanned<'a, T>,
    ) -> Result<ValueType, Spanned<'a, TypeError>> {
        self.get_type(name.fragment()).cloned().ok_or_else(|| {
            let e = TypeError::UndefinedVar((*name.fragment()).to_owned());
            name.copy_with_extra(e)
        })
    }

    /// Processes an isolated expression.
    pub fn process_expr<'a, T>(
        &mut self,
        expr: &SpannedExpr<'a, T>,
    ) -> Result<ValueType, Spanned<'a, TypeError>>
    where
        T: Grammar<Type = ValueType>,
    {
        let mut substitutions = Substitutions::default();
        self.process_expr_inner(&mut substitutions, expr)
    }

    /// Processes an lvalue type by replacing `Any` types with newly created type vars.
    fn process_lvalue<'a>(
        &mut self,
        substitutions: &mut Substitutions,
        lvalue: &SpannedLvalue<'a, ValueType>,
    ) -> Result<ValueType, Spanned<'a, TypeError>> {
        match &lvalue.extra {
            Lvalue::Variable { ty } => {
                let mut value_type = if let Some(ty) = ty {
                    // `ty` may contain `Any` elements, so we need to replace them with type vars.
                    ty.extra.clone()
                } else {
                    ValueType::Any
                };
                substitutions
                    .assign_new_type(&mut value_type)
                    .map_err(|err| ty.as_ref().unwrap().copy_with_extra(err))?;
                // `unwrap` is safe: an error can only occur with a type hint present.

                self.scopes
                    .last_mut()
                    .unwrap()
                    .variables
                    .insert((*lvalue.fragment()).to_string(), value_type.clone());
                Ok(value_type)
            }

            Lvalue::Tuple(destructure) => {
                let element_types = self.process_destructure(substitutions, destructure)?;
                Ok(ValueType::Tuple(element_types))
            }

            _ => {
                let err = TypeError::unsupported(lvalue.extra.ty());
                Err(lvalue.copy_with_extra(err))
            }
        }
    }

    #[inline]
    fn process_destructure<'a>(
        &mut self,
        substitutions: &mut Substitutions,
        destructure: &Destructure<'a, ValueType>,
    ) -> Result<Vec<ValueType>, Spanned<'a, TypeError>> {
        if let Some(middle) = &destructure.middle {
            // TODO: allow middles with explicitly set type.
            let err = middle.copy_with_extra(TypeError::UnsupportedDestructure);
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
        substitutions: &mut Substitutions,
        call_expr: &SpannedExpr<'a, T>,
        fn_type: ValueType,
        args: impl Iterator<Item = &'it SpannedExpr<'a, T>>,
    ) -> Result<ValueType, Spanned<'a, TypeError>>
    where
        T: Grammar<Type = ValueType>,
    {
        let arg_types: Result<Vec<_>, _> = args
            .map(|arg| self.process_expr_inner(substitutions, arg))
            .collect();
        let arg_types = arg_types?;
        let return_type = substitutions
            .unify_fn_call(&fn_type, arg_types)
            .map_err(|e| call_expr.copy_with_extra(e))?;

        Ok(return_type)
    }

    #[inline]
    fn process_unary_op<'a, T>(
        &mut self,
        substitutions: &mut Substitutions,
        op: &Spanned<'a, UnaryOp>,
        inner: &SpannedExpr<'a, T>,
    ) -> Result<ValueType, Spanned<'a, TypeError>>
    where
        T: Grammar<Type = ValueType>,
    {
        let inner_type = self.process_expr_inner(substitutions, inner)?;
        match op.extra {
            UnaryOp::Not => {
                substitutions.unify_spanned_expr(&inner_type, inner, ValueType::Bool)?;
                Ok(ValueType::Bool)
            }

            UnaryOp::Neg => {
                substitutions
                    .mark_as_linear(&inner_type)
                    .map_err(|e| inner.copy_with_extra(e))?;
                Ok(inner_type)
            }

            _ => Err(op.copy_with_extra(TypeError::unsupported(op.extra))),
        }
    }

    #[inline]
    fn process_binary_op<'a, T>(
        &mut self,
        substitutions: &mut Substitutions,
        binary_expr: &SpannedExpr<'a, T>,
        op: BinaryOp,
        lhs: &SpannedExpr<'a, T>,
        rhs: &SpannedExpr<'a, T>,
    ) -> Result<ValueType, Spanned<'a, TypeError>>
    where
        T: Grammar<Type = ValueType>,
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
                substitutions.unify_spanned_expr(&lhs_ty, lhs, ValueType::Bool)?;
                substitutions.unify_spanned_expr(&rhs_ty, rhs, ValueType::Bool)?;
                Ok(ValueType::Bool)
            }

            // FIXME: optionally support order comparisons
            _ => Err(binary_expr.copy_with_extra(TypeError::unsupported(op))),
        }
    }

    fn process_fn_def<'a, T>(
        &mut self,
        substitutions: &mut Substitutions,
        def: &FnDefinition<'a, T>,
    ) -> Result<FnType, Spanned<'a, TypeError>>
    where
        T: Grammar<Type = ValueType>,
    {
        self.scopes.push(TypeScope::default());

        let arg_types = self.process_destructure(substitutions, &def.args.extra)?;

        let was_in_function = mem::replace(&mut self.is_in_function, true);
        for statement in &def.body.statements {
            self.process_statement(substitutions, statement)?;
        }
        let return_type = if let Some(ref return_value) = def.body.return_value {
            self.process_expr_inner(substitutions, return_value)?
        } else {
            ValueType::void()
        };
        // TODO: do we need to pop a scope on error?
        self.scopes.pop();
        self.is_in_function = was_in_function;

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

    fn process_statement<'a, T>(
        &mut self,
        substitutions: &mut Substitutions,
        statement: &SpannedStatement<'a, T>,
    ) -> Result<ValueType, Spanned<'a, TypeError>>
    where
        T: Grammar<Type = ValueType>,
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
                let err = TypeError::unsupported(statement.extra.ty());
                Err(statement.copy_with_extra(err))
            }
        }
    }

    /// Processes statements. After processing, the context will contain type info
    /// about newly declared vars.
    pub fn process_statements<'a, T>(
        &mut self,
        statements: &[SpannedStatement<'a, T>],
    ) -> Result<(), Spanned<'a, TypeError>>
    where
        T: Grammar<Type = ValueType>,
    {
        let mut substitutions = Substitutions::default();

        let result = statements.iter().try_for_each(|statement| {
            self.process_statement(&mut substitutions, statement)
                .map(drop)
        });

        // We need to resolve vars even if an error occurred.
        let scope = self.scopes.last_mut().unwrap();
        for var_type in scope.variables.values_mut() {
            *var_type = substitutions.resolve(var_type);
        }
        result
    }
}

impl ops::Index<&str> for TypeEnvironment {
    type Output = ValueType;

    fn index(&self, name: &str) -> &Self::Output {
        self.get_type(name)
            .unwrap_or_else(|| panic!("Variable `{}` is not defined", name))
    }
}
