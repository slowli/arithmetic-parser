//! The meat of the entire crate: `TypeProcessor`.

use std::{collections::HashMap, fmt, iter, mem};

use crate::{
    arith::{BinaryOpContext, UnaryOpContext},
    ast::{AstConversionState, SpannedTypeAst, TypeAst},
    env::{FullArithmetic, TypeEnvironment},
    error::{Error, ErrorContext, ErrorKind, Errors, OpErrors, TupleContext},
    visit::VisitMut,
    FnType, PrimitiveType, Slice, Tuple, Type,
};
use arithmetic_parser::{
    grammars::Grammar, BinaryOp, Block, Destructure, DestructureRest, Expr, FnDefinition, Lvalue,
    Spanned, SpannedExpr, SpannedLvalue, SpannedStatement, Statement, UnaryOp,
};

/// Processor for deriving type information.
pub(super) struct TypeProcessor<'a, 'env, Val, Prim: PrimitiveType> {
    env: &'env mut TypeEnvironment<Prim>,
    scopes: Vec<HashMap<String, Type<Prim>>>,
    scope_before_first_error: Option<HashMap<String, Type<Prim>>>,
    arithmetic: &'env dyn FullArithmetic<Val, Prim>,
    is_in_function: bool,
    errors: Errors<'a, Prim>,
}

impl<'env, Val, Prim: PrimitiveType> TypeProcessor<'_, 'env, Val, Prim> {
    pub fn new(
        env: &'env mut TypeEnvironment<Prim>,
        arithmetic: &'env dyn FullArithmetic<Val, Prim>,
    ) -> Self {
        Self {
            env,
            scopes: vec![HashMap::new()],
            scope_before_first_error: None,
            arithmetic,
            is_in_function: false,
            errors: Errors::new(),
        }
    }
}

impl<'a, Val, Prim> TypeProcessor<'a, '_, Val, Prim>
where
    Val: fmt::Debug + Clone,
    Prim: PrimitiveType,
{
    fn get_type(&self, name: &str) -> Option<&Type<Prim>> {
        self.scopes
            .iter()
            .rev()
            .flat_map(|scope| scope.get(name))
            .next()
            .or_else(|| self.env.get(name))
    }

    fn insert_type(&mut self, name: &str, ty: Type<Prim>) {
        let scope = self.scopes.last_mut().unwrap();
        scope.insert(name.to_owned(), ty);
    }

    /// Creates a new type variable.
    fn new_type(&mut self) -> Type<Prim> {
        self.env.substitutions.new_type_var()
    }

    #[allow(clippy::option_if_let_else)] // false positive; `self` is moved into both clauses
    fn process_annotation(&mut self, ty: Option<&SpannedTypeAst<'a>>) -> Type<Prim> {
        if let Some(ty) = ty {
            AstConversionState::new(&mut self.env, &mut self.errors).convert_type(ty)
        } else {
            self.new_type()
        }
    }

    fn process_expr_inner<T>(&mut self, expr: &SpannedExpr<'a, T>) -> Type<Prim>
    where
        T: Grammar<'a, Lit = Val, Type = TypeAst<'a>>,
    {
        match &expr.extra {
            Expr::Variable => self.process_var(expr),

            Expr::Literal(lit) => Type::Prim(self.arithmetic.type_of_literal(lit)),

            Expr::Tuple(ref terms) => {
                let elements: Vec<_> = terms
                    .iter()
                    .map(|term| self.process_expr_inner(term))
                    .collect();
                Type::Tuple(elements.into())
            }

            Expr::Function { name, args } => {
                let fn_type = self.process_expr_inner(name);
                self.process_fn_call(expr, fn_type, args.iter())
            }

            Expr::Method {
                name,
                receiver,
                args,
            } => {
                let fn_type = self.process_var(name);
                let all_args = iter::once(receiver.as_ref()).chain(args);
                self.process_fn_call(expr, fn_type, all_args)
            }

            Expr::Block(block) => {
                self.scopes.push(HashMap::new());
                let result = self.process_block(block);
                self.scopes.pop(); // intentionally called even on failure
                result
            }

            Expr::FnDefinition(def) => self.process_fn_def(def).into(),

            Expr::Cast { value, ty } => {
                let ty = self.process_annotation(Some(ty));
                let original_ty = self.process_expr_inner(value);
                let mut errors = OpErrors::new();
                self.env
                    .substitutions
                    .unify(&ty, &original_ty, errors.by_ref());
                let context = ErrorContext::TypeCast {
                    source: original_ty,
                    target: ty.clone(),
                };
                self.errors.extend(errors.contextualize(expr, context));
                ty
            }

            Expr::Unary { op, inner } => self.process_unary_op(expr, op.extra, inner),

            Expr::Binary { lhs, rhs, op } => self.process_binary_op(expr, op.extra, lhs, rhs),

            _ => {
                self.errors.push(Error::unsupported(expr.extra.ty(), expr));
                // No better choice than to go with `Some` type.
                self.new_type()
            }
        }
    }

    #[inline]
    #[allow(clippy::option_if_let_else)] // false positive
    fn process_var<T>(&mut self, name: &Spanned<'a, T>) -> Type<Prim> {
        let var_name = *name.fragment();

        if let Some(ty) = self.get_type(var_name) {
            ty.clone()
        } else {
            self.errors.push(Error::undefined_var(name));
            // No better choice than to go with `Some` type.
            self.new_type()
        }
    }

    fn process_block<T>(&mut self, block: &Block<'a, T>) -> Type<Prim>
    where
        T: Grammar<'a, Lit = Val, Type = TypeAst<'a>>,
    {
        for statement in &block.statements {
            self.process_statement(statement);
        }
        block
            .return_value
            .as_ref()
            .map_or_else(Type::void, |return_value| {
                self.process_expr_inner(return_value)
            })
    }

    /// Processes an lvalue type by replacing `Some` types with newly created type vars.
    fn process_lvalue(
        &mut self,
        lvalue: &SpannedLvalue<'a, TypeAst<'a>>,
        mut errors: OpErrors<'_, Prim>,
    ) -> Type<Prim> {
        match &lvalue.extra {
            Lvalue::Variable { ty } => {
                let type_instance = self.process_annotation(ty.as_ref());
                self.insert_type(lvalue.fragment(), type_instance.clone());
                type_instance
            }

            Lvalue::Tuple(destructure) => {
                let element_types =
                    self.process_destructure(destructure, TupleContext::Generic, errors);
                Type::Tuple(element_types)
            }

            _ => {
                errors.push(ErrorKind::unsupported(lvalue.extra.ty()));
                // No better choice than to go with `Some` type.
                self.new_type()
            }
        }
    }

    #[inline]
    fn process_destructure(
        &mut self,
        destructure: &Destructure<'a, TypeAst<'a>>,
        context: TupleContext,
        mut errors: OpErrors<'_, Prim>,
    ) -> Tuple<Prim> {
        let start = destructure
            .start
            .iter()
            .enumerate()
            .map(|(i, element)| {
                let loc = context.element(i);
                self.process_lvalue(element, errors.with_location(loc))
            })
            .collect();

        let middle = destructure
            .middle
            .as_ref()
            .map(|middle| self.process_destructure_rest(&middle.extra));

        let end = destructure
            .end
            .iter()
            .enumerate()
            .map(|(i, element)| {
                let loc = context.end_element(i);
                self.process_lvalue(element, errors.with_location(loc))
            })
            .collect();

        Tuple::from_parts(start, middle, end)
    }

    fn process_destructure_rest(&mut self, rest: &DestructureRest<'a, TypeAst<'a>>) -> Slice<Prim> {
        let ty = match rest {
            DestructureRest::Unnamed => None,
            DestructureRest::Named { ty, .. } => ty.as_ref(),
        };
        let element = self.process_annotation(ty);
        let length = self.env.substitutions.new_len_var();

        if let DestructureRest::Named { variable, .. } = rest {
            self.insert_type(variable.fragment(), Type::slice(element.clone(), length));
        }
        Slice::new(element, length)
    }

    fn process_fn_call<'it, T>(
        &mut self,
        call_expr: &SpannedExpr<'a, T>,
        definition: Type<Prim>,
        args: impl Iterator<Item = &'it SpannedExpr<'a, T>>,
    ) -> Type<Prim>
    where
        'a: 'it,
        T: Grammar<'a, Lit = Val, Type = TypeAst<'a>>,
    {
        let arg_types: Vec<_> = args.map(|arg| self.process_expr_inner(arg)).collect();
        let return_type = self.new_type();
        let call_signature = FnType::new(arg_types.into(), return_type.clone()).into();

        let mut errors = OpErrors::new();
        self.env
            .substitutions
            .unify(&call_signature, &definition, errors.by_ref());
        let context = ErrorContext::FnCall {
            definition,
            call_signature,
        };
        self.errors.extend(errors.contextualize(call_expr, context));
        return_type
    }

    #[inline]
    fn process_unary_op<T>(
        &mut self,
        unary_expr: &SpannedExpr<'a, T>,
        op: UnaryOp,
        inner: &SpannedExpr<'a, T>,
    ) -> Type<Prim>
    where
        T: Grammar<'a, Lit = Val, Type = TypeAst<'a>>,
    {
        let inner_ty = self.process_expr_inner(inner);
        let context = UnaryOpContext { op, arg: inner_ty };

        let mut errors = OpErrors::new();
        let output = self.arithmetic.process_unary_op(
            &mut self.env.substitutions,
            &context,
            errors.by_ref(),
        );
        self.errors
            .extend(errors.contextualize(unary_expr, context));
        output
    }

    #[inline]
    fn process_binary_op<T>(
        &mut self,
        binary_expr: &SpannedExpr<'a, T>,
        op: BinaryOp,
        lhs: &SpannedExpr<'a, T>,
        rhs: &SpannedExpr<'a, T>,
    ) -> Type<Prim>
    where
        T: Grammar<'a, Lit = Val, Type = TypeAst<'a>>,
    {
        let lhs_ty = self.process_expr_inner(lhs);
        let rhs_ty = self.process_expr_inner(rhs);
        let context = BinaryOpContext {
            op,
            lhs: lhs_ty,
            rhs: rhs_ty,
        };

        let mut errors = OpErrors::new();
        let output = self.arithmetic.process_binary_op(
            &mut self.env.substitutions,
            &context,
            errors.by_ref(),
        );
        self.errors
            .extend(errors.contextualize(binary_expr, context));
        output
    }

    fn process_fn_def<T>(&mut self, def: &FnDefinition<'a, T>) -> FnType<Prim>
    where
        T: Grammar<'a, Lit = Val, Type = TypeAst<'a>>,
    {
        self.scopes.push(HashMap::new());
        let was_in_function = mem::replace(&mut self.is_in_function, true);

        let mut errors = OpErrors::new();
        let arg_types =
            self.process_destructure(&def.args.extra, TupleContext::FnArgs, errors.by_ref());
        let errors = errors.contextualize_destructure(&def.args, || ErrorContext::FnDefinition {
            args: arg_types.clone(),
        });
        self.errors.extend(errors);

        let return_type = self.process_block(&def.body);
        self.scopes.pop();
        self.is_in_function = was_in_function;

        let mut fn_type = FnType::new(arg_types, return_type);
        let substitutions = &self.env.substitutions;
        substitutions.resolver().visit_function_mut(&mut fn_type);

        if !self.is_in_function {
            fn_type.finalize(substitutions);
        }
        fn_type
    }

    fn process_statement<T>(&mut self, statement: &SpannedStatement<'a, T>) -> Type<Prim>
    where
        T: Grammar<'a, Lit = Val, Type = TypeAst<'a>>,
    {
        // Backup assignments in the root scope if there are no errors yet.
        let backup = if self.scopes.len() == 1 && self.errors.is_empty() {
            self.scopes.first().cloned()
        } else {
            None
        };

        let output = match &statement.extra {
            Statement::Expr(expr) => self.process_expr_inner(expr),

            Statement::Assignment { lhs, rhs } => {
                let rhs_ty = self.process_expr_inner(rhs);

                let mut errors = OpErrors::new();
                let lhs_ty = self.process_lvalue(lhs, errors.by_ref());
                self.env
                    .substitutions
                    .unify(&lhs_ty, &rhs_ty, errors.by_ref());
                let context = ErrorContext::Assignment {
                    lhs: lhs_ty,
                    rhs: rhs_ty,
                };
                self.errors
                    .extend(errors.contextualize_assignment(lhs, &context));
                Type::void()
            }

            _ => {
                self.errors
                    .push(Error::unsupported(statement.extra.ty(), statement));
                // No better choice than to go with `Some` type.
                self.new_type()
            }
        };

        if backup.is_some() && !self.errors.is_empty() {
            self.scope_before_first_error = backup;
        }
        output
    }

    pub fn process_statements<T>(
        mut self,
        block: &Block<'a, T>,
    ) -> Result<Type<Prim>, Errors<'a, Prim>>
    where
        T: Grammar<'a, Lit = Val, Type = TypeAst<'a>>,
    {
        let mut return_value = self.process_block(block);

        let mut scopes = self.scopes;
        debug_assert_eq!(scopes.len(), 1);
        let mut new_root_vars = self
            .scope_before_first_error
            .unwrap_or_else(|| scopes.pop().unwrap());

        let mut resolver = self.env.substitutions.resolver();
        for var_type in new_root_vars.values_mut() {
            resolver.visit_type_mut(var_type);
        }
        resolver.visit_type_mut(&mut return_value);
        self.env.variables.extend(new_root_vars);

        if self.errors.is_empty() {
            Ok(return_value)
        } else {
            self.errors.post_process(&mut resolver);
            Err(self.errors)
        }
    }
}