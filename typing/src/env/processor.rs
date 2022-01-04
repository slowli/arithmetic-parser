//! The meat of the entire crate: `TypeProcessor`.

use std::{collections::HashMap, fmt, iter, mem};

use crate::{
    arith::{BinaryOpContext, UnaryOpContext},
    ast::{AstConversionState, SpannedTypeAst, TypeAst},
    env::{FullArithmetic, TypeEnvironment},
    error::{Error, ErrorContext, ErrorKind, Errors, OpErrors, TupleContext},
    types::IndexError,
    visit::VisitMut,
    Function, Object, PrimitiveType, Slice, Tuple, TupleLen, Type,
};
use arithmetic_parser::{
    grammars::Grammar, is_valid_variable_name, BinaryOp, Block, Destructure, DestructureRest, Expr,
    FnDefinition, Lvalue, MethodCallSeparator, ObjectDestructure, ObjectExpr, Spanned, SpannedExpr,
    SpannedLvalue, SpannedStatement, Statement, UnaryOp,
};

/// Processor for deriving type information.
pub(super) struct TypeProcessor<'a, 'env, Val, Prim: PrimitiveType> {
    env: &'env mut TypeEnvironment<Prim>,
    scopes: Vec<HashMap<String, Type<Prim>>>,
    scope_before_first_error: Option<HashMap<String, Type<Prim>>>,
    arithmetic: &'env dyn FullArithmetic<Val, Prim>,
    /// Are we currently evaluating a function?
    is_in_function: bool,
    /// Variables assigned within the current lvalue (if it is being processed).
    /// Used to determine duplicate vars.
    lvalue_vars: Option<HashMap<&'a str, Spanned<'a>>>,
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
            lvalue_vars: None,
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
            .find_map(|scope| scope.get(name))
            .or_else(|| self.env.get(name))
    }

    fn insert_type(&mut self, name_span: Spanned<'a>, ty: Type<Prim>) {
        let name = *name_span.fragment();
        if name != "_" {
            if let Some(lvalue_vars) = &mut self.lvalue_vars {
                if lvalue_vars.insert(name, name_span).is_some() {
                    self.errors.push(Error::repeated_assignment(name_span));
                }
            }
            let scope = self.scopes.last_mut().unwrap();
            scope.insert(name.to_owned(), ty);
        }
    }

    /// Creates a new type variable.
    fn new_type(&mut self) -> Type<Prim> {
        self.env.substitutions.new_type_var()
    }

    fn process_annotation(&mut self, ty: Option<&SpannedTypeAst<'a>>) -> Type<Prim> {
        if let Some(ty) = ty {
            AstConversionState::new(self.env, &mut self.errors).convert_type(ty)
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

            Expr::FieldAccess { name, receiver } => {
                let receiver = self.process_expr_inner(receiver);
                self.process_field_access(expr, &receiver, name)
            }

            Expr::Method {
                name,
                receiver,
                args,
                separator,
            } => match separator.extra {
                MethodCallSeparator::Dot => {
                    let fn_type = self.process_var(name);
                    let all_args = iter::once(receiver.as_ref()).chain(args);
                    self.process_fn_call(expr, fn_type, all_args)
                }
                MethodCallSeparator::Colon2 => {
                    let receiver = self.process_expr_inner(receiver);
                    let fn_field = self.process_field_access(expr, &receiver, name);
                    self.process_fn_call(expr, fn_field, args.iter())
                }
            },

            Expr::Block(block) => {
                self.scopes.push(HashMap::new());
                let result = self.process_block(block);
                self.scopes.pop(); // intentionally called even on failure
                result
            }

            Expr::Object(object) => self.process_object(object).into(),

            Expr::FnDefinition(def) => self.process_fn_def(def).into(),

            Expr::TypeCast { value, ty } => {
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
        let top_level = self.scopes.len() == 1;
        for (i, statement) in block.statements.iter().enumerate() {
            // Backup assignments in the root scope if there are no errors yet.
            let backup = if top_level && self.errors.is_empty() {
                self.scopes.first().cloned()
            } else {
                None
            };

            self.process_statement(statement);

            if backup.is_some() && !self.errors.is_empty() {
                self.errors.set_first_failing_statement(i);
                self.scope_before_first_error = backup;
            }
        }

        block
            .return_value
            .as_ref()
            .map_or_else(Type::void, |return_value| {
                let no_errors = self.errors.is_empty();
                let return_type = self.process_expr_inner(return_value);
                if top_level && no_errors && !self.errors.is_empty() {
                    self.errors
                        .set_first_failing_statement(block.statements.len());
                }
                return_type
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
                self.insert_type(lvalue.with_no_extra(), type_instance.clone());
                type_instance
            }

            Lvalue::Tuple(destructure) => {
                let element_types =
                    self.process_destructure(destructure, TupleContext::Generic, errors);
                Type::Tuple(element_types)
            }

            Lvalue::Object(destructure) => self.process_object_destructure(destructure, errors),

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
            self.insert_type(*variable, Type::slice(element.clone(), length));
        }
        Slice::new(element, length)
    }

    fn process_object_destructure(
        &mut self,
        destructure: &ObjectDestructure<'a, TypeAst<'a>>,
        mut errors: OpErrors<'_, Prim>,
    ) -> Type<Prim> {
        let mut object_fields = HashMap::new();
        for field in &destructure.fields {
            let field_str = *field.field_name.fragment();
            if object_fields.insert(field_str, field.field_name).is_some() {
                // We push directly into `self.errors` to preserve the error span.
                self.errors.push(Error::repeated_field(field.field_name));
            }
        }

        let fields = destructure.fields.iter().map(|field| {
            let field_name = *field.field_name.fragment();

            // We still process lvalues even if they correspond to duplicate fields.
            let field_type = if let Some(binding) = &field.binding {
                self.process_lvalue(binding, errors.with_location(field_name))
            } else {
                let new_type = self.new_type();
                if object_fields[field_name] == field.field_name {
                    // Skip inserting a field if we know it's a duplicate; this will just lead to
                    // an additional error.
                    self.insert_type(field.field_name, new_type.clone());
                }
                new_type
            };
            (field_name.to_owned(), field_type)
        });
        let object = Object::from_map(fields.collect());

        let object_ty = self.new_type();
        object.apply_as_constraint(&object_ty, &mut self.env.substitutions, errors.by_ref());
        object_ty
    }

    fn process_field_access<T>(
        &mut self,
        access_expr: &SpannedExpr<'a, T>,
        receiver: &Type<Prim>,
        field_name: &Spanned<'a>,
    ) -> Type<Prim>
    where
        T: Grammar<'a, Lit = Val, Type = TypeAst<'a>>,
    {
        let field_str = *field_name.fragment();
        if let Ok(index) = field_str.parse::<usize>() {
            self.process_indexing(access_expr, receiver, index)
        } else if is_valid_variable_name(field_str) {
            self.process_object_access(access_expr, receiver, field_str)
        } else {
            self.errors.push(Error::invalid_field_name(*field_name));
            self.new_type()
        }
    }

    fn process_indexing<T>(
        &mut self,
        access_expr: &SpannedExpr<'a, T>,
        receiver: &Type<Prim>,
        index: usize,
    ) -> Type<Prim>
    where
        T: Grammar<'a, Lit = Val, Type = TypeAst<'a>>,
    {
        let receiver = self.env.substitutions.fast_resolve(receiver);
        match receiver {
            Type::Tuple(tuple) => {
                let middle_len = tuple.parts().1.map_or(TupleLen::ZERO, Slice::len);
                let middle_len = self.env.substitutions.resolve_len(middle_len);

                match tuple.get_element(index, middle_len) {
                    Ok(ty) => return ty.clone(),
                    Err(IndexError::OutOfBounds) => {
                        self.errors.push(Error::index_out_of_bounds(
                            tuple.clone(),
                            access_expr,
                            index,
                        ));
                        return self.new_type();
                    }
                    Err(IndexError::NoInfo) => { /* An error will be added below. */ }
                }
            }
            Type::Function(_) | Type::Prim(_) | Type::Dyn(_) => {
                self.errors
                    .push(Error::cannot_index(receiver.clone(), access_expr));
                return self.new_type();
            }
            Type::Any => {
                return self.new_type();
            }
            Type::Var(var) => {
                if let Some(object) = self.env.substitutions.object_constraint(*var) {
                    self.errors
                        .push(Error::cannot_index(object.into(), access_expr));
                    return self.new_type();
                }
            }
            _ => { /* An error will be added below. */ }
        }

        self.errors
            .push(Error::unsupported_index(receiver.clone(), access_expr));
        self.new_type()
    }

    fn process_object<T>(&mut self, object: &ObjectExpr<'a, T>) -> Object<Prim>
    where
        T: Grammar<'a, Lit = Val, Type = TypeAst<'a>>,
    {
        // Check that all field names are unique.
        let mut object_fields = HashMap::new();
        for (name, _) in &object.fields {
            let field_str = *name.fragment();
            if object_fields.insert(field_str, *name).is_some() {
                self.errors.push(Error::repeated_field(*name));
            }
        }

        let fields = object.fields.iter().map(|(name, field_expr)| {
            let name_string = (*name.fragment()).to_owned();
            if let Some(field_expr) = field_expr {
                (name_string, self.process_expr_inner(field_expr))
            } else {
                (name_string, self.process_var(name))
            }
        });
        Object::from_map(fields.collect())
    }

    fn process_object_access<T>(
        &mut self,
        access_expr: &SpannedExpr<'a, T>,
        receiver: &Type<Prim>,
        field_name: &str,
    ) -> Type<Prim>
    where
        T: Grammar<'a, Lit = Val, Type = TypeAst<'a>>,
    {
        let mut errors = OpErrors::new();
        let return_type = self.new_type();
        Object::just(field_name, return_type.clone()).apply_as_constraint(
            receiver,
            &mut self.env.substitutions,
            errors.by_ref(),
        );

        let context = ErrorContext::ObjectFieldAccess {
            ty: receiver.clone(),
        };
        self.errors
            .extend(errors.contextualize(access_expr, context));
        return_type
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
        let call_signature = Function::new(arg_types.into(), return_type.clone()).into();

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

    fn process_fn_def<T>(&mut self, def: &FnDefinition<'a, T>) -> Function<Prim>
    where
        T: Grammar<'a, Lit = Val, Type = TypeAst<'a>>,
    {
        self.scopes.push(HashMap::new());
        let was_in_function = mem::replace(&mut self.is_in_function, true);

        self.lvalue_vars = Some(HashMap::new());
        let mut errors = OpErrors::new();
        let arg_types =
            self.process_destructure(&def.args.extra, TupleContext::FnArgs, errors.by_ref());
        let errors = errors.contextualize_destructure(&def.args, || ErrorContext::FnDefinition {
            args: arg_types.clone(),
        });
        self.errors.extend(errors);
        self.lvalue_vars.take();

        let return_type = self.process_block(&def.body);
        self.scopes.pop();
        self.is_in_function = was_in_function;

        let mut fn_type = Function::new(arg_types, return_type);
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
        match &statement.extra {
            Statement::Expr(expr) => self.process_expr_inner(expr),

            Statement::Assignment { lhs, rhs } => {
                let rhs_ty = self.process_expr_inner(rhs);

                self.lvalue_vars = Some(HashMap::new());
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
                self.lvalue_vars.take();

                Type::void()
            }

            _ => {
                self.errors
                    .push(Error::unsupported(statement.extra.ty(), statement));
                // No better choice than to go with `Some` type.
                self.new_type()
            }
        }
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
