//! `TypeEnvironment` and related types.

use std::{
    collections::HashMap,
    fmt,
    iter::{self, FromIterator},
    mem, ops,
};

use crate::ast::SpannedTypeAst;
use crate::{
    arith::{BinaryOpContext, MapPrimitiveType, NumArithmetic, TypeArithmetic, UnaryOpContext},
    ast::{AstConversionState, TypeAst},
    error::{ErrorContext, ErrorLocation, OpTypeErrors, TypeError, TypeErrorKind, TypeErrors},
    types::{ParamConstraints, ParamQuantifier},
    visit::VisitMut,
    FnType, Num, PrimitiveType, Slice, Substitutions, Tuple, Type,
};
use arithmetic_parser::{
    grammars::Grammar, BinaryOp, Block, Destructure, DestructureRest, Expr, FnDefinition, Lvalue,
    Spanned, SpannedExpr, SpannedLvalue, SpannedStatement, Statement, UnaryOp,
};

#[cfg(test)]
mod tests;

/// Environment containing type information on named variables.
///
/// # Examples
///
/// See [the crate docs](index.html#examples) for examples of usage.
///
/// # Concrete and partially specified types
///
/// The environment retains full info on the types even if the type is not
/// [concrete](Type::is_concrete()). Non-concrete types are tied to an environment.
/// An environment will panic on inserting a non-concrete type via [`Self::insert()`]
/// or other methods.
///
/// ```
/// # use arithmetic_parser::grammars::{NumGrammar, Parse, Typed};
/// # use arithmetic_typing::{Annotated, Prelude, TypeEnvironment};
/// # type Parser = Typed<Annotated<NumGrammar<f32>>>;
/// # fn main() -> anyhow::Result<()> {
/// // An easy way to get a non-concrete type is to involve `any`.
/// let code = r#"
///     lin: any Lin = (1, 2, 3);
///     (x, ...) = lin;
/// "#;
/// let code = Parser::parse_statements(code)?;
///
/// let mut env: TypeEnvironment = Prelude::iter().collect();
/// env.process_statements(&code)?;
/// assert!(!env["x"].is_concrete());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct TypeEnvironment<Prim: PrimitiveType = Num> {
    pub(crate) substitutions: Substitutions<Prim>,
    variables: HashMap<String, Type<Prim>>,
}

impl<Prim: PrimitiveType> Default for TypeEnvironment<Prim> {
    fn default() -> Self {
        Self {
            variables: HashMap::new(),
            substitutions: Substitutions::default(),
        }
    }
}

impl<Prim: PrimitiveType> TypeEnvironment<Prim> {
    /// Creates an empty environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Gets type of the specified variable.
    pub fn get(&self, name: &str) -> Option<&Type<Prim>> {
        self.variables.get(name)
    }

    /// Iterates over variables contained in this env.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Type<Prim>)> + '_ {
        self.variables.iter().map(|(name, ty)| (name.as_str(), ty))
    }

    fn prepare_type(ty: impl Into<Type<Prim>>) -> Type<Prim> {
        let mut ty = ty.into();
        assert!(ty.is_concrete(), "Type {} is not concrete", ty);

        if let Type::Function(function) = &mut ty {
            if function.params.is_none() {
                ParamQuantifier::set_params(function, ParamConstraints::default());
            }
        }
        ty
    }

    /// Sets type of a variable.
    ///
    /// # Panics
    ///
    /// - Will panic if `ty` is not [concrete](Type::is_concrete()). Non-concrete
    ///   types are tied to the environment; inserting them into an env is a logical error.
    pub fn insert(&mut self, name: &str, ty: impl Into<Type<Prim>>) -> &mut Self {
        self.variables
            .insert(name.to_owned(), Self::prepare_type(ty));
        self
    }

    /// Processes statements with the default type arithmetic. After processing, the environment
    /// will contain type info about newly declared vars.
    ///
    /// This method is a shortcut for calling `process_with_arithmetic` with
    /// [`NumArithmetic::without_comparisons()`].
    pub fn process_statements<'a, T>(
        &mut self,
        block: &Block<'a, T>,
    ) -> Result<Type<Prim>, TypeErrors<'a, Prim>>
    where
        T: Grammar<'a, Type = TypeAst<'a>>,
        NumArithmetic: MapPrimitiveType<T::Lit, Prim = Prim> + TypeArithmetic<Prim>,
    {
        self.process_with_arithmetic(&NumArithmetic::without_comparisons(), block)
    }

    /// Processes statements with a given `arithmetic`. After processing, the environment
    /// will contain type info about newly declared vars.
    ///
    /// # Errors
    ///
    /// Even if there are any type errors, all statements in the `block` will be executed
    /// to completion and all errors will be reported. However, the environment will **not**
    /// include any vars beyond the first failing statement.
    pub fn process_with_arithmetic<'a, T, A>(
        &mut self,
        arithmetic: &A,
        block: &Block<'a, T>,
    ) -> Result<Type<Prim>, TypeErrors<'a, Prim>>
    where
        T: Grammar<'a, Type = TypeAst<'a>>,
        A: MapPrimitiveType<T::Lit, Prim = Prim> + TypeArithmetic<Prim>,
    {
        TypeProcessor::new(self, arithmetic).process_statements(block)
    }
}

impl<Prim: PrimitiveType> ops::Index<&str> for TypeEnvironment<Prim> {
    type Output = Type<Prim>;

    fn index(&self, name: &str) -> &Self::Output {
        self.get(name)
            .unwrap_or_else(|| panic!("Variable `{}` is not defined", name))
    }
}

fn convert_iter<Prim: PrimitiveType, S, Ty, I>(
    iter: I,
) -> impl Iterator<Item = (String, Type<Prim>)>
where
    I: IntoIterator<Item = (S, Ty)>,
    S: Into<String>,
    Ty: Into<Type<Prim>>,
{
    iter.into_iter()
        .map(|(name, ty)| (name.into(), TypeEnvironment::prepare_type(ty)))
}

impl<Prim: PrimitiveType, S, Ty> FromIterator<(S, Ty)> for TypeEnvironment<Prim>
where
    S: Into<String>,
    Ty: Into<Type<Prim>>,
{
    fn from_iter<I: IntoIterator<Item = (S, Ty)>>(iter: I) -> Self {
        Self {
            variables: convert_iter(iter).collect(),
            substitutions: Substitutions::default(),
        }
    }
}

impl<Prim: PrimitiveType, S, Ty> Extend<(S, Ty)> for TypeEnvironment<Prim>
where
    S: Into<String>,
    Ty: Into<Type<Prim>>,
{
    fn extend<I: IntoIterator<Item = (S, Ty)>>(&mut self, iter: I) {
        self.variables.extend(convert_iter(iter))
    }
}

// Helper trait to wrap type mapper and arithmetic.
trait FullArithmetic<Val, Prim: PrimitiveType>:
    MapPrimitiveType<Val, Prim = Prim> + TypeArithmetic<Prim>
{
}

impl<Val, Prim: PrimitiveType, T> FullArithmetic<Val, Prim> for T where
    T: MapPrimitiveType<Val, Prim = Prim> + TypeArithmetic<Prim>
{
}

/// Processor for deriving type information.
struct TypeProcessor<'a, 'env, Val, Prim: PrimitiveType> {
    env: &'env mut TypeEnvironment<Prim>,
    scopes: Vec<HashMap<String, Type<Prim>>>,
    scope_before_first_error: Option<HashMap<String, Type<Prim>>>,
    arithmetic: &'env dyn FullArithmetic<Val, Prim>,
    is_in_function: bool,
    errors: TypeErrors<'a, Prim>,
}

impl<'env, Val, Prim: PrimitiveType> TypeProcessor<'_, 'env, Val, Prim> {
    fn new(
        env: &'env mut TypeEnvironment<Prim>,
        arithmetic: &'env dyn FullArithmetic<Val, Prim>,
    ) -> Self {
        Self {
            env,
            scopes: vec![HashMap::new()],
            scope_before_first_error: None,
            arithmetic,
            is_in_function: false,
            errors: TypeErrors::new(),
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

            Expr::Unary { op, inner } => self.process_unary_op(expr, op.extra, inner),

            Expr::Binary { lhs, rhs, op } => self.process_binary_op(expr, op.extra, lhs, rhs),

            _ => {
                self.errors
                    .push(TypeError::unsupported(expr.extra.ty(), expr));
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
            self.errors.push(TypeError::undefined_var(name));
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
        mut errors: OpTypeErrors<'_, Prim>,
    ) -> Type<Prim> {
        match &lvalue.extra {
            Lvalue::Variable { ty } => {
                let type_instance = self.process_annotation(ty.as_ref());
                self.insert_type(lvalue.fragment(), type_instance.clone());
                type_instance
            }

            Lvalue::Tuple(destructure) => {
                let element_types = self.process_destructure(destructure, errors);
                Type::Tuple(element_types)
            }

            _ => {
                errors.push(TypeErrorKind::unsupported(lvalue.extra.ty()));
                // No better choice than to go with `Some` type.
                self.new_type()
            }
        }
    }

    #[inline]
    fn process_destructure(
        &mut self,
        destructure: &Destructure<'a, TypeAst<'a>>,
        mut errors: OpTypeErrors<'_, Prim>,
    ) -> Tuple<Prim> {
        let start = destructure
            .start
            .iter()
            .enumerate()
            .map(|(i, element)| {
                let loc = ErrorLocation::TupleElement(i);
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
                let loc = ErrorLocation::TupleEnd(i);
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

        let mut errors = OpTypeErrors::new();
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

        let mut errors = OpTypeErrors::new();
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

        let mut errors = OpTypeErrors::new();
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

        let mut errors = OpTypeErrors::new();
        let arg_types = self.process_destructure(&def.args.extra, errors.by_ref());
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

                let mut errors = OpTypeErrors::new();
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
                    .push(TypeError::unsupported(statement.extra.ty(), statement));
                // No better choice than to go with `Some` type.
                self.new_type()
            }
        };

        if backup.is_some() && !self.errors.is_empty() {
            self.scope_before_first_error = backup;
        }
        output
    }

    fn process_statements<T>(
        mut self,
        block: &Block<'a, T>,
    ) -> Result<Type<Prim>, TypeErrors<'a, Prim>>
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
