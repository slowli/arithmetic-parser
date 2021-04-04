//! `TypeEnvironment` and related types.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    iter::{self, FromIterator},
    mem, ops,
};

use crate::{
    arith::{BinaryOpSpans, MapPrimitiveType, NumArithmetic, TypeArithmetic, UnaryOpSpans},
    visit::VisitMut,
    FnType, Num, PrimitiveType, Slice, Substitutions, Tuple, TypeError, TypeErrorKind, TypeResult,
    UnknownLen, ValueType,
};
use arithmetic_parser::{
    grammars::Grammar, BinaryOp, Block, Destructure, DestructureRest, Expr, FnDefinition, Lvalue,
    Spanned, SpannedExpr, SpannedLvalue, SpannedStatement, Statement, UnaryOp,
};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod type_annotation_tests;

type FnArgsAndOutput<Prim> = (Tuple<Prim>, ValueType<Prim>);

/// Environment containing type information on named variables.
///
/// # Concrete and partially specified types
///
/// The environment retains full info on the types even if the type is not
/// [concrete](ValueType::is_concrete()). Consider the following example:
///
/// ```
/// # use arithmetic_parser::grammars::{NumGrammar, Typed, Parse};
/// # use arithmetic_typing::{arith::NumArithmetic, Annotated, Prelude, TypeEnvironment};
/// # fn main() -> anyhow::Result<()> {
/// type Parser = Typed<Annotated<NumGrammar<f32>>>;
/// let code = r#"
///     xs = (1, 2, 3, 4, 5);
///     filtered = xs.filter(|x| x > 1);
///     mapped = filtered.map(|x| x * 2);
///     (filtered, mapped)
/// "#;
///
/// let mut env: TypeEnvironment = Prelude::iter().collect();
/// let output = env.process_with_arithmetic(
///     &NumArithmetic::with_comparisons(),
///     &Parser::parse_statements(code)?,
/// )?;
/// assert_eq!(output.to_string(), "([Num; _], [Num; _])");
///
/// // We have additional information in `env` that both elements
/// // of the tuple have the same length (albeit a dynamic one).
/// assert_eq!(env["filtered"], env["mapped"]);
/// // This means that the following code works.
/// let output = env.process_with_arithmetic(
///     &NumArithmetic::with_comparisons(),
///     &Parser::parse_statements("filtered + mapped")?,
/// )?;
/// assert_eq!(output.to_string(), "[Num; _]");
/// # Ok(())
/// # }
/// ```
///
/// Non-concrete types are tied to an environment. An environment will panic
/// on inserting a non-concrete type via [`Self::insert()`] or other methods.
#[derive(Debug, Clone)]
pub struct TypeEnvironment<Prim: PrimitiveType = Num> {
    variables: HashMap<String, ValueType<Prim>>,
    substitutions: Substitutions<Prim>,
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
    pub fn get(&self, name: &str) -> Option<&ValueType<Prim>> {
        self.variables.get(name)
    }

    /// Iterates over variables contained in this env.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ValueType<Prim>)> + '_ {
        self.variables.iter().map(|(name, ty)| (name.as_str(), ty))
    }

    /// Sets type of a variable.
    ///
    /// # Panics
    ///
    /// - Will panic if `value_type` is not [concrete](ValueType::is_concrete()). Non-concrete
    ///   types are tied to the environment; inserting them into an env is a logical error.
    pub fn insert(&mut self, name: &str, value_type: impl Into<ValueType<Prim>>) -> &mut Self {
        let value_type = value_type.into();
        assert!(
            value_type.is_concrete(),
            "Type {} is not concrete",
            value_type
        );
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
    ) -> Result<ValueType<Prim>, TypeError<'a, Prim>>
    where
        T: Grammar<Type = ValueType<Prim>>,
        NumArithmetic: MapPrimitiveType<T::Lit, Prim = Prim> + TypeArithmetic<Prim>,
    {
        self.process_with_arithmetic(&NumArithmetic::without_comparisons(), block)
    }

    /// Processes statements with a given `arithmetic`. After processing, the context will contain
    /// type info about newly declared vars.
    pub fn process_with_arithmetic<'a, T, A>(
        &mut self,
        arithmetic: &A,
        block: &Block<'a, T>,
    ) -> Result<ValueType<Prim>, TypeError<'a, Prim>>
    where
        T: Grammar<Type = ValueType<Prim>>,
        A: MapPrimitiveType<T::Lit, Prim = Prim> + TypeArithmetic<Prim>,
    {
        TypeProcessor::new(self, arithmetic).process_statements(block)
    }
}

impl<Prim: PrimitiveType> ops::Index<&str> for TypeEnvironment<Prim> {
    type Output = ValueType<Prim>;

    fn index(&self, name: &str) -> &Self::Output {
        self.get(name)
            .unwrap_or_else(|| panic!("Variable `{}` is not defined", name))
    }
}

fn convert_iter<Prim: PrimitiveType, S, Ty, I>(
    iter: I,
) -> impl Iterator<Item = (String, ValueType<Prim>)>
where
    I: IntoIterator<Item = (S, Ty)>,
    S: Into<String>,
    Ty: Into<ValueType<Prim>>,
{
    iter.into_iter().map(|(name, ty)| {
        let ty: ValueType<Prim> = ty.into();
        assert!(ty.is_concrete(), "Type {} is not concrete", ty);
        (name.into(), ty)
    })
}

impl<Prim: PrimitiveType, S, Ty> FromIterator<(S, Ty)> for TypeEnvironment<Prim>
where
    S: Into<String>,
    Ty: Into<ValueType<Prim>>,
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
    Ty: Into<ValueType<Prim>>,
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
struct TypeProcessor<'a, Val, Prim: PrimitiveType> {
    root_scope: &'a mut TypeEnvironment<Prim>,
    unresolved_root_vars: HashSet<String>,
    inner_scopes: Vec<HashMap<String, ValueType<Prim>>>,
    arithmetic: &'a dyn FullArithmetic<Val, Prim>,
    is_in_function: bool,
}

impl<'a, Val, Prim: PrimitiveType> TypeProcessor<'a, Val, Prim> {
    fn new(
        env: &'a mut TypeEnvironment<Prim>,
        arithmetic: &'a dyn FullArithmetic<Val, Prim>,
    ) -> Self {
        Self {
            root_scope: env,
            unresolved_root_vars: HashSet::new(),
            inner_scopes: vec![],
            arithmetic,
            is_in_function: false,
        }
    }
}

impl<Val: fmt::Debug + Clone, Prim: PrimitiveType> TypeProcessor<'_, Val, Prim> {
    fn get_type(&self, name: &str) -> Option<&ValueType<Prim>> {
        self.inner_scopes
            .iter()
            .rev()
            .flat_map(|scope| scope.get(name))
            .next()
            .or_else(|| self.root_scope.get(name))
    }

    fn insert_type(&mut self, name: &str, ty: ValueType<Prim>) {
        let scope = self
            .inner_scopes
            .last_mut()
            .unwrap_or(&mut self.root_scope.variables);
        scope.insert(name.to_owned(), ty);

        if self.inner_scopes.is_empty() {
            self.unresolved_root_vars.insert(name.to_owned());
        }
    }

    fn process_expr_inner<'a, T>(&mut self, expr: &SpannedExpr<'a, T>) -> TypeResult<'a, Prim>
    where
        T: Grammar<Lit = Val, Type = ValueType<Prim>>,
    {
        match &expr.extra {
            Expr::Variable => self.process_var(expr),

            Expr::Literal(lit) => Ok(ValueType::Prim(self.arithmetic.type_of_literal(lit))),

            Expr::Tuple(ref terms) => {
                let elements: Result<Vec<_>, _> = terms
                    .iter()
                    .map(|term| self.process_expr_inner(term))
                    .collect();
                let elements = elements?;
                Ok(ValueType::Tuple(elements.into()))
            }

            Expr::Function { name, args } => {
                let fn_type = self.process_expr_inner(name)?;
                self.process_fn_call(expr, &fn_type, args.iter())
            }

            Expr::Method {
                name,
                receiver,
                args,
            } => {
                let fn_type = self.process_var(name)?;
                let all_args = iter::once(receiver.as_ref()).chain(args);
                self.process_fn_call(expr, &fn_type, all_args)
            }

            Expr::Block(block) => {
                self.inner_scopes.push(HashMap::new());
                let result = self.process_block(block);
                self.inner_scopes.pop(); // intentionally called even on failure
                result
            }

            Expr::FnDefinition(def) => self
                .process_fn_def(def)
                .map(|fn_type| ValueType::Function(Box::new(fn_type))),

            Expr::Unary { op, inner } => self.process_unary_op(expr, *op, inner),

            Expr::Binary { lhs, rhs, op } => self.process_binary_op(expr, *op, lhs, rhs),

            _ => Err(TypeErrorKind::unsupported(expr.extra.ty()).with_span(expr)),
        }
    }

    // TODO: handle `Some` type specially? (Assign new type on each call.)
    #[inline]
    fn process_var<'a, T>(&self, name: &Spanned<'a, T>) -> TypeResult<'a, Prim> {
        let var_name = *name.fragment();
        self.get_type(var_name)
            .cloned()
            .ok_or_else(|| TypeErrorKind::UndefinedVar(var_name.to_owned()).with_span(name))
    }

    fn process_block<'a, T>(&mut self, block: &Block<'a, T>) -> TypeResult<'a, Prim>
    where
        T: Grammar<Lit = Val, Type = ValueType<Prim>>,
    {
        for statement in &block.statements {
            self.process_statement(statement)?;
        }
        block.return_value.as_ref().map_or_else(
            || Ok(ValueType::void()),
            |return_value| self.process_expr_inner(return_value),
        )
    }

    /// Processes an lvalue type by replacing `Any` types with newly created type vars.
    fn process_lvalue<'a>(
        &mut self,
        lvalue: &SpannedLvalue<'a, ValueType<Prim>>,
    ) -> TypeResult<'a, Prim> {
        match &lvalue.extra {
            Lvalue::Variable { ty } => {
                let mut value_type = ty.as_ref().map_or(ValueType::Some, |ty| ty.extra.clone());
                self.root_scope
                    .substitutions
                    .assign_new_type(&mut value_type)
                    .map_err(|err| err.with_span(ty.as_ref().unwrap()))?;
                // `unwrap` is safe: an error can only occur with a type annotation present.

                self.insert_type(lvalue.fragment(), value_type.clone());
                Ok(value_type)
            }

            Lvalue::Tuple(destructure) => {
                let element_types = self.process_destructure(destructure, false)?;
                Ok(ValueType::Tuple(element_types))
            }

            _ => Err(TypeErrorKind::unsupported(lvalue.extra.ty()).with_span(lvalue)),
        }
    }

    #[inline]
    fn process_destructure<'a>(
        &mut self,
        destructure: &Destructure<'a, ValueType<Prim>>,
        is_fn_args: bool,
    ) -> Result<Tuple<Prim>, TypeError<'a, Prim>> {
        let start = destructure
            .start
            .iter()
            .map(|element| self.process_lvalue(element))
            .collect::<Result<Vec<_>, _>>()?;

        let middle = if let Some(middle) = &destructure.middle {
            Some(self.process_destructure_rest(&middle.extra, is_fn_args)?)
        } else {
            None
        };

        let end = destructure
            .end
            .iter()
            .map(|element| self.process_lvalue(element))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Tuple::from_parts(start, middle, end))
    }

    fn process_destructure_rest<'a>(
        &mut self,
        rest: &DestructureRest<'a, ValueType<Prim>>,
        is_fn_args: bool,
    ) -> Result<Slice<Prim>, TypeError<'a, Prim>> {
        let ty = match rest {
            DestructureRest::Unnamed => None,
            DestructureRest::Named { ty, .. } => ty.as_ref(),
        };
        let mut element = ty.map_or(ValueType::Some, |ty| ty.extra.to_owned());

        self.root_scope
            .substitutions
            .assign_new_type(&mut element)
            .map_err(|err| err.with_span(ty.as_ref().unwrap()))?;
        // `unwrap` is safe: an error can only occur with a type annotation present.

        let mut length = if is_fn_args {
            UnknownLen::Dynamic.into()
        } else {
            UnknownLen::Some.into()
        };
        self.root_scope.substitutions.assign_new_len(&mut length);

        if let DestructureRest::Named { variable, .. } = rest {
            self.insert_type(
                variable.fragment(),
                ValueType::slice(element.clone(), length),
            );
        }
        Ok(Slice::new(element, length))
    }

    fn process_fn_call<'it, 'a: 'it, T>(
        &mut self,
        call_expr: &SpannedExpr<'a, T>,
        fn_type: &ValueType<Prim>,
        args: impl Iterator<Item = &'it SpannedExpr<'a, T>>,
    ) -> TypeResult<'a, Prim>
    where
        T: Grammar<Lit = Val, Type = ValueType<Prim>>,
    {
        let arg_types: Result<Vec<_>, _> = args.map(|arg| self.process_expr_inner(arg)).collect();
        let arg_types = arg_types?;
        let return_type = self
            .root_scope
            .substitutions
            .unify_fn_call(fn_type, arg_types)
            .map_err(|err| err.with_span(call_expr))?;

        Ok(return_type)
    }

    #[inline]
    fn process_unary_op<'a, T>(
        &mut self,
        unary_expr: &SpannedExpr<'a, T>,
        op: Spanned<'a, UnaryOp>,
        inner: &SpannedExpr<'a, T>,
    ) -> TypeResult<'a, Prim>
    where
        T: Grammar<Lit = Val, Type = ValueType<Prim>>,
    {
        let inner_ty = self.process_expr_inner(inner)?;
        let spans = UnaryOpSpans {
            total: unary_expr.with_no_extra(),
            op,
            inner: inner.copy_with_extra(inner_ty),
        };
        self.arithmetic
            .process_unary_op(&mut self.root_scope.substitutions, spans)
    }

    #[inline]
    fn process_binary_op<'a, T>(
        &mut self,
        binary_expr: &SpannedExpr<'a, T>,
        op: Spanned<'a, BinaryOp>,
        lhs: &SpannedExpr<'a, T>,
        rhs: &SpannedExpr<'a, T>,
    ) -> TypeResult<'a, Prim>
    where
        T: Grammar<Lit = Val, Type = ValueType<Prim>>,
    {
        let lhs_ty = self.process_expr_inner(lhs)?;
        let rhs_ty = self.process_expr_inner(rhs)?;
        let spans = BinaryOpSpans {
            total: binary_expr.with_no_extra(),
            op,
            lhs: lhs.copy_with_extra(lhs_ty),
            rhs: rhs.copy_with_extra(rhs_ty),
        };
        self.arithmetic
            .process_binary_op(&mut self.root_scope.substitutions, spans)
    }

    fn process_fn_def<'a, T>(
        &mut self,
        def: &FnDefinition<'a, T>,
    ) -> Result<FnType<Prim>, TypeError<'a, Prim>>
    where
        T: Grammar<Lit = Val, Type = ValueType<Prim>>,
    {
        self.inner_scopes.push(HashMap::new());
        let was_in_function = mem::replace(&mut self.is_in_function, true);
        let result = self.process_fn_def_inner(def);
        // Perform basic finalization in any case.
        self.inner_scopes.pop();
        self.is_in_function = was_in_function;

        let (arg_types, return_type) = result?;
        let mut fn_type = FnType::new(arg_types, return_type);
        let substitutions = &self.root_scope.substitutions;
        substitutions.resolver().visit_function_mut(&mut fn_type);

        if !self.is_in_function {
            fn_type.finalize(substitutions);
        }

        Ok(fn_type)
    }

    /// Fallible part of fn definition processing.
    fn process_fn_def_inner<'a, T>(
        &mut self,
        def: &FnDefinition<'a, T>,
    ) -> Result<FnArgsAndOutput<Prim>, TypeError<'a, Prim>>
    where
        T: Grammar<Lit = Val, Type = ValueType<Prim>>,
    {
        let arg_types = self.process_destructure(&def.args.extra, true)?;
        let return_type = self.process_block(&def.body)?;
        Ok((arg_types, return_type))
    }

    fn process_statement<'a, T>(
        &mut self,
        statement: &SpannedStatement<'a, T>,
    ) -> TypeResult<'a, Prim>
    where
        T: Grammar<Lit = Val, Type = ValueType<Prim>>,
    {
        match &statement.extra {
            Statement::Expr(expr) => self.process_expr_inner(expr),

            Statement::Assignment { lhs, rhs } => {
                let rhs_ty = self.process_expr_inner(rhs)?;
                let lhs_ty = self.process_lvalue(lhs)?;
                self.root_scope
                    .substitutions
                    .unify(&lhs_ty, &rhs_ty)
                    .map(|()| ValueType::void())
                    .map_err(|err| err.with_span(statement))
            }

            _ => Err(TypeErrorKind::unsupported(statement.extra.ty()).with_span(statement)),
        }
    }

    fn process_statements<'a, T>(&mut self, block: &Block<'a, T>) -> TypeResult<'a, Prim>
    where
        T: Grammar<Lit = Val, Type = ValueType<Prim>>,
    {
        let result = self.process_block(block);

        // We need to resolve vars even if an error occurred.
        debug_assert!(self.inner_scopes.is_empty());
        let mut resolver = self.root_scope.substitutions.resolver();
        for (name, var_type) in &mut self.root_scope.variables {
            if self.unresolved_root_vars.contains(name) {
                resolver.visit_type_mut(var_type);
            }
        }

        result.map(|mut ty| {
            resolver.visit_type_mut(&mut ty);
            ty
        })
    }
}
