//! `TypeContext` and related types.

use std::{collections::HashMap, fmt, iter, mem};

use crate::{substitutions::Substitutions, FnType, TypeError, ValueType};
use arithmetic_parser::{
    grammars::Grammar, BinaryOp, Block, Expr, FnDefinition, Lvalue, Spanned, SpannedExpr,
    SpannedLvalue, SpannedStatement, Statement, UnaryOp,
};

/// Analogue of `Scope` for type information.
#[derive(Debug, Default)]
struct TypeScope {
    variables: HashMap<String, ValueType>,
}

/// Context for deriving type information.
pub struct TypeContext {
    scopes: Vec<TypeScope>,
    is_in_function: bool,
}

impl fmt::Debug for TypeContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TypeContext")
            .field("scopes", &self.scopes)
            .finish()
    }
}

impl Default for TypeContext {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeContext {
    /// Creates a type context based on the interpreter context.
    pub fn new() -> Self {
        TypeContext {
            scopes: vec![TypeScope::default()],
            is_in_function: false,
        }
    }

    /// Gets type of the specified variable.
    pub fn get_type(&self, name: &str) -> Option<ValueType> {
        self.scopes
            .iter()
            .rev()
            .flat_map(|scope| scope.variables.get(name).cloned())
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
            Expr::Variable => self.get_type(expr.fragment()).ok_or_else(|| {
                let e = TypeError::UndefinedVar((*expr.fragment()).to_owned());
                expr.copy_with_extra(e)
            }),

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
                let fn_type = self.get_type(name.fragment()).ok_or_else(|| {
                    let e = TypeError::UndefinedVar((*expr.fragment()).to_owned());
                    expr.copy_with_extra(e)
                })?;
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
            Lvalue::Variable { ref ty } => {
                let mut value_type = if let Some(ty) = ty {
                    // `ty` may contain `Any` elements, so we need to replace them with type vars.
                    ty.extra.clone()
                } else {
                    ValueType::Any
                };
                substitutions.assign_new_type(&mut value_type);

                self.scopes
                    .last_mut()
                    .unwrap()
                    .variables
                    .insert((*lvalue.fragment()).to_string(), value_type.clone());
                Ok(value_type)
            }

            // FIXME: deal with middle
            Lvalue::Tuple(destructure) => {
                let element_types: Result<Vec<_>, _> = destructure
                    .start
                    .iter()
                    .chain(&destructure.end)
                    .map(|fragment| self.process_lvalue(substitutions, fragment))
                    .collect();
                Ok(ValueType::Tuple(element_types?))
            }

            _ => {
                let err = TypeError::unsupported(lvalue.extra.ty());
                Err(lvalue.copy_with_extra(err))
            }
        }
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
                // This only handles T op T == T.
                // TODO: We should handle num op T == T and T op num == T as well!
                substitutions.unify(&lhs_ty, &rhs_ty).map_err(|e| {
                    let e = e.into_op_mismatch(substitutions, &lhs_ty, &rhs_ty, op);
                    binary_expr.copy_with_extra(e)
                })?;
                substitutions
                    .mark_as_linear(&lhs_ty)
                    .map_err(|e| binary_expr.copy_with_extra(e))?;
                Ok(rhs_ty)
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

        let args = &def.args.extra;
        // FIXME: deal with middle
        let arg_types: Result<Vec<_>, _> = args
            .start
            .iter()
            .chain(&args.end)
            .map(|arg| self.process_lvalue(substitutions, arg))
            .collect();
        let arg_types = arg_types?;

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
    /// about newly declared vars / functions.
    pub fn process_statements<'a, T>(
        &mut self,
        statements: &[SpannedStatement<'a, T>],
    ) -> Result<(), Spanned<'a, TypeError>>
    where
        T: Grammar<Type = ValueType>,
    {
        let mut substitutions = Substitutions::default();
        for statement in statements {
            self.process_statement(&mut substitutions, statement)?;
        }

        let scope = self.scopes.last_mut().unwrap();
        for var_type in scope.variables.values_mut() {
            *var_type = substitutions.resolve(var_type);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FnArgs;
    use std::collections::BTreeMap;

    use arithmetic_parser::{
        grammars::{Grammar, NumLiteral, Parse, ParseLiteral, Typed},
        InputSpan, NomResult,
    };
    use assert_matches::assert_matches;

    fn hash_fn_type() -> FnType {
        FnType {
            args: FnArgs::Any,
            return_type: ValueType::Number,
            type_params: BTreeMap::new(),
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct NumGrammar;

    impl ParseLiteral for NumGrammar {
        type Lit = f32;

        fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
            f32::parse(input)
        }
    }

    impl Grammar for NumGrammar {
        type Type = ValueType;

        fn parse_type(_input: InputSpan<'_>) -> NomResult<'_, Self::Type> {
            unimplemented!()
        }
    }

    #[test]
    fn statements_with_a_block() {
        let code = "y = { x = 3; 2 * x }; x ^ y == 6 * x;";
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();

        let mut type_context = TypeContext::new();
        type_context.insert_type("x", ValueType::Number);
        type_context.process_statements(&block.statements).unwrap();
        assert_eq!(type_context.get_type("y").unwrap(), ValueType::Number);
    }

    #[test]
    fn boolean_statements() {
        let code = "y = x == x ^ 2; y = y || { x = 3; x != 7 };";
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();

        let mut type_context = TypeContext::new();
        type_context.insert_type("x", ValueType::Number);
        type_context.process_statements(&block.statements).unwrap();
        assert_eq!(type_context.get_type("y").unwrap(), ValueType::Bool);
    }

    #[test]
    fn function_definition() {
        // FIXME: revert to single `hash()`
        let code = r#"
            sign = |x, msg| {
                (r, R) = (hash(), hash()) * (1, 3);
                c = hash(R, msg);
                (R, r + c * x)
            };
        "#;

        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();

        let hash_type = hash_fn_type();
        type_context.insert_type("hash", ValueType::Function(Box::new(hash_type)));

        type_context.process_statements(&block.statements).unwrap();
        assert_eq!(
            type_context.get_type("sign").unwrap().to_string(),
            "fn<T: ?Lin>(Num, T) -> (Num, Num)"
        );
    }

    #[test]
    fn non_linear_types_in_function() {
        let code = r#"
            compare = |x, y| x == y;
            compare_hash = |x, z| x == 2 ^ hash(z);
            add_hashes = |x, y| hash(x, y) + hash(y, x);
        "#;

        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();

        let hash_type = hash_fn_type();
        type_context.insert_type("hash", ValueType::Function(Box::new(hash_type)));

        type_context.process_statements(&block.statements).unwrap();
        assert_eq!(
            type_context.get_type("compare").unwrap().to_string(),
            "fn<T: ?Lin>(T, T) -> Bool"
        );
        assert_eq!(
            type_context.get_type("compare_hash").unwrap().to_string(),
            "fn<T: ?Lin>(Num, T) -> Bool"
        );
        assert_eq!(
            type_context.get_type("add_hashes").unwrap().to_string(),
            "fn<T: ?Lin, U: ?Lin>(T, U) -> Num"
        );
    }

    #[test]
    fn type_recursion() {
        let code = "bog = |x| x + (x, 2);";
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        let err = type_context
            .process_statements(&block.statements)
            .unwrap_err();
        assert_eq!(*err.fragment(), "x + (x, 2)");
        assert_matches!(err.extra, TypeError::RecursiveType(ref ty) if ty.to_string() == "(T, Num)");
    }

    #[test]
    fn indirect_type_recursion() {
        let code = r#"
            add = |x, y| x + y; // this function is fine
            bog = |x| add(x, (1, x)); // ...but its application is not
        "#;

        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        let err = type_context
            .process_statements(&block.statements)
            .unwrap_err();
        assert_matches!(
            err.extra,
            TypeError::RecursiveType(ref ty) if ty.to_string() == "(Num, T)"
        );
    }

    #[test]
    fn recursion_via_fn() {
        let code = "func = |bog| bog(1, bog);";
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        let err = type_context
            .process_statements(&block.statements)
            .unwrap_err();
        assert_matches!(
            err.extra,
            TypeError::RecursiveType(ref ty) if ty.to_string() == "fn(Num, T) -> _"
        );
    }

    #[test]
    fn inferring_value_type_from_embedded_function() {
        let code = "double = |x| { (x, || (x, 2 * x)) };";
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        type_context.process_statements(&block.statements).unwrap();
        assert_eq!(
            type_context.get_type("double").unwrap().to_string(),
            "fn(Num) -> (Num, fn() -> (Num, Num))"
        );
    }

    #[test]
    fn free_and_bound_type_vars() {
        let code = r#"
            concat = |x| { |y| (x, y) };
            x = concat(2)(5);
            partial = concat(3);
            y = (partial(2), partial((1, 1)));
        "#;
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        type_context.process_statements(&block.statements).unwrap();

        assert_eq!(
            type_context.get_type("concat").unwrap().to_string(),
            "fn<T: ?Lin>(T) -> fn<U: ?Lin>(U) -> (T, U)"
        );
        assert_eq!(
            type_context.get_type("x").unwrap().to_string(),
            "(Num, Num)"
        );
        assert_eq!(
            type_context.get_type("partial").unwrap().to_string(),
            "fn<U: ?Lin>(U) -> (Num, U)"
        );
        assert_eq!(
            type_context.get_type("y").unwrap().to_string(),
            "((Num, Num), (Num, (Num, Num)))"
        );
    }

    #[test]
    fn attributing_type_vars_to_correct_fn() {
        let code = "double = |x| { (x, || (x, x)) };";
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        type_context.process_statements(&block.statements).unwrap();
        assert_eq!(
            type_context.get_type("double").unwrap().to_string(),
            "fn<T: ?Lin>(T) -> (T, fn() -> (T, T))"
        );
    }

    #[test]
    fn defining_and_calling_embedded_function() {
        let code = r#"
            call_double = |x| {
                double = |x| (x, x);
                double(x) == (1, 3)
            };
        "#;
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        type_context.process_statements(&block.statements).unwrap();
        assert_eq!(
            type_context.get_type("call_double").unwrap().to_string(),
            "fn(Num) -> Bool"
        );
    }

    #[test]
    fn incorrect_function_arity() {
        let code = "double = |x| (x, x); (z,) = double(5);";
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        let err = type_context
            .process_statements(&block.statements)
            .unwrap_err();
        assert_matches!(err.extra, TypeError::TupleLenMismatch(2, 1));
    }

    #[test]
    fn function_as_arg() {
        let code = "mapper = |(x, y), map| (map(x), map(y));";
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        type_context.process_statements(&block.statements).unwrap();
        assert_eq!(
            type_context.get_type("mapper").unwrap().to_string(),
            "fn<T: ?Lin, U: ?Lin>((T, T), fn(T) -> U) -> (U, U)"
        );
    }

    #[test]
    fn function_as_arg_with_more_constraints() {
        let code = "mapper = |(x, y), map| map(x) + map(y);";
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        type_context.process_statements(&block.statements).unwrap();
        assert_eq!(
            type_context.get_type("mapper").unwrap().to_string(),
            "fn<T: ?Lin, U>((T, T), fn(T) -> U) -> U"
        );
    }

    #[test]
    fn function_as_arg_with_even_more_constraints() {
        let code = "mapper = |(x, y), map| map(x * map(y));";
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        type_context.process_statements(&block.statements).unwrap();
        assert_eq!(
            type_context.get_type("mapper").unwrap().to_string(),
            "fn<T>((T, T), fn(T) -> T) -> T"
        );
    }

    #[test]
    fn function_arg_with_multiple_args() {
        let code = "test_fn = |x, fun| fun(x, x * 3);";
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        type_context.process_statements(&block.statements).unwrap();
        assert_eq!(
            type_context.get_type("test_fn").unwrap().to_string(),
            "fn<T: ?Lin>(Num, fn(Num, Num) -> T) -> T"
        );
    }

    #[test]
    fn function_as_arg_within_tuple() {
        let code = r#"
            test_fn = |struct, y| {
                (fn, x) = struct;
                fn(x / 3) * y
            };
        "#;
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        type_context.process_statements(&block.statements).unwrap();

        assert_eq!(
            type_context.get_type("test_fn").unwrap().to_string(),
            "fn<T>((fn(Num) -> T, Num), T) -> T"
        );
    }

    #[test]
    fn function_instantiations_are_independent() {
        let code = r#"
            identity = |x| x;
            x = (identity(5), identity(1 == 2));
        "#;
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        type_context.process_statements(&block.statements).unwrap();

        assert_eq!(
            type_context.get_type("x").unwrap().to_string(),
            "(Num, Bool)"
        );
    }

    #[test]
    fn function_passed_as_arg() {
        let code = r#"
            mapper = |(x, y), map| (map(x), map(y));
            tuple = mapper((1, 2), |x| x + 3);
            create_fn = |x| { || x };
            tuple_of_fns = mapper((1, 2), create_fn);
        "#;
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        type_context.process_statements(&block.statements).unwrap();

        assert_eq!(
            type_context.get_type("tuple").unwrap().to_string(),
            "(Num, Num)"
        );
        assert_eq!(
            type_context.get_type("tuple_of_fns").unwrap().to_string(),
            "(fn() -> Num, fn() -> Num)"
        );
    }

    #[test]
    fn curried_function_passed_as_arg() {
        let code = r#"
            mapper = |(x, y), map| (map(x), map(y));
            concat = |x| { |y| (x, y) };
            r = mapper((1, 2), concat(1 == 1));
        "#;
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        type_context.process_statements(&block.statements).unwrap();

        assert_eq!(
            type_context.get_type("r").unwrap().to_string(),
            "((Bool, Num), (Bool, Num))"
        );
    }

    #[test]
    fn parametric_fn_passed_as_arg_with_different_constraints() {
        let code = r#"
            concat = |x| { |y| (x, y) };
            partial = concat(3); // fn<U>(U) -> (Num, U)

            first = |fun| fun(5);
            r = first(partial); // (Num, Num)
            second = |fun, b| fun(b) == (3, b);
            second(partial, 1 == 1);
        "#;
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        type_context.process_statements(&block.statements).unwrap();

        assert_eq!(
            type_context.get_type("r").unwrap().to_string(),
            "(Num, Num)"
        );
    }

    #[test]
    fn parametric_fn_passed_as_arg_with_unsatisfiable_requirements() {
        let code = r#"
            concat = |x| { |y| (x, y) };
            partial = concat(3); // fn<U>(U) -> (Num, U)

            bogus = |fun| fun(1) == 4;
            bogus(partial);
        "#;

        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        let err = type_context
            .process_statements(&block.statements)
            .unwrap_err();

        assert_matches!(err.extra, TypeError::IncompatibleTypes(_, _));
    }

    #[test]
    fn parametric_fn_passed_as_arg_with_recursive_requirements() {
        let code = r#"
            concat = |x| { |y| (x, y) };
            partial = concat(3); // fn<U>(U) -> (Num, U)
            bogus = |fun| { |x| fun(x) == x };
            bogus(partial);
        "#;

        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        let err = type_context
            .process_statements(&block.statements)
            .unwrap_err();

        assert_matches!(err.extra, TypeError::RecursiveType(_));
    }

    #[test]
    fn type_param_is_placed_correctly_with_fn_arg() {
        let code = "foo = |fun| { |x| fun(x) == x };";
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();

        type_context.process_statements(&block.statements).unwrap();
        assert_eq!(
            type_context.get_type("foo").unwrap().to_string(),
            "fn<T: ?Lin>(fn(T) -> T) -> fn(T) -> Bool"
        );
    }

    #[test]
    fn type_params_in_fn_with_multiple_fn_args() {
        let code = "test = |x, foo, bar| foo(x) == bar(x * x);";
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();

        type_context.process_statements(&block.statements).unwrap();
        assert_eq!(
            type_context.get_type("test").unwrap().to_string(),
            "fn<T, U: ?Lin>(T, fn(T) -> U, fn(T) -> U) -> Bool"
        );
    }

    #[test]
    fn function_passed_as_arg_invalid_arity() {
        let code = r#"
            mapper = |(x, y), map| (map(x), map(y));
            mapper((1, 2), |x, y| x + y);
        "#;
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        let err = type_context
            .process_statements(&block.statements)
            .unwrap_err();

        assert_matches!(
            err.extra,
            TypeError::ArgLenMismatch {
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn function_passed_as_arg_invalid_arg_type() {
        let code = r#"
            mapper = |(x, y), map| (map(x), map(y));
            mapper((1, 2), |(x, _)| x);
        "#;
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        let err = type_context
            .process_statements(&block.statements)
            .unwrap_err();

        assert_matches!(err.extra, TypeError::IncompatibleTypes(..));
    }

    #[test]
    fn function_passed_as_arg_invalid_input() {
        let code = r#"
            mapper = |(x, y), map| (map(x), map(y));
            mapper((1, 2 != 3), |x| x + 2);
        "#;
        let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
        let mut type_context = TypeContext::new();
        let err = type_context
            .process_statements(&block.statements)
            .unwrap_err();

        assert_matches!(err.extra, TypeError::IncompatibleTypes(..));
    }
}
