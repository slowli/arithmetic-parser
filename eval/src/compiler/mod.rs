//! Transformation of AST output by the parser into non-recursive format.

use crate::{
    alloc::{Arc, HashMap, String, ToOwned},
    exec::{Atom, Command, CompiledExpr, Executable, ExecutableModule, FieldName, ModuleId},
    Error, ErrorKind,
};
use arithmetic_parser::{
    grammars::Grammar, BinaryOp, Block, Destructure, FnDefinition, InputSpan, Lvalue,
    ObjectDestructure, Spanned, SpannedLvalue, UnaryOp,
};

mod captures;
mod expr;

pub(crate) use self::captures::Captures;
use self::captures::{CapturesExtractor, CompilerExtTarget};

#[derive(Debug)]
pub(crate) struct Compiler {
    /// Mapping between registers and named variables.
    vars_to_registers: HashMap<String, usize>,
    scope_depth: usize,
    register_count: usize,
    module_id: Arc<dyn ModuleId>,
}

impl Compiler {
    fn new(module_id: Arc<dyn ModuleId>) -> Self {
        Self {
            vars_to_registers: HashMap::new(),
            scope_depth: 0,
            register_count: 0,
            module_id,
        }
    }

    fn from_env(module_id: Arc<dyn ModuleId>, env: &Captures) -> Self {
        Self {
            vars_to_registers: env.variables_map().clone(),
            register_count: env.len(),
            scope_depth: 0,
            module_id,
        }
    }

    /// Backups this instance. This effectively clones all fields.
    fn backup(&mut self) -> Self {
        Self {
            vars_to_registers: self.vars_to_registers.clone(),
            scope_depth: self.scope_depth,
            register_count: self.register_count,
            module_id: self.module_id.clone(),
        }
    }

    fn create_error<T>(&self, span: &Spanned<'_, T>, err: ErrorKind) -> Error {
        Error::new(self.module_id.clone(), span, err)
    }

    fn check_unary_op(&self, op: &Spanned<'_, UnaryOp>) -> Result<UnaryOp, Error> {
        match op.extra {
            UnaryOp::Neg | UnaryOp::Not => Ok(op.extra),
            _ => Err(self.create_error(op, ErrorKind::unsupported(op.extra))),
        }
    }

    fn check_binary_op(&self, op: &Spanned<'_, BinaryOp>) -> Result<BinaryOp, Error> {
        match op.extra {
            BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::Div
            | BinaryOp::Power
            | BinaryOp::And
            | BinaryOp::Or
            | BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Gt
            | BinaryOp::Ge
            | BinaryOp::Lt
            | BinaryOp::Le => Ok(op.extra),

            _ => Err(self.create_error(op, ErrorKind::unsupported(op.extra))),
        }
    }

    fn get_var(&self, name: &str) -> usize {
        *self
            .vars_to_registers
            .get(name)
            .expect("Captures must created during module compilation")
    }

    fn push_assignment<T, U>(
        &mut self,
        executable: &mut Executable<T>,
        rhs: CompiledExpr<T>,
        rhs_span: &Spanned<'_, U>,
    ) -> usize {
        let register = self.register_count;
        let command = Command::Push(rhs);
        executable.push_command(rhs_span.copy_with_extra(command));
        self.register_count += 1;
        register
    }

    pub fn compile_module<'a, Id: ModuleId, T: Grammar<'a>>(
        module_id: Id,
        block: &Block<'a, T>,
    ) -> Result<ExecutableModule<T::Lit>, Error> {
        let module_id = Arc::new(module_id) as Arc<dyn ModuleId>;
        let captures = Self::extract_captures(module_id.clone(), block)?;
        let mut compiler = Self::from_env(module_id.clone(), &captures);

        let mut executable = Executable::new(module_id);
        let empty_span = InputSpan::new("");
        let last_atom = compiler
            .compile_block_inner(&mut executable, block)?
            .map_or(Atom::Void, |spanned| spanned.extra);
        // Push the last variable to a register to be popped during execution.
        compiler.push_assignment(
            &mut executable,
            CompiledExpr::Atom(last_atom),
            &empty_span.into(),
        );

        executable.finalize_block(compiler.register_count);
        Ok(ExecutableModule::from_parts(executable, captures))
    }

    fn extract_captures<'a, T: Grammar<'a>>(
        module_id: Arc<dyn ModuleId>,
        block: &Block<'a, T>,
    ) -> Result<Captures, Error> {
        let mut extractor = CapturesExtractor::new(module_id);
        extractor.eval_block(block)?;
        Ok(extractor.into_captures())
    }

    fn assign<T, Ty>(
        &mut self,
        executable: &mut Executable<T>,
        lhs: &SpannedLvalue<'_, Ty>,
        rhs_register: usize,
    ) -> Result<(), Error> {
        match &lhs.extra {
            Lvalue::Variable { .. } => {
                self.insert_var(executable, lhs.with_no_extra(), rhs_register);
            }

            Lvalue::Tuple(destructure) => {
                let span = lhs.with_no_extra();
                self.destructure(executable, destructure, span, rhs_register)?;
            }

            Lvalue::Object(destructure) => {
                let span = lhs.with_no_extra();
                self.destructure_object(executable, destructure, span, rhs_register)?;
            }

            _ => {
                let err = ErrorKind::unsupported(lhs.extra.ty());
                return Err(self.create_error(lhs, err));
            }
        }

        Ok(())
    }

    fn insert_var<T>(
        &mut self,
        executable: &mut Executable<T>,
        var_span: Spanned<'_>,
        register: usize,
    ) {
        let var_name = *var_span.fragment();
        if var_name != "_" {
            self.vars_to_registers.insert(var_name.to_owned(), register);

            // It does not make sense to annotate vars in the inner scopes, since
            // they cannot be accessed externally.
            if self.scope_depth == 0 {
                let command = Command::Annotate {
                    register,
                    name: var_name.to_owned(),
                };
                executable.push_command(var_span.copy_with_extra(command));
            }
        }
    }

    fn destructure<'a, T, Ty>(
        &mut self,
        executable: &mut Executable<T>,
        destructure: &Destructure<'a, Ty>,
        span: Spanned<'a>,
        rhs_register: usize,
    ) -> Result<(), Error> {
        let command = Command::Destructure {
            source: rhs_register,
            start_len: destructure.start.len(),
            end_len: destructure.end.len(),
            lvalue_len: destructure.len(),
            unchecked: false,
        };
        executable.push_command(span.copy_with_extra(command));
        let start_register = self.register_count;
        self.register_count += destructure.start.len() + destructure.end.len() + 1;

        for (i, lvalue) in (start_register..).zip(&destructure.start) {
            self.assign(executable, lvalue, i)?;
        }

        let start_register = start_register + destructure.start.len();
        if let Some(middle) = &destructure.middle {
            if let Some(lvalue) = middle.extra.to_lvalue() {
                self.assign(executable, &lvalue, start_register)?;
            }
        }

        let start_register = start_register + 1;
        for (i, lvalue) in (start_register..).zip(&destructure.end) {
            self.assign(executable, lvalue, i)?;
        }

        Ok(())
    }

    fn destructure_object<'a, T, Ty>(
        &mut self,
        executable: &mut Executable<T>,
        destructure: &ObjectDestructure<'a, Ty>,
        span: Spanned<'a>,
        rhs_register: usize,
    ) -> Result<(), Error> {
        for field in &destructure.fields {
            let field_name = FieldName::Name((*field.field_name.fragment()).to_owned());
            let field_access = CompiledExpr::FieldAccess {
                receiver: span.copy_with_extra(Atom::Register(rhs_register)).into(),
                field: field_name,
            };
            let register = self.push_assignment(executable, field_access, &field.field_name);
            if let Some(binding) = &field.binding {
                self.assign(executable, binding, register)?;
            } else {
                self.insert_var(executable, field.field_name, register);
            }
        }
        Ok(())
    }
}

/// Compiler extensions defined for some AST nodes, most notably, `Block`.
///
/// # Examples
///
/// ```
/// use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// use arithmetic_eval::exec::CompilerExt;
/// # use std::{collections::HashSet, iter::FromIterator};
///
/// # fn main() -> anyhow::Result<()> {
/// let block = "x = sin(0.5) / PI; y = x * E; (x, y)";
/// let block = Untyped::<F32Grammar>::parse_statements(block)?;
/// let undefined_vars = block.undefined_variables()?;
/// assert_eq!(
///     undefined_vars.keys().copied().collect::<HashSet<_>>(),
///     HashSet::from_iter(vec!["sin", "PI", "E"])
/// );
/// assert_eq!(undefined_vars["PI"].location_offset(), 15);
/// # Ok(())
/// # }
/// ```
pub trait CompilerExt<'a> {
    /// Returns variables not defined within the AST node, together with the span of their first
    /// occurrence.
    ///
    /// # Errors
    ///
    /// - Returns an error if the AST is intrinsically malformed. This may be the case if it
    ///   contains destructuring with the same variable on left-hand side,
    ///   such as `(x, x) = ...`.
    ///
    /// The fact that an error is *not* returned does not guarantee that the AST node will evaluate
    /// successfully if all variables are assigned.
    fn undefined_variables(&self) -> Result<HashMap<&'a str, Spanned<'a>>, Error>;
}

impl<'a, T: Grammar<'a>> CompilerExt<'a> for Block<'a, T> {
    fn undefined_variables(&self) -> Result<HashMap<&'a str, Spanned<'a>>, Error> {
        CompilerExtTarget::Block(self).get_undefined_variables()
    }
}

impl<'a, T: Grammar<'a>> CompilerExt<'a> for FnDefinition<'a, T> {
    fn undefined_variables(&self) -> Result<HashMap<&'a str, Spanned<'a>>, Error> {
        CompilerExtTarget::FnDefinition(self).get_undefined_variables()
    }
}

#[cfg(test)]
mod tests {
    use arithmetic_parser::{
        grammars::{F32Grammar, Parse, ParseLiteral, Typed, Untyped},
        Expr, Location, NomResult,
    };

    use super::*;
    use crate::{exec::WildcardId, Environment, Value};

    #[test]
    fn compilation_basics() {
        let block = "x = 3; 1 + { y = 2; y * x } == 7";
        let block = Untyped::<F32Grammar>::parse_statements(block).unwrap();
        let module = Compiler::compile_module(WildcardId, &block).unwrap();
        let value = module.with_env(&Environment::new()).unwrap().run().unwrap();
        assert_eq!(value, Value::Bool(true));
    }

    #[test]
    fn compiled_function() {
        let block = "add = |x, y| x + y; add(2, 3) == 5";
        let block = Untyped::<F32Grammar>::parse_statements(block).unwrap();
        let module = Compiler::compile_module(WildcardId, &block).unwrap();
        let value = module.with_env(&Environment::new()).unwrap().run().unwrap();
        assert_eq!(value, Value::Bool(true));
    }

    #[test]
    fn compiled_function_with_capture() {
        let block = "A = 2; add = |x, y| x + y / A; add(2, 3) == 3.5";
        let block = Untyped::<F32Grammar>::parse_statements(block).unwrap();
        let module = Compiler::compile_module(WildcardId, &block).unwrap();
        let value = module.with_env(&Environment::new()).unwrap().run().unwrap();
        assert_eq!(value, Value::Bool(true));
    }

    #[test]
    fn variable_extraction() {
        let def = "|a, b| ({ x = a * b + y; x - 2 }, a / b)";
        let def = Untyped::<F32Grammar>::parse_statements(def)
            .unwrap()
            .return_value
            .unwrap();
        let Expr::FnDefinition(def) = def.extra else {
            panic!("Unexpected function parsing result: {def:?}");
        };

        let captures = def.undefined_variables().unwrap();
        assert_eq!(captures["y"].location_offset(), 22);
        assert!(!captures.contains_key("x"));
    }

    #[test]
    fn variable_extraction_with_scoping() {
        let def = "|a, b| ({ x = a * b + y; x - 2 }, a / x)";
        let def = Untyped::<F32Grammar>::parse_statements(def)
            .unwrap()
            .return_value
            .unwrap();
        let Expr::FnDefinition(def) = def.extra else {
            panic!("Unexpected function parsing result: {def:?}");
        };

        let captures = def.undefined_variables().unwrap();
        assert_eq!(captures["y"].location_offset(), 22);
        assert_eq!(captures["x"].location_offset(), 38);
    }

    #[test]
    fn extracting_captures() {
        let program = "y = 5 * x; y - 3 + x";
        let module = Untyped::<F32Grammar>::parse_statements(program).unwrap();
        let captures = Compiler::extract_captures(Arc::new(WildcardId), &module).unwrap();

        let captures: Vec<_> = captures.iter().collect();
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0], ("x", &Location::from_str(program, 8..9)));
    }

    #[test]
    fn extracting_captures_with_inner_fns() {
        let program = r#"
            y = 5 * x;          // x is a capture
            fun = |z| {         // z is not a capture
                z * x + y * PI  // y is not a capture for the entire module, PI is
            };
        "#;
        let module = Untyped::<F32Grammar>::parse_statements(program).unwrap();

        let captures = Compiler::extract_captures(Arc::new(WildcardId), &module).unwrap();
        assert_eq!(captures.len(), 2);

        assert!(captures.contains("PI"));
        let x_location = captures.location("x").unwrap();
        assert_eq!(x_location.location_line(), 2); // should be the first mention
    }

    #[test]
    fn type_casts_are_ignored() -> anyhow::Result<()> {
        struct TypedGrammar;

        impl ParseLiteral for TypedGrammar {
            type Lit = f32;

            fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
                F32Grammar::parse_literal(input)
            }
        }

        impl Grammar<'_> for TypedGrammar {
            type Type = ();

            fn parse_type(input: InputSpan<'_>) -> NomResult<'_, Self::Type> {
                use nom::{bytes::complete::tag, combinator::map};
                map(tag("Num"), drop)(input)
            }
        }

        let block = "x = 3 as Num; 1 + { y = 2; y * x as Num } == 7";
        let block = Typed::<TypedGrammar>::parse_statements(block)?;
        let module = Compiler::compile_module(WildcardId, &block)?;
        let value = module.with_env(&Environment::new())?.run()?;
        assert_eq!(value, Value::Bool(true));
        Ok(())
    }
}
