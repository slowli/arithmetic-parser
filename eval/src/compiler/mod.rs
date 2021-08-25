//! Transformation of AST output by the parser into non-recursive format.

use hashbrown::HashMap;

use crate::{
    alloc::{Box, String, ToOwned},
    executable::{Atom, Command, CompiledExpr, Executable, ExecutableModule, FieldName, Registers},
    Error, ErrorKind, ModuleId, Value,
};
use arithmetic_parser::{
    grammars::Grammar, BinaryOp, Block, Destructure, FnDefinition, InputSpan, Lvalue,
    ObjectDestructure, Spanned, SpannedLvalue, UnaryOp,
};

mod captures;
mod expr;

use self::captures::{CapturesExtractor, CompilerExtTarget};

pub(crate) type ImportSpans<'a> = HashMap<String, Spanned<'a>>;

#[derive(Debug)]
pub(crate) struct Compiler {
    /// Mapping between registers and named variables.
    vars_to_registers: HashMap<String, usize>,
    scope_depth: usize,
    register_count: usize,
    module_id: Box<dyn ModuleId>,
}

impl Compiler {
    fn new(module_id: Box<dyn ModuleId>) -> Self {
        Self {
            vars_to_registers: HashMap::new(),
            scope_depth: 0,
            register_count: 0,
            module_id,
        }
    }

    fn from_env<T>(module_id: Box<dyn ModuleId>, env: &Registers<'_, T>) -> Self {
        Self {
            vars_to_registers: env.variables_map().clone(),
            register_count: env.register_count(),
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
            module_id: self.module_id.clone_boxed(),
        }
    }

    fn create_error<'a, T>(&self, span: &Spanned<'a, T>, err: ErrorKind) -> Error<'a> {
        Error::new(self.module_id.as_ref(), span, err)
    }

    fn check_unary_op<'a>(&self, op: &Spanned<'a, UnaryOp>) -> Result<UnaryOp, Error<'a>> {
        match op.extra {
            UnaryOp::Neg | UnaryOp::Not => Ok(op.extra),
            _ => Err(self.create_error(op, ErrorKind::unsupported(op.extra))),
        }
    }

    fn check_binary_op<'a>(&self, op: &Spanned<'a, BinaryOp>) -> Result<BinaryOp, Error<'a>> {
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

    fn push_assignment<'a, T, U>(
        &mut self,
        executable: &mut Executable<'a, T>,
        rhs: CompiledExpr<'a, T>,
        rhs_span: &Spanned<'a, U>,
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
    ) -> Result<(ExecutableModule<'a, T::Lit>, ImportSpans<'a>), Error<'a>> {
        let module_id = Box::new(module_id) as Box<dyn ModuleId>;
        let (captures, import_spans) = Self::extract_captures(module_id.clone_boxed(), block)?;
        let mut compiler = Self::from_env(module_id.clone_boxed(), &captures);

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
        let module = ExecutableModule::from_parts(executable, captures);
        Ok((module, import_spans))
    }

    fn extract_captures<'a, T: Grammar<'a>>(
        module_id: Box<dyn ModuleId>,
        block: &Block<'a, T>,
    ) -> Result<(Registers<'a, T::Lit>, ImportSpans<'a>), Error<'a>> {
        let mut extractor = CapturesExtractor::new(module_id);
        extractor.eval_block(block)?;

        let mut captures = Registers::new();
        for &var_name in extractor.captures.keys() {
            captures.insert_var(var_name, Value::void());
        }

        let import_spans = extractor
            .captures
            .into_iter()
            .map(|(var_name, var_span)| (var_name.to_owned(), var_span))
            .collect();

        Ok((captures, import_spans))
    }

    fn assign<'a, T, Ty>(
        &mut self,
        executable: &mut Executable<'a, T>,
        lhs: &SpannedLvalue<'a, Ty>,
        rhs_register: usize,
    ) -> Result<(), Error<'a>> {
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

    fn insert_var<'a, T>(
        &mut self,
        executable: &mut Executable<'a, T>,
        var_span: Spanned<'a>,
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
        executable: &mut Executable<'a, T>,
        destructure: &Destructure<'a, Ty>,
        span: Spanned<'a>,
        rhs_register: usize,
    ) -> Result<(), Error<'a>> {
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
        executable: &mut Executable<'a, T>,
        destructure: &ObjectDestructure<'a, Ty>,
        span: Spanned<'a>,
        rhs_register: usize,
    ) -> Result<(), Error<'a>> {
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
/// use arithmetic_eval::CompilerExt;
/// # use hashbrown::HashSet;
/// # use core::iter::FromIterator;
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
    fn undefined_variables(&self) -> Result<HashMap<&'a str, Spanned<'a>>, Error<'a>>;
}

impl<'a, T: Grammar<'a>> CompilerExt<'a> for Block<'a, T> {
    fn undefined_variables(&self) -> Result<HashMap<&'a str, Spanned<'a>>, Error<'a>> {
        CompilerExtTarget::Block(self).get_undefined_variables()
    }
}

impl<'a, T: Grammar<'a>> CompilerExt<'a> for FnDefinition<'a, T> {
    fn undefined_variables(&self) -> Result<HashMap<&'a str, Spanned<'a>>, Error<'a>> {
        CompilerExtTarget::FnDefinition(self).get_undefined_variables()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Value, WildcardId};

    use arithmetic_parser::grammars::{F32Grammar, Parse, ParseLiteral, Typed, Untyped};
    use arithmetic_parser::{Expr, NomResult};

    #[test]
    fn compilation_basics() {
        let block = "x = 3; 1 + { y = 2; y * x } == 7";
        let block = Untyped::<F32Grammar>::parse_statements(block).unwrap();
        let (module, _) = Compiler::compile_module(WildcardId, &block).unwrap();
        let value = module.run().unwrap();
        assert_eq!(value, Value::Bool(true));
    }

    #[test]
    fn compiled_function() {
        let block = "add = |x, y| x + y; add(2, 3) == 5";
        let block = Untyped::<F32Grammar>::parse_statements(block).unwrap();
        let (module, _) = Compiler::compile_module(WildcardId, &block).unwrap();
        assert_eq!(module.run().unwrap(), Value::Bool(true));
    }

    #[test]
    fn compiled_function_with_capture() {
        let block = "A = 2; add = |x, y| x + y / A; add(2, 3) == 3.5";
        let block = Untyped::<F32Grammar>::parse_statements(block).unwrap();
        let (module, _) = Compiler::compile_module(WildcardId, &block).unwrap();
        assert_eq!(module.run().unwrap(), Value::Bool(true));
    }

    #[test]
    fn variable_extraction() {
        let def = "|a, b| ({ x = a * b + y; x - 2 }, a / b)";
        let def = Untyped::<F32Grammar>::parse_statements(def)
            .unwrap()
            .return_value
            .unwrap();
        let def = match def.extra {
            Expr::FnDefinition(def) => def,
            other => panic!("Unexpected function parsing result: {:?}", other),
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
        let def = match def.extra {
            Expr::FnDefinition(def) => def,
            other => panic!("Unexpected function parsing result: {:?}", other),
        };

        let captures = def.undefined_variables().unwrap();
        assert_eq!(captures["y"].location_offset(), 22);
        assert_eq!(captures["x"].location_offset(), 38);
    }

    #[test]
    fn extracting_captures() {
        let program = "y = 5 * x; y - 3 + x";
        let module = Untyped::<F32Grammar>::parse_statements(program).unwrap();
        let (registers, import_spans) =
            Compiler::extract_captures(Box::new(WildcardId), &module).unwrap();

        assert_eq!(registers.register_count(), 1);
        assert_eq!(*registers.get_var("x").unwrap(), Value::void());
        assert_eq!(import_spans.len(), 1);
        assert_eq!(import_spans["x"], Spanned::from_str(program, 8..9));
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

        let (registers, import_spans) =
            Compiler::extract_captures(Box::new(WildcardId), &module).unwrap();
        assert_eq!(registers.register_count(), 2);
        assert!(registers.variables_map().contains_key("x"));
        assert!(registers.variables_map().contains_key("PI"));
        assert_eq!(import_spans["x"].location_line(), 2); // should be the first mention
    }

    #[test]
    fn type_casts_are_ignored() {
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
        let block = Typed::<TypedGrammar>::parse_statements(block).unwrap();
        let (module, _) = Compiler::compile_module(WildcardId, &block).unwrap();
        let value = module.run().unwrap();
        assert_eq!(value, Value::Bool(true));
    }
}
