//! Transformation of AST output by the parser into non-recursive format.

use hashbrown::HashMap;

use core::iter;

use crate::{
    alloc::{vec, Box, String, ToOwned, Vec},
    error::RepeatedAssignmentContext,
    executable::{
        Atom, Command, ComparisonOp, CompiledExpr, Executable, ExecutableFn, ExecutableModule,
        Registers, SpannedAtom,
    },
    Error, ErrorKind, ModuleId, Value,
};
use arithmetic_parser::{
    grammars::Grammar, is_valid_variable_name, BinaryOp, Block, Destructure, Expr, FnDefinition,
    InputSpan, Lvalue, MaybeSpanned, Spanned, SpannedExpr, SpannedLvalue, SpannedStatement,
    Statement, UnaryOp,
};

mod captures;

use self::captures::{extract_vars_iter, CapturesExtractor, CompilerExtTarget};

/// Name of the comparison function used in desugaring order comparisons.
pub(crate) const CMP_FUNCTION_NAME: &str = "cmp";

pub(crate) type ImportSpans<'a> = HashMap<String, Spanned<'a>>;

#[derive(Debug)]
pub(crate) struct Compiler {
    vars_to_registers: HashMap<String, usize>,
    scope_depth: usize,
    register_count: usize,
    module_id: Box<dyn ModuleId>,
}

impl Clone for Compiler {
    fn clone(&self) -> Self {
        Self {
            vars_to_registers: self.vars_to_registers.clone(),
            scope_depth: self.scope_depth,
            register_count: self.register_count,
            module_id: self.module_id.clone_boxed(),
        }
    }
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
            vars_to_registers: env.variables_map().to_owned(),
            register_count: env.register_count(),
            scope_depth: 0,
            module_id,
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

    fn compile_expr<'a, T: Grammar>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        expr: &SpannedExpr<'a, T>,
    ) -> Result<SpannedAtom<'a, T::Lit>, Error<'a>> {
        let atom = match &expr.extra {
            Expr::Literal(lit) => Atom::Constant(lit.clone()),

            Expr::Variable => {
                let var_name = *expr.fragment();
                let register = self.vars_to_registers.get(var_name).ok_or_else(|| {
                    let err = ErrorKind::Undefined(var_name.to_owned());
                    self.create_error(expr, err)
                })?;
                Atom::Register(*register)
            }

            Expr::Tuple(tuple) => {
                let registers = tuple
                    .iter()
                    .map(|elem| {
                        self.compile_expr(executable, elem)
                            .map(|spanned| spanned.extra)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let register =
                    self.push_assignment(executable, CompiledExpr::Tuple(registers), expr);
                Atom::Register(register)
            }

            Expr::Unary { op, inner } => {
                let inner = self.compile_expr(executable, inner)?;
                let register = self.push_assignment(
                    executable,
                    CompiledExpr::Unary {
                        op: self.check_unary_op(op)?,
                        inner,
                    },
                    expr,
                );
                Atom::Register(register)
            }

            Expr::Binary { op, lhs, rhs } => {
                self.compile_binary_expr(executable, expr, op, lhs, rhs)?
            }
            Expr::Function { name, args } => self.compile_fn_call(executable, expr, name, args)?,

            Expr::Method {
                name,
                receiver,
                args,
            } => self.compile_method_call(executable, expr, name, receiver, args)?,

            Expr::Block(block) => self.compile_block(executable, expr, block)?,
            Expr::FnDefinition(def) => self.compile_fn_definition(executable, expr, def)?,

            _ => {
                let err = ErrorKind::unsupported(expr.extra.ty());
                return Err(self.create_error(expr, err));
            }
        };

        Ok(expr.copy_with_extra(atom).into())
    }

    fn compile_binary_expr<'a, T: Grammar>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        binary_expr: &SpannedExpr<'a, T>,
        op: &Spanned<'a, BinaryOp>,
        lhs: &SpannedExpr<'a, T>,
        rhs: &SpannedExpr<'a, T>,
    ) -> Result<Atom<T::Lit>, Error<'a>> {
        let lhs = self.compile_expr(executable, lhs)?;
        let rhs = self.compile_expr(executable, rhs)?;

        let compiled = if op.extra.is_order_comparison() {
            let cmp_function = self.get_var(CMP_FUNCTION_NAME);
            let cmp_function = op.copy_with_extra(Atom::Register(cmp_function));

            let cmp_invocation = CompiledExpr::Function {
                original_name: Some((*cmp_function.fragment()).to_owned()),
                name: cmp_function.into(),
                args: vec![lhs, rhs],
            };
            let cmp_register = self.push_assignment(executable, cmp_invocation, binary_expr);

            CompiledExpr::Compare {
                inner: binary_expr
                    .copy_with_extra(Atom::Register(cmp_register))
                    .into(),
                op: ComparisonOp::from(op.extra),
            }
        } else {
            CompiledExpr::Binary {
                op: self.check_binary_op(op)?,
                lhs,
                rhs,
            }
        };

        let register = self.push_assignment(executable, compiled, binary_expr);
        Ok(Atom::Register(register))
    }

    fn compile_fn_call<'a, T: Grammar>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        call_expr: &SpannedExpr<'a, T>,
        name: &SpannedExpr<'a, T>,
        args: &[SpannedExpr<'a, T>],
    ) -> Result<Atom<T::Lit>, Error<'a>> {
        let original_name = *name.fragment();
        let original_name = if is_valid_variable_name(original_name) {
            Some(original_name.to_owned())
        } else {
            None
        };

        let name = self.compile_expr(executable, name)?;

        let args = args
            .iter()
            .map(|arg| self.compile_expr(executable, arg))
            .collect::<Result<Vec<_>, _>>()?;
        let function = CompiledExpr::Function {
            name,
            original_name,
            args,
        };
        let register = self.push_assignment(executable, function, call_expr);
        Ok(Atom::Register(register))
    }

    fn compile_method_call<'a, T: Grammar>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        call_expr: &SpannedExpr<'a, T>,
        name: &Spanned<'a>,
        receiver: &SpannedExpr<'a, T>,
        args: &[SpannedExpr<'a, T>],
    ) -> Result<Atom<T::Lit>, Error<'a>> {
        let original_name = Some((*name.fragment()).to_owned());
        let name: MaybeSpanned<_> = name
            .copy_with_extra(Atom::Register(self.vars_to_registers[*name.fragment()]))
            .into();
        let args = iter::once(receiver)
            .chain(args)
            .map(|arg| self.compile_expr(executable, arg))
            .collect::<Result<Vec<_>, _>>()?;

        let function = CompiledExpr::Function {
            name,
            original_name,
            args,
        };
        let register = self.push_assignment(executable, function, call_expr);
        Ok(Atom::Register(register))
    }

    fn compile_block<'a, T: Grammar>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        block_expr: &SpannedExpr<'a, T>,
        block: &Block<'a, T>,
    ) -> Result<Atom<T::Lit>, Error<'a>> {
        let backup_state = self.clone();
        if self.scope_depth == 0 {
            let command = Command::StartInnerScope;
            executable.push_command(block_expr.copy_with_extra(command));
        }
        self.scope_depth += 1;

        let return_value = self
            .compile_block_inner(executable, block)?
            .unwrap_or_else(|| block_expr.copy_with_extra(Atom::Void).into());

        // Move the return value to the next register.
        let new_register = if let Atom::Register(ret_register) = return_value.extra {
            let command = Command::Copy {
                source: ret_register,
                destination: backup_state.register_count,
            };
            executable.push_command(block_expr.copy_with_extra(command));
            true
        } else {
            false
        };

        // Return to the previous state. This will erase register mapping
        // for the inner scope and set the `scope_depth`.
        *self = backup_state;
        if new_register {
            self.register_count += 1;
        }
        if self.scope_depth == 0 {
            let command = Command::EndInnerScope;
            executable.push_command(block_expr.copy_with_extra(command));
        }
        executable.push_command(
            block_expr.copy_with_extra(Command::TruncateRegisters(self.register_count)),
        );

        Ok(if new_register {
            Atom::Register(self.register_count - 1)
        } else {
            Atom::Void
        })
    }

    fn compile_block_inner<'a, T: Grammar>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        block: &Block<'a, T>,
    ) -> Result<Option<SpannedAtom<'a, T::Lit>>, Error<'a>> {
        for statement in &block.statements {
            self.compile_statement(executable, statement)?;
        }

        Ok(if let Some(ref return_value) = block.return_value {
            Some(self.compile_expr(executable, return_value)?)
        } else {
            None
        })
    }

    fn compile_fn_definition<'a, T: Grammar>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        def_expr: &SpannedExpr<'a, T>,
        def: &FnDefinition<'a, T>,
    ) -> Result<Atom<T::Lit>, Error<'a>> {
        let module_id = self.module_id.clone_boxed();

        let mut extractor = CapturesExtractor::new(module_id);
        extractor.eval_function(def)?;
        let captures = self.get_captures(extractor);

        let fn_executable = self.compile_function(def, &captures)?;
        let fn_executable = ExecutableFn {
            inner: fn_executable,
            def_span: def_expr.with_no_extra().into(),
            arg_count: def.args.extra.len(),
        };

        let ptr = executable.push_child_fn(fn_executable);
        let (capture_names, captures) = captures
            .into_iter()
            .map(|(name, value)| (name.to_owned(), value))
            .unzip();
        let register = self.push_assignment(
            executable,
            CompiledExpr::DefineFunction {
                ptr,
                captures,
                capture_names,
            },
            def_expr,
        );
        Ok(Atom::Register(register))
    }

    fn get_captures<'a, T>(
        &self,
        extractor: CapturesExtractor<'a>,
    ) -> HashMap<&'a str, SpannedAtom<'a, T>> {
        extractor
            .captures
            .into_iter()
            .map(|(var_name, var_span)| {
                let register = self.get_var(var_name);
                let capture = var_span.copy_with_extra(Atom::Register(register));
                (var_name, capture.into())
            })
            .collect()
    }

    fn compile_function<'a, T: Grammar>(
        &self,
        def: &FnDefinition<'a, T>,
        captures: &HashMap<&'a str, SpannedAtom<'a, T::Lit>>,
    ) -> Result<Executable<'a, T::Lit>, Error<'a>> {
        // Allocate registers for captures.
        let mut this = Self::new(self.module_id.clone_boxed());
        this.scope_depth = 1; // Disable generating variable annotations.

        for (i, &name) in captures.keys().enumerate() {
            this.vars_to_registers.insert(name.to_owned(), i);
        }
        this.register_count = captures.len() + 1; // one additional register for args

        let mut executable = Executable::new(self.module_id.clone_boxed());
        let args_span = def.args.with_no_extra();
        this.destructure(&mut executable, &def.args.extra, args_span, captures.len())?;

        for statement in &def.body.statements {
            this.compile_statement(&mut executable, statement)?;
        }
        if let Some(ref return_value) = def.body.return_value {
            let return_atom = this.compile_expr(&mut executable, return_value)?;
            let return_span = return_atom.with_no_extra();
            let command = Command::Push(CompiledExpr::Atom(return_atom.extra));
            executable.push_command(return_span.copy_with_extra(command));
        }

        executable.finalize_function(this.register_count);
        Ok(executable)
    }

    fn compile_statement<'a, T: Grammar>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        statement: &SpannedStatement<'a, T>,
    ) -> Result<Option<SpannedAtom<'a, T::Lit>>, Error<'a>> {
        Ok(match &statement.extra {
            Statement::Expr(expr) => Some(self.compile_expr(executable, expr)?),

            Statement::Assignment { lhs, rhs } => {
                extract_vars_iter(
                    self.module_id.as_ref(),
                    &mut HashMap::new(),
                    iter::once(lhs),
                    RepeatedAssignmentContext::Assignment,
                )?;

                let rhs = self.compile_expr(executable, rhs)?;
                // Allocate the register for the constant if necessary.
                let rhs_register = match rhs.extra {
                    Atom::Constant(_) | Atom::Void => {
                        self.push_assignment(executable, CompiledExpr::Atom(rhs.extra), statement)
                    }
                    Atom::Register(register) => register,
                };
                self.assign(executable, lhs, rhs_register)?;
                None
            }

            _ => {
                let err = ErrorKind::unsupported(statement.extra.ty());
                return Err(self.create_error(statement, err));
            }
        })
    }

    pub fn compile_module<'a, Id: ModuleId, T: Grammar>(
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

    fn extract_captures<'a, T: Grammar>(
        module_id: Box<dyn ModuleId>,
        block: &Block<'a, T>,
    ) -> Result<(Registers<'a, T::Lit>, ImportSpans<'a>), Error<'a>> {
        let mut extractor = CapturesExtractor::new(module_id);
        extractor.eval_block(&block)?;

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
                let var_name = *lhs.fragment();
                if var_name != "_" {
                    self.vars_to_registers
                        .insert(var_name.to_owned(), rhs_register);

                    // It does not make sense to annotate vars in the inner scopes, since
                    // they cannot be accessed externally.
                    if self.scope_depth == 0 {
                        let command = Command::Annotate {
                            register: rhs_register,
                            name: var_name.to_owned(),
                        };
                        executable.push_command(lhs.copy_with_extra(command));
                    }
                }
            }

            Lvalue::Tuple(destructure) => {
                let span = lhs.with_no_extra();
                self.destructure(executable, destructure, span, rhs_register)?;
            }

            _ => {
                let err = ErrorKind::unsupported(lhs.extra.ty());
                return Err(self.create_error(lhs, err));
            }
        }

        Ok(())
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
        if let Some(ref middle) = destructure.middle {
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

impl<'a, T: Grammar> CompilerExt<'a> for Block<'a, T> {
    fn undefined_variables(&self) -> Result<HashMap<&'a str, Spanned<'a>>, Error<'a>> {
        CompilerExtTarget::Block(self).get_undefined_variables()
    }
}

impl<'a, T: Grammar> CompilerExt<'a> for FnDefinition<'a, T> {
    fn undefined_variables(&self) -> Result<HashMap<&'a str, Spanned<'a>>, Error<'a>> {
        CompilerExtTarget::FnDefinition(self).get_undefined_variables()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Value, WildcardId};

    use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};

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
}
