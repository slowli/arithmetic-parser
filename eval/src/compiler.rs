//! Transformation of AST output by the parser into non-recursive format.

use hashbrown::HashMap;

use core::iter;

use crate::{
    alloc::{vec, ToOwned, Vec},
    executable::{
        Atom, Command, CompiledExpr, Env, Executable, ExecutableFn, ExecutableModule, SpannedAtom,
    },
    AuxErrorInfo, EvalError, RepeatedAssignmentContext, SpannedEvalError,
};
use arithmetic_parser::{
    create_span_ref, Block, Destructure, Expr, FnDefinition, Grammar, Lvalue, Span, Spanned,
    SpannedExpr, SpannedLvalue, SpannedStatement, Statement,
};

#[derive(Debug, Clone, Default)]
pub struct Compiler {
    vars_to_registers: HashMap<String, usize>,
    scope_depth: usize,
    register_count: usize,
}

impl Compiler {
    pub fn new() -> Self {
        Self::default()
    }

    fn from_env<T: Grammar>(env: &Env<'_, T>) -> Self {
        Self {
            vars_to_registers: env.variables_map(),
            register_count: env.register_count(),
            scope_depth: 0,
        }
    }

    fn get_var(&self, name: &str) -> Option<usize> {
        self.vars_to_registers.get(name).copied()
    }

    fn push_assignment<'a, T: Grammar, U>(
        &mut self,
        executable: &mut Executable<'a, T>,
        rhs: CompiledExpr<'a, T>,
        rhs_span: &Spanned<'a, U>,
    ) -> usize {
        let register = self.register_count;
        let command = Command::Push(rhs);
        executable.push_command(create_span_ref(rhs_span, command));
        self.register_count += 1;
        register
    }

    fn compile_expr<'a, T: Grammar>(
        &mut self,
        executable: &mut Executable<'a, T>,
        expr: &SpannedExpr<'a, T>,
    ) -> Result<SpannedAtom<'a, T>, SpannedEvalError<'a>> {
        let atom = match &expr.extra {
            Expr::Literal(lit) => Atom::Constant(lit.clone()),

            Expr::Variable => {
                let register = self.vars_to_registers.get(expr.fragment).ok_or_else(|| {
                    let err = EvalError::Undefined(expr.fragment.to_owned());
                    SpannedEvalError::new(expr, err)
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
                        op: op.extra,
                        inner,
                    },
                    expr,
                );
                Atom::Register(register)
            }

            Expr::Binary { op, lhs, rhs } => {
                let lhs = self.compile_expr(executable, lhs)?;
                let rhs = self.compile_expr(executable, rhs)?;
                let register = self.push_assignment(
                    executable,
                    CompiledExpr::Binary {
                        op: op.extra,
                        lhs,
                        rhs,
                    },
                    expr,
                );
                Atom::Register(register)
            }

            Expr::Function { name, args } => {
                let name = self.compile_expr(executable, name)?;
                let args = args
                    .iter()
                    .map(|arg| self.compile_expr(executable, arg))
                    .collect::<Result<Vec<_>, _>>()?;
                let register =
                    self.push_assignment(executable, CompiledExpr::Function { name, args }, expr);
                Atom::Register(register)
            }

            Expr::Method {
                name,
                receiver,
                args,
            } => {
                let name =
                    create_span_ref(name, Atom::Register(self.vars_to_registers[name.fragment]));
                let args = iter::once(receiver.as_ref())
                    .chain(args)
                    .map(|arg| self.compile_expr(executable, arg))
                    .collect::<Result<Vec<_>, _>>()?;

                let register =
                    self.push_assignment(executable, CompiledExpr::Function { name, args }, expr);
                Atom::Register(register)
            }

            Expr::Block(block) => {
                let backup_state = self.clone();
                if self.scope_depth == 0 {
                    let command = Command::StartInnerScope;
                    executable.push_command(create_span_ref(&expr, command));
                }
                self.scope_depth += 1;

                let return_value = self
                    .compile_block_inner(executable, block)?
                    .unwrap_or_else(|| create_span_ref(&expr, Atom::Void));

                // Move the return value to the next register.
                let new_register = if let Atom::Register(ret_register) = return_value.extra {
                    let command = Command::Copy {
                        source: ret_register,
                        destination: backup_state.register_count,
                    };
                    executable.push_command(create_span_ref(expr, command));
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
                    executable.push_command(create_span_ref(&expr, command));
                }
                executable.push_command(create_span_ref(
                    expr,
                    Command::TruncateRegisters(self.register_count),
                ));

                if new_register {
                    Atom::Register(self.register_count - 1)
                } else {
                    Atom::Void
                }
            }

            Expr::FnDefinition(def) => {
                let mut extractor = CapturesExtractor::new();
                extractor.eval_function(def, self)?;
                let captures = extractor
                    .captures
                    .values()
                    .map(|capture| create_span_ref(capture, Atom::Register(capture.extra)))
                    .collect();
                let fn_executable = Self::compile_function(def, &extractor.captures)?;
                let fn_executable = ExecutableFn {
                    inner: fn_executable,
                    def_span: create_span_ref(expr, ()),
                    arg_count: def.args.extra.len(),
                };

                let ptr = executable.push_child_fn(fn_executable);
                let register = self.push_assignment(
                    executable,
                    CompiledExpr::DefineFunction { ptr, captures },
                    expr,
                );
                Atom::Register(register)
            }
        };
        Ok(create_span_ref(expr, atom))
    }

    fn compile_function<'a, T: Grammar>(
        def: &FnDefinition<'a, T>,
        captures: &HashMap<&'a str, Spanned<'a, usize>>,
    ) -> Result<Executable<'a, T>, SpannedEvalError<'a>> {
        // Allocate registers for captures.
        let mut this = Self::new();
        this.scope_depth = 1; // Disable generating variable annotations.

        for (i, &name) in captures.keys().enumerate() {
            this.vars_to_registers.insert(name.to_owned(), i);
        }
        this.register_count = captures.len() + 1; // one additional register for args

        let mut executable = Executable::new();
        let args_span = create_span_ref(&def.args, ());
        this.destructure(&mut executable, &def.args.extra, args_span, captures.len());

        for statement in &def.body.statements {
            this.compile_statement(&mut executable, statement)?;
        }
        if let Some(ref return_value) = def.body.return_value {
            let return_atom = this.compile_expr(&mut executable, return_value)?;
            let return_span = create_span_ref(&return_atom, ());
            let command = Command::Push(CompiledExpr::Atom(return_atom.extra));
            executable.push_command(create_span_ref(&return_span, command));
        }

        executable.finalize_function(this.register_count);
        Ok(executable)
    }

    fn compile_statement<'a, T: Grammar>(
        &mut self,
        executable: &mut Executable<'a, T>,
        statement: &SpannedStatement<'a, T>,
    ) -> Result<Option<SpannedAtom<'a, T>>, SpannedEvalError<'a>> {
        Ok(match &statement.extra {
            Statement::Expr(expr) => Some(self.compile_expr(executable, expr)?),

            Statement::Assignment { lhs, rhs } => {
                extract_vars_iter(
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
                self.assign(executable, lhs, rhs_register);
                None
            }
        })
    }

    pub(super) fn compile_module<'a, T: Grammar>(
        env: &Env<'a, T>,
        block: &Block<'a, T>,
        execute_in_env: bool,
    ) -> Result<ExecutableModule<'a, T>, SpannedEvalError<'a>> {
        let mut compiler = Self::from_env(env);
        let captures = if execute_in_env {
            // We don't care about captures since we won't execute the module with them anyway.
            Env::new()
        } else {
            let mut extractor = CapturesExtractor::new();
            extractor.local_vars.push(HashMap::new());
            extractor.eval_block(&block, &compiler)?;

            let mut captures = Env::new();
            for &var_name in extractor.captures.keys() {
                captures.push_var(var_name, env.get_var(var_name).unwrap().clone());
            }
            compiler = Self::from_env(&captures);
            captures
        };

        let mut executable = Executable::new();
        let empty_span = Span::new("");
        let last_atom = compiler
            .compile_block_inner(&mut executable, block)?
            .map(|spanned| spanned.extra)
            .unwrap_or(Atom::Void);
        // Push the last variable to a register to be popped during execution.
        compiler.push_assignment(&mut executable, CompiledExpr::Atom(last_atom), &empty_span);

        executable.finalize_block(compiler.register_count);
        Ok(ExecutableModule::new(executable, captures))
    }

    fn compile_block_inner<'a, T: Grammar>(
        &mut self,
        executable: &mut Executable<'a, T>,
        block: &Block<'a, T>,
    ) -> Result<Option<SpannedAtom<'a, T>>, SpannedEvalError<'a>> {
        for statement in &block.statements {
            self.compile_statement(executable, statement)?;
        }

        Ok(if let Some(ref return_value) = block.return_value {
            Some(self.compile_expr(executable, return_value)?)
        } else {
            None
        })
    }

    fn assign<'a, T: Grammar>(
        &mut self,
        executable: &mut Executable<'a, T>,
        lhs: &SpannedLvalue<'a, T::Type>,
        rhs_register: usize,
    ) {
        match &lhs.extra {
            Lvalue::Variable { .. } => {
                if lhs.fragment != "_" {
                    self.vars_to_registers
                        .insert(lhs.fragment.to_owned(), rhs_register);

                    // It does not make sense to annotate vars in the inner scopes, since
                    // they cannot be accessed externally.
                    if self.scope_depth == 0 {
                        let command = Command::Annotate {
                            register: rhs_register,
                            name: lhs.fragment,
                        };
                        executable.push_command(create_span_ref(lhs, command));
                    }
                }
            }

            Lvalue::Tuple(destructure) => {
                let span = create_span_ref(&lhs, ());
                self.destructure(executable, destructure, span, rhs_register);
            }
        }
    }

    fn destructure<'a, T: Grammar>(
        &mut self,
        executable: &mut Executable<'a, T>,
        destructure: &Destructure<'a, T::Type>,
        span: Span<'a>,
        rhs_register: usize,
    ) {
        let command = Command::Destructure {
            source: rhs_register,
            start_len: destructure.start.len(),
            end_len: destructure.end.len(),
            lvalue_len: destructure.len(),
            unchecked: false,
        };
        executable.push_command(create_span_ref(&span, command));
        let start_register = self.register_count;
        self.register_count += destructure.start.len() + destructure.end.len() + 1;

        for (i, lvalue) in (start_register..).zip(&destructure.start) {
            self.assign(executable, lvalue, i);
        }

        let start_register = start_register + destructure.start.len();
        if let Some(ref middle) = destructure.middle {
            if let Some(lvalue) = middle.extra.to_lvalue() {
                self.assign(executable, &lvalue, start_register);
            }
        }

        let start_register = start_register + 1;
        for (i, lvalue) in (start_register..).zip(&destructure.end) {
            self.assign(executable, lvalue, i);
        }
    }
}

/// Helper context for symbolic execution of a function body or a block in order to determine
/// variables captured by it.
#[derive(Debug)]
struct CapturesExtractor<'a> {
    local_vars: Vec<HashMap<&'a str, Span<'a>>>,
    captures: HashMap<&'a str, Spanned<'a, usize>>,
}

impl<'a> CapturesExtractor<'a> {
    fn new() -> Self {
        Self {
            local_vars: vec![],
            captures: HashMap::new(),
        }
    }

    /// Collects variables captured by the function into a single `Scope`.
    fn eval_function<T: Grammar>(
        &mut self,
        definition: &FnDefinition<'a, T>,
        context: &Compiler,
    ) -> Result<(), SpannedEvalError<'a>> {
        self.local_vars.push(HashMap::new());
        extract_vars(
            self.local_vars.last_mut().unwrap(),
            &definition.args.extra,
            RepeatedAssignmentContext::FnArgs,
        )?;
        self.eval_block(&definition.body, context)
    }

    fn has_var(&self, var_name: &str) -> bool {
        self.captures.contains_key(var_name)
            || self.local_vars.iter().any(|set| set.contains_key(var_name))
    }

    /// Processes a local variable in the rvalue position.
    fn eval_local_var<T>(
        &mut self,
        var_span: &Spanned<'a, T>,
        context: &Compiler,
    ) -> Result<(), EvalError> {
        let var_name = var_span.fragment;

        if self.has_var(var_name) {
            // No action needs to be performed.
        } else if let Some(register) = context.get_var(var_name) {
            self.captures
                .insert(var_name, create_span_ref(var_span, register));
        } else {
            return Err(EvalError::Undefined(var_name.to_owned()));
        }

        Ok(())
    }

    /// Evaluates an expression with the function validation semantics, i.e., to determine
    /// function captures.
    fn eval<T: Grammar>(
        &mut self,
        expr: &SpannedExpr<'a, T>,
        context: &Compiler,
    ) -> Result<(), SpannedEvalError<'a>> {
        match &expr.extra {
            Expr::Variable => {
                self.eval_local_var(expr, context)
                    .map_err(|e| SpannedEvalError::new(expr, e))?;
            }

            Expr::Literal(_) => { /* no action */ }

            Expr::Tuple(fragments) => {
                for fragment in fragments {
                    self.eval(fragment, context)?;
                }
            }
            Expr::Unary { inner, .. } => {
                self.eval(inner, context)?;
            }
            Expr::Binary { lhs, rhs, .. } => {
                self.eval(lhs, context)?;
                self.eval(rhs, context)?;
            }

            Expr::Function { args, name } => {
                for arg in args {
                    self.eval(arg, context)?;
                }
                self.eval(name, context)?;
            }

            Expr::Method {
                args,
                receiver,
                name,
            } => {
                self.eval(receiver, context)?;
                for arg in args {
                    self.eval(arg, context)?;
                }

                self.eval_local_var(name, context)
                    .map_err(|e| SpannedEvalError::new(name, e))?;
            }

            Expr::Block(block) => {
                self.local_vars.push(HashMap::new());
                self.eval_block(block, context)?;
            }

            Expr::FnDefinition(def) => {
                self.eval_function(def, context)?;
            }
        }
        Ok(())
    }

    /// Evaluates a statement using the provided context.
    fn eval_statement<T: Grammar>(
        &mut self,
        statement: &SpannedStatement<'a, T>,
        context: &Compiler,
    ) -> Result<(), SpannedEvalError<'a>> {
        match &statement.extra {
            Statement::Expr(expr) => self.eval(expr, context),
            Statement::Assignment { lhs, rhs } => {
                self.eval(rhs, context)?;
                let mut new_vars = HashMap::new();
                extract_vars_iter(
                    &mut new_vars,
                    iter::once(lhs),
                    RepeatedAssignmentContext::Assignment,
                )?;
                self.local_vars.last_mut().unwrap().extend(&new_vars);
                Ok(())
            }
        }
    }

    fn eval_block<T: Grammar>(
        &mut self,
        block: &Block<'a, T>,
        context: &Compiler,
    ) -> Result<(), SpannedEvalError<'a>> {
        self.local_vars.push(HashMap::new());
        for statement in &block.statements {
            self.eval_statement(statement, context)?;
        }
        if let Some(ref return_expr) = block.return_value {
            self.eval(return_expr, context)?;
        }
        self.local_vars.pop();
        Ok(())
    }
}

fn extract_vars<'a, T>(
    vars: &mut HashMap<&'a str, Span<'a>>,
    lvalues: &Destructure<'a, T>,
    context: RepeatedAssignmentContext,
) -> Result<(), SpannedEvalError<'a>> {
    let middle = lvalues
        .middle
        .as_ref()
        .and_then(|rest| rest.extra.to_lvalue());
    let all_lvalues = lvalues
        .start
        .iter()
        .chain(middle.as_ref())
        .chain(&lvalues.end);
    extract_vars_iter(vars, all_lvalues, context)
}

fn extract_vars_iter<'it, 'a: 'it, T: 'it>(
    vars: &mut HashMap<&'a str, Span<'a>>,
    lvalues: impl Iterator<Item = &'it SpannedLvalue<'a, T>>,
    context: RepeatedAssignmentContext,
) -> Result<(), SpannedEvalError<'a>> {
    for lvalue in lvalues {
        match &lvalue.extra {
            Lvalue::Variable { .. } => {
                if lvalue.fragment != "_" {
                    let var_span = create_span_ref(lvalue, ());
                    if let Some(prev_span) = vars.insert(lvalue.fragment, var_span) {
                        let err = EvalError::RepeatedAssignment { context };
                        return Err(SpannedEvalError::new(lvalue, err)
                            .with_span(&prev_span, AuxErrorInfo::PrevAssignment));
                    }
                }
            }

            Lvalue::Tuple(fragments) => {
                extract_vars(vars, fragments, context)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Value;

    use arithmetic_parser::{grammars::F32Grammar, GrammarExt, Span};

    use core::iter::FromIterator;

    #[test]
    fn compilation_basics() {
        let block = Span::new("x = 3; 1 + { y = 2; y * x } == 7");
        let block = F32Grammar::parse_statements(block).unwrap();
        let module = Compiler::compile_module(&Env::new(), &block, false).unwrap();
        let value = module.run().unwrap();
        assert_eq!(value, Value::Bool(true));
    }

    #[test]
    fn compiled_function() {
        let block = Span::new("add = |x, y| x + y; add(2, 3) == 5");
        let block = F32Grammar::parse_statements(block).unwrap();
        let value = Compiler::compile_module(&Env::new(), &block, false)
            .unwrap()
            .run()
            .unwrap();
        assert_eq!(value, Value::Bool(true));
    }

    #[test]
    fn compiled_function_with_capture() {
        let block = "A = 2; add = |x, y| x + y / A; add(2, 3) == 3.5";
        let block = F32Grammar::parse_statements(Span::new(block)).unwrap();
        let value = Compiler::compile_module(&Env::new(), &block, false)
            .unwrap()
            .run()
            .unwrap();
        assert_eq!(value, Value::Bool(true));
    }

    #[test]
    fn variable_extraction() {
        let compiler = Compiler {
            vars_to_registers: HashMap::from_iter(vec![("x".to_owned(), 0), ("y".to_owned(), 1)]),
            scope_depth: 0,
            register_count: 2,
        };

        let def = "|a, b| ({ x = a * b + y; x - 2 }, a / b)";
        let def = F32Grammar::parse_statements(Span::new(def))
            .unwrap()
            .return_value
            .unwrap();
        let def = match def.extra {
            Expr::FnDefinition(def) => def,
            other => panic!("Unexpected function parsing result: {:?}", other),
        };

        let mut validator = CapturesExtractor::new();
        validator.eval_function(&def, &compiler).unwrap();
        assert!(validator.captures.contains_key("y"));
        assert!(!validator.captures.contains_key("x"));
    }

    #[test]
    fn variable_extraction_with_scoping() {
        let compiler = Compiler {
            vars_to_registers: HashMap::from_iter(vec![("x".to_owned(), 0), ("y".to_owned(), 1)]),
            scope_depth: 0,
            register_count: 2,
        };

        let def = "|a, b| ({ x = a * b + y; x - 2 }, a / x)";
        let def = F32Grammar::parse_statements(Span::new(def))
            .unwrap()
            .return_value
            .unwrap();
        let def = match def.extra {
            Expr::FnDefinition(def) => def,
            other => panic!("Unexpected function parsing result: {:?}", other),
        };

        let mut validator = CapturesExtractor::new();
        validator.eval_function(&def, &compiler).unwrap();
        assert!(validator.captures.contains_key("y"));
        assert!(validator.captures.contains_key("x"));
    }

    #[test]
    fn module_imports() {
        let mut env = Env::new();
        env.push_var("x", Value::<F32Grammar>::Number(1.0));
        env.push_var("y", Value::Number(-3.0));

        let module = "y = 5 * x; y - 3";
        let module = F32Grammar::parse_statements(Span::new(module)).unwrap();
        let mut module = Compiler::compile_module(&env, &module, false).unwrap();

        let imports = module.imports().iter().collect::<Vec<_>>();
        assert_eq!(imports, &[("x", &Value::Number(1.0))]);
        let value = module.run().unwrap();
        assert_eq!(value, Value::Number(2.0));
        module.set_import("x", Value::Number(2.0));
        let value = module.run().unwrap();
        assert_eq!(value, Value::Number(7.0));
    }
}
