//! Transformation of AST output by the parser into non-recursive format.

use hashbrown::HashMap;

use core::iter;

use crate::{
    alloc::{vec, ToOwned, Vec},
    executable::{
        Atom, Command, ComparisonOp, CompiledExpr, Env, Executable, ExecutableFn, ExecutableModule,
        SpannedAtom,
    },
    AuxErrorInfo, EvalError, RepeatedAssignmentContext, SpannedEvalError,
};
use arithmetic_parser::{
    BinaryOp, Block, Destructure, Expr, FnDefinition, Grammar, InputSpan, Lvalue, Spanned,
    SpannedExpr, SpannedLvalue, SpannedStatement, Statement,
};

/// Name of the comparison function used in desugaring order comparisons.
const CMP_FUNCTION_NAME: &str = "cmp";

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
        executable.push_command(rhs_span.copy_with_extra(command));
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
                let var_name = *expr.fragment();
                let register = self.vars_to_registers.get(var_name).ok_or_else(|| {
                    let err = EvalError::Undefined(var_name.to_owned());
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

                let compiled = if op.extra.is_order_comparison() {
                    let cmp_function = self.get_var(CMP_FUNCTION_NAME).ok_or_else(|| {
                        let err = EvalError::MissingCmpFunction {
                            name: CMP_FUNCTION_NAME.to_owned(),
                        };
                        SpannedEvalError::new(expr, err)
                    })?;
                    let cmp_function = op.copy_with_extra(Atom::Register(cmp_function));

                    let cmp_invocation = CompiledExpr::Function {
                        name: cmp_function,
                        args: vec![lhs, rhs],
                    };
                    let cmp_register = self.push_assignment(executable, cmp_invocation, expr);
                    CompiledExpr::Compare {
                        inner: expr.copy_with_extra(Atom::Register(cmp_register)),
                        op: ComparisonOp::from(op.extra),
                    }
                } else {
                    CompiledExpr::Binary {
                        op: op.extra,
                        lhs,
                        rhs,
                    }
                };

                let register = self.push_assignment(executable, compiled, expr);
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
                    name.copy_with_extra(Atom::Register(self.vars_to_registers[*name.fragment()]));
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
                    executable.push_command(expr.copy_with_extra(command));
                }
                self.scope_depth += 1;

                let return_value = self
                    .compile_block_inner(executable, block)?
                    .unwrap_or_else(|| expr.copy_with_extra(Atom::Void));

                // Move the return value to the next register.
                let new_register = if let Atom::Register(ret_register) = return_value.extra {
                    let command = Command::Copy {
                        source: ret_register,
                        destination: backup_state.register_count,
                    };
                    executable.push_command(expr.copy_with_extra(command));
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
                    executable.push_command(expr.copy_with_extra(command));
                }
                executable.push_command(
                    expr.copy_with_extra(Command::TruncateRegisters(self.register_count)),
                );

                if new_register {
                    Atom::Register(self.register_count - 1)
                } else {
                    Atom::Void
                }
            }

            Expr::FnDefinition(def) => {
                let mut captures = HashMap::new();
                let mut extractor = CapturesExtractor::new(|var_name, var_span| {
                    if let Some(register) = self.get_var(var_name) {
                        captures
                            .insert(var_name, var_span.copy_with_extra(Atom::Register(register)));
                        Ok(())
                    } else {
                        Err(EvalError::Undefined(var_name.to_owned()))
                    }
                });

                extractor.eval_function(def)?;
                let fn_executable = Self::compile_function(def, &captures)?;
                let fn_executable = ExecutableFn {
                    inner: fn_executable,
                    def_span: expr.with_no_extra(),
                    arg_count: def.args.extra.len(),
                };

                let ptr = executable.push_child_fn(fn_executable);
                let register = self.push_assignment(
                    executable,
                    CompiledExpr::DefineFunction {
                        ptr,
                        captures: captures.into_iter().map(|(_, value)| value).collect(),
                    },
                    expr,
                );
                Atom::Register(register)
            }
        };
        Ok(expr.copy_with_extra(atom))
    }

    fn compile_function<'a, T: Grammar>(
        def: &FnDefinition<'a, T>,
        captures: &HashMap<&'a str, SpannedAtom<'a, T>>,
    ) -> Result<Executable<'a, T>, SpannedEvalError<'a>> {
        // Allocate registers for captures.
        let mut this = Self::new();
        this.scope_depth = 1; // Disable generating variable annotations.

        for (i, &name) in captures.keys().enumerate() {
            this.vars_to_registers.insert(name.to_owned(), i);
        }
        this.register_count = captures.len() + 1; // one additional register for args

        let mut executable = Executable::new();
        let args_span = def.args.with_no_extra();
        this.destructure(&mut executable, &def.args.extra, args_span, captures.len());

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
            let mut captures = Env::new();
            let mut extractor = CapturesExtractor::new(|var_name, _| {
                if let Some(value) = env.get_var(var_name) {
                    captures.push_var(var_name, value.clone());
                    Ok(())
                } else {
                    Err(EvalError::Undefined(var_name.to_owned()))
                }
            });
            extractor.local_vars.push(HashMap::new());
            extractor.eval_block(&block)?;

            compiler = Self::from_env(&captures);
            captures
        };

        let mut executable = Executable::new();
        let empty_span = InputSpan::new("");
        let last_atom = compiler
            .compile_block_inner(&mut executable, block)?
            .map(|spanned| spanned.extra)
            .unwrap_or(Atom::Void);
        // Push the last variable to a register to be popped during execution.
        compiler.push_assignment(
            &mut executable,
            CompiledExpr::Atom(last_atom),
            &empty_span.into(),
        );

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
                let var_name = *lhs.fragment();
                if var_name != "_" {
                    self.vars_to_registers
                        .insert(var_name.to_owned(), rhs_register);

                    // It does not make sense to annotate vars in the inner scopes, since
                    // they cannot be accessed externally.
                    if self.scope_depth == 0 {
                        let command = Command::Annotate {
                            register: rhs_register,
                            name: var_name,
                        };
                        executable.push_command(lhs.copy_with_extra(command));
                    }
                }
            }

            Lvalue::Tuple(destructure) => {
                let span = lhs.with_no_extra();
                self.destructure(executable, destructure, span, rhs_register);
            }
        }
    }

    fn destructure<'a, T: Grammar>(
        &mut self,
        executable: &mut Executable<'a, T>,
        destructure: &Destructure<'a, T::Type>,
        span: Spanned<'a>,
        rhs_register: usize,
    ) {
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
struct CapturesExtractor<'a, F> {
    local_vars: Vec<HashMap<&'a str, Spanned<'a>>>,
    action: F,
}

impl<'a, F> CapturesExtractor<'a, F>
where
    F: FnMut(&'a str, Spanned<'a>) -> Result<(), EvalError>,
{
    fn new(action: F) -> Self {
        Self {
            local_vars: vec![],
            action,
        }
    }

    /// Collects variables captured by the function into a single `Scope`.
    fn eval_function<T: Grammar>(
        &mut self,
        definition: &FnDefinition<'a, T>,
    ) -> Result<(), SpannedEvalError<'a>> {
        self.local_vars.push(HashMap::new());
        extract_vars(
            self.local_vars.last_mut().unwrap(),
            &definition.args.extra,
            RepeatedAssignmentContext::FnArgs,
        )?;
        self.eval_block(&definition.body)
    }

    fn has_var(&self, var_name: &str) -> bool {
        self.local_vars.iter().any(|set| set.contains_key(var_name))
    }

    /// Processes a local variable in the rvalue position.
    fn eval_local_var<T>(&mut self, var_span: &Spanned<'a, T>) -> Result<(), EvalError> {
        if self.has_var(var_span.fragment()) {
            // No action needs to be performed.
            Ok(())
        } else {
            (self.action)(var_span.fragment(), var_span.with_no_extra())
        }
    }

    fn eval_cmp(&mut self, op_span: &Spanned<'a, BinaryOp>) -> Result<(), EvalError> {
        if self.has_var(CMP_FUNCTION_NAME) {
            Ok(())
        } else {
            (self.action)(CMP_FUNCTION_NAME, op_span.with_no_extra())
        }
    }

    /// Evaluates an expression with the function validation semantics, i.e., to determine
    /// function captures.
    fn eval<T: Grammar>(&mut self, expr: &SpannedExpr<'a, T>) -> Result<(), SpannedEvalError<'a>> {
        match &expr.extra {
            Expr::Variable => {
                self.eval_local_var(expr)
                    .map_err(|e| SpannedEvalError::new(expr, e))?;
            }

            Expr::Literal(_) => { /* no action */ }

            Expr::Tuple(fragments) => {
                for fragment in fragments {
                    self.eval(fragment)?;
                }
            }
            Expr::Unary { inner, .. } => {
                self.eval(inner)?;
            }
            Expr::Binary { lhs, rhs, op } => {
                self.eval(lhs)?;
                self.eval(rhs)?;

                if op.extra.is_order_comparison() {
                    self.eval_cmp(op)
                        .map_err(|e| SpannedEvalError::new(op, e))?;
                }
            }

            Expr::Function { args, name } => {
                for arg in args {
                    self.eval(arg)?;
                }
                self.eval(name)?;
            }

            Expr::Method {
                args,
                receiver,
                name,
            } => {
                self.eval(receiver)?;
                for arg in args {
                    self.eval(arg)?;
                }

                self.eval_local_var(name)
                    .map_err(|e| SpannedEvalError::new(name, e))?;
            }

            Expr::Block(block) => {
                self.local_vars.push(HashMap::new());
                self.eval_block(block)?;
            }

            Expr::FnDefinition(def) => {
                self.eval_function(def)?;
            }
        }
        Ok(())
    }

    /// Evaluates a statement using the provided context.
    fn eval_statement<T: Grammar>(
        &mut self,
        statement: &SpannedStatement<'a, T>,
    ) -> Result<(), SpannedEvalError<'a>> {
        match &statement.extra {
            Statement::Expr(expr) => self.eval(expr),
            Statement::Assignment { lhs, rhs } => {
                self.eval(rhs)?;
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

    fn eval_block<T: Grammar>(&mut self, block: &Block<'a, T>) -> Result<(), SpannedEvalError<'a>> {
        self.local_vars.push(HashMap::new());
        for statement in &block.statements {
            self.eval_statement(statement)?;
        }
        if let Some(ref return_expr) = block.return_value {
            self.eval(return_expr)?;
        }
        self.local_vars.pop();
        Ok(())
    }
}

fn extract_vars<'a, T>(
    vars: &mut HashMap<&'a str, Spanned<'a>>,
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
    vars: &mut HashMap<&'a str, Spanned<'a>>,
    lvalues: impl Iterator<Item = &'it SpannedLvalue<'a, T>>,
    context: RepeatedAssignmentContext,
) -> Result<(), SpannedEvalError<'a>> {
    for lvalue in lvalues {
        match &lvalue.extra {
            Lvalue::Variable { .. } => {
                let var_name = *lvalue.fragment();
                if var_name != "_" {
                    let var_span = lvalue.with_no_extra();
                    if let Some(prev_span) = vars.insert(var_name, var_span) {
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

/// Compiler extensions defined for some AST nodes, most notably, `Block`.
///
/// # Examples
///
/// ```
/// use arithmetic_parser::{grammars::F32Grammar, GrammarExt, InputSpan};
/// use arithmetic_eval::CompilerExt;
/// # use hashbrown::HashSet;
/// # use core::iter::FromIterator;
///
/// let block = InputSpan::new("x = sin(0.5) / PI; y = x * E; (x, y)");
/// let block = F32Grammar::parse_statements(block).unwrap();
/// let undefined_vars = block.undefined_variables().unwrap();
/// assert_eq!(
///     undefined_vars.keys().copied().collect::<HashSet<_>>(),
///     HashSet::from_iter(vec!["sin", "PI", "E"])
/// );
/// assert_eq!(undefined_vars["PI"].location_offset(), 15);
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
    fn undefined_variables(&self) -> Result<HashMap<&'a str, Spanned<'a>>, SpannedEvalError<'a>>;
}

impl<'a, T: Grammar> CompilerExt<'a> for Block<'a, T> {
    fn undefined_variables(&self) -> Result<HashMap<&'a str, Spanned<'a>>, SpannedEvalError<'a>> {
        CompilerExtTarget::Block(self).get_undefined_variables()
    }
}

impl<'a, T: Grammar> CompilerExt<'a> for FnDefinition<'a, T> {
    fn undefined_variables(&self) -> Result<HashMap<&'a str, Spanned<'a>>, SpannedEvalError<'a>> {
        CompilerExtTarget::FnDefinition(self).get_undefined_variables()
    }
}

/// Helper enum for `CompilerExt` implementations that allows to reduce code duplication.
#[derive(Debug)]
enum CompilerExtTarget<'r, 'a, T: Grammar> {
    Block(&'r Block<'a, T>),
    FnDefinition(&'r FnDefinition<'a, T>),
}

impl<'a, T: Grammar> CompilerExtTarget<'_, 'a, T> {
    fn get_undefined_variables(
        self,
    ) -> Result<HashMap<&'a str, Spanned<'a>>, SpannedEvalError<'a>> {
        let mut undefined_vars = HashMap::new();
        let mut extractor = CapturesExtractor::new(|var_name, var_span| {
            if !undefined_vars.contains_key(var_name) {
                undefined_vars.insert(var_name, var_span);
            }
            Ok(())
        });

        match self {
            Self::Block(block) => extractor.eval_block(block)?,
            Self::FnDefinition(definition) => extractor.eval_function(definition)?,
        }

        Ok(undefined_vars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Value;

    use arithmetic_parser::{grammars::F32Grammar, GrammarExt, InputSpan};

    #[test]
    fn compilation_basics() {
        let block = InputSpan::new("x = 3; 1 + { y = 2; y * x } == 7");
        let block = F32Grammar::parse_statements(block).unwrap();
        let module = Compiler::compile_module(&Env::new(), &block, false).unwrap();
        let value = module.run().unwrap();
        assert_eq!(value, Value::Bool(true));
    }

    #[test]
    fn compiled_function() {
        let block = InputSpan::new("add = |x, y| x + y; add(2, 3) == 5");
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
        let block = F32Grammar::parse_statements(InputSpan::new(block)).unwrap();
        let value = Compiler::compile_module(&Env::new(), &block, false)
            .unwrap()
            .run()
            .unwrap();
        assert_eq!(value, Value::Bool(true));
    }

    #[test]
    fn variable_extraction() {
        let def = "|a, b| ({ x = a * b + y; x - 2 }, a / b)";
        let def = F32Grammar::parse_statements(InputSpan::new(def))
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
        let def = F32Grammar::parse_statements(InputSpan::new(def))
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
    fn module_imports() {
        let mut env = Env::new();
        env.push_var("x", Value::<F32Grammar>::Number(1.0));
        env.push_var("y", Value::Number(-3.0));

        let module = "y = 5 * x; y - 3";
        let module = F32Grammar::parse_statements(InputSpan::new(module)).unwrap();
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
