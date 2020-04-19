//! Transformation of AST output by the parser into non-recursive format.

#![allow(missing_docs)]

use num_traits::{Num, Pow};
use smallvec::{smallvec, SmallVec};

use std::{collections::HashMap, iter, ops, rc::Rc};

use crate::{
    helpers::create_span_ref,
    interpreter::{
        error::{
            AuxErrorInfo, Backtrace, ErrorWithBacktrace, EvalError, EvalResult,
            RepeatedAssignmentContext, SpannedEvalError, TupleLenMismatchContext,
        },
        values::{CallContext, InterpretedFn, Value},
    },
    BinaryOp, Block, Destructure, Expr, FnDefinition, Grammar, Lvalue, LvalueLen, Span, Spanned,
    SpannedExpr, SpannedLvalue, SpannedStatement, Statement, UnaryOp,
};

#[derive(Debug, Clone, Copy)]
enum Atom<T: Grammar> {
    Constant(T::Lit),
    Register(usize),
    Void,
}

type SpannedAtom<'a, T> = Spanned<'a, Atom<T>>;

#[derive(Debug, Clone)]
enum CompiledExpr<'a, T: Grammar> {
    Atom(SpannedAtom<'a, T>),
    Tuple(Vec<SpannedAtom<'a, T>>),
    Unary {
        op: UnaryOp,
        inner: SpannedAtom<'a, T>,
    },
    Binary {
        op: BinaryOp,
        lhs: SpannedAtom<'a, T>,
        rhs: SpannedAtom<'a, T>,
    },
    Function {
        name: SpannedAtom<'a, T>,
        args: Vec<SpannedAtom<'a, T>>,
    },
    DefineFunction {
        ptr: usize,
        captures: Vec<SpannedAtom<'a, T>>,
    },
}

#[derive(Debug)]
enum Command<'a, T: Grammar> {
    Push(CompiledExpr<'a, T>),

    Destructure {
        source: usize,
        start_len: usize,
        end_len: usize,
        lvalue_len: LvalueLen,
    },

    Copy {
        source: usize,
        destination: usize,
    },

    Annotate {
        register: usize,
        name: &'a str,
    },

    StartInnerScope,
    EndInnerScope,
    TruncateRegisters(usize),
}

type SpannedCommand<'a, T> = Spanned<'a, Command<'a, T>>;

#[derive(Debug)]
pub(super) struct ExecutableFn<'a, T: Grammar> {
    pub inner: Executable<'a, T>,
    pub def_span: Span<'a>,
    pub arg_count: LvalueLen,
}

#[derive(Debug)]
pub(super) struct Executable<'a, T: Grammar> {
    commands: Vec<SpannedCommand<'a, T>>,
    child_fns: Vec<Rc<ExecutableFn<'a, T>>>,
    // Hint how many registers the executable requires.
    register_capacity: usize,
}

impl<T: Grammar> Executable<'_, T> {
    pub(super) fn new() -> Self {
        Self {
            commands: vec![],
            child_fns: vec![],
            register_capacity: 0,
        }
    }
}

impl<'a, T> Executable<'a, T>
where
    T: Grammar,
    T::Lit: Num + ops::Neg<Output = T::Lit> + Pow<T::Lit, Output = T::Lit>,
{
    pub(super) fn call_function(
        &self,
        captures: Vec<Value<'a, T>>,
        args: Vec<Value<'a, T>>,
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        let mut registers = SmallVec::from(captures);
        registers.push(Value::Tuple(args));
        let mut env = Env {
            registers,
            ..Env::new()
        };
        env.execute(self, ctx.backtrace())
    }
}

#[derive(Debug)]
pub(super) struct Env<'a, T: Grammar> {
    registers: SmallVec<[Value<'a, T>; 16]>,
    // Maps variables to registers. Variables are mapped only from the global scope;
    // thus, we don't need to remove them on error in an inner scope.
    vars: HashMap<&'a str, usize>,
    // Marks the start of a first inner scope currently being evaluated. This is used
    // to quickly remove registers from the inner scopes on error.
    inner_scope_start: Option<usize>,
}

impl<T: Grammar> Clone for Env<'_, T> {
    fn clone(&self) -> Self {
        Self {
            registers: self.registers.clone(),
            vars: self.vars.clone(),
            inner_scope_start: self.inner_scope_start,
        }
    }
}

impl<'a, T: Grammar> Env<'a, T> {
    pub fn new() -> Self {
        Self {
            registers: smallvec![],
            vars: HashMap::new(),
            inner_scope_start: None,
        }
    }

    pub fn get_var(&self, name: &str) -> Option<&Value<'a, T>> {
        let register = *self.vars.get(name)?;
        Some(&self.registers[register])
    }

    pub fn variables(&self) -> impl Iterator<Item = (&'a str, &Value<'a, T>)> + '_ {
        self.vars
            .iter()
            .map(move |(name, register)| (*name, &self.registers[*register]))
    }

    fn set_var(&mut self, name: &str, value: Value<'a, T>) {
        let register = *self.vars.get(name).unwrap_or_else(|| {
            panic!("Variable `{}` is not defined", name);
        });
        self.registers[register] = value;
    }

    pub fn push_var(&mut self, name: &'a str, value: Value<'a, T>) {
        let register = self.registers.len();
        self.registers.push(value);
        self.vars.insert(name, register);
    }
}

impl<'a, T> Env<'a, T>
where
    T: Grammar,
    T::Lit: Num + ops::Neg<Output = T::Lit> + Pow<T::Lit, Output = T::Lit>,
{
    pub fn execute(
        &mut self,
        executable: &Executable<'a, T>,
        backtrace: Option<&mut Backtrace<'a>>,
    ) -> EvalResult<'a, T> {
        self.execute_inner(executable, backtrace).map_err(|err| {
            if let Some(scope_start) = self.inner_scope_start.take() {
                self.registers.truncate(scope_start);
            }
            err
        })
    }

    fn execute_inner(
        &mut self,
        executable: &Executable<'a, T>,
        mut backtrace: Option<&mut Backtrace<'a>>,
    ) -> EvalResult<'a, T> {
        if let Some(additional_capacity) = executable
            .register_capacity
            .checked_sub(self.registers.len())
        {
            self.registers.reserve(additional_capacity);
        }

        for command in &executable.commands {
            match &command.extra {
                Command::Push(expr) => {
                    let expr_span = create_span_ref(command, ());
                    let expr_value =
                        self.execute_expr(expr_span, expr, executable, backtrace.as_deref_mut())?;
                    self.registers.push(expr_value);
                }

                Command::Copy {
                    source,
                    destination,
                } => {
                    self.registers[*destination] = self.registers[*source].clone();
                }

                Command::TruncateRegisters(size) => {
                    self.registers.truncate(*size);
                }

                Command::Destructure {
                    source,
                    start_len,
                    end_len,
                    lvalue_len,
                } => {
                    let source = self.registers[*source].clone();
                    if let Value::Tuple(mut elements) = source {
                        if !lvalue_len.matches(elements.len()) {
                            let err = EvalError::TupleLenMismatch {
                                lhs: *lvalue_len,
                                rhs: elements.len(),
                                context: TupleLenMismatchContext::Assignment,
                            };
                            return Err(SpannedEvalError::new(command, err));
                        }

                        let mut tail = elements.split_off(*start_len);
                        self.registers.extend(elements);
                        let end = tail.split_off(tail.len() - *end_len);
                        self.registers.push(Value::Tuple(tail));
                        self.registers.extend(end);
                    } else {
                        let err = EvalError::CannotDestructure;
                        return Err(SpannedEvalError::new(command, err));
                    }
                }

                Command::Annotate { register, name } => {
                    self.vars.insert(*name, *register);
                }

                Command::StartInnerScope => {
                    debug_assert!(self.inner_scope_start.is_none());
                    self.inner_scope_start = Some(self.registers.len());
                }
                Command::EndInnerScope => {
                    debug_assert!(self.inner_scope_start.is_some());
                    self.inner_scope_start = None;
                }
            }
        }

        Ok(self.registers.pop().unwrap_or_else(Value::void))
    }

    fn execute_expr(
        &self,
        span: Span<'a>,
        expr: &CompiledExpr<'a, T>,
        executable: &Executable<'a, T>,
        mut backtrace: Option<&mut Backtrace<'a>>,
    ) -> EvalResult<'a, T> {
        match expr {
            CompiledExpr::Atom(atom) => Ok(self.resolve_atom(atom)),
            CompiledExpr::Tuple(atoms) => {
                let atoms = atoms.iter().map(|atom| self.resolve_atom(atom)).collect();
                Ok(Value::Tuple(atoms))
            }

            CompiledExpr::Unary { op, inner } => {
                let inner_value = self.resolve_atom(inner);
                match op {
                    UnaryOp::Neg => inner_value.try_neg(),
                    UnaryOp::Not => inner_value.try_not(),
                }
                .map_err(|err| SpannedEvalError::new(&span, err))
            }

            CompiledExpr::Binary { op, lhs, rhs } => {
                let lhs_value = create_span_ref(lhs, self.resolve_atom(lhs));
                let rhs_value = create_span_ref(rhs, self.resolve_atom(rhs));
                match op {
                    BinaryOp::Add => Value::try_add(span, lhs_value, rhs_value),
                    BinaryOp::Sub => Value::try_sub(span, lhs_value, rhs_value),
                    BinaryOp::Mul => Value::try_mul(span, lhs_value, rhs_value),
                    BinaryOp::Div => Value::try_div(span, lhs_value, rhs_value),
                    BinaryOp::Power => Value::try_pow(span, lhs_value, rhs_value),

                    BinaryOp::Eq => Ok(Value::Bool(lhs_value.extra.compare(&rhs_value.extra))),
                    BinaryOp::NotEq => Ok(Value::Bool(!lhs_value.extra.compare(&rhs_value.extra))),

                    BinaryOp::And => Value::try_and(&lhs_value, &rhs_value),
                    BinaryOp::Or => Value::try_or(&lhs_value, &rhs_value),
                }
            }

            CompiledExpr::Function { name, args } => {
                if let Value::Function(function) = self.resolve_atom(name) {
                    let arg_values = args
                        .iter()
                        .map(|arg| create_span_ref(arg, self.resolve_atom(arg)))
                        .collect();

                    if let Some(backtrace) = backtrace.as_deref_mut() {
                        // FIXME: distinguish between interpreted and native calls.
                        backtrace.push_call(name.fragment, function.def_span(), span);
                    }
                    let mut context =
                        CallContext::new(create_span_ref(name, ()), span, backtrace.as_deref_mut());

                    function.evaluate(arg_values, &mut context).map(|value| {
                        if let Some(backtrace) = backtrace {
                            backtrace.pop_call();
                        }
                        value
                    })
                } else {
                    Err(SpannedEvalError::new(&span, EvalError::CannotCall))
                }
            }

            CompiledExpr::DefineFunction { ptr, captures } => {
                let fn_executable = executable.child_fns[*ptr].clone();
                let captured_values: Vec<_> = captures
                    .iter()
                    .map(|capture| self.resolve_atom(capture))
                    .collect();
                let capture_spans: Vec<_> = captures
                    .iter()
                    .map(|capture| create_span_ref(capture, ()))
                    .collect();

                let function = InterpretedFn::new(fn_executable, captured_values, capture_spans);
                Ok(Value::interpreted_fn(function))
            }
        }
    }

    #[inline]
    fn resolve_atom(&self, atom: &SpannedAtom<'a, T>) -> Value<'a, T> {
        match &atom.extra {
            Atom::Register(index) => self.registers[*index].clone(),
            Atom::Constant(value) => Value::Number(value.clone()),
            Atom::Void => Value::void(),
        }
    }
}

/// Executable module together with its imports.
#[derive(Debug)]
pub struct ExecutableModule<'a, T: Grammar> {
    inner: Executable<'a, T>,
    imports: Env<'a, T>,
}

impl<'a, T: Grammar> ExecutableModule<'a, T> {
    /// Sets the value of an imported variable.
    pub fn set_import(&mut self, name: &str, value: Value<'a, T>) -> &mut Self {
        self.imports.set_var(name, value);
        self
    }

    pub(super) fn set_imports(&mut self, env: &Env<'a, T>) {
        self.imports = env.clone();
    }

    pub(super) fn inner(&self) -> &Executable<'a, T> {
        &self.inner
    }
}

impl<'a, T: Grammar> ExecutableModule<'a, T>
where
    T::Lit: Num + ops::Neg<Output = T::Lit> + Pow<T::Lit, Output = T::Lit>,
{
    pub fn run(&self) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        let mut backtrace = Backtrace::default();
        self.imports
            .clone()
            .execute(&self.inner, Some(&mut backtrace))
            .map_err(|err| ErrorWithBacktrace::new(err, backtrace))
    }
}

#[derive(Debug, Clone, Default)]
pub struct Compiler<'a> {
    vars_to_registers: HashMap<&'a str, usize>,
    scope_depth: usize,
    register_count: usize,
}

impl<'a> Compiler<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub(super) fn from_env<T: Grammar>(env: &Env<'a, T>) -> Self {
        Self {
            vars_to_registers: env.vars.clone(),
            register_count: env.registers.len(),
            scope_depth: 0,
        }
    }

    fn get_var(&self, name: &str) -> Option<usize> {
        self.vars_to_registers.get(name).copied()
    }

    fn push_assignment<T: Grammar, U>(
        &mut self,
        compiled: &mut Executable<'a, T>,
        rhs: CompiledExpr<'a, T>,
        rhs_span: &Spanned<'a, U>,
    ) -> usize {
        let register = self.register_count;
        let command = Command::Push(rhs);
        compiled.commands.push(create_span_ref(rhs_span, command));
        self.register_count += 1;
        register
    }

    fn compile_expr<T: Grammar>(
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
                    .map(|elem| self.compile_expr(executable, elem))
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
                    executable.commands.push(create_span_ref(&expr, command));
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
                    executable.commands.push(create_span_ref(expr, command));
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
                    executable.commands.push(create_span_ref(&expr, command));
                }
                executable.commands.push(create_span_ref(
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
                let mut validator = FnValidator::new();
                validator.eval_function(def, self)?;
                let captures = validator
                    .captures
                    .values()
                    .map(|capture| create_span_ref(capture, Atom::Register(capture.extra)))
                    .collect();
                let fn_executable = Self::compile_function(def, &validator.captures)?;
                let fn_executable = ExecutableFn {
                    inner: fn_executable,
                    def_span: create_span_ref(expr, ()),
                    arg_count: def.args.extra.len(),
                };

                let ptr = executable.child_fns.len();
                executable.child_fns.push(Rc::new(fn_executable));
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

    fn compile_function<T: Grammar>(
        def: &FnDefinition<'a, T>,
        captures: &HashMap<&'a str, Spanned<'a, usize>>,
    ) -> Result<Executable<'a, T>, SpannedEvalError<'a>> {
        // Allocate registers for captures.
        let mut this = Self::new();
        this.scope_depth = 1; // Disable generating variable annotations.

        for (i, &name) in captures.keys().enumerate() {
            this.vars_to_registers.insert(name, i);
        }
        this.register_count = captures.len() + 1; // one additional register for args

        let mut executable = Executable::new();
        let args_span = create_span_ref(&def.args, ());
        this.destructure(&mut executable, &def.args.extra, args_span, captures.len());

        for statement in &def.body.statements {
            this.compile_statement(&mut executable, statement)?;
        }
        if let Some(ref return_value) = def.body.return_value {
            this.compile_expr(&mut executable, return_value)?;
        }

        executable.register_capacity = this.register_count;
        Ok(executable)
    }

    fn compile_statement<T: Grammar>(
        &mut self,
        compiled: &mut Executable<'a, T>,
        statement: &SpannedStatement<'a, T>,
    ) -> Result<Option<SpannedAtom<'a, T>>, SpannedEvalError<'a>> {
        Ok(match &statement.extra {
            Statement::Expr(expr) => Some(self.compile_expr(compiled, expr)?),

            Statement::Assignment { lhs, rhs } => {
                let rhs = self.compile_expr(compiled, rhs)?;
                // Allocate the register for the constant if necessary.
                let rhs_register = match rhs.extra {
                    Atom::Constant(_) | Atom::Void => {
                        self.push_assignment(compiled, CompiledExpr::Atom(rhs), statement)
                    }
                    Atom::Register(register) => register,
                };
                self.assign(compiled, lhs, rhs_register);
                None
            }
        })
    }

    pub(super) fn compile_module<T: Grammar>(
        &mut self,
        block: &Block<'a, T>,
    ) -> Result<ExecutableModule<'a, T>, SpannedEvalError<'a>> {
        let mut executable = Executable::new();
        let empty_span = Span::new("");
        let last_atom = self
            .compile_block_inner(&mut executable, block)?
            .unwrap_or_else(|| create_span_ref(&empty_span, Atom::Void));

        // Push the last variable to a register to be popped during execution.
        self.push_assignment(&mut executable, CompiledExpr::Atom(last_atom), &empty_span);

        executable.register_capacity = self.register_count;
        Ok(ExecutableModule {
            inner: executable,
            imports: Env::new(),
        })
    }

    fn compile_block_inner<T: Grammar>(
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

    fn assign<T: Grammar>(
        &mut self,
        compiled: &mut Executable<'a, T>,
        lhs: &SpannedLvalue<'a, T::Type>,
        rhs_register: usize,
    ) {
        match &lhs.extra {
            Lvalue::Variable { .. } => {
                if lhs.fragment != "_" {
                    self.vars_to_registers.insert(lhs.fragment, rhs_register);

                    // It does not make sense to annotate vars in the inner scopes, since
                    // they cannot be accessed externally.
                    if self.scope_depth == 0 {
                        let command = Command::Annotate {
                            register: rhs_register,
                            name: lhs.fragment,
                        };
                        compiled.commands.push(create_span_ref(lhs, command));
                    }
                }
            }

            Lvalue::Tuple(destructure) => {
                let span = create_span_ref(&lhs, ());
                self.destructure(compiled, destructure, span, rhs_register);
            }
        }
    }

    fn destructure<T: Grammar>(
        &mut self,
        compiled: &mut Executable<'a, T>,
        destructure: &Destructure<'a, T::Type>,
        span: Span<'a>,
        rhs_register: usize,
    ) {
        let command = Command::Destructure {
            source: rhs_register,
            start_len: destructure.start.len(),
            end_len: destructure.end.len(),
            lvalue_len: destructure.len(),
        };
        compiled.commands.push(create_span_ref(&span, command));
        let start_register = self.register_count;
        self.register_count += destructure.start.len() + destructure.end.len() + 1;

        for (i, lvalue) in (start_register..).zip(&destructure.start) {
            self.assign(compiled, lvalue, i);
        }

        let start_register = start_register + destructure.start.len();
        if let Some(ref middle) = destructure.middle {
            if let Some(lvalue) = middle.extra.to_lvalue() {
                self.assign(compiled, &lvalue, start_register);
            }
        }

        let start_register = start_register + 1;
        for (i, lvalue) in (start_register..).zip(&destructure.end) {
            self.assign(compiled, lvalue, i);
        }
    }
}

/// Helper context for symbolic execution of a function body in order to determine
/// variables captured by the function.
#[derive(Debug)]
struct FnValidator<'a> {
    local_vars: Vec<HashMap<&'a str, Span<'a>>>,
    captures: HashMap<&'a str, Spanned<'a, usize>>,
}

impl<'a> FnValidator<'a> {
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
        context: &Compiler<'a>,
    ) -> Result<(), SpannedEvalError<'a>> {
        self.local_vars.push(HashMap::new());

        extract_vars(
            self.local_vars.last_mut().unwrap(),
            &definition.args.extra,
            RepeatedAssignmentContext::FnArgs,
        )?;
        for statement in &definition.body.statements {
            self.eval_statement(statement, context)?;
        }
        if let Some(ref return_expr) = definition.body.return_value {
            self.eval(return_expr, context)?;
        }

        // Remove local vars defined *within* the function.
        self.local_vars.pop();
        Ok(())
    }

    fn has_var(&self, var_name: &str) -> bool {
        self.captures.contains_key(var_name)
            || self.local_vars.iter().any(|set| set.contains_key(var_name))
    }

    /// Processes a local variable in the rvalue position.
    fn eval_local_var<T>(
        &mut self,
        var_span: &Spanned<'a, T>,
        context: &Compiler<'a>,
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
        context: &Compiler<'a>,
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
                for statement in &block.statements {
                    self.eval_statement(statement, context)?;
                }
                if let Some(ref return_expr) = block.return_value {
                    self.eval(return_expr, context)?;
                }
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
        context: &Compiler<'a>,
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
    use crate::{grammars::F32Grammar, GrammarExt, Span};

    #[test]
    fn compilation_basics() {
        let block = Span::new("x = 3; 1 + { y = 2; y * x } == 7");
        let block = F32Grammar::parse_statements(block).unwrap();
        let module = Compiler::new().compile_module(&block).unwrap();
        let value = module.run().unwrap();
        assert_eq!(value, Value::Bool(true));
    }

    #[test]
    fn compiled_function() {
        let block = Span::new("add = |x, y| x + y; add(2, 3) == 5");
        let block = F32Grammar::parse_statements(block).unwrap();
        let value = Compiler::new()
            .compile_module(&block)
            .unwrap()
            .run()
            .unwrap();
        assert_eq!(value, Value::Bool(true));
    }

    #[test]
    fn compiled_function_with_capture() {
        let block = "A = 2; add = |x, y| x + y / A; add(2, 3) == 3.5";
        let block = F32Grammar::parse_statements(Span::new(block)).unwrap();
        let value = Compiler::new()
            .compile_module(&block)
            .unwrap()
            .run()
            .unwrap();
        assert_eq!(value, Value::Bool(true));
    }

    #[test]
    fn iterative_evaluation() {
        let mut env = Env::new();
        env.push_var("x", Value::<F32Grammar>::Number(5.0));

        let block = F32Grammar::parse_statements(Span::new("x")).unwrap();
        let mut module = Compiler::from_env(&env).compile_module(&block).unwrap();
        assert_eq!(module.inner.register_capacity, 2);
        assert_eq!(module.inner.commands.len(), 1); // push `x` from r0 to r1
        module.imports = env;
        let value = module.run().unwrap();
        assert_eq!(value, Value::Number(5.0));
    }
}
