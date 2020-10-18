//! Execution `Env` and closely related types.

use hashbrown::HashMap;

use crate::error::CodeInModule;
use crate::{
    alloc::{vec, Rc, Vec},
    executable::{
        command::{Atom, Command, CompiledExpr, SpannedCommand},
        ExecutableFn,
    },
    Backtrace, CallContext, EvalError, EvalResult, Function, InterpretedFn, ModuleId, Number,
    SpannedEvalError, SpannedValue, TupleLenMismatchContext, Value,
};
use arithmetic_parser::{BinaryOp, Grammar, MaybeSpanned, StripCode, UnaryOp};

#[derive(Debug)]
pub(crate) struct Executable<'a, T: Grammar> {
    id: Box<dyn ModuleId>,
    commands: Vec<SpannedCommand<'a, T>>,
    child_fns: Vec<Rc<ExecutableFn<'a, T>>>,
    // Hint how many registers the executable requires.
    register_capacity: usize,
}

impl<'a, T: Grammar> Clone for Executable<'a, T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone_boxed(),
            commands: self.commands.clone(),
            child_fns: self.child_fns.clone(),
            register_capacity: self.register_capacity,
        }
    }
}

impl<T: Grammar> StripCode for Executable<'_, T> {
    type Stripped = Executable<'static, T>;

    fn strip_code(&self) -> Self::Stripped {
        Executable {
            id: self.id.clone_boxed(),
            commands: self
                .commands
                .iter()
                .map(|command| {
                    command
                        .copy_with_extra(command.extra.strip_code())
                        .strip_code()
                })
                .collect(),
            child_fns: self
                .child_fns
                .iter()
                .map(|function| Rc::new(function.strip_code()))
                .collect(),
            register_capacity: self.register_capacity,
        }
    }
}

impl<'a, T: Grammar> Executable<'a, T> {
    pub fn new(id: Box<dyn ModuleId>) -> Self {
        Self {
            id,
            commands: vec![],
            child_fns: vec![],
            register_capacity: 0,
        }
    }

    pub fn id(&self) -> &dyn ModuleId {
        self.id.as_ref()
    }

    fn create_error<U>(&self, span: &MaybeSpanned<'a, U>, err: EvalError) -> SpannedEvalError<'a> {
        SpannedEvalError::new(self.id.as_ref(), span, err)
    }

    pub fn push_command(&mut self, command: impl Into<SpannedCommand<'a, T>>) {
        self.commands.push(command.into());
    }

    pub fn push_child_fn(&mut self, child_fn: ExecutableFn<'a, T>) -> usize {
        let fn_ptr = self.child_fns.len();
        self.child_fns.push(Rc::new(child_fn));
        fn_ptr
    }

    pub fn finalize_function(&mut self, register_count: usize) {
        // We check number of arguments in `InterpretedFn::evaluate()` in order to provide
        // a more precise error.
        match &mut self.commands[0].extra {
            Command::Destructure { unchecked, .. } => {
                *unchecked = true;
            }
            _ => unreachable!(),
        }
        self.register_capacity = register_count;
    }

    pub fn finalize_block(&mut self, register_count: usize) {
        self.register_capacity = register_count;
    }
}

impl<'a, T> Executable<'a, T>
where
    T: Grammar,
    T::Lit: Number,
{
    pub fn call_function(
        &self,
        captures: Vec<Value<'a, T>>,
        args: Vec<Value<'a, T>>,
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        let mut registers = captures;
        registers.push(Value::Tuple(args));
        let mut env = Env {
            registers,
            ..Env::new()
        };
        env.execute(self, ctx.backtrace())
    }
}

// TODO: restore `SmallVec` wrapped into a covariant wrapper.
type Registers<'a, T> = Vec<Value<'a, T>>;

#[derive(Debug)]
pub(crate) struct Env<'a, T: Grammar> {
    registers: Registers<'a, T>,
    // Maps variables to registers. Variables are mapped only from the global scope;
    // thus, we don't need to remove them on error in an inner scope.
    // TODO: investigate using stack-hosted small strings for keys.
    vars: HashMap<String, usize>,
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

impl<T: Grammar> StripCode for Env<'_, T> {
    type Stripped = Env<'static, T>;

    fn strip_code(&self) -> Self::Stripped {
        Env {
            registers: self.registers.iter().map(StripCode::strip_code).collect(),
            vars: self.vars.clone(),
            inner_scope_start: self.inner_scope_start,
        }
    }
}

impl<'a, T: Grammar> Env<'a, T> {
    pub fn new() -> Self {
        Self {
            registers: vec![],
            vars: HashMap::new(),
            inner_scope_start: None,
        }
    }

    pub fn get_var(&self, name: &str) -> Option<&Value<'a, T>> {
        let register = *self.vars.get(name)?;
        Some(&self.registers[register])
    }

    pub fn get_var_mut(&mut self, name: &str) -> Option<&mut Value<'a, T>> {
        let register = *self.vars.get(name)?;
        Some(&mut self.registers[register])
    }

    pub fn variables(&self) -> impl Iterator<Item = (&str, &Value<'a, T>)> + '_ {
        self.vars
            .iter()
            .map(move |(name, register)| (name.as_str(), &self.registers[*register]))
    }

    pub fn variables_map(&self) -> &HashMap<String, usize> {
        &self.vars
    }

    pub fn register_count(&self) -> usize {
        self.registers.len()
    }

    pub fn set_var(&mut self, name: &str, value: Value<'a, T>) {
        let register = *self.vars.get(name).unwrap_or_else(|| {
            panic!("Variable `{}` is not defined", name);
        });
        self.registers[register] = value;
    }

    /// Allocates a new register with the specified name. If the name was previously assigned,
    /// the name association is updated, but the old register itself remains intact.
    pub fn push_var(&mut self, name: &str, value: Value<'a, T>) {
        let register = self.registers.len();
        self.registers.push(value);
        self.vars.insert(name.to_owned(), register);
    }

    /// Retains only registers corresponding to named variables.
    pub fn compress(&mut self) {
        let mut registers = Vec::with_capacity(self.vars.len());
        for (i, register) in self.vars.values_mut().enumerate() {
            registers.push(self.registers[*register].clone());
            *register = i;
        }
        self.registers = registers;
    }
}

impl<'a, T> Env<'a, T>
where
    T: Grammar,
    T::Lit: Number,
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
                    let expr_span = command.with_no_extra();
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
                    unchecked,
                } => {
                    let source = self.registers[*source].clone();
                    if let Value::Tuple(mut elements) = source {
                        if !*unchecked && !lvalue_len.matches(elements.len()) {
                            let err = EvalError::TupleLenMismatch {
                                lhs: *lvalue_len,
                                rhs: elements.len(),
                                context: TupleLenMismatchContext::Assignment,
                            };
                            return Err(executable.create_error(command, err));
                        }

                        let mut tail = elements.split_off(*start_len);
                        self.registers.extend(elements);
                        let end = tail.split_off(tail.len() - *end_len);
                        self.registers.push(Value::Tuple(tail));
                        self.registers.extend(end);
                    } else {
                        let err = EvalError::CannotDestructure;
                        return Err(executable.create_error(command, err));
                    }
                }

                Command::Annotate { register, name } => {
                    self.vars.insert((*name).to_owned(), *register);
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
        span: MaybeSpanned<'a>,
        expr: &CompiledExpr<'a, T>,
        executable: &Executable<'a, T>,
        backtrace: Option<&mut Backtrace<'a>>,
    ) -> EvalResult<'a, T> {
        match expr {
            CompiledExpr::Atom(atom) => Ok(self.resolve_atom(atom)),
            CompiledExpr::Tuple(atoms) => {
                let values = atoms.iter().map(|atom| self.resolve_atom(atom)).collect();
                Ok(Value::Tuple(values))
            }

            CompiledExpr::Unary { op, inner } => {
                let inner_value = self.resolve_atom(&inner.extra);
                match op {
                    UnaryOp::Neg => inner_value.try_neg(),
                    UnaryOp::Not => inner_value.try_not(),
                    _ => unreachable!("Checked during compilation"),
                }
                .map_err(|err| executable.create_error(&span, err))
            }

            CompiledExpr::Binary { op, lhs, rhs } => {
                let lhs_value = lhs.copy_with_extra(self.resolve_atom(&lhs.extra));
                let rhs_value = rhs.copy_with_extra(self.resolve_atom(&rhs.extra));
                let module_id = executable.id();

                match op {
                    BinaryOp::Add => Value::try_add(module_id, span, lhs_value, rhs_value),
                    BinaryOp::Sub => Value::try_sub(module_id, span, lhs_value, rhs_value),
                    BinaryOp::Mul => Value::try_mul(module_id, span, lhs_value, rhs_value),
                    BinaryOp::Div => Value::try_div(module_id, span, lhs_value, rhs_value),
                    BinaryOp::Power => Value::try_pow(module_id, span, lhs_value, rhs_value),

                    BinaryOp::Eq => Ok(Value::Bool(lhs_value.extra == rhs_value.extra)),
                    BinaryOp::NotEq => Ok(Value::Bool(lhs_value.extra != rhs_value.extra)),

                    BinaryOp::And => Value::try_and(module_id, &lhs_value, &rhs_value),
                    BinaryOp::Or => Value::try_or(module_id, &lhs_value, &rhs_value),

                    BinaryOp::Gt | BinaryOp::Lt | BinaryOp::Ge | BinaryOp::Le => {
                        unreachable!("Must be desugared by the compiler")
                    }

                    _ => unreachable!("Checked during compilation"),
                }
            }

            CompiledExpr::Compare { inner, op } => {
                let inner_value = self.resolve_atom(&inner.extra);
                op.compare(&inner_value)
                    .map(Value::Bool)
                    .ok_or_else(|| executable.create_error(&span, EvalError::InvalidCmpResult))
            }

            CompiledExpr::Function {
                name,
                original_name,
                args,
            } => {
                if let Value::Function(function) = self.resolve_atom(&name.extra) {
                    let fn_name = original_name.as_deref().unwrap_or("(anonymous function)");
                    let arg_values = args
                        .iter()
                        .map(|arg| arg.copy_with_extra(self.resolve_atom(&arg.extra)))
                        .collect();
                    Self::eval_function(
                        &function,
                        fn_name,
                        executable.id.as_ref(),
                        span,
                        arg_values,
                        backtrace,
                    )
                } else {
                    Err(executable.create_error(&span, EvalError::CannotCall))
                }
            }

            CompiledExpr::DefineFunction { ptr, captures } => {
                let fn_executable = executable.child_fns[*ptr].clone();
                let captured_values = captures
                    .iter()
                    .map(|capture| self.resolve_atom(&capture.extra))
                    .collect();
                let capture_names = captures
                    .iter()
                    .map(|capture| capture.code_or_location("var"))
                    .collect();

                let function = InterpretedFn::new(fn_executable, captured_values, capture_names);
                Ok(Value::interpreted_fn(function))
            }
        }
    }

    fn eval_function(
        function: &Function<'a, T>,
        fn_name: &str,
        module_id: &dyn ModuleId,
        call_span: MaybeSpanned<'a>,
        arg_values: Vec<SpannedValue<'a, T>>,
        mut backtrace: Option<&mut Backtrace<'a>>,
    ) -> EvalResult<'a, T> {
        let full_call_span = CodeInModule::new(module_id, call_span);
        if let Some(backtrace) = backtrace.as_deref_mut() {
            backtrace.push_call(fn_name, function.def_span(), full_call_span.clone());
        }
        let mut context = CallContext::new(full_call_span, backtrace.as_deref_mut());

        function.evaluate(arg_values, &mut context).map(|value| {
            if let Some(backtrace) = backtrace {
                backtrace.pop_call();
            }
            value
        })
    }

    #[inline]
    fn resolve_atom(&self, atom: &Atom<T>) -> Value<'a, T> {
        match atom {
            Atom::Register(index) => self.registers[*index].clone(),
            Atom::Constant(value) => Value::Number(*value),
            Atom::Void => Value::void(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compiler::Compiler, executable::ModuleImports, WildcardId};
    use arithmetic_parser::{grammars::F32Grammar, GrammarExt, InputSpan};

    #[test]
    fn env_compression() {
        let mut env = Env::new();
        env.push_var("x", Value::<F32Grammar>::Number(5.0));

        let block = "y = x + 2 * (x + 1) + 1; y";
        let block = F32Grammar::parse_statements(InputSpan::new(block)).unwrap();
        let module = Compiler::compile_module(WildcardId, &env, &block, true).unwrap();
        let value = env.execute(&module.inner, None).unwrap();
        assert_eq!(value, Value::Number(18.0));

        assert!(env.registers.len() > 2);
        env.compress();
        assert_eq!(env.registers.len(), 2);
        assert_eq!(env.vars.len(), 2);
        assert!(env.vars.contains_key("x"));
        assert!(env.vars.contains_key("y"));
    }

    #[test]
    fn iterative_evaluation() {
        let mut env = Env::new();
        env.push_var("x", Value::<F32Grammar>::Number(5.0));

        let block = F32Grammar::parse_statements(InputSpan::new("x")).unwrap();
        let mut module = Compiler::compile_module(WildcardId, &env, &block, true).unwrap();
        assert_eq!(module.inner.register_capacity, 2);
        assert_eq!(module.inner.commands.len(), 1); // push `x` from r0 to r1
        module.imports = ModuleImports { inner: env };
        let value = module.run().unwrap();
        assert_eq!(value, Value::Number(5.0));
    }
}
