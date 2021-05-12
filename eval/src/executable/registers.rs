//! `Registers` for executing commands and closely related types.

use hashbrown::HashMap;

use crate::{
    alloc::{vec, Box, Rc, String, ToOwned, Vec},
    arith::OrdArithmetic,
    error::{Backtrace, CodeInModule, EvalResult, TupleLenMismatchContext},
    executable::command::{Atom, Command, CompiledExpr, SpannedCommand},
    CallContext, Environment, Error, ErrorKind, Function, InterpretedFn, ModuleId, SpannedValue,
    Value,
};
use arithmetic_parser::{BinaryOp, LvalueLen, MaybeSpanned, StripCode, UnaryOp};

/// Sequence of instructions that can be executed with the `Registers`.
#[derive(Debug)]
pub(crate) struct Executable<'a, T> {
    id: Box<dyn ModuleId>,
    commands: Vec<SpannedCommand<'a, T>>,
    child_fns: Vec<Rc<ExecutableFn<'a, T>>>,
    // Hint how many registers the executable requires.
    register_capacity: usize,
}

impl<'a, T: Clone> Clone for Executable<'a, T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone_boxed(),
            commands: self.commands.clone(),
            child_fns: self.child_fns.clone(),
            register_capacity: self.register_capacity,
        }
    }
}

impl<T: 'static + Clone> StripCode for Executable<'_, T> {
    type Stripped = Executable<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        Executable {
            id: self.id,
            commands: self
                .commands
                .into_iter()
                .map(|command| command.map_extra(StripCode::strip_code).strip_code())
                .collect(),
            child_fns: self
                .child_fns
                .into_iter()
                .map(|function| Rc::new(function.to_stripped_code()))
                .collect(),
            register_capacity: self.register_capacity,
        }
    }
}

impl<'a, T> Executable<'a, T> {
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

    fn create_error<U>(&self, span: &MaybeSpanned<'a, U>, err: ErrorKind) -> Error<'a> {
        Error::new(self.id.as_ref(), span, err)
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

impl<'a, T: Clone> Executable<'a, T> {
    pub fn call_function(
        &self,
        captures: Vec<Value<'a, T>>,
        args: Vec<Value<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        let mut registers = captures;
        registers.push(Value::Tuple(args));
        let mut env = Registers {
            registers,
            ..Registers::new()
        };
        env.execute(self, ctx.arithmetic(), ctx.backtrace())
    }
}

/// `Executable` together with function-specific info.
#[derive(Debug)]
pub(crate) struct ExecutableFn<'a, T> {
    pub inner: Executable<'a, T>,
    pub def_span: MaybeSpanned<'a>,
    pub arg_count: LvalueLen,
}

impl<T: 'static + Clone> ExecutableFn<'_, T> {
    pub fn to_stripped_code(&self) -> ExecutableFn<'static, T> {
        ExecutableFn {
            inner: self.inner.clone().strip_code(),
            def_span: self.def_span.strip_code(),
            arg_count: self.arg_count,
        }
    }
}

impl<T: 'static + Clone> StripCode for ExecutableFn<'_, T> {
    type Stripped = ExecutableFn<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        ExecutableFn {
            inner: self.inner.strip_code(),
            def_span: self.def_span.strip_code(),
            arg_count: self.arg_count,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Registers<'a, T> {
    // TODO: restore `SmallVec` wrapped into a covariant wrapper.
    registers: Vec<Value<'a, T>>,
    // Maps variables to registers. Variables are mapped only from the global scope;
    // thus, we don't need to remove them on error in an inner scope.
    // TODO: investigate using stack-hosted small strings for keys.
    vars: HashMap<String, usize>,
    // Marks the start of a first inner scope currently being evaluated. This is used
    // to quickly remove registers from the inner scopes on error.
    inner_scope_start: Option<usize>,
}

impl<T: Clone> Clone for Registers<'_, T> {
    fn clone(&self) -> Self {
        Self {
            registers: self.registers.clone(),
            vars: self.vars.clone(),
            inner_scope_start: self.inner_scope_start,
        }
    }
}

impl<T: 'static + Clone> StripCode for Registers<'_, T> {
    type Stripped = Registers<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        Registers {
            registers: self
                .registers
                .into_iter()
                .map(StripCode::strip_code)
                .collect(),
            vars: self.vars,
            inner_scope_start: self.inner_scope_start,
        }
    }
}

impl<'a, T> Registers<'a, T> {
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

    /// Allocates a new register with the specified name if the name was not allocated previously.
    pub fn insert_var(&mut self, name: &str, value: Value<'a, T>) -> bool {
        if self.vars.contains_key(name) {
            false
        } else {
            let register = self.registers.len();
            self.registers.push(value);
            self.vars.insert(name.to_owned(), register);

            true
        }
    }
}

impl<'a, T: Clone> Registers<'a, T> {
    /// Updates from the specified environment. Updates are performed in place.
    pub fn update_from_env(&mut self, env: &Environment<'a, T>) {
        for (var_name, register) in &self.vars {
            if let Some(value) = env.get(var_name) {
                self.registers[*register] = value.clone();
            }
        }
    }

    /// Updates environment from this instance.
    pub fn update_env(&self, env: &mut Environment<'a, T>) {
        for (var_name, register) in &self.vars {
            let value = self.registers[*register].clone();
            // ^-- We cannot move `value` from `registers` because multiple names may be pointing
            // to the same register.

            env.insert(var_name, value);
        }
    }

    pub fn into_variables(self) -> impl Iterator<Item = (String, Value<'a, T>)> {
        let registers = self.registers;
        // Moving out of `registers` is not sound because of possible aliasing.
        self.vars
            .into_iter()
            .map(move |(name, register)| (name, registers[register].clone()))
    }
}

impl<'a, T: Clone> Registers<'a, T> {
    pub fn execute(
        &mut self,
        executable: &Executable<'a, T>,
        arithmetic: &dyn OrdArithmetic<T>,
        backtrace: Option<&mut Backtrace<'a>>,
    ) -> EvalResult<'a, T> {
        self.execute_inner(executable, arithmetic, backtrace)
            .map_err(|err| {
                if let Some(scope_start) = self.inner_scope_start.take() {
                    self.registers.truncate(scope_start);
                }
                err
            })
    }

    fn execute_inner(
        &mut self,
        executable: &Executable<'a, T>,
        arithmetic: &dyn OrdArithmetic<T>,
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
                    let expr_value = self.execute_expr(
                        expr_span,
                        expr,
                        executable,
                        arithmetic,
                        backtrace.as_deref_mut(),
                    )?;
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
                            let err = ErrorKind::TupleLenMismatch {
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
                        let err = ErrorKind::CannotDestructure;
                        return Err(executable.create_error(command, err));
                    }
                }

                Command::Annotate { register, name } => {
                    self.vars.insert(name.clone(), *register);
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
        arithmetic: &dyn OrdArithmetic<T>,
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
                    UnaryOp::Neg => inner_value.try_neg(arithmetic),
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
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Power => {
                        Value::try_binary_op(module_id, span, lhs_value, rhs_value, *op, arithmetic)
                    }

                    BinaryOp::Eq | BinaryOp::NotEq => {
                        let is_eq = lhs_value
                            .extra
                            .eq_by_arithmetic(&rhs_value.extra, arithmetic);
                        Ok(Value::Bool(if *op == BinaryOp::Eq {
                            is_eq
                        } else {
                            !is_eq
                        }))
                    }

                    BinaryOp::And => Value::try_and(module_id, &lhs_value, &rhs_value),
                    BinaryOp::Or => Value::try_or(module_id, &lhs_value, &rhs_value),

                    BinaryOp::Gt | BinaryOp::Lt | BinaryOp::Ge | BinaryOp::Le => {
                        Value::compare(module_id, &lhs_value, &rhs_value, *op, arithmetic)
                    }

                    _ => unreachable!("Checked during compilation"),
                }
            }

            CompiledExpr::FieldAccess { receiver, index } => {
                if let Value::Tuple(mut tuple) = self.resolve_atom(&receiver.extra) {
                    let len = tuple.len();
                    if *index >= len {
                        Err(executable.create_error(
                            &span,
                            ErrorKind::IndexOutOfBounds { index: *index, len },
                        ))
                    } else {
                        Ok(tuple.swap_remove(*index))
                    }
                } else {
                    Err(executable.create_error(&span, ErrorKind::CannotIndex))
                }
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
                        arithmetic,
                        backtrace,
                    )
                } else {
                    Err(executable.create_error(&span, ErrorKind::CannotCall))
                }
            }

            CompiledExpr::DefineFunction {
                ptr,
                captures,
                capture_names,
            } => {
                let fn_executable = Rc::clone(&executable.child_fns[*ptr]);
                let captured_values = captures
                    .iter()
                    .map(|capture| self.resolve_atom(&capture.extra))
                    .collect();

                let function =
                    InterpretedFn::new(fn_executable, captured_values, capture_names.clone());
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
        arithmetic: &dyn OrdArithmetic<T>,
        mut backtrace: Option<&mut Backtrace<'a>>,
    ) -> EvalResult<'a, T> {
        let full_call_span = CodeInModule::new(module_id, call_span);
        if let Some(backtrace) = backtrace.as_deref_mut() {
            backtrace.push_call(fn_name, function.def_span(), full_call_span.clone());
        }
        let mut context = CallContext::new(full_call_span, backtrace.as_deref_mut(), arithmetic);

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
            Atom::Constant(value) => Value::Number(value.clone()),
            Atom::Void => Value::void(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compiler::Compiler, executable::ModuleImports, WildcardId};
    use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};

    #[test]
    fn iterative_evaluation() {
        let block = Untyped::<F32Grammar>::parse_statements("x").unwrap();
        let (mut module, _) = Compiler::compile_module(WildcardId, &block).unwrap();
        assert_eq!(module.inner.register_capacity, 2);
        assert_eq!(module.inner.commands.len(), 1); // push `x` from r0 to r1

        let mut env = Registers::new();
        env.insert_var("x", Value::Number(5.0));
        module.imports = ModuleImports { inner: env };
        let value = module.run().unwrap();
        assert_eq!(value, Value::Number(5.0));
    }
}
