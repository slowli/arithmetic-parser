//! `Registers` for executing commands and closely related types.

use crate::{
    alloc::{vec, Box, HashMap, Rc, String, ToOwned, Vec},
    arith::OrdArithmetic,
    error::{Backtrace, EvalResult, LocationInModule, TupleLenMismatchContext},
    exec::command::{Atom, Command, CompiledExpr, FieldName, LocatedAtom, LocatedCommand},
    exec::ModuleId,
    CallContext, Environment, Error, ErrorKind, Function, InterpretedFn, SpannedValue, Value,
};
use arithmetic_parser::{BinaryOp, Location, LvalueLen, UnaryOp};

/// Sequence of instructions that can be executed with the `Registers`.
#[derive(Debug)]
pub(crate) struct Executable<T> {
    id: Box<dyn ModuleId>, // FIXME: consider using `Rc<_>`?
    commands: Vec<LocatedCommand<T>>,
    child_fns: Vec<Rc<ExecutableFn<T>>>,
    // Hint how many registers the executable requires.
    register_capacity: usize,
}

impl<T: Clone> Clone for Executable<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone_boxed(),
            commands: self.commands.clone(),
            child_fns: self.child_fns.clone(),
            register_capacity: self.register_capacity,
        }
    }
}

impl<T> Executable<T> {
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

    fn create_error<U>(&self, location: &Location<U>, err: ErrorKind) -> Error {
        Error::new(self.id.as_ref(), location, err)
    }

    pub fn push_command(&mut self, command: impl Into<LocatedCommand<T>>) {
        self.commands.push(command.into());
    }

    pub fn push_child_fn(&mut self, child_fn: ExecutableFn<T>) -> usize {
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

impl<T: 'static + Clone> Executable<T> {
    pub fn call_function(
        &self,
        captures: Vec<Value<T>>,
        args: Vec<Value<T>>,
        ctx: &mut CallContext<'_, T>,
    ) -> EvalResult<T> {
        let mut registers = captures;
        registers.push(Value::Tuple(args.into()));
        let mut env = Registers {
            registers,
            ..Registers::new()
        };
        let operations = Operations::from(ctx.arithmetic());
        env.execute(self, operations, ctx.backtrace())
    }
}

/// `Executable` together with function-specific info.
#[derive(Debug)]
pub(crate) struct ExecutableFn<T> {
    pub inner: Executable<T>,
    pub def_location: Location,
    pub arg_count: LvalueLen,
}

/// Encompasses all irreducible operations defined externally for `Value`s; for now, these are just
/// arithmetic ops.
#[derive(Debug)]
pub(crate) struct Operations<'r, T> {
    pub arithmetic: &'r dyn OrdArithmetic<T>,
}

impl<T> Clone for Operations<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Operations<'_, T> {}

impl<'r, T> From<&'r dyn OrdArithmetic<T>> for Operations<'r, T> {
    fn from(arithmetic: &'r dyn OrdArithmetic<T>) -> Self {
        Self { arithmetic }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Registers<T> {
    // TODO: restore `SmallVec` wrapped into a covariant wrapper.
    registers: Vec<Value<T>>,
    // Maps variables to registers. Variables are mapped only from the global scope;
    // thus, we don't need to remove them on error in an inner scope.
    // TODO: investigate using stack-hosted small strings for keys.
    vars: HashMap<String, usize>,
    // Marks the start of a first inner scope currently being evaluated. This is used
    // to quickly remove registers from the inner scopes on error.
    inner_scope_start: Option<usize>,
}

impl<T> Registers<T> {
    pub fn new() -> Self {
        Self {
            registers: vec![],
            vars: HashMap::new(),
            inner_scope_start: None,
        }
    }

    pub fn variables(&self) -> impl Iterator<Item = (&str, &Value<T>)> + '_ {
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

    /// Allocates a new register with the specified name if the name was not allocated previously.
    pub fn insert_var(&mut self, name: &str, value: Value<T>) -> bool {
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

impl<T: Clone> Registers<T> {
    /// Updates from the specified environment. Updates are performed in place.
    pub fn update_from_env(&mut self, env: &Environment<T>) {
        for (var_name, register) in &self.vars {
            if let Some(value) = env.get(var_name) {
                self.registers[*register] = value.clone();
            }
        }
    }

    /// Updates environment from this instance.
    pub fn update_env(&self, env: &mut Environment<T>) {
        for (var_name, register) in &self.vars {
            let value = self.registers[*register].clone();
            // ^-- We cannot move `value` from `registers` because multiple names may be pointing
            // to the same register.

            env.insert(var_name, value);
        }
    }
}

impl<T: 'static + Clone> Registers<T> {
    pub fn execute(
        &mut self,
        executable: &Executable<T>,
        operations: Operations<'_, T>,
        backtrace: Option<&mut Backtrace>,
    ) -> EvalResult<T> {
        self.execute_inner(executable, operations, backtrace)
            .map_err(|err| {
                if let Some(scope_start) = self.inner_scope_start.take() {
                    self.registers.truncate(scope_start);
                }
                err
            })
    }

    #[allow(clippy::needless_option_as_deref)] // false positive
    fn execute_inner(
        &mut self,
        executable: &Executable<T>,
        operations: Operations<'_, T>,
        mut backtrace: Option<&mut Backtrace>,
    ) -> EvalResult<T> {
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
                        operations,
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
                    if let Value::Tuple(elements) = source {
                        if !*unchecked && !lvalue_len.matches(elements.len()) {
                            let err = ErrorKind::TupleLenMismatch {
                                lhs: *lvalue_len,
                                rhs: elements.len(),
                                context: TupleLenMismatchContext::Assignment,
                            };
                            return Err(executable.create_error(command, err));
                        }

                        let mut elements = Vec::from(elements);
                        let mut tail = elements.split_off(*start_len);
                        self.registers.extend(elements);
                        let end = tail.split_off(tail.len() - *end_len);
                        self.registers.push(Value::Tuple(tail.into()));
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
        location: Location,
        expr: &CompiledExpr<T>,
        executable: &Executable<T>,
        operations: Operations<'_, T>,
        backtrace: Option<&mut Backtrace>,
    ) -> EvalResult<T> {
        match expr {
            CompiledExpr::Atom(atom) => Ok(self.resolve_atom(atom)),

            CompiledExpr::Tuple(atoms) => {
                let values = atoms.iter().map(|atom| self.resolve_atom(atom)).collect();
                Ok(Value::Tuple(values))
            }
            CompiledExpr::Object(fields) => {
                let fields = fields
                    .iter()
                    .map(|(name, atom)| (name.clone(), self.resolve_atom(atom)));
                Ok(Value::Object(fields.collect()))
            }

            CompiledExpr::Unary { op, inner } => {
                let inner_value = self.resolve_atom(&inner.extra);
                match op {
                    UnaryOp::Neg => inner_value.try_neg(operations.arithmetic),
                    UnaryOp::Not => inner_value.try_not(),
                    _ => unreachable!("Checked during compilation"),
                }
                .map_err(|err| executable.create_error(&location, err))
            }

            CompiledExpr::Binary { op, lhs, rhs } => {
                let arith = operations.arithmetic;
                self.execute_binary_expr(executable.id(), location, *op, lhs, rhs, arith)
            }

            CompiledExpr::FieldAccess {
                receiver,
                field: FieldName::Index(index),
            } => self
                .access_index_field(&receiver.extra, *index)
                .map_err(|err| executable.create_error(&location, err)),

            CompiledExpr::FieldAccess {
                receiver,
                field: FieldName::Name(name),
            } => self
                .access_named_field(&receiver.extra, name)
                .map_err(|err| executable.create_error(&location, err)),

            CompiledExpr::FunctionCall {
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
                        location,
                        arg_values,
                        operations,
                        backtrace,
                    )
                } else {
                    Err(executable.create_error(&location, ErrorKind::CannotCall))
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

    fn execute_binary_expr(
        &self,
        module_id: &dyn ModuleId,
        location: Location,
        op: BinaryOp,
        lhs: &LocatedAtom<T>,
        rhs: &LocatedAtom<T>,
        arithmetic: &dyn OrdArithmetic<T>,
    ) -> EvalResult<T> {
        let lhs_value = lhs.copy_with_extra(self.resolve_atom(&lhs.extra));
        let rhs_value = rhs.copy_with_extra(self.resolve_atom(&rhs.extra));

        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Power => {
                Value::try_binary_op(module_id, location, lhs_value, rhs_value, op, arithmetic)
            }

            BinaryOp::Eq | BinaryOp::NotEq => {
                let is_eq = lhs_value
                    .extra
                    .eq_by_arithmetic(&rhs_value.extra, arithmetic);
                Ok(Value::Bool(if op == BinaryOp::Eq { is_eq } else { !is_eq }))
            }

            BinaryOp::And => Value::try_and(module_id, &lhs_value, &rhs_value),
            BinaryOp::Or => Value::try_or(module_id, &lhs_value, &rhs_value),

            BinaryOp::Gt | BinaryOp::Lt | BinaryOp::Ge | BinaryOp::Le => {
                Value::compare(module_id, &lhs_value, &rhs_value, op, arithmetic)
            }

            _ => unreachable!("Checked during compilation"),
        }
    }

    fn access_index_field(&self, receiver: &Atom<T>, index: usize) -> Result<Value<T>, ErrorKind> {
        let receiver = match receiver {
            Atom::Register(idx) => &self.registers[*idx],
            Atom::Constant(_) => {
                return Err(ErrorKind::CannotIndex);
            }
            Atom::Void => {
                return Err(ErrorKind::IndexOutOfBounds { index, len: 0 });
            }
        };

        if let Value::Tuple(tuple) = receiver {
            let len = tuple.len();
            if index >= len {
                Err(ErrorKind::IndexOutOfBounds { index, len })
            } else {
                Ok(tuple[index].clone())
            }
        } else {
            Err(ErrorKind::CannotIndex)
        }
    }

    fn access_named_field(&self, receiver: &Atom<T>, name: &str) -> Result<Value<T>, ErrorKind> {
        let Atom::Register(idx) = receiver else {
            return Err(ErrorKind::CannotAccessFields);
        };
        let receiver = &self.registers[*idx];
        let object = receiver.as_object().ok_or(ErrorKind::CannotAccessFields)?;
        object.get(name).cloned().ok_or_else(|| ErrorKind::NoField {
            field: name.to_owned(),
            available_fields: object.field_names().map(String::from).collect(),
        })
    }

    #[allow(clippy::needless_option_as_deref)] // false positive
    fn eval_function(
        function: &Function<T>,
        fn_name: &str,
        module_id: &dyn ModuleId,
        call_location: Location,
        arg_values: Vec<SpannedValue<T>>,
        operations: Operations<'_, T>,
        mut backtrace: Option<&mut Backtrace>,
    ) -> EvalResult<T> {
        let full_call_span = LocationInModule::new(module_id, call_location);
        if let Some(backtrace) = &mut backtrace {
            backtrace.push_call(fn_name, function.def_location(), full_call_span.clone());
        }
        let mut context = CallContext::new(full_call_span, backtrace.as_deref_mut(), operations);

        function.evaluate(arg_values, &mut context).map(|value| {
            if let Some(backtrace) = backtrace {
                backtrace.pop_call();
            }
            value
        })
    }

    #[inline]
    fn resolve_atom(&self, atom: &Atom<T>) -> Value<T> {
        match atom {
            Atom::Register(index) => self.registers[*index].clone(),
            Atom::Constant(value) => Value::Prim(value.clone()),
            Atom::Void => Value::void(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compiler::Compiler, exec::WildcardId};
    use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};

    #[test]
    fn iterative_evaluation() {
        let block = Untyped::<F32Grammar>::parse_statements("x").unwrap();
        let module = Compiler::compile_module(WildcardId, &block).unwrap();
        assert_eq!(module.inner.register_capacity, 2);
        assert_eq!(module.inner.commands.len(), 1); // push `x` from r0 to r1

        let mut env = Environment::new();
        env.insert("x", Value::Prim(5.0));
        let value = module.with_env(&env).unwrap().run().unwrap();
        assert_eq!(value, Value::Prim(5.0));
    }
}
