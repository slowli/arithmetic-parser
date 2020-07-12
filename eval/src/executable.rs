//! Executables output by a `Compiler` and related types.

use hashbrown::HashMap;
use num_traits::{Num, Pow};
use smallvec::{smallvec, SmallVec};

use core::ops;

use crate::{
    alloc::{vec, Rc, Vec},
    Backtrace, CallContext, ErrorWithBacktrace, EvalError, EvalResult, InterpretedFn,
    SpannedEvalError, TupleLenMismatchContext, Value,
};
use arithmetic_parser::{create_span_ref, BinaryOp, Grammar, LvalueLen, Span, Spanned, UnaryOp};

/// Pointer to a register or constant.
#[derive(Debug, Clone, Copy)]
pub enum Atom<T: Grammar> {
    Constant(T::Lit),
    Register(usize),
    Void,
}

pub type SpannedAtom<'a, T> = Spanned<'a, Atom<T>>;

/// Atomic operation on registers and/or constants.
#[derive(Debug, Clone)]
pub enum CompiledExpr<'a, T: Grammar> {
    Atom(Atom<T>),
    Tuple(Vec<Atom<T>>),
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

/// Commands for a primitive register VM used to execute compiled programs.
#[derive(Debug)]
pub enum Command<'a, T: Grammar> {
    /// Create a new register and push the result of the specified computation there.
    Push(CompiledExpr<'a, T>),

    /// Destructure a tuple value. This will push `start_len` starting elements from the tuple,
    /// the middle of the tuple (as a tuple), and `end_len` ending elements from the tuple
    /// as new registers, in this order.
    Destructure {
        /// Index of the register with the value.
        source: usize,
        /// Number of starting arguments to place in separate registers.
        start_len: usize,
        /// Number of ending arguments to place in separate registers.
        end_len: usize,
        /// Acceptable length(s) of the source.
        lvalue_len: LvalueLen,
        /// Does `lvalue_len` should be checked? When destructuring arguments for functions,
        /// this check was performed previously.
        unchecked: bool,
    },

    /// Copies the source register into the destination. The destination register must exist.
    Copy { source: usize, destination: usize },

    /// Annotates a register as containing the specified variable.
    Annotate { register: usize, name: &'a str },

    /// Signals that the following commands are executed in the inner scope.
    StartInnerScope,
    /// Signals that the following commands are executed in the global scope.
    EndInnerScope,
    /// Signals to truncate registers to the specified number.
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

impl<'a, T: Grammar> Executable<'a, T> {
    pub fn new() -> Self {
        Self {
            commands: vec![],
            child_fns: vec![],
            register_capacity: 0,
        }
    }

    pub fn push_command(&mut self, command: SpannedCommand<'a, T>) {
        self.commands.push(command);
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

type Registers<'a, T> = SmallVec<[Value<'a, T>; 16]>;

#[derive(Debug)]
pub(super) struct Env<'a, T: Grammar> {
    registers: Registers<'a, T>,
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

    pub fn variables_map(&self) -> HashMap<&'a str, usize> {
        self.vars.clone()
    }

    pub fn register_count(&self) -> usize {
        self.registers.len()
    }

    fn set_var(&mut self, name: &str, value: Value<'a, T>) {
        let register = *self.vars.get(name).unwrap_or_else(|| {
            panic!("Variable `{}` is not defined", name);
        });
        self.registers[register] = value;
    }

    /// Allocates a new register with the specified name. If the name was previously assigned,
    /// the name association is updated, but the old register itself remains intact.
    pub fn push_var(&mut self, name: &'a str, value: Value<'a, T>) {
        let register = self.registers.len();
        self.registers.push(value);
        self.vars.insert(name, register);
    }

    /// Retains only registers corresponding to named variables.
    pub fn compress(&mut self) {
        let mut registers = SmallVec::with_capacity(self.vars.len());
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
                let values = atoms.iter().map(|atom| self.resolve_atom(atom)).collect();
                Ok(Value::Tuple(values))
            }

            CompiledExpr::Unary { op, inner } => {
                let inner_value = self.resolve_atom(&inner.extra);
                match op {
                    UnaryOp::Neg => inner_value.try_neg(),
                    UnaryOp::Not => inner_value.try_not(),
                }
                .map_err(|err| SpannedEvalError::new(&span, err))
            }

            CompiledExpr::Binary { op, lhs, rhs } => {
                let lhs_value = create_span_ref(lhs, self.resolve_atom(&lhs.extra));
                let rhs_value = create_span_ref(rhs, self.resolve_atom(&rhs.extra));
                match op {
                    BinaryOp::Add => Value::try_add(span, lhs_value, rhs_value),
                    BinaryOp::Sub => Value::try_sub(span, lhs_value, rhs_value),
                    BinaryOp::Mul => Value::try_mul(span, lhs_value, rhs_value),
                    BinaryOp::Div => Value::try_div(span, lhs_value, rhs_value),
                    BinaryOp::Power => Value::try_pow(span, lhs_value, rhs_value),

                    BinaryOp::Eq => Ok(Value::Bool(lhs_value.extra == rhs_value.extra)),
                    BinaryOp::NotEq => Ok(Value::Bool(lhs_value.extra != rhs_value.extra)),

                    BinaryOp::And => Value::try_and(&lhs_value, &rhs_value),
                    BinaryOp::Or => Value::try_or(&lhs_value, &rhs_value),
                }
            }

            CompiledExpr::Function { name, args } => {
                if let Value::Function(function) = self.resolve_atom(&name.extra) {
                    let arg_values = args
                        .iter()
                        .map(|arg| create_span_ref(arg, self.resolve_atom(&arg.extra)))
                        .collect();

                    if let Some(backtrace) = backtrace.as_deref_mut() {
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
                let captured_values = captures
                    .iter()
                    .map(|capture| self.resolve_atom(&capture.extra))
                    .collect();
                let capture_spans = captures
                    .iter()
                    .map(|capture| create_span_ref(capture, ()))
                    .collect();

                let function = InterpretedFn::new(fn_executable, captured_values, capture_spans);
                Ok(Value::interpreted_fn(function))
            }
        }
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

/// Executable module together with its imports.
///
/// # Examples
///
/// ```
/// use arithmetic_parser::{grammars::F32Grammar, GrammarExt, Span};
/// use arithmetic_eval::{fns, Interpreter, Value};
/// # use std::{collections::HashSet, f32, iter::FromIterator};
///
/// let mut interpreter = Interpreter::new();
/// interpreter
///     .insert_native_fn(
///         "max",
///         fns::Binary::new(|x: f32, y: f32| if x > y { x } else { y }),
///     )
///     .insert_native_fn("fold", fns::Fold)
///     .insert_var("INFINITY", Value::Number(f32::INFINITY))
///     .insert_var("xs", Value::Tuple(vec![]));
///
/// let module = "xs.fold(-INFINITY, max)";
/// let module = F32Grammar::parse_statements(Span::new(module)).unwrap();
/// let mut module = interpreter.compile(&module).unwrap();
///
/// // With the original imports, the returned value is `-INFINITY`.
/// assert_eq!(module.run().unwrap(), Value::Number(f32::NEG_INFINITY));
///
/// // Imports can be changed. Let's check that `xs` is indeed an import.
/// let imports: HashSet<_> = module.imports().map(|(name, _)| name).collect();
/// assert!(imports.is_superset(&HashSet::from_iter(vec!["max", "fold", "xs"])));
///
/// // Change the `xs` import and run the module again.
/// let array = [1.0, -3.0, 2.0, 0.5].iter().copied().map(Value::Number).collect();
/// module.set_import("xs", Value::Tuple(array));
/// assert_eq!(module.run().unwrap(), Value::Number(2.0));
/// ```
#[derive(Debug)]
pub struct ExecutableModule<'a, T: Grammar> {
    inner: Executable<'a, T>,
    imports: Env<'a, T>,
}

impl<'a, T: Grammar> ExecutableModule<'a, T> {
    pub(super) fn new(inner: Executable<'a, T>, imports: Env<'a, T>) -> Self {
        Self { inner, imports }
    }

    /// Sets the value of an imported variable.
    ///
    /// # Panics
    ///
    /// Panics if the variable with the specified name is not an import.
    pub fn set_import(&mut self, name: &str, value: Value<'a, T>) -> &mut Self {
        self.imports.set_var(name, value);
        self
    }

    /// Enumerates imports of this module together with their current values.
    pub fn imports(&self) -> impl Iterator<Item = (&'a str, &Value<'a, T>)> + '_ {
        self.imports.variables()
    }

    pub(super) fn inner(&self) -> &Executable<'a, T> {
        &self.inner
    }
}

impl<'a, T: Grammar> ExecutableModule<'a, T>
where
    T::Lit: Num + ops::Neg<Output = T::Lit> + Pow<T::Lit, Output = T::Lit>,
{
    /// Runs the module with the current values of imports.
    pub fn run(&self) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        let mut backtrace = Backtrace::default();
        self.imports
            .clone()
            .execute(&self.inner, Some(&mut backtrace))
            .map_err(|err| ErrorWithBacktrace::new(err, backtrace))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::Compiler;

    use arithmetic_parser::{grammars::F32Grammar, GrammarExt};

    #[test]
    fn iterative_evaluation() {
        let mut env = Env::new();
        env.push_var("x", Value::<F32Grammar>::Number(5.0));

        let block = F32Grammar::parse_statements(Span::new("x")).unwrap();
        let mut module = Compiler::compile_module(&env, &block, true).unwrap();
        assert_eq!(module.inner.register_capacity, 2);
        assert_eq!(module.inner.commands.len(), 1); // push `x` from r0 to r1
        module.imports = env;
        let value = module.run().unwrap();
        assert_eq!(value, Value::Number(5.0));
    }

    #[test]
    fn env_compression() {
        let mut env = Env::new();
        env.push_var("x", Value::<F32Grammar>::Number(5.0));

        let block = "y = x + 2 * (x + 1) + 1; y";
        let block = F32Grammar::parse_statements(Span::new(block)).unwrap();
        let module = Compiler::compile_module(&env, &block, true).unwrap();
        let value = env.execute(&module.inner, None).unwrap();
        assert_eq!(value, Value::Number(18.0));

        assert!(env.registers.len() > 2);
        env.compress();
        assert_eq!(env.registers.len(), 2);
        assert_eq!(env.vars.len(), 2);
        assert!(env.vars.contains_key("x"));
        assert!(env.vars.contains_key("y"));
    }
}
