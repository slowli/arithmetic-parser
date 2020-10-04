//! Executables output by a `Compiler` and related types.

use hashbrown::HashMap;
use smallvec::{smallvec, SmallVec};

use core::{cmp::Ordering, ops};

use crate::{
    alloc::{vec, Rc, Vec},
    Backtrace, CallContext, ErrorWithBacktrace, EvalError, EvalResult, Function, InterpretedFn,
    Number, SpannedEvalError, SpannedValue, TupleLenMismatchContext, Value,
};
use arithmetic_parser::{create_span_ref, BinaryOp, Grammar, LvalueLen, Span, Spanned, UnaryOp};
use num_traits::{One, Zero};

/// Pointer to a register or constant.
#[derive(Debug)]
pub enum Atom<T: Grammar> {
    Constant(T::Lit),
    Register(usize),
    Void,
}

impl<T: Grammar> Clone for Atom<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Constant(literal) => Self::Constant(literal.clone()),
            Self::Register(index) => Self::Register(*index),
            Self::Void => Self::Void,
        }
    }
}

pub type SpannedAtom<'a, T> = Spanned<'a, Atom<T>>;

/// Atomic operation on registers and/or constants.
#[derive(Debug)]
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
    Compare {
        inner: SpannedAtom<'a, T>,
        op: ComparisonOp,
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

impl<T: Grammar> Clone for CompiledExpr<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Atom(atom) => Self::Atom(atom.clone()),
            Self::Tuple(atoms) => Self::Tuple(atoms.clone()),

            Self::Unary { op, inner } => Self::Unary {
                op: *op,
                inner: inner.clone(),
            },

            Self::Binary { op, lhs, rhs } => Self::Binary {
                op: *op,
                lhs: lhs.clone(),
                rhs: rhs.clone(),
            },

            Self::Compare { inner, op } => Self::Compare {
                inner: inner.clone(),
                op: *op,
            },

            Self::Function { name, args } => Self::Function {
                name: name.clone(),
                args: args.clone(),
            },

            Self::DefineFunction { ptr, captures } => Self::DefineFunction {
                ptr: *ptr,
                captures: captures.clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ComparisonOp {
    Gt,
    Lt,
    Ge,
    Le,
}

impl ComparisonOp {
    pub fn from(op: BinaryOp) -> Self {
        match op {
            BinaryOp::Gt => Self::Gt,
            BinaryOp::Lt => Self::Lt,
            BinaryOp::Ge => Self::Ge,
            BinaryOp::Le => Self::Le,
            _ => unreachable!("Never called with other variants"),
        }
    }

    fn compare<T>(self, cmp_value: Value<'_, T>) -> Option<bool>
    where
        T: Grammar,
        T::Lit: Number,
    {
        let ordering = match cmp_value {
            Value::Number(num) if num.is_one() => Ordering::Greater,
            Value::Number(num) if num.is_zero() => Ordering::Equal,
            Value::Number(num) if (-num).is_one() => Ordering::Less,
            _ => return None,
        };
        Some(match self {
            Self::Gt => ordering == Ordering::Greater,
            Self::Lt => ordering == Ordering::Less,
            Self::Ge => ordering != Ordering::Less,
            Self::Le => ordering != Ordering::Greater,
        })
    }
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

impl<T: Grammar> Clone for Command<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Push(expr) => Self::Push(expr.clone()),

            Self::Destructure {
                source,
                start_len,
                end_len,
                lvalue_len,
                unchecked,
            } => Self::Destructure {
                source: *source,
                start_len: *start_len,
                end_len: *end_len,
                lvalue_len: *lvalue_len,
                unchecked: *unchecked,
            },

            Self::Copy {
                source,
                destination,
            } => Self::Copy {
                source: *source,
                destination: *destination,
            },

            Self::Annotate { register, name } => Self::Annotate {
                register: *register,
                name,
            },

            Self::StartInnerScope => Self::StartInnerScope,
            Self::EndInnerScope => Self::EndInnerScope,
            Self::TruncateRegisters(size) => Self::TruncateRegisters(*size),
        }
    }
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

impl<'a, T: Grammar> Clone for Executable<'a, T> {
    fn clone(&self) -> Self {
        Self {
            commands: self.commands.clone(),
            child_fns: self.child_fns.clone(),
            register_capacity: self.register_capacity,
        }
    }
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
    T::Lit: Number,
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

    pub fn get_var_mut(&mut self, name: &str) -> Option<&mut Value<'a, T>> {
        let register = *self.vars.get(name)?;
        Some(&mut self.registers[register])
    }

    pub fn variables(&self) -> impl Iterator<Item = (&str, &Value<'a, T>)> + '_ {
        self.vars
            .iter()
            .map(move |(name, register)| (name.as_str(), &self.registers[*register]))
    }

    pub fn variables_map(&self) -> HashMap<String, usize> {
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
    pub fn push_var(&mut self, name: &str, value: Value<'a, T>) {
        let register = self.registers.len();
        self.registers.push(value);
        self.vars.insert(name.to_owned(), register);
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
        span: Span<'a>,
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

                    BinaryOp::Gt | BinaryOp::Lt | BinaryOp::Ge | BinaryOp::Le => {
                        unreachable!("Must be desugared by the compiler")
                    }
                }
            }

            CompiledExpr::Compare { inner, op } => {
                let inner_value = self.resolve_atom(&inner.extra);
                op.compare(inner_value)
                    .map(Value::Bool)
                    .ok_or_else(|| SpannedEvalError::new(&span, EvalError::InvalidCmpResult))
            }

            CompiledExpr::Function { name, args } => {
                if let Value::Function(function) = self.resolve_atom(&name.extra) {
                    let fn_name = create_span_ref(name, ());
                    let arg_values = args
                        .iter()
                        .map(|arg| create_span_ref(arg, self.resolve_atom(&arg.extra)))
                        .collect();
                    Self::eval_function(&function, fn_name, span, arg_values, backtrace)
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

    fn eval_function(
        function: &Function<'a, T>,
        fn_name: Span<'a>,
        call_span: Span<'a>,
        arg_values: Vec<SpannedValue<'a, T>>,
        mut backtrace: Option<&mut Backtrace<'a>>,
    ) -> EvalResult<'a, T> {
        if let Some(backtrace) = backtrace.as_deref_mut() {
            backtrace.push_call(fn_name.fragment(), function.def_span(), call_span);
        }
        let mut context = CallContext::new(call_span, backtrace.as_deref_mut());

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

/// Executable module together with its imports.
///
/// # Examples
///
/// ```
/// use arithmetic_parser::{grammars::F32Grammar, GrammarExt, Span};
/// use arithmetic_eval::{fns, Interpreter, Value, ValueType};
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
/// assert!(module.imports().contains("xs"));
/// // ...or even
/// assert!(module.imports()["fold"].is_function());
/// // It's possible to iterate over imports, too.
/// let imports: HashSet<_> = module.imports().iter().map(|(name, _)| name).collect();
/// assert!(imports.is_superset(&HashSet::from_iter(vec!["max", "fold", "xs"])));
///
/// // Change the `xs` import and run the module again.
/// let array = [1.0, -3.0, 2.0, 0.5].iter().copied().map(Value::Number).collect();
/// module.set_import("xs", Value::Tuple(array));
/// assert_eq!(module.run().unwrap(), Value::Number(2.0));
/// ```
///
/// The same module can be run with multiple imports:
///
/// ```
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt, Span};
/// # use arithmetic_eval::{Interpreter, Value};
/// let mut interpreter = Interpreter::new();
/// interpreter
///     .insert_var("x", Value::Number(3.0))
///     .insert_var("y", Value::Number(5.0));
/// let module = "x + y";
/// let module = F32Grammar::parse_statements(Span::new(module)).unwrap();
/// let mut module = interpreter.compile(&module).unwrap();
/// assert_eq!(module.run().unwrap(), Value::Number(8.0));
///
/// let mut imports = module.imports().to_owned();
/// imports["x"] = Value::Number(-1.0);
/// assert_eq!(module.run_with_imports(imports).unwrap(), Value::Number(4.0));
/// ```
#[derive(Debug)]
pub struct ExecutableModule<'a, T: Grammar> {
    inner: Executable<'a, T>,
    imports: ModuleImports<'a, T>,
}

impl<T: Grammar> Clone for ExecutableModule<'_, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            imports: self.imports.clone(),
        }
    }
}

impl<'a, T: Grammar> ExecutableModule<'a, T> {
    pub(super) fn new(inner: Executable<'a, T>, imports: Env<'a, T>) -> Self {
        Self {
            inner,
            imports: ModuleImports { inner: imports },
        }
    }

    /// Sets the value of an imported variable.
    ///
    /// # Panics
    ///
    /// Panics if the variable with the specified name is not an import. Check
    /// that the import exists beforehand via [`imports().contains()`] if this is
    /// unknown at compile time.
    ///
    /// [`imports().contains()`]: struct.ModuleImports.html#method.contains
    pub fn set_import(&mut self, name: &str, value: Value<'a, T>) -> &mut Self {
        self.imports.set(name, value);
        self
    }

    /// Returns shared reference to imports of this module.
    pub fn imports(&self) -> &ModuleImports<'a, T> {
        &self.imports
    }

    pub(super) fn inner(&self) -> &Executable<'a, T> {
        &self.inner
    }
}

impl<'a, T: Grammar> ExecutableModule<'a, T>
where
    T::Lit: Number,
{
    /// Runs the module with the current values of imports.
    pub fn run(&self) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        self.run_with_imports_unchecked(self.imports.clone())
    }

    /// Runs the module with the specified imports.
    ///
    /// # Panics
    ///
    /// - Panics if the imports are not compatible with the module.
    pub fn run_with_imports(
        &self,
        imports: ModuleImports<'a, T>,
    ) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        assert!(
            imports.is_compatible(self),
            "Cannot run module with incompatible imports"
        );
        self.run_with_imports_unchecked(imports)
    }

    /// Runs the module with the specified imports. Unlike [`run_with_imports`], this method
    /// does not check if the imports are compatible with the module; it is the caller's
    /// responsibility to ensure this.
    ///
    /// # Safety
    ///
    /// If the module and imports are incompatible, the module execution may lead to panics
    /// or unpredictable results.
    ///
    /// [`run_with_imports`]: #method.run_with_imports
    pub fn run_with_imports_unchecked(
        &self,
        mut imports: ModuleImports<'a, T>,
    ) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        let mut backtrace = Backtrace::default();
        imports
            .inner
            .execute(&self.inner, Some(&mut backtrace))
            .map_err(|err| ErrorWithBacktrace::new(err, backtrace))
    }
}

/// Imports of an [`ExecutableModule`].
///
/// Note that imports implement [`Index`] / [`IndexMut`] traits, which allows to eloquently
/// access or modify imports.
///
/// [`ExecutableModule`]: struct.ExecutableModule.html
/// [`Index`]: https://doc.rust-lang.org/std/ops/trait.Index.html
/// [`IndexMut`]: https://doc.rust-lang.org/std/ops/trait.IndexMut.html
#[derive(Debug)]
pub struct ModuleImports<'a, T: Grammar> {
    inner: Env<'a, T>,
}

impl<T: Grammar> Clone for ModuleImports<'_, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<'a, T: Grammar> ModuleImports<'a, T> {
    /// Checks if the imports contain a variable with the specified name.
    pub fn contains(&self, name: &str) -> bool {
        self.inner.vars.contains_key(name)
    }

    /// Gets the current value of the import with the specified name, or `None` if the import
    /// is not defined.
    pub fn get(&self, name: &str) -> Option<&Value<'a, T>> {
        self.inner.get_var(name)
    }

    /// Sets the value of an imported variable.
    ///
    /// # Panics
    ///
    /// Panics if the variable with the specified name is not an import.
    pub fn set(&mut self, name: &str, value: Value<'a, T>) -> &mut Self {
        self.inner.set_var(name, value);
        self
    }

    /// Iterates over imported variables.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value<'a, T>)> + '_ {
        self.inner.variables()
    }

    /// Checks if these imports could be compatible with the provided module.
    ///
    /// Imports produced by cloning imports of a module and then changing variables
    /// via [`set`](#method.set) are guaranteed to remain compatible with the module.
    /// Imports taken from another module are almost always incompatible with the module.
    ///
    /// The compatibility does not guarantee that the module execution will succeed; instead,
    /// it guarantees that the execution will not lead to a panic or unpredictable results.
    pub fn is_compatible(&self, module: &ExecutableModule<'a, T>) -> bool {
        self.inner.vars == module.imports.inner.vars
    }
}

impl<'a, T: Grammar> ops::Index<&str> for ModuleImports<'a, T> {
    type Output = Value<'a, T>;

    fn index(&self, index: &str) -> &Self::Output {
        self.inner
            .get_var(index)
            .unwrap_or_else(|| panic!("Import `{}` is not defined", index))
    }
}

impl<'a, T: Grammar> ops::IndexMut<&str> for ModuleImports<'a, T> {
    fn index_mut(&mut self, index: &str) -> &mut Self::Output {
        self.inner
            .get_var_mut(index)
            .unwrap_or_else(|| panic!("Import `{}` is not defined", index))
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
        module.imports = ModuleImports { inner: env };
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

    #[test]
    fn cloning_module() {
        let mut env = Env::new();
        env.push_var("x", Value::<F32Grammar>::Number(5.0));

        let block = "y = x + 2 * (x + 1) + 1; y";
        let block = F32Grammar::parse_statements(Span::new(block)).unwrap();
        let module = Compiler::compile_module(&env, &block, false).unwrap();

        let mut module_copy = module.clone();
        module_copy.set_import("x", Value::Number(10.0));
        let value = module_copy.run().unwrap();
        assert_eq!(value, Value::Number(33.0));
        let value = module.run().unwrap();
        assert_eq!(value, Value::Number(18.0));
    }

    #[test]
    fn checking_import_compatibility() {
        let mut env = Env::new();
        env.push_var("x", Value::<F32Grammar>::Number(5.0));
        env.push_var("y", Value::Bool(true));

        let block = "x + y";
        let block = F32Grammar::parse_statements(Span::new(block)).unwrap();
        let module = Compiler::compile_module(&env, &block, false).unwrap();

        let mut imports = module.imports().to_owned();
        assert!(imports.is_compatible(&module));
        imports.set("x", Value::Number(-1.0));
        assert!(imports.is_compatible(&module));

        let mut other_env = Env::new();
        other_env.push_var("y", Value::<F32Grammar>::Number(1.0));
        assert!(!ModuleImports { inner: other_env }.is_compatible(&module));
    }

    #[test]
    #[should_panic(expected = "Cannot run module with incompatible imports")]
    fn running_module_with_incompatible_imports() {
        let mut env = Env::new();
        env.push_var("x", Value::<F32Grammar>::Number(5.0));
        env.push_var("y", Value::Number(1.0));

        let block = "x + y";
        let block = F32Grammar::parse_statements(Span::new(block)).unwrap();
        let module = Compiler::compile_module(&env, &block, false).unwrap();

        let mut other_env = Env::new();
        other_env.push_var("y", Value::<F32Grammar>::Number(1.0));
        module
            .run_with_imports(ModuleImports { inner: other_env })
            .ok();
    }
}
