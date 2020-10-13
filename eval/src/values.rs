//! Values used by the interpreter.

use hashbrown::HashMap;
use num_traits::Pow;

use core::fmt;

use crate::{
    alloc::{vec, Rc, Vec},
    executable::ExecutableFn,
    AuxErrorInfo, Backtrace, CodeInModule, EvalError, EvalResult, ModuleId, Number,
    SpannedEvalError, TupleLenMismatchContext,
};
use arithmetic_parser::{
    BinaryOp, Grammar, InputSpan, LvalueLen, MaybeSpanned, Op, Spanned, StripCode, UnaryOp,
};

/// Opaque context for native calls.
#[derive(Debug)]
pub struct CallContext<'r, 'a> {
    call_span: MaybeSpanned<'a>,
    backtrace: Option<&'r mut Backtrace<'a>>,
}

impl<'r, 'a> CallContext<'r, 'a> {
    /// Creates a mock call context.
    pub fn mock() -> Self {
        Self {
            call_span: Spanned::from(InputSpan::new("")).into(),
            backtrace: None,
        }
    }

    pub(crate) fn new(
        call_span: MaybeSpanned<'a>,
        backtrace: Option<&'r mut Backtrace<'a>>,
    ) -> Self {
        Self {
            call_span,
            backtrace,
        }
    }

    pub(crate) fn backtrace(&mut self) -> Option<&mut Backtrace<'a>> {
        self.backtrace.as_deref_mut()
    }

    /// Returns the call span.
    pub fn apply_call_span<T>(&self, value: T) -> MaybeSpanned<'a, T> {
        self.call_span.copy_with_extra(value)
    }

    /// Creates the error spanning the call site.
    pub fn call_site_error(&self, error: EvalError) -> SpannedEvalError<'a> {
        SpannedEvalError::new(&self.call_span, error)
    }

    /// Checks argument count and returns an error if it doesn't match.
    pub fn check_args_count<T: Grammar>(
        &self,
        args: &[SpannedValue<'a, T>],
        expected_count: impl Into<LvalueLen>,
    ) -> Result<(), SpannedEvalError<'a>> {
        let expected_count = expected_count.into();
        if expected_count.matches(args.len()) {
            Ok(())
        } else {
            Err(self.call_site_error(EvalError::ArgsLenMismatch {
                def: expected_count,
                call: args.len(),
            }))
        }
    }
}

/// Function on zero or more `Value`s.
pub trait NativeFn<T: Grammar> {
    /// Executes the function on the specified arguments.
    fn evaluate<'a>(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        context: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T>;
}

impl<T: Grammar, F: 'static> NativeFn<T> for F
where
    F: for<'a> Fn(Vec<SpannedValue<'a, T>>, &mut CallContext<'_, 'a>) -> EvalResult<'a, T>,
{
    fn evaluate<'a>(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        context: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        self(args, context)
    }
}

impl<T: Grammar> fmt::Debug for dyn NativeFn<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("NativeFn").finish()
    }
}

impl<T: Grammar> dyn NativeFn<T> {
    /// Extracts a data pointer from this trait object reference.
    pub(crate) fn data_ptr(&self) -> *const () {
        // `*const dyn Trait as *const ()` extracts the data pointer,
        // see https://github.com/rust-lang/rust/issues/27751. This is seemingly
        // the simplest way to extract the data pointer; `TraitObject` in `std::raw` is
        // a more future-proof alternative, but it is unstable.
        self as *const dyn NativeFn<T> as *const ()
    }
}

/// Function defined within the interpreter.
#[derive(Debug)]
pub struct InterpretedFn<'a, T: Grammar> {
    definition: Rc<ExecutableFn<'a, T>>,
    captures: Vec<Value<'a, T>>,
    capture_names: Vec<String>,
}

impl<T: Grammar> StripCode for InterpretedFn<'_, T> {
    type Stripped = InterpretedFn<'static, T>;

    fn strip_code(&self) -> Self::Stripped {
        InterpretedFn {
            definition: Rc::new(self.definition.strip_code()),
            captures: self.captures.iter().map(StripCode::strip_code).collect(),
            capture_names: self.capture_names.clone(),
        }
    }
}

impl<'a, T: Grammar> InterpretedFn<'a, T> {
    pub(crate) fn new(
        definition: Rc<ExecutableFn<'a, T>>,
        captures: Vec<Value<'a, T>>,
        capture_names: Vec<String>,
    ) -> Self {
        Self {
            definition,
            captures,
            capture_names,
        }
    }

    /// Returns ID of the module defining this function.
    pub fn module_id(&self) -> &dyn ModuleId {
        self.definition.inner.id()
    }

    /// Returns the number of arguments for this function.
    pub fn arg_count(&self) -> LvalueLen {
        self.definition.arg_count
    }

    /// Returns values captured by this function.
    pub fn captures(&self) -> HashMap<&str, &Value<'a, T>> {
        self.capture_names
            .iter()
            .zip(&self.captures)
            .map(|(name, val)| (name.as_str(), val))
            .collect()
    }
}

impl<'a, T> InterpretedFn<'a, T>
where
    T: Grammar,
    T::Lit: Number,
{
    /// Evaluates this function with the provided arguments and the execution context.
    pub fn evaluate(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        if !self.arg_count().matches(args.len()) {
            let err = EvalError::ArgsLenMismatch {
                def: self.arg_count(),
                call: args.len(),
            };
            return Err(SpannedEvalError::new(&ctx.call_span, err));
        }

        let args = args.into_iter().map(|arg| arg.extra).collect();
        self.definition
            .inner
            .call_function(self.captures.clone(), args, ctx)
    }
}

/// Function definition. Functions can be either native (defined in the Rust code) or defined
/// in the interpreter.
#[derive(Debug)]
pub enum Function<'a, T>
where
    T: Grammar,
{
    /// Native function.
    Native(Rc<dyn NativeFn<T>>),
    /// Interpreted function.
    Interpreted(Rc<InterpretedFn<'a, T>>),
}

impl<T: Grammar> Clone for Function<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Native(function) => Self::Native(Rc::clone(&function)),
            Self::Interpreted(function) => Self::Interpreted(Rc::clone(&function)),
        }
    }
}

impl<T: Grammar> StripCode for Function<'_, T> {
    type Stripped = Function<'static, T>;

    fn strip_code(&self) -> Self::Stripped {
        match self {
            Self::Native(function) => Function::Native(Rc::clone(function)),
            Self::Interpreted(function) => Function::Interpreted(Rc::new(function.strip_code())),
        }
    }
}

impl<'a, T: Grammar> Function<'a, T> {
    /// Creates a native function.
    pub fn native(function: impl NativeFn<T> + 'static) -> Self {
        Self::Native(Rc::new(function))
    }

    /// Checks if the provided function is the same as this one.
    pub fn is_same_function(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Native(this), Self::Native(other)) => this.data_ptr() == other.data_ptr(),
            (Self::Interpreted(this), Self::Interpreted(other)) => Rc::ptr_eq(this, other),
            _ => false,
        }
    }
}

impl<'a, T> Function<'a, T>
where
    T: Grammar,
    T::Lit: Number,
{
    /// Evaluates the function on the specified arguments.
    pub fn evaluate(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        match self {
            Self::Native(function) => function.evaluate(args, ctx),
            Self::Interpreted(function) => function.evaluate(args, ctx),
        }
    }

    pub(crate) fn def_span(&self) -> Option<CodeInModule<'a>> {
        match self {
            Self::Native(_) => None,
            Self::Interpreted(function) => Some(CodeInModule::new(
                function.module_id(),
                function.definition.def_span,
            )),
        }
    }
}

/// Possible high-level types of [`Value`]s.
///
/// [`Value`]: enum.Value.html
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ValueType {
    /// Number.
    Number,
    /// Boolean value.
    Bool,
    /// Function value.
    Function,
    /// Tuple of a specific size.
    Tuple(usize),
    /// Array (a tuple of arbitrary size).
    ///
    /// This variant is never returned from [`Value::value_type()`]; at the same time, it is
    /// used for error reporting etc.
    ///
    /// [`Value::value_type()`]: enum.Value.html#method.value_type
    Array,
}

impl fmt::Display for ValueType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Number => formatter.write_str("number"),
            Self::Bool => formatter.write_str("boolean value"),
            Self::Function => formatter.write_str("function"),
            Self::Tuple(1) => write!(formatter, "tuple with 1 element"),
            Self::Tuple(size) => write!(formatter, "tuple with {} elements", size),
            Self::Array => formatter.write_str("array"),
        }
    }
}

/// Values produced by expressions during their interpretation.
#[derive(Debug)]
#[non_exhaustive]
pub enum Value<'a, T: Grammar> {
    /// Primitive value: a single literal.
    Number(T::Lit),
    /// Boolean value.
    Bool(bool),
    /// Function.
    Function(Function<'a, T>),
    /// Tuple of zero or more values.
    Tuple(Vec<Value<'a, T>>),
}

/// Value together with a span that has produced it.
pub type SpannedValue<'a, T> = MaybeSpanned<'a, Value<'a, T>>;

impl<'a, T: Grammar> Value<'a, T> {
    /// Creates a value for a native function.
    pub fn native_fn(function: impl NativeFn<T> + 'static) -> Self {
        Self::Function(Function::Native(Rc::new(function)))
    }

    /// Creates a value for an interpreted function.
    pub(crate) fn interpreted_fn(function: InterpretedFn<'a, T>) -> Self {
        Self::Function(Function::Interpreted(Rc::new(function)))
    }

    /// Creates a void value (an empty tuple).
    pub fn void() -> Self {
        Self::Tuple(vec![])
    }

    /// Returns the type of this value.
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::Number(_) => ValueType::Number,
            Self::Bool(_) => ValueType::Bool,
            Self::Function(_) => ValueType::Function,
            Self::Tuple(elements) => ValueType::Tuple(elements.len()),
        }
    }

    /// Checks if the value is void.
    pub fn is_void(&self) -> bool {
        matches!(self, Self::Tuple(tuple) if tuple.is_empty())
    }

    /// Checks if this value is a function.
    pub fn is_function(&self) -> bool {
        matches!(self, Self::Function(_))
    }
}

impl<T: Grammar> Clone for Value<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Number(lit) => Self::Number(lit.clone()),
            Self::Bool(bool) => Self::Bool(*bool),
            Self::Function(function) => Self::Function(function.clone()),
            Self::Tuple(tuple) => Self::Tuple(tuple.clone()),
        }
    }
}

impl<T: Grammar> StripCode for Value<'_, T> {
    type Stripped = Value<'static, T>;

    fn strip_code(&self) -> Self::Stripped {
        match self {
            Self::Number(lit) => Value::Number(lit.clone()),
            Self::Bool(bool) => Value::Bool(*bool),
            Self::Function(function) => Value::Function(function.strip_code()),
            Self::Tuple(tuple) => Value::Tuple(tuple.iter().map(StripCode::strip_code).collect()),
        }
    }
}

impl<T: Grammar> PartialEq for Value<'_, T>
where
    T::Lit: PartialEq,
{
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Number(this), Self::Number(other)) => this == other,
            (Self::Bool(this), Self::Bool(other)) => this == other,
            (Self::Tuple(this), Self::Tuple(other)) => this == other,
            (Self::Function(this), Self::Function(other)) => this.is_same_function(other),
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum OpSide {
    Lhs,
    Rhs,
}

#[derive(Debug)]
struct BinaryOpError {
    inner: EvalError,
    side: Option<OpSide>,
}

impl BinaryOpError {
    fn new(op: BinaryOp) -> Self {
        Self {
            inner: EvalError::UnexpectedOperand { op: Op::Binary(op) },
            side: None,
        }
    }

    fn tuple(op: BinaryOp, lhs: usize, rhs: usize) -> Self {
        Self {
            inner: EvalError::TupleLenMismatch {
                lhs: lhs.into(),
                rhs,
                context: TupleLenMismatchContext::BinaryOp(op),
            },
            side: Some(OpSide::Lhs),
        }
    }

    fn with_side(mut self, side: OpSide) -> Self {
        self.side = Some(side);
        self
    }

    fn span<'a>(
        self,
        total_span: MaybeSpanned<'a>,
        lhs_span: MaybeSpanned<'a>,
        rhs_span: MaybeSpanned<'a>,
    ) -> SpannedEvalError<'a> {
        let main_span = match self.side {
            Some(OpSide::Lhs) => lhs_span,
            Some(OpSide::Rhs) => rhs_span,
            None => total_span,
        };

        let aux_info = if let EvalError::TupleLenMismatch { rhs, .. } = self.inner {
            Some(AuxErrorInfo::UnbalancedRhs(rhs))
        } else {
            None
        };

        let mut err = SpannedEvalError::new(&main_span, self.inner);
        if let Some(aux_info) = aux_info {
            err = err.with_span(&rhs_span, aux_info);
        }
        err
    }
}

impl<'a, T> Value<'a, T>
where
    T: Grammar,
    T::Lit: Number,
{
    fn try_binary_op_inner(
        self,
        rhs: Self,
        op: BinaryOp,
        primitive_op: fn(T::Lit, T::Lit) -> T::Lit,
    ) -> Result<Self, BinaryOpError> {
        match (self, rhs) {
            (Self::Number(this), Self::Number(other)) => {
                Ok(Self::Number(primitive_op(this, other)))
            }

            (this @ Self::Number(_), Self::Tuple(other)) => {
                let output: Result<Vec<_>, _> = other
                    .into_iter()
                    .map(|y| this.clone().try_binary_op_inner(y, op, primitive_op))
                    .collect();
                output.map(Self::Tuple)
            }

            (Self::Tuple(this), other @ Self::Number(_)) => {
                let output: Result<Vec<_>, _> = this
                    .into_iter()
                    .map(|x| x.try_binary_op_inner(other.clone(), op, primitive_op))
                    .collect();
                output.map(Self::Tuple)
            }

            (Self::Tuple(this), Self::Tuple(other)) => {
                if this.len() == other.len() {
                    let output: Result<Vec<_>, _> = this
                        .into_iter()
                        .zip(other)
                        .map(|(x, y)| x.try_binary_op_inner(y, op, primitive_op))
                        .collect();
                    output.map(Self::Tuple)
                } else {
                    Err(BinaryOpError::tuple(op, this.len(), other.len()))
                }
            }

            (Self::Number(_), _) | (Self::Tuple(_), _) => {
                Err(BinaryOpError::new(op).with_side(OpSide::Rhs))
            }
            _ => Err(BinaryOpError::new(op).with_side(OpSide::Lhs)),
        }
    }

    #[inline]
    fn try_binary_op(
        total_span: MaybeSpanned<'a>,
        lhs: MaybeSpanned<'a, Self>,
        rhs: MaybeSpanned<'a, Self>,
        op: BinaryOp,
        primitive_op: fn(T::Lit, T::Lit) -> T::Lit,
    ) -> Result<Self, SpannedEvalError<'a>> {
        let lhs_span = lhs.with_no_extra();
        let rhs_span = rhs.with_no_extra();
        lhs.extra
            .try_binary_op_inner(rhs.extra, op, primitive_op)
            .map_err(|e| e.span(total_span, lhs_span, rhs_span))
    }

    pub(crate) fn try_add(
        total_span: MaybeSpanned<'a>,
        lhs: MaybeSpanned<'a, Self>,
        rhs: MaybeSpanned<'a, Self>,
    ) -> Result<Self, SpannedEvalError<'a>> {
        Self::try_binary_op(total_span, lhs, rhs, BinaryOp::Add, |x, y| x + y)
    }

    pub(crate) fn try_sub(
        total_span: MaybeSpanned<'a>,
        lhs: MaybeSpanned<'a, Self>,
        rhs: MaybeSpanned<'a, Self>,
    ) -> Result<Self, SpannedEvalError<'a>> {
        Self::try_binary_op(total_span, lhs, rhs, BinaryOp::Sub, |x, y| x - y)
    }

    pub(crate) fn try_mul(
        total_span: MaybeSpanned<'a>,
        lhs: MaybeSpanned<'a, Self>,
        rhs: MaybeSpanned<'a, Self>,
    ) -> Result<Self, SpannedEvalError<'a>> {
        Self::try_binary_op(total_span, lhs, rhs, BinaryOp::Mul, |x, y| x * y)
    }

    pub(crate) fn try_div(
        total_span: MaybeSpanned<'a>,
        lhs: MaybeSpanned<'a, Self>,
        rhs: MaybeSpanned<'a, Self>,
    ) -> Result<Self, SpannedEvalError<'a>> {
        Self::try_binary_op(total_span, lhs, rhs, BinaryOp::Div, |x, y| x / y)
    }

    pub(crate) fn try_pow(
        total_span: MaybeSpanned<'a>,
        lhs: MaybeSpanned<'a, Self>,
        rhs: MaybeSpanned<'a, Self>,
    ) -> Result<Self, SpannedEvalError<'a>> {
        Self::try_binary_op(total_span, lhs, rhs, BinaryOp::Power, Pow::pow)
    }

    pub(crate) fn try_neg(self) -> Result<Self, EvalError> {
        match self {
            Self::Number(val) => Ok(Self::Number(-val)),
            Self::Tuple(tuple) => {
                let res: Result<Vec<_>, _> = tuple.into_iter().map(Value::try_neg).collect();
                res.map(Self::Tuple)
            }

            _ => Err(EvalError::UnexpectedOperand {
                op: UnaryOp::Neg.into(),
            }),
        }
    }

    pub(crate) fn try_not(self) -> Result<Self, EvalError> {
        match self {
            Self::Bool(val) => Ok(Self::Bool(!val)),
            Self::Tuple(tuple) => {
                let res: Result<Vec<_>, _> = tuple.into_iter().map(Value::try_not).collect();
                res.map(Self::Tuple)
            }

            _ => Err(EvalError::UnexpectedOperand {
                op: UnaryOp::Not.into(),
            }),
        }
    }

    pub(crate) fn try_and(
        lhs: &MaybeSpanned<'a, Self>,
        rhs: &MaybeSpanned<'a, Self>,
    ) -> Result<Self, SpannedEvalError<'a>> {
        match (&lhs.extra, &rhs.extra) {
            (Value::Bool(this), Value::Bool(other)) => Ok(Value::Bool(*this && *other)),
            (Value::Bool(_), _) => {
                let err = EvalError::UnexpectedOperand {
                    op: BinaryOp::And.into(),
                };
                Err(SpannedEvalError::new(&rhs, err))
            }
            _ => {
                let err = EvalError::UnexpectedOperand {
                    op: BinaryOp::And.into(),
                };
                Err(SpannedEvalError::new(&lhs, err))
            }
        }
    }

    pub(crate) fn try_or(
        lhs: &MaybeSpanned<'a, Self>,
        rhs: &MaybeSpanned<'a, Self>,
    ) -> Result<Self, SpannedEvalError<'a>> {
        match (&lhs.extra, &rhs.extra) {
            (Value::Bool(this), Value::Bool(other)) => Ok(Value::Bool(*this || *other)),
            (Value::Bool(_), _) => {
                let err = EvalError::UnexpectedOperand {
                    op: BinaryOp::Or.into(),
                };
                Err(SpannedEvalError::new(&rhs, err))
            }
            _ => {
                let err = EvalError::UnexpectedOperand {
                    op: BinaryOp::Or.into(),
                };
                Err(SpannedEvalError::new(&lhs, err))
            }
        }
    }
}
