//! Evaluation errors.

use derive_more::Display;

use core::fmt;

use crate::{
    alloc::{format, vec, String, ToOwned, Vec},
    fns::FromValueError,
    ModuleId, Value,
};
use arithmetic_parser::{
    BinaryOp, CodeFragment, ExprType, LocatedSpan, LvalueLen, LvalueType, MaybeSpanned, Op,
    StatementType, StripCode, UnaryOp,
};

/// Context for [`EvalError::TupleLenMismatch`].
///
/// [`EvalError::TupleLenMismatch`]: enum.EvalError.html#variant.TupleLenMismatch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TupleLenMismatchContext {
    /// An error has occurred when evaluating a binary operation.
    BinaryOp(BinaryOp),
    /// An error has occurred during assignment.
    Assignment,
}

impl fmt::Display for TupleLenMismatchContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BinaryOp(op) => write!(formatter, "{}", op),
            Self::Assignment => formatter.write_str("assignment"),
        }
    }
}

/// Context for [`EvalError::RepeatedAssignment`].
///
/// [`EvalError::RepeatedAssignment`]: enum.EvalError.html#variant.RepeatedAssignment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatedAssignmentContext {
    /// A duplicated variable is in function args.
    FnArgs,
    /// A duplicated variable is in an lvalue of an assignment.
    Assignment,
}

impl fmt::Display for RepeatedAssignmentContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::FnArgs => "function args",
            Self::Assignment => "assignment",
        })
    }
}

/// Errors that can occur during interpreting expressions and statements.
#[derive(Debug, Clone, Display)]
#[non_exhaustive]
pub enum EvalError {
    /// Mismatch between length of tuples in a binary operation or assignment.
    #[display(
        fmt = "Mismatch between length of tuples in {}: LHS has {} element(s), whereas RHS has {}",
        context,
        lhs,
        rhs
    )]
    TupleLenMismatch {
        /// Length of a tuple on the left-hand side.
        lhs: LvalueLen,
        /// Length of a tuple on the right-hand side.
        rhs: usize,
        /// Context in which the error has occurred.
        context: TupleLenMismatchContext,
    },

    /// Mismatch between the number of arguments in the function definition and its call.
    #[display(
        fmt = "Mismatch between the number of arguments in the function definition and its call: \
            definition requires {} arg(s), call has {}",
        def,
        call
    )]
    ArgsLenMismatch {
        /// Number of args at the function definition.
        def: LvalueLen,
        /// Number of args at the function call.
        call: usize,
    },

    /// Cannot destructure a non-tuple variable.
    #[display(fmt = "Cannot destructure a non-tuple variable")]
    CannotDestructure,

    /// Repeated assignment to the same variable in function args or tuple destructuring.
    #[display(fmt = "Repeated assignment to the same variable in {}", context)]
    RepeatedAssignment {
        /// Context in which the error has occurred.
        context: RepeatedAssignmentContext,
    },

    /// Variable with the enclosed name is not defined.
    #[display(fmt = "Variable `{}` is not defined", _0)]
    Undefined(String),

    /// Value is not callable (i.e., is not a function).
    #[display(fmt = "Value is not callable")]
    CannotCall,

    /// Generic error during execution of a native function.
    #[display(fmt = "Failed executing native function: {}", _0)]
    NativeCall(String),

    /// Error while converting arguments for [`FnWrapper`].
    ///
    /// [`FnWrapper`]: fns/struct.FnWrapper.html
    #[display(
        fmt = "Failed converting arguments for native function wrapper: {}",
        _0
    )]
    Wrapper(FromValueError),

    /// Unexpected operand type for the specified operation.
    #[display(fmt = "Unexpected operand type for {}", op)]
    UnexpectedOperand {
        /// Operation which failed.
        op: Op,
    },

    /// Missing comparison function.
    #[display(fmt = "Missing comparison function {}", name)]
    MissingCmpFunction {
        /// Expected function name.
        name: String,
    },

    /// Unexpected result of a comparison function invocation. The comparison function should
    /// always return -1, 0, or 1.
    #[display(
        fmt = "Unexpected result of a comparison function invocation. The comparison function \
            should only return -1, 0, or 1."
    )]
    InvalidCmpResult,

    /// Construct not supported by the interpreter.
    #[display(fmt = "Unsupported {}", _0)]
    Unsupported(UnsupportedType),
}

impl EvalError {
    /// Creates a native error.
    pub fn native(message: impl Into<String>) -> Self {
        Self::NativeCall(message.into())
    }

    /// Creates an error for an lvalue type not supported by the interpreter.
    pub fn unsupported<T: Into<UnsupportedType>>(ty: T) -> Self {
        Self::Unsupported(ty.into())
    }

    /// Returned shortened error cause.
    pub fn to_short_string(&self) -> String {
        match self {
            Self::TupleLenMismatch { context, .. } => {
                format!("Mismatch between length of tuples in {}", context)
            }
            Self::ArgsLenMismatch { .. } => {
                "Mismatch between the number of arguments in the function definition and its call"
                    .to_owned()
            }
            Self::CannotDestructure => "Cannot destructure a non-tuple variable".to_owned(),
            Self::RepeatedAssignment { context } => {
                format!("Repeated assignment to the same variable in {}", context)
            }
            Self::Undefined(name) => format!("Variable `{}` is not defined", name),
            Self::CannotCall => "Value is not callable".to_owned(),
            Self::NativeCall(message) => message.to_owned(),
            Self::Wrapper(err) => err.to_string(),
            Self::UnexpectedOperand { op } => format!("Unexpected operand type for {}", op),
            Self::MissingCmpFunction { .. } => "Missing comparison function".to_owned(),
            Self::InvalidCmpResult => "Invalid comparison result".to_owned(),
            Self::Unsupported(_) => "Grammar construct not supported".to_owned(),
        }
    }

    /// Returns a short description of the spanned information.
    pub fn main_span_info(&self) -> String {
        match self {
            Self::TupleLenMismatch { context, lhs, .. } => {
                format!("LHS of {} with {} element(s)", context, lhs)
            }
            Self::ArgsLenMismatch { call, .. } => format!("Called with {} arg(s) here", call),
            Self::CannotDestructure => "Failed destructuring".to_owned(),
            Self::RepeatedAssignment { .. } => "Re-assigned variable".to_owned(),
            Self::Undefined(_) => "Undefined variable occurrence".to_owned(),
            Self::CannotCall | Self::NativeCall(_) | Self::Wrapper(_) => "Failed call".to_owned(),
            Self::UnexpectedOperand { .. } => "Operand of wrong type".to_owned(),
            Self::MissingCmpFunction { name } => {
                format!("Function with name {} should exist in the context", name)
            }
            Self::InvalidCmpResult => "Comparison function must return -1, 0 or 1".to_owned(),
            Self::Unsupported(ty) => format!("Unsupported {}", ty),
        }
    }

    /// Returns information helping fix the error.
    pub fn help(&self) -> Option<String> {
        Some(match self {
            Self::TupleLenMismatch { context, .. } => format!(
                "If both args of {} are tuples, the number of elements in them must agree",
                context
            ),
            Self::CannotDestructure => {
                "Only tuples can be destructured; numbers, functions and booleans cannot".to_owned()
            }
            Self::RepeatedAssignment { context } => format!(
                "In {}, all assigned variables must have different names",
                context
            ),
            Self::CannotCall => "Only functions are callable, i.e., can be used as `fn_name` \
                in `fn_name(...)` expressions"
                .to_owned(),
            Self::UnexpectedOperand { op: Op::Binary(op) } if op.is_arithmetic() => {
                "Operands of binary arithmetic ops must be numbers or tuples containing numbers"
                    .to_owned()
            }
            Self::UnexpectedOperand { op: Op::Binary(op) } if op.is_comparison() => {
                "Operands of comparison ops must be numbers or tuples containing numbers".to_owned()
            }
            Self::UnexpectedOperand { op: Op::Binary(_) } => {
                "Operands of binary boolean ops must be boolean".to_owned()
            }
            Self::UnexpectedOperand {
                op: Op::Unary(UnaryOp::Neg),
            } => "Operand of negation must be a number or a tuple".to_owned(),
            Self::UnexpectedOperand {
                op: Op::Unary(UnaryOp::Not),
            } => "Operand of boolean negation must be boolean".to_owned(),

            _ => return None,
        })
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EvalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Wrapper(error) => Some(error),
            _ => None,
        }
    }
}

/// Description of a construct not supported by the interpreter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnsupportedType {
    /// Unary operation.
    UnaryOp(UnaryOp),
    /// Binary operation.
    BinaryOp(BinaryOp),
    /// Expression.
    Expr(ExprType),
    /// Statement.
    Statement(StatementType),
    /// Lvalue.
    Lvalue(LvalueType),
}

impl fmt::Display for UnsupportedType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnaryOp(op) => write!(formatter, "unary op: {}", op),
            Self::BinaryOp(op) => write!(formatter, "binary op: {}", op),
            Self::Expr(expr) => write!(formatter, "expression: {}", expr),
            Self::Statement(statement) => write!(formatter, "statement: {}", statement),
            Self::Lvalue(lvalue) => write!(formatter, "lvalue: {}", lvalue),
        }
    }
}

impl From<UnaryOp> for UnsupportedType {
    fn from(value: UnaryOp) -> Self {
        Self::UnaryOp(value)
    }
}

impl From<BinaryOp> for UnsupportedType {
    fn from(value: BinaryOp) -> Self {
        Self::BinaryOp(value)
    }
}

impl From<ExprType> for UnsupportedType {
    fn from(value: ExprType) -> Self {
        Self::Expr(value)
    }
}

impl From<StatementType> for UnsupportedType {
    fn from(value: StatementType) -> Self {
        Self::Statement(value)
    }
}

impl From<LvalueType> for UnsupportedType {
    fn from(value: LvalueType) -> Self {
        Self::Lvalue(value)
    }
}

/// Auxiliary information about error.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum AuxErrorInfo {
    /// Function arguments declaration for [`ArgsLenMismatch`].
    ///
    /// [`ArgsLenMismatch`]: enum.EvalError.html#variant.ArgsLenMismatch
    FnArgs,

    /// Previous variable assignment for [`RepeatedAssignment`].
    ///
    /// [`RepeatedAssignment`]: enum.EvalError.html#variant.RepeatedAssignment
    PrevAssignment,

    /// Rvalue containing an invalid assignment for [`CannotDestructure`] or [`TupleLenMismatch`].
    ///
    /// [`CannotDestructure`]: enum.EvalError.html#variant.CannotDestructure
    /// [`TupleLenMismatch`]: enum.EvalError.html#variant.TupleLenMismatch
    Rvalue,

    /// RHS of a binary operation on differently sized tuples.
    UnbalancedRhs(usize),

    /// Invalid argument.
    InvalidArg,
}

impl fmt::Display for AuxErrorInfo {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FnArgs => formatter.write_str("Function arguments declared here"),
            Self::PrevAssignment => formatter.write_str("Previous declaration"),
            Self::Rvalue => formatter.write_str("RHS containing the invalid assignment"),
            Self::UnbalancedRhs(size) => write!(formatter, "RHS with the {}-element tuple", size),
            Self::InvalidArg => formatter.write_str("Invalid argument"),
        }
    }
}

/// Evaluation error together with one or more relevant code spans.
#[derive(Debug)]
pub struct SpannedEvalError<'a> {
    error: EvalError,
    main_span: CodeInModule<'a>,
    aux_spans: Vec<CodeInModule<'a, AuxErrorInfo>>,
}

impl<'a> SpannedEvalError<'a> {
    pub(crate) fn new<Span, T>(
        module_id: &dyn ModuleId,
        main_span: &LocatedSpan<Span, T>,
        error: EvalError,
    ) -> Self
    where
        Span: Copy + Into<CodeFragment<'a>>,
    {
        Self {
            error,
            main_span: CodeInModule::new(
                module_id,
                main_span.with_no_extra().map_fragment(Into::into),
            ),
            aux_spans: vec![],
        }
    }

    pub(crate) fn from_parts(main_span: CodeInModule<'a>, error: EvalError) -> Self {
        Self {
            error,
            main_span,
            aux_spans: vec![],
        }
    }

    #[doc(hidden)] // used in `wrap_fn` macro
    pub fn with_span(mut self, span: CodeInModule<'a>, info: AuxErrorInfo) -> Self {
        self.aux_spans.push(CodeInModule {
            module_id: span.module_id,
            code: span.code.copy_with_extra(info),
        });
        self
    }

    /// Returns the source of the error.
    pub fn kind(&self) -> &EvalError {
        &self.error
    }

    /// Returns the main span for this error.
    pub fn main_span(&self) -> &CodeInModule<'a> {
        &self.main_span
    }

    /// Returns auxiliary spans for this error.
    pub fn aux_spans(&self) -> &[CodeInModule<'a, AuxErrorInfo>] {
        &self.aux_spans
    }
}

impl fmt::Display for SpannedEvalError<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.main_span.fmt_location(formatter)?;
        write!(formatter, ": {}", self.error)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SpannedEvalError<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl StripCode for SpannedEvalError<'_> {
    type Stripped = SpannedEvalError<'static>;

    fn strip_code(&self) -> Self::Stripped {
        SpannedEvalError {
            error: self.error.clone(),
            main_span: self.main_span.strip_code(),
            aux_spans: self.aux_spans.iter().map(StripCode::strip_code).collect(),
        }
    }
}

/// Result of an expression evaluation.
pub type EvalResult<'a, T> = Result<Value<'a, T>, SpannedEvalError<'a>>;

/// Code fragment together with information about the module containing the fragment.
#[derive(Debug)]
pub struct CodeInModule<'a, T = ()> {
    module_id: Box<dyn ModuleId>,
    code: MaybeSpanned<'a, T>,
}

impl<T: Clone> Clone for CodeInModule<'_, T> {
    fn clone(&self) -> Self {
        Self {
            module_id: self.module_id.clone_boxed(),
            code: self.code.clone(),
        }
    }
}

impl<'a> CodeInModule<'a> {
    pub(crate) fn new(module_id: &dyn ModuleId, span: MaybeSpanned<'a>) -> Self {
        Self {
            module_id: module_id.clone_boxed(),
            code: span,
        }
    }
}

impl<'a, T> CodeInModule<'a, T> {
    /// Returns the ID of the module containing this fragment.
    pub fn module_id(&self) -> &dyn ModuleId {
        self.module_id.as_ref()
    }

    /// Returns the code fragment within the module. The fragment may be stripped
    /// (i.e., contain only location info, not the code string itself).
    pub fn code(&self) -> &MaybeSpanned<'a, T> {
        &self.code
    }

    fn fmt_location(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}:{}:{}",
            self.module_id,
            self.code.location_line(),
            self.code.get_column()
        )
    }
}

impl<T: Clone + 'static> StripCode for CodeInModule<'_, T> {
    type Stripped = CodeInModule<'static, T>;

    fn strip_code(&self) -> Self::Stripped {
        CodeInModule {
            module_id: self.module_id.clone_boxed(),
            code: self.code.strip_code(),
        }
    }
}

/// Function call.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BacktraceElement<'a> {
    /// Function name.
    pub fn_name: String,
    /// Code span of the function definition.
    pub def_span: Option<CodeInModule<'a>>,
    /// Code span of the function call.
    pub call_span: CodeInModule<'a>,
}

impl StripCode for BacktraceElement<'_> {
    type Stripped = BacktraceElement<'static>;

    fn strip_code(&self) -> Self::Stripped {
        BacktraceElement {
            fn_name: self.fn_name.clone(),
            def_span: self.def_span.as_ref().map(StripCode::strip_code),
            call_span: self.call_span.strip_code(),
        }
    }
}

/// Call backtrace.
#[derive(Debug, Default)]
pub struct Backtrace<'a> {
    calls: Vec<BacktraceElement<'a>>,
}

impl<'a> Backtrace<'a> {
    /// Iterates over the backtrace, starting from the most recent call.
    pub fn calls(&self) -> impl Iterator<Item = BacktraceElement<'a>> + '_ {
        self.calls.iter().rev().cloned()
    }

    /// Appends a function call into the backtrace.
    pub(crate) fn push_call(
        &mut self,
        fn_name: &str,
        def_span: Option<CodeInModule<'a>>,
        call_span: CodeInModule<'a>,
    ) {
        self.calls.push(BacktraceElement {
            fn_name: fn_name.to_owned(),
            def_span,
            call_span,
        });
    }

    /// Pops a function call.
    pub(crate) fn pop_call(&mut self) {
        self.calls.pop();
    }
}

impl StripCode for Backtrace<'_> {
    type Stripped = Backtrace<'static>;

    fn strip_code(&self) -> Self::Stripped {
        Backtrace {
            calls: self.calls.iter().map(StripCode::strip_code).collect(),
        }
    }
}

/// Error with the associated backtrace.
#[derive(Debug)]
pub struct ErrorWithBacktrace<'a> {
    inner: SpannedEvalError<'a>,
    backtrace: Backtrace<'a>,
}

impl<'a> ErrorWithBacktrace<'a> {
    pub(crate) fn new(inner: SpannedEvalError<'a>, backtrace: Backtrace<'a>) -> Self {
        Self { inner, backtrace }
    }

    // FIXME: rework; backtrace must not be empty.
    pub(crate) fn with_empty_trace(inner: SpannedEvalError<'a>) -> Self {
        Self {
            inner,
            backtrace: Backtrace::default(),
        }
    }

    /// Returns the source of the error.
    pub fn source(&self) -> &SpannedEvalError<'a> {
        &self.inner
    }

    /// Returns error backtrace.
    pub fn backtrace(&self) -> &Backtrace<'a> {
        &self.backtrace
    }
}

impl fmt::Display for ErrorWithBacktrace<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, formatter)?;

        if formatter.alternate() && !self.backtrace.calls.is_empty() {
            writeln!(formatter, "\nBacktrace (most recent call last):")?;
            for (index, call) in self.backtrace.calls.iter().enumerate() {
                write!(formatter, "{:>4}: {} ", index + 1, call.fn_name)?;

                if let Some(ref def_span) = call.def_span {
                    write!(formatter, "(module `{}`)", def_span.module_id)?;
                } else {
                    formatter.write_str("(native)")?;
                }

                write!(formatter, " called at ")?;
                call.call_span.fmt_location(formatter)?;
                writeln!(formatter)?;
            }
        }
        Ok(())
    }
}

impl StripCode for ErrorWithBacktrace<'_> {
    type Stripped = ErrorWithBacktrace<'static>;

    fn strip_code(&self) -> Self::Stripped {
        ErrorWithBacktrace {
            inner: self.inner.strip_code(),
            backtrace: self.backtrace.strip_code(),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ErrorWithBacktrace<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        std::error::Error::source(&self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::ToString;

    #[test]
    fn display_for_eval_error() {
        let err = EvalError::Undefined("test".to_owned());
        assert_eq!(err.to_string(), "Variable `test` is not defined");

        let err = EvalError::ArgsLenMismatch {
            def: LvalueLen::AtLeast(2),
            call: 1,
        };
        assert!(err
            .to_string()
            .ends_with("definition requires at least 2 arg(s), call has 1"));
    }

    #[test]
    fn display_for_spanned_eval_error() {
        let input = "(_, test) = (1, 2);";
        let main_span = MaybeSpanned::from_str(input, 4..8);
        let err = SpannedEvalError::new(
            &"test_module",
            &main_span,
            EvalError::Undefined("test".to_owned()),
        );
        let err_string = err.to_string();
        assert_eq!(
            err_string,
            "test_module:1:5: Variable `test` is not defined"
        );
    }

    #[test]
    fn display_for_error_with_backtrace() {
        let input = "(_, test) = (1, 2);";
        let main_span = MaybeSpanned::from_str(input, 4..8);
        let err =
            SpannedEvalError::new(&"test", &main_span, EvalError::Undefined("test".to_owned()));

        let mut err = ErrorWithBacktrace::with_empty_trace(err);
        let call_span = CodeInModule::new(&"test", MaybeSpanned::from_str(input, ..));
        err.backtrace.push_call("test_fn", None, call_span);

        let err_string = err.to_string();
        assert_eq!(err_string, "test:1:5: Variable `test` is not defined");

        let expanded_err_string = format!("{:#}", err);
        assert!(expanded_err_string.starts_with("test:1:5: Variable `test` is not defined"));
        assert!(expanded_err_string.contains("\nBacktrace"));
        assert!(expanded_err_string.contains("\n   1: test_fn (native) called at test:1:1"));
    }
}
