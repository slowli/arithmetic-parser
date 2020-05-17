//! Evaluation errors.

use thiserror::Error;

use std::fmt;

use crate::{
    eval::Value, helpers::create_span_ref, BinaryOp, LvalueLen, Op, Span, Spanned, UnaryOp,
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
#[derive(Debug, Error)]
pub enum EvalError {
    /// Mismatch between length of tuples in a binary operation or assignment.
    #[error(
        "Mismatch between length of tuples in {context}: \
         LHS has {lhs} element(s), whereas RHS has {rhs}"
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
    #[error(
        "Mismatch between the number of arguments in the function definition and its call: \
         definition requires {def} arg(s), call has {call}"
    )]
    ArgsLenMismatch {
        /// Number of args at the function definition.
        def: LvalueLen,
        /// Number of args at the function call.
        call: usize,
    },

    /// Cannot destructure a non-tuple variable.
    #[error("Cannot destructure a non-tuple variable.")]
    CannotDestructure,

    /// Repeated assignment to the same variable in function args or tuple destructuring.
    #[error("Repeated assignment to the same variable in {context}")]
    RepeatedAssignment {
        /// Context in which the error has occurred.
        context: RepeatedAssignmentContext,
    },

    /// Variable with the enclosed name is not defined.
    #[error("Variable `{0}` is not defined")]
    Undefined(String),

    /// Value is not callable (i.e., is not a function).
    #[error("Value is not callable")]
    CannotCall,

    /// Generic error during execution of a native function.
    #[error("Failed executing native function: {0}")]
    NativeCall(String),

    /// Unexpected operand type for the specified operation.
    #[error("Unexpected operand type for {op}")]
    UnexpectedOperand {
        /// Operation which failed.
        op: Op,
    },
}

impl EvalError {
    /// Creates a native error.
    pub fn native(message: impl Into<String>) -> Self {
        Self::NativeCall(message.into())
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
            Self::UnexpectedOperand { op } => format!("Unexpected operand type for {}", op),
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
            Self::CannotCall | Self::NativeCall(_) => "Failed call".to_owned(),
            Self::UnexpectedOperand { .. } => "Operand of wrong type".to_owned(),
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

/// Auxiliary information about error.
#[derive(Debug)]
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
    main_span: Span<'a>,
    aux_spans: Vec<Spanned<'a, AuxErrorInfo>>,
}

impl<'a> SpannedEvalError<'a> {
    pub(super) fn new<T>(main_span: &Spanned<'a, T>, error: EvalError) -> Self {
        Self {
            error,
            main_span: create_span_ref(main_span, ()),
            aux_spans: vec![],
        }
    }

    pub(super) fn with_span<T>(mut self, span: &Spanned<'a, T>, info: AuxErrorInfo) -> Self {
        self.aux_spans.push(create_span_ref(span, info));
        self
    }
}

/// Result of an expression evaluation.
pub type EvalResult<'a, T> = Result<Value<'a, T>, SpannedEvalError<'a>>;

/// Call backtrace.
#[derive(Debug, Default)]
pub struct Backtrace<'a> {
    calls: Vec<BacktraceElement<'a>>,
}

/// Function call.
#[derive(Debug, Clone, Copy)]
pub struct BacktraceElement<'a> {
    /// Function name.
    pub fn_name: &'a str,
    /// Code span of the function definition.
    pub def_span: Option<Span<'a>>,
    /// Code span of the function call.
    pub call_span: Span<'a>,
}

impl<'a> Backtrace<'a> {
    /// Iterates over the backtrace, starting from the most recent call.
    pub fn calls(&self) -> impl Iterator<Item = BacktraceElement<'a>> + '_ {
        self.calls.iter().rev().cloned()
    }

    /// Appends a function call into the backtrace.
    pub(super) fn push_call(
        &mut self,
        fn_name: &'a str,
        def_span: Option<Span<'a>>,
        call_span: Span<'a>,
    ) {
        self.calls.push(BacktraceElement {
            fn_name,
            def_span,
            call_span,
        });
    }

    /// Pops a function call.
    pub(super) fn pop_call(&mut self) {
        self.calls.pop();
    }
}

/// Error with the associated backtrace.
#[derive(Debug)]
pub struct ErrorWithBacktrace<'a> {
    /// Error.
    inner: SpannedEvalError<'a>,
    /// Backtrace information.
    backtrace: Backtrace<'a>,
}

impl<'a> ErrorWithBacktrace<'a> {
    pub(super) fn new(inner: SpannedEvalError<'a>, backtrace: Backtrace<'a>) -> Self {
        Self { inner, backtrace }
    }

    pub(super) fn with_empty_trace(inner: SpannedEvalError<'a>) -> Self {
        Self {
            inner,
            backtrace: Backtrace::default(),
        }
    }

    /// Returns the source of the error.
    pub fn source(&self) -> &EvalError {
        &self.inner.error
    }

    /// Returns the main span for this error.
    pub fn main_span(&self) -> Span<'a> {
        self.inner.main_span
    }

    /// Returns auxiliary spans for this error.
    pub fn aux_spans(&self) -> &[Spanned<'a, AuxErrorInfo>] {
        &self.inner.aux_spans
    }

    /// Returns error backtrace.
    pub fn backtrace(&self) -> &Backtrace<'a> {
        &self.backtrace
    }
}
