//! Evaluation errors.

use thiserror::Error;

use crate::{helpers::create_span_ref, interpreter::Value, Op, Span, Spanned};

/// Errors that can occur during interpreting expressions and statements.
#[derive(Debug, Error)]
pub enum EvalError {
    /// Mismatch between length of tuples in a binary operation or assignment.
    #[error(
        "Mismatch between length of tuples in a binary operation or assignment: \
         LHS has {lhs} element(s), whereas RHS has {rhs}"
    )]
    TupleLenMismatch {
        /// Length of a tuple on the left-hand side.
        lhs: usize,
        /// Length of a tuple on the right-hand side.
        rhs: usize,
    },

    /// Mismatch between the number of arguments in the function definition and its call.
    #[error(
        "Mismatch between the number of arguments in the function definition and its call: \
         definition requires {def} arg(s), call has {call}"
    )]
    ArgsLenMismatch {
        /// Number of args at the function definition.
        def: usize,
        /// Number of args at the function call.
        call: usize,
    },

    /// Cannot destructure a no-tuple variable.
    #[error("Cannot destructure a no-tuple variable.")]
    CannotDestructure,

    /// Repeated assignment to the same variable in function args or tuple destructuring.
    #[error("Repeated assignment to the same variable in function args or tuple destructuring.")]
    RepeatedAssignment,

    /// Variable with the enclosed name is not defined.
    #[error("Variable {0} is not defined")]
    Undefined(String),

    /// Value is not callable (i.e., is not a function).
    #[error("Value is not callable")]
    CannotCall,

    /// Generic error during execution of a native function.
    #[error("Failed executing native function: {0}")]
    NativeCall(String),

    /// Unexpected operand type(s) for the specified operation.
    #[error("Unexpected operand type(s) for {op}")]
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
}

/// Evaluation error together with one or more relevant code spans.
#[derive(Debug)]
pub struct SpannedEvalError<'a> {
    error: EvalError,
    main_span: Span<'a>,
}

impl<'a> SpannedEvalError<'a> {
    pub(super) fn new<U>(main_span: &Spanned<'a, U>, error: EvalError) -> Self {
        Self {
            error,
            main_span: create_span_ref(main_span, ()),
        }
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
    pub def_span: Span<'a>,
    /// Code span of the function call.
    pub call_span: Span<'a>,
}

impl<'a> Backtrace<'a> {
    /// Iterates over the backtrace, starting from the most recent call.
    pub fn calls(&self) -> impl Iterator<Item = BacktraceElement<'a>> + '_ {
        self.calls.iter().rev().cloned()
    }

    /// Appends a function call into the backtrace.
    pub(super) fn push_call(&mut self, fn_name: &'a str, def_span: Span<'a>, call_span: Span<'a>) {
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

    /// Returns the source of the error.
    pub fn source(&self) -> &EvalError {
        &self.inner.error
    }

    /// Returns the main span for this error.
    pub fn main_span(&self) -> Span<'a> {
        self.inner.main_span
    }

    /// Returns error backtrace.
    pub fn backtrace(&self) -> &Backtrace<'a> {
        &self.backtrace
    }
}
