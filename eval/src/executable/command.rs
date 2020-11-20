//! Executable `Command` and its building blocks.

use crate::alloc::{String, Vec};
use arithmetic_parser::{BinaryOp, LvalueLen, MaybeSpanned, StripCode, UnaryOp};

/// Pointer to a register or constant.
#[derive(Debug)]
pub(crate) enum Atom<T> {
    Constant(T),
    Register(usize),
    Void,
}

impl<T: Clone> Clone for Atom<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Constant(literal) => Self::Constant(literal.clone()),
            Self::Register(index) => Self::Register(*index),
            Self::Void => Self::Void,
        }
    }
}

pub(crate) type SpannedAtom<'a, T> = MaybeSpanned<'a, Atom<T>>;

/// Atomic operation on registers and/or constants.
#[derive(Debug)]
pub(crate) enum CompiledExpr<'a, T> {
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
        // Original function name if it is a proper variable name.
        original_name: Option<String>,
        args: Vec<SpannedAtom<'a, T>>,
    },
    DefineFunction {
        ptr: usize,
        captures: Vec<SpannedAtom<'a, T>>,
        // Original capture names.
        capture_names: Vec<String>,
    },
}

impl<T: Clone> Clone for CompiledExpr<'_, T> {
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

            Self::Function {
                name,
                original_name,
                args,
            } => Self::Function {
                name: name.clone(),
                original_name: original_name.clone(),
                args: args.clone(),
            },

            Self::DefineFunction {
                ptr,
                captures,
                capture_names,
            } => Self::DefineFunction {
                ptr: *ptr,
                captures: captures.clone(),
                capture_names: capture_names.clone(),
            },
        }
    }
}

impl<T: 'static + Clone> StripCode for CompiledExpr<'_, T> {
    type Stripped = CompiledExpr<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        match self {
            Self::Atom(atom) => CompiledExpr::Atom(atom),
            Self::Tuple(atoms) => CompiledExpr::Tuple(atoms),

            Self::Unary { op, inner } => CompiledExpr::Unary {
                op,
                inner: inner.strip_code(),
            },

            Self::Binary { op, lhs, rhs } => CompiledExpr::Binary {
                op,
                lhs: lhs.strip_code(),
                rhs: rhs.strip_code(),
            },

            Self::Function {
                name,
                original_name,
                args,
            } => CompiledExpr::Function {
                name: name.strip_code(),
                original_name,
                args: args.into_iter().map(StripCode::strip_code).collect(),
            },

            Self::DefineFunction {
                ptr,
                captures,
                capture_names,
            } => CompiledExpr::DefineFunction {
                ptr,
                captures: captures.into_iter().map(StripCode::strip_code).collect(),
                capture_names,
            },
        }
    }
}

/// Commands for a primitive register VM used to execute compiled programs.
#[derive(Debug)]
pub(crate) enum Command<'a, T> {
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
    Annotate { register: usize, name: String },

    /// Signals that the following commands are executed in the inner scope.
    StartInnerScope,
    /// Signals that the following commands are executed in the global scope.
    EndInnerScope,
    /// Signals to truncate registers to the specified number.
    TruncateRegisters(usize),
}

impl<T: Clone> Clone for Command<'_, T> {
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
                name: name.clone(),
            },

            Self::StartInnerScope => Self::StartInnerScope,
            Self::EndInnerScope => Self::EndInnerScope,
            Self::TruncateRegisters(size) => Self::TruncateRegisters(*size),
        }
    }
}

impl<T: 'static + Clone> StripCode for Command<'_, T> {
    type Stripped = Command<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        match self {
            Self::Push(expr) => Command::Push(expr.strip_code()),

            Self::Destructure {
                source,
                start_len,
                end_len,
                lvalue_len,
                unchecked,
            } => Command::Destructure {
                source,
                start_len,
                end_len,
                lvalue_len,
                unchecked,
            },

            Self::Copy {
                source,
                destination,
            } => Command::Copy {
                source,
                destination,
            },

            Self::Annotate { register, name } => Command::Annotate { register, name },

            Self::StartInnerScope => Command::StartInnerScope,
            Self::EndInnerScope => Command::EndInnerScope,
            Self::TruncateRegisters(size) => Command::TruncateRegisters(size),
        }
    }
}

pub(crate) type SpannedCommand<'a, T> = MaybeSpanned<'a, Command<'a, T>>;
