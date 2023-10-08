//! Executable `Command` and its building blocks.

use crate::alloc::{String, Vec};
use arithmetic_parser::{BinaryOp, Location, LvalueLen, UnaryOp};

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

pub(crate) type LocatedAtom<T> = Location<Atom<T>>;

#[derive(Debug, Clone)]
pub(crate) enum FieldName {
    Index(usize),
    Name(String),
}

/// Atomic operation on registers and/or constants.
#[derive(Debug, Clone)]
pub(crate) enum CompiledExpr<T> {
    Atom(Atom<T>),
    Tuple(Vec<Atom<T>>),
    Object(Vec<(String, Atom<T>)>),
    Unary {
        op: UnaryOp,
        inner: LocatedAtom<T>,
    },
    Binary {
        op: BinaryOp,
        lhs: LocatedAtom<T>,
        rhs: LocatedAtom<T>,
    },
    FieldAccess {
        receiver: LocatedAtom<T>,
        field: FieldName,
    },
    FunctionCall {
        name: LocatedAtom<T>,
        // Original function name if it is a proper variable name.
        original_name: Option<String>,
        args: Vec<LocatedAtom<T>>,
    },
    DefineFunction {
        ptr: usize,
        captures: Vec<LocatedAtom<T>>,
        // Original capture names.
        capture_names: Vec<String>,
    },
}

/// Commands for a primitive register VM used to execute compiled programs.
#[derive(Debug, Clone)]
pub(crate) enum Command<T> {
    /// Create a new register and push the result of the specified computation there.
    Push(CompiledExpr<T>),

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

pub(crate) type LocatedCommand<T> = Location<Command<T>>;
