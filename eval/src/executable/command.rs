//! Executable `Command` and its building blocks.

use num_traits::{One, Zero};

use std::cmp::Ordering;

use crate::{Number, Value};
use arithmetic_parser::{BinaryOp, Grammar, LvalueLen, MaybeSpanned, StripCode, UnaryOp};

/// Pointer to a register or constant.
#[derive(Debug)]
pub(crate) enum Atom<T: Grammar> {
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

pub(crate) type SpannedAtom<'a, T> = MaybeSpanned<'a, Atom<T>>;

/// Atomic operation on registers and/or constants.
#[derive(Debug)]
pub(crate) enum CompiledExpr<'a, T: Grammar> {
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
        // Original function name if it is a proper variable name.
        original_name: Option<String>,
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

            Self::Function {
                name,
                original_name,
                args,
            } => Self::Function {
                name: name.clone(),
                original_name: original_name.clone(),
                args: args.clone(),
            },

            Self::DefineFunction { ptr, captures } => Self::DefineFunction {
                ptr: *ptr,
                captures: captures.clone(),
            },
        }
    }
}

impl<T: Grammar> StripCode for CompiledExpr<'_, T> {
    type Stripped = CompiledExpr<'static, T>;

    fn strip_code(&self) -> Self::Stripped {
        match self {
            Self::Atom(atom) => CompiledExpr::Atom(atom.clone()),
            Self::Tuple(atoms) => CompiledExpr::Tuple(atoms.clone()),

            Self::Unary { op, inner } => CompiledExpr::Unary {
                op: *op,
                inner: inner.strip_code(),
            },

            Self::Binary { op, lhs, rhs } => CompiledExpr::Binary {
                op: *op,
                lhs: lhs.strip_code(),
                rhs: rhs.strip_code(),
            },

            Self::Compare { inner, op } => CompiledExpr::Compare {
                inner: inner.strip_code(),
                op: *op,
            },

            Self::Function {
                name,
                original_name,
                args,
            } => CompiledExpr::Function {
                name: name.strip_code(),
                original_name: original_name.clone(),
                args: args.iter().map(StripCode::strip_code).collect(),
            },

            Self::DefineFunction { ptr, captures } => CompiledExpr::DefineFunction {
                ptr: *ptr,
                captures: captures.iter().map(StripCode::strip_code).collect(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ComparisonOp {
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

    pub fn compare<T>(self, cmp_value: &Value<'_, T>) -> Option<bool>
    where
        T: Grammar,
        T::Lit: Number,
    {
        let ordering = match cmp_value {
            Value::Number(num) if num.is_one() => Ordering::Greater,
            Value::Number(num) if num.is_zero() => Ordering::Equal,
            Value::Number(num) if (-*num).is_one() => Ordering::Less,
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
pub(crate) enum Command<'a, T: Grammar> {
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
                name: name.clone(),
            },

            Self::StartInnerScope => Self::StartInnerScope,
            Self::EndInnerScope => Self::EndInnerScope,
            Self::TruncateRegisters(size) => Self::TruncateRegisters(*size),
        }
    }
}

impl<T: Grammar> StripCode for Command<'_, T> {
    type Stripped = Command<'static, T>;

    fn strip_code(&self) -> Self::Stripped {
        match self {
            Self::Push(expr) => Command::Push(expr.strip_code()),

            Self::Destructure {
                source,
                start_len,
                end_len,
                lvalue_len,
                unchecked,
            } => Command::Destructure {
                source: *source,
                start_len: *start_len,
                end_len: *end_len,
                lvalue_len: *lvalue_len,
                unchecked: *unchecked,
            },

            Self::Copy {
                source,
                destination,
            } => Command::Copy {
                source: *source,
                destination: *destination,
            },

            Self::Annotate { register, name } => Command::Annotate {
                register: *register,
                name: name.clone(),
            },

            Self::StartInnerScope => Command::StartInnerScope,
            Self::EndInnerScope => Command::EndInnerScope,
            Self::TruncateRegisters(size) => Command::TruncateRegisters(*size),
        }
    }
}

pub(crate) type SpannedCommand<'a, T> = MaybeSpanned<'a, Command<'a, T>>;
