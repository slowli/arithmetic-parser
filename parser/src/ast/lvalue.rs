//! Lvalues for arithmetic expressions.

use core::fmt;

use crate::spans::Spanned;

/// Length of an assigned lvalue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LvalueLen {
    /// Exact length.
    Exact(usize),
    /// Minimum length.
    AtLeast(usize),
}

impl LvalueLen {
    /// Checks if this length matches the provided length of the rvalue.
    pub fn matches(self, value: usize) -> bool {
        match self {
            Self::Exact(len) => value == len,
            Self::AtLeast(len) => value >= len,
        }
    }
}

impl fmt::Display for LvalueLen {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(len) => write!(formatter, "{len}"),
            Self::AtLeast(len) => write!(formatter, "at least {len}"),
        }
    }
}

impl From<usize> for LvalueLen {
    fn from(value: usize) -> Self {
        Self::Exact(value)
    }
}

/// Tuple destructuring, such as `(a, b, ..., c)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Destructure<'a, T> {
    /// Start part of the destructuring, e.g, `a` and `b` in `(a, b, ..., c)`.
    pub start: Vec<SpannedLvalue<'a, T>>,
    /// Middle part of the destructuring, e.g., `rest` in `(a, b, ...rest, _)`.
    pub middle: Option<Spanned<'a, DestructureRest<'a, T>>>,
    /// End part of the destructuring, e.g., `c` in `(a, b, ..., c)`.
    pub end: Vec<SpannedLvalue<'a, T>>,
}

impl<T> Destructure<'_, T> {
    /// Returns the length of destructured elements.
    pub fn len(&self) -> LvalueLen {
        if self.middle.is_some() {
            LvalueLen::AtLeast(self.start.len() + self.end.len())
        } else {
            LvalueLen::Exact(self.start.len())
        }
    }

    /// Checks if the destructuring is empty.
    pub fn is_empty(&self) -> bool {
        self.start.is_empty()
    }
}

/// Rest syntax, such as `...rest` in `(a, ...rest, b)`.
#[derive(Debug, Clone, PartialEq)]
pub enum DestructureRest<'a, T> {
    /// Unnamed rest syntax, i.e., `...`.
    Unnamed,
    /// Named rest syntax, e.g., `...rest`.
    Named {
        /// Variable span, e.g., `rest`.
        variable: Spanned<'a>,
        /// Type annotation of the value.
        ty: Option<Spanned<'a, T>>,
    },
}

impl<'a, T> DestructureRest<'a, T> {
    /// Tries to convert this rest declaration into an lvalue. Return `None` if the rest declaration
    /// is unnamed.
    pub fn to_lvalue(&self) -> Option<SpannedLvalue<'a, T>> {
        match self {
            Self::Named { variable, .. } => {
                Some(variable.copy_with_extra(Lvalue::Variable { ty: None }))
            }
            Self::Unnamed => None,
        }
    }
}

/// Object destructuring, such as `{ x, y: new_y }`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ObjectDestructure<'a, T> {
    /// Fields mentioned in the destructuring.
    pub fields: Vec<ObjectDestructureField<'a, T>>,
}

/// Single field in [`ObjectDestructure`], such as `x` and `y: new_y` in `{ x, y: new_y }`.
///
/// In addition to the "ordinary" `field: lvalue` syntax for a field with binding,
/// an alternative one is supported: `field -> lvalue`. This makes the case
/// of a field with type annotation easier to recognize (for humans); `field -> lvalue: Type` is
/// arguably more readable than `field: lvalue: Type` (although the latter is still valid syntax).
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectDestructureField<'a, T> {
    /// Field name, such as `xs` in `xs: (x, ...tail)`.
    pub field_name: Spanned<'a>,
    /// Binding for the field, such as `(x, ...tail)` in `xs: (x, ...tail)`.
    pub binding: Option<SpannedLvalue<'a, T>>,
}

/// Assignable value.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Lvalue<'a, T> {
    /// Simple variable, e.g., `x`.
    Variable {
        /// Type annotation of the value.
        ty: Option<Spanned<'a, T>>,
    },
    /// Tuple destructuring, e.g., `(x, y)`.
    Tuple(Destructure<'a, T>),
    /// Object destructuring, e.g., `{ x, y }`.
    Object(ObjectDestructure<'a, T>),
}

impl<T> Lvalue<'_, T> {
    /// Returns type of this lvalue.
    pub fn ty(&self) -> LvalueType {
        match self {
            Self::Variable { .. } => LvalueType::Variable,
            Self::Tuple(_) => LvalueType::Tuple,
            Self::Object(_) => LvalueType::Object,
        }
    }
}

/// [`Lvalue`] with the associated code span.
pub type SpannedLvalue<'a, T> = Spanned<'a, Lvalue<'a, T>>;

/// Type of an [`Lvalue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LvalueType {
    /// Simple variable, e.g., `x`.
    Variable,
    /// Tuple destructuring, e.g., `(x, y)`.
    Tuple,
    /// Object destructuring, e.g., `{ x, y }`.
    Object,
}

impl fmt::Display for LvalueType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Variable => "simple variable",
            Self::Tuple => "tuple destructuring",
            Self::Object => "object destructuring",
        })
    }
}
