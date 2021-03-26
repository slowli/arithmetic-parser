//! Tuple types.

use std::{borrow::Cow, cmp, fmt, iter};

use crate::{Num, PrimitiveType, ValueType};

/// Length of a tuple.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TupleLength {
    /// Wildcard length.
    Some {
        /// Is this length dynamic (can vary at runtime)?
        is_dynamic: bool,
    },
    /// Exact known length.
    Exact(usize),
    /// Length parameter in a function definition.
    Param(usize),

    /// Length variable. In contrast to `Param`s, `Var`s are used exclusively during
    /// inference and cannot occur in standalone function signatures.
    Var(usize),
}

impl fmt::Display for TupleLength {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Some { is_dynamic: false } | Self::Var(_) => formatter.write_str("_"),
            Self::Some { is_dynamic: true } => formatter.write_str("*"),
            Self::Exact(len) => fmt::Display::fmt(len, formatter),
            Self::Param(idx) => formatter.write_str(Self::const_param(*idx).as_ref()),
        }
    }
}

impl TupleLength {
    pub(crate) fn const_param(index: usize) -> Cow<'static, str> {
        const PARAM_NAMES: &str = "NMLKJI";
        PARAM_NAMES.get(index..=index).map_or_else(
            || Cow::from(format!("N{}", index - PARAM_NAMES.len())),
            Cow::from,
        )
    }

    fn is_concrete(&self) -> bool {
        matches!(self, Self::Param(_) | Self::Exact(_))
    }
}

/// Kind of a length parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum LengthKind {
    /// Length is static (can be found during type inference / "in compile time").
    Static,
    /// Length is dynamic (can vary at runtime).
    Dynamic,
}

/// Tuple type.
#[derive(Debug, Clone)]
pub struct Tuple<Prim: PrimitiveType = Num> {
    start: Vec<ValueType<Prim>>,
    middle: Option<Slice<Prim>>,
    end: Vec<ValueType<Prim>>,
}

impl<Prim: PrimitiveType> PartialEq for Tuple<Prim> {
    fn eq(&self, other: &Self) -> bool {
        let this_len = self.len();
        if this_len != other.len() {
            false
        } else if this_len.1.is_none() {
            self.equal_elements_static(other, this_len.0)
                .all(|(x, y)| x == y)
        } else {
            self.equal_elements_dyn(other).all(|(x, y)| x == y)
        }
    }
}

impl<Prim: PrimitiveType> fmt::Display for Tuple<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(slice) = self.as_slice() {
            if !matches!(slice.length, TupleLength::Exact(_)) {
                return fmt::Display::fmt(slice, formatter);
            }
        }
        self.format_as_tuple(formatter)
    }
}

impl<Prim: PrimitiveType> Tuple<Prim> {
    pub(crate) fn new(
        start: Vec<ValueType<Prim>>,
        middle: Option<Slice<Prim>>,
        end: Vec<ValueType<Prim>>,
    ) -> Self {
        Self { start, middle, end }
    }

    pub(crate) fn empty() -> Self {
        Self {
            start: Vec::new(),
            middle: None,
            end: Vec::new(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.start.is_empty()
            && self.end.is_empty()
            && self.resolved_middle_len() == TupleLength::Exact(0)
    }

    pub(crate) fn is_concrete(&self) -> bool {
        self.start
            .iter()
            .chain(&self.end)
            .all(ValueType::is_concrete)
            && self.middle.as_ref().map_or(true, Slice::is_concrete)
    }

    /// Returns this tuple as slice if it fits (has no start or end components).
    pub fn as_slice(&self) -> Option<&Slice<Prim>> {
        self.middle
            .as_ref()
            .filter(|_| self.start.is_empty() && self.end.is_empty())
    }

    pub(crate) fn format_as_tuple(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("(")?;

        for (i, element) in self.start.iter().enumerate() {
            fmt::Display::fmt(element, formatter)?;
            if i + 1 < self.start.len() || self.middle.is_some() {
                formatter.write_str(", ")?;
            }
        }

        if let Some(middle) = &self.middle {
            if let TupleLength::Exact(len) = middle.length {
                // Write the slice inline, not separating it into square brackets.
                for i in 0..len {
                    fmt::Display::fmt(&middle.element, formatter)?;
                    if i + 1 < len {
                        formatter.write_str(", ")?;
                    }
                }
            } else {
                formatter.write_str("...")?;
                fmt::Display::fmt(middle, formatter)?;
            }
        }
        if !self.end.is_empty() {
            formatter.write_str(", ")?;
        }

        for (i, element) in self.end.iter().enumerate() {
            fmt::Display::fmt(element, formatter)?;
            if i + 1 < self.end.len() {
                formatter.write_str(", ")?;
            }
        }

        formatter.write_str(")")
    }

    fn resolved_middle_len(&self) -> TupleLength {
        self.middle
            .as_ref()
            .map_or(TupleLength::Exact(0), |middle| middle.length)
    }

    fn middle_element(&self) -> &ValueType<Prim> {
        self.middle
            .as_ref()
            .map_or(&ValueType::Var(0), |middle| middle.element.as_ref())
    }

    pub(crate) fn middle_len_mut(&mut self) -> Option<&mut TupleLength> {
        self.middle.as_mut().map(|middle| &mut middle.length)
    }

    /// Returns components of this length in the form `(exact, variable)`.
    pub(crate) fn len(&self) -> (usize, Option<TupleLength>) {
        let exact = self.start.len() + self.end.len();
        let middle_len = self.resolved_middle_len();
        if let TupleLength::Exact(middle_len) = middle_len {
            (exact + middle_len, None)
        } else {
            (exact, Some(middle_len))
        }
    }

    pub(crate) fn push(&mut self, element: ValueType<Prim>) {
        self.start.push(element);
    }

    /// Returns pairs of elements of this and `other` tuple that should be equal to each other.
    ///
    /// This method is specialized for the case when the length of middles is known.
    pub(crate) fn equal_elements_static<'a>(
        &'a self,
        other: &'a Self,
        total_len: usize,
    ) -> impl Iterator<Item = (&'a ValueType<Prim>, &'a ValueType<Prim>)> + 'a {
        let middle_len = total_len - self.start.len() - self.end.len();
        let other_middle_len = total_len - other.start.len() - other.end.len();

        let this_iter = self
            .start
            .iter()
            .chain(iter::repeat(self.middle_element()).take(middle_len))
            .chain(&self.end);
        let other_iter = other
            .start
            .iter()
            .chain(iter::repeat(other.middle_element()).take(other_middle_len))
            .chain(&other.end);
        this_iter.zip(other_iter)
    }

    /// Returns pairs of elements of this and `other` tuple that should be equal to each other.
    ///
    /// This method is specialized for the case when the length of middles is unknown,
    /// i.e., `self.len()` returns `(_, Some(_))`.
    pub(crate) fn equal_elements_dyn<'a>(
        &'a self,
        other: &'a Self,
    ) -> impl Iterator<Item = (&'a ValueType<Prim>, &'a ValueType<Prim>)> + 'a {
        let middle_elem = self.middle.as_ref().unwrap().element.as_ref();
        let other_middle_elem = other.middle.as_ref().unwrap().element.as_ref();
        // `unwrap`s are safe due to checks above.
        let iter = iter::once((middle_elem, other_middle_elem));

        let borders_iter = self
            .start
            .iter()
            .zip(&other.start)
            .chain(self.end.iter().rev().zip(other.end.iter().rev()));
        let iter = iter.chain(borders_iter);

        let skip_at_start = cmp::min(self.start.len(), other.start.len());
        let skip_at_end = cmp::min(self.end.len(), other.end.len());
        let middle = self
            .start
            .iter()
            .skip(skip_at_start)
            .chain(self.end.iter().rev().skip(skip_at_end));
        let iter = iter.chain(middle.map(move |elem| (elem, middle_elem)));

        let other_middle = other
            .start
            .iter()
            .skip(skip_at_start)
            .chain(self.end.iter().rev().skip(skip_at_end));
        iter.chain(other_middle.map(move |elem| (elem, middle_elem)))
    }

    /// Iterates over all distinct elements in this tuple.
    pub(crate) fn element_types(&self) -> impl Iterator<Item = &ValueType<Prim>> + '_ {
        let middle_element = self.middle.as_ref().map(|slice| slice.element.as_ref());
        self.start.iter().chain(middle_element).chain(&self.end)
    }

    pub(crate) fn element_types_mut(&mut self) -> impl Iterator<Item = &mut ValueType<Prim>> + '_ {
        let middle_element = self.middle.as_mut().map(|slice| slice.element.as_mut());
        self.start
            .iter_mut()
            .chain(middle_element)
            .chain(&mut self.end)
    }

    pub(crate) fn map_types(
        &self,
        mut map_fn: impl FnMut(&ValueType<Prim>) -> ValueType<Prim>,
    ) -> Self {
        Self {
            start: self.start.iter().map(&mut map_fn).collect(),
            middle: self.middle.as_ref().map(|middle| Slice {
                element: Box::new(map_fn(&middle.element)),
                length: middle.length,
            }),
            end: self.end.iter().map(&mut map_fn).collect(),
        }
    }
}

impl<Prim: PrimitiveType> From<Vec<ValueType<Prim>>> for Tuple<Prim> {
    fn from(elements: Vec<ValueType<Prim>>) -> Self {
        Self {
            start: elements,
            middle: None,
            end: Vec::new(),
        }
    }
}

/// Slice type. Unlike in Rust, slices are a subset of tuples. If `length` is
/// [`Exact`](TupleLength::Exact), the slice is completely equivalent
/// to the corresponding tuple.
#[derive(Debug, Clone, PartialEq)]
pub struct Slice<Prim: PrimitiveType = Num> {
    element: Box<ValueType<Prim>>,
    length: TupleLength,
}

impl<Prim: PrimitiveType> fmt::Display for Slice<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "[{}; {}]", self.element, self.length)
    }
}

impl<Prim: PrimitiveType> Slice<Prim> {
    pub(crate) fn new(element: ValueType<Prim>, length: TupleLength) -> Self {
        Self {
            element: Box::new(element),
            length,
        }
    }

    /// Returns the element type of this slice.
    pub fn element(&self) -> &ValueType<Prim> {
        self.element.as_ref()
    }

    /// Returns the length of this slice.
    pub fn len(&self) -> TupleLength {
        self.length
    }

    fn is_concrete(&self) -> bool {
        self.length.is_concrete() && self.element.is_concrete()
    }
}

impl<Prim: PrimitiveType> From<Slice<Prim>> for Tuple<Prim> {
    fn from(slice: Slice<Prim>) -> Self {
        Self {
            start: Vec::new(),
            middle: Some(slice),
            end: Vec::new(),
        }
    }
}

// FIXME: test `Display`
// FIXME: test equal items iterators
