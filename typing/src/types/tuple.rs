//! Tuple types.

use std::{borrow::Cow, cmp, fmt, iter, ops};

use crate::{Num, PrimitiveType, ValueType};

/// Unknown / variable length, e.g., of a tuple.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum UnknownLen {
    /// Wildcard length, i.e. some length that is not specified. Similar to `_` in type annotations
    /// in Rust. Unlike [`Self::Dynamic`], this length can be found during type inference.
    Some,
    /// *Dynamic* wildcard length. Unlike [`Self::Some`], this length can vary at runtime,
    /// i.e., it cannot be unified with any other length during type inference.
    Dynamic,
    /// Length parameter in a function definition.
    Param(usize),

    /// Length variable. In contrast to `Param`s, `Var`s are used exclusively during
    /// inference and cannot occur in standalone function signatures.
    Var(usize),
}

impl fmt::Display for UnknownLen {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Some | Self::Var(_) => formatter.write_str("_"),
            Self::Dynamic => formatter.write_str("*"),
            Self::Param(idx) => formatter.write_str(Self::const_param(*idx).as_ref()),
        }
    }
}

impl UnknownLen {
    pub(crate) fn const_param(index: usize) -> Cow<'static, str> {
        const PARAM_NAMES: &str = "NMLKJI";
        PARAM_NAMES.get(index..=index).map_or_else(
            || Cow::from(format!("N{}", index - PARAM_NAMES.len())),
            Cow::from,
        )
    }
}

impl ops::Add<usize> for UnknownLen {
    type Output = TupleLen;

    fn add(self, rhs: usize) -> Self::Output {
        TupleLen {
            var: Some(self),
            exact: rhs,
        }
    }
}

/// Generic tuple length.
///
/// A tuple length consists of the two components: an unknown / variable length,
/// such as [`UnknownLen::Param`], and a non-negative increment.
/// These components can be obtained via [`Self::components()`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TupleLen {
    var: Option<UnknownLen>,
    exact: usize,
}

impl fmt::Display for TupleLen {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.var, self.exact) {
            (Some(var), 0) => fmt::Display::fmt(var, formatter),
            (Some(var), exact) => write!(formatter, "{} + {}", var, exact),
            (None, exact) => fmt::Display::fmt(&exact, formatter),
        }
    }
}

impl ops::Add<usize> for TupleLen {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        Self {
            var: self.var,
            exact: self.exact + rhs,
        }
    }
}

impl From<UnknownLen> for TupleLen {
    fn from(var: UnknownLen) -> Self {
        Self {
            var: Some(var),
            exact: 0,
        }
    }
}

impl From<usize> for TupleLen {
    fn from(exact: usize) -> Self {
        Self { var: None, exact }
    }
}

impl TupleLen {
    /// Zero length.
    pub(crate) const ZERO: Self = Self {
        var: None,
        exact: 0,
    };

    fn is_concrete(&self) -> bool {
        !matches!(&self.var, Some(UnknownLen::Var(_)))
    }

    /// Returns components of this length.
    pub fn components(&self) -> (Option<UnknownLen>, usize) {
        (self.var, self.exact)
    }

    /// Returns mutable references to the components of this length.
    pub fn components_mut(&mut self) -> (Option<&mut UnknownLen>, &mut usize) {
        (self.var.as_mut(), &mut self.exact)
    }
}

/// Kind of a length parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum LengthKind {
    /// Length is static (can be found during type inference / "in compile time").
    Static,
    /// Length is dynamic (can vary at runtime). Dynamic lengths cannot be unified with
    /// any other length during type inference.
    Dynamic,
}

/// Tuple type.
///
/// Most generally, a tuple type consists of three fragments: *start*,
/// *middle* and *end*. Types at the start and at the end are heterogeneous,
/// while the middle always contains items of the same type (but the number
/// of these items can generally vary). A [`Slice`] is a partial case of a tuple type;
/// i.e., a type with the empty start and end. Likewise, a Rust-like tuple is a tuple
/// that only has a start.
///
/// # Notation
///
/// A tuple type is denoted like `(T, U, ...[V; _], X, Y)`, where `T` and `U` are types
/// at the start, `V` is the middle type, and `X`, `Y` are types at the end.
/// The number of middle elements can be parametric, such as `N`.
/// If a tuple only has a start, this notation collapses into Rust-like `(T, U)`.
/// If a tuple only has a middle part ([`Self::as_slice()`] returns `Some(_)`),
/// it is denoted as the corresponding slice, something like `[T; N]`.
///
/// # Examples
///
/// Simple tuples can be created using the [`From`] trait. Complex tuples can be created
/// via [`Self::new()`].
///
/// ```
/// # use arithmetic_typing::{Slice, Tuple, UnknownLen, ValueType};
/// # use assert_matches::assert_matches;
/// let simple_tuple = Tuple::from(vec![ValueType::NUM, ValueType::BOOL]);
/// assert_matches!(simple_tuple.parts(), ([_, _], None, []));
/// assert!(simple_tuple.as_slice().is_none());
/// assert_eq!(simple_tuple.to_string(), "(Num, Bool)");
///
/// let slice_tuple = Tuple::from(
///    ValueType::NUM.repeat(UnknownLen::Param(0)),
/// );
/// assert_matches!(slice_tuple.parts(), ([], Some(_), []));
/// assert!(slice_tuple.as_slice().is_some());
/// assert_eq!(slice_tuple.to_string(), "[Num; N]");
///
/// let complex_tuple = Tuple::new(
///     vec![ValueType::NUM],
///     ValueType::NUM.repeat(UnknownLen::Param(0)),
///     vec![ValueType::BOOL, ValueType::Some],
/// );
/// assert_matches!(complex_tuple.parts(), ([_], Some(_), [_, _]));
/// assert_eq!(complex_tuple.to_string(), "(Num, ...[Num; N], Bool, _)");
/// ```
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
        } else if let (None, len) = this_len.components() {
            self.equal_elements_static(other, len).all(|(x, y)| x == y)
        } else {
            self.equal_elements_dyn(other).all(|(x, y)| x == y)
        }
    }
}

impl<Prim: PrimitiveType> fmt::Display for Tuple<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(slice) = self.as_slice() {
            if let (Some(_), _) = slice.length.components() {
                return fmt::Display::fmt(slice, formatter);
            }
        }
        self.format_as_tuple(formatter)
    }
}

impl<Prim: PrimitiveType> Tuple<Prim> {
    pub(crate) fn from_parts(
        start: Vec<ValueType<Prim>>,
        middle: Option<Slice<Prim>>,
        end: Vec<ValueType<Prim>>,
    ) -> Self {
        Self { start, middle, end }
    }

    /// Creates a new complex tuple.
    pub fn new(
        start: Vec<ValueType<Prim>>,
        middle: Slice<Prim>,
        end: Vec<ValueType<Prim>>,
    ) -> Self {
        Self::from_parts(start, Some(middle), end)
    }

    pub(crate) fn empty() -> Self {
        Self {
            start: Vec::new(),
            middle: None,
            end: Vec::new(),
        }
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
            if let (None, len) = middle.length.components() {
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

    fn resolved_middle_len(&self) -> TupleLen {
        self.middle
            .as_ref()
            .map_or(TupleLen::ZERO, |middle| middle.length)
    }

    fn middle_element(&self) -> &ValueType<Prim> {
        self.middle
            .as_ref()
            .map_or(&ValueType::Var(0), |middle| middle.element.as_ref())
    }

    /// Returns shared references to the parts comprising this tuple: start, middle, and end.
    #[allow(clippy::type_complexity)]
    pub fn parts(&self) -> (&[ValueType<Prim>], Option<&Slice<Prim>>, &[ValueType<Prim>]) {
        (&self.start, self.middle.as_ref(), &self.end)
    }

    /// Returns exclusive references to the parts comprising this tuple: start, middle, and end.
    #[allow(clippy::type_complexity)]
    pub fn parts_mut(
        &mut self,
    ) -> (
        &mut [ValueType<Prim>],
        Option<&mut Slice<Prim>>,
        &mut [ValueType<Prim>],
    ) {
        (&mut self.start, self.middle.as_mut(), &mut self.end)
    }

    /// Returns the length of this tuple.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arithmetic_typing::{Slice, Tuple, ValueType, UnknownLen, TupleLen};
    /// let tuple = Tuple::from(vec![ValueType::NUM, ValueType::BOOL]);
    /// assert_eq!(tuple.len(), TupleLen::from(2));
    ///
    /// let slice = Slice::new(ValueType::NUM, UnknownLen::Param(0));
    /// let tuple = Tuple::from(slice.clone());
    /// assert_eq!(tuple.len(), TupleLen::from(UnknownLen::Param(0)));
    ///
    /// let tuple = Tuple::new(vec![], slice, vec![ValueType::BOOL]);
    /// assert_eq!(tuple.len(), UnknownLen::Param(0) + 1);
    /// ```
    pub fn len(&self) -> TupleLen {
        let increment = self.start.len() + self.end.len();
        self.resolved_middle_len().to_owned() + increment
    }

    /// Returns `true` iff this tuple is guaranteed to be empty.
    pub fn is_empty(&self) -> bool {
        self.start.is_empty() && self.end.is_empty() && self.resolved_middle_len() == TupleLen::ZERO
    }

    pub(crate) fn push(&mut self, element: ValueType<Prim>) {
        if self.middle.is_some() {
            self.end.push(element);
        } else {
            self.start.push(element);
        }
    }

    pub(crate) fn set_middle(&mut self, element: ValueType<Prim>, len: TupleLen) {
        self.middle = Some(Slice::new(element, len));
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
    /// This method is specialized for the case when the length of middles is unknown.
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
        let iter = iter.chain(middle.map(move |elem| (middle_elem, elem)));

        let other_middle = other
            .start
            .iter()
            .skip(skip_at_start)
            .chain(other.end.iter().rev().skip(skip_at_end));
        iter.chain(other_middle.map(move |elem| (middle_elem, elem)))
    }

    /// Iterates over all distinct elements in this tuple. The iteration is performed in order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arithmetic_typing::{Slice, Tuple, UnknownLen, ValueType};
    /// let complex_tuple = Tuple::new(
    ///     vec![ValueType::NUM],
    ///     Slice::new(ValueType::NUM, UnknownLen::Param(0)),
    ///     vec![ValueType::BOOL, ValueType::Some],
    /// );
    /// let elements: Vec<_> = complex_tuple.element_types().collect();
    /// assert_eq!(
    ///     elements.as_slice(),
    ///     &[&ValueType::NUM, &ValueType::NUM, &ValueType::BOOL, &ValueType::Some]
    /// );
    /// ```
    pub fn element_types(&self) -> impl Iterator<Item = &ValueType<Prim>> + '_ {
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
/// [`Exact`](TupleLen::Exact), the slice is completely equivalent
/// to the corresponding tuple.
///
/// # Notation
///
/// A slice is denoted as `[T; N]` where `T` is the slice [element](Self::element())
/// and `N` is the slice [length](Self::len()). A special case is `[T]`, a slice
/// with a dynamic length.
///
/// # Examples
///
/// ```
/// use arithmetic_parser::grammars::{NumGrammar, Parse, Typed};
/// use arithmetic_typing::{Annotated, TupleLen, TypeEnvironment, ValueType};
///
/// # fn main() -> anyhow::Result<()> {
/// type Parser = Typed<Annotated<NumGrammar<f32>>>;
/// let ast = Parser::parse_statements("xs: [Num; _] = (1, 2, 3);")?;
/// let mut env = TypeEnvironment::new();
/// env.process_statements(&ast)?;
/// // Slices with fixed length are equivalent to tuples.
/// assert_eq!(env["xs"].to_string(), "(Num, Num, Num)");
///
/// let ast = Parser::parse_statements(r#"
///     xs: [Num] = (1, 2, 3);
///     ys = xs + 1; // works fine: despite `xs` having unknown length,
///                  // it's always possible to add a number to it
///     (_, _, z) = xs; // does not work: the tuple length is erased
/// "#)?;
/// let err = env.process_statements(&ast).unwrap_err();
/// assert_eq!(*err.span().fragment(), "(_, _, z) = xs");
/// assert_eq!(env["ys"], env["xs"]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Slice<Prim: PrimitiveType = Num> {
    element: Box<ValueType<Prim>>,
    length: TupleLen,
}

impl<Prim: PrimitiveType> fmt::Display for Slice<Prim> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.length == TupleLen::from(UnknownLen::Dynamic) {
            write!(formatter, "[{}]", self.element)
        } else {
            write!(formatter, "[{}; {}]", self.element, self.length)
        }
    }
}

impl<Prim: PrimitiveType> Slice<Prim> {
    /// Creates a new slice.
    pub fn new(element: ValueType<Prim>, length: impl Into<TupleLen>) -> Self {
        Self {
            element: Box::new(element),
            length: length.into(),
        }
    }

    /// Returns the element type of this slice.
    pub fn element(&self) -> &ValueType<Prim> {
        self.element.as_ref()
    }

    /// Returns the length of this slice.
    pub fn len(&self) -> TupleLen {
        self.length
    }

    pub(crate) fn len_mut(&mut self) -> &mut TupleLen {
        &mut self.length
    }

    /// Returns `true` iff this slice is definitely empty.
    pub fn is_empty(&self) -> bool {
        self.length == TupleLen::ZERO
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuple_length_display() {
        let len = TupleLen::from(3);
        assert_eq!(len.to_string(), "3");
        let len = UnknownLen::Param(0) + 2;
        assert_eq!(len.to_string(), "N + 2");
    }

    #[test]
    fn slice_display() {
        let slice = Slice::new(ValueType::NUM, UnknownLen::Param(0));
        assert_eq!(slice.to_string(), "[Num; N]");
        let slice = Slice::new(ValueType::NUM, UnknownLen::Var(0));
        assert_eq!(slice.to_string(), "[Num; _]");
        let slice = Slice::new(ValueType::NUM, TupleLen::from(3));
        assert_eq!(slice.to_string(), "[Num; 3]");
    }

    #[test]
    fn tuple_display() {
        // Simple tuples.
        let tuple = Tuple::from(vec![ValueType::NUM, ValueType::BOOL]);
        assert_eq!(tuple.to_string(), "(Num, Bool)");
        let tuple = Tuple::from(Slice::new(ValueType::NUM, UnknownLen::Param(0)));
        assert_eq!(tuple.to_string(), "[Num; N]");
        let tuple = Tuple::from(Slice::new(ValueType::NUM, TupleLen::from(3)));
        assert_eq!(tuple.to_string(), "(Num, Num, Num)");

        let tuple = Tuple {
            start: vec![ValueType::NUM, ValueType::BOOL],
            middle: Some(Slice::new(ValueType::NUM, UnknownLen::Param(0))),
            end: vec![],
        };
        assert_eq!(tuple.to_string(), "(Num, Bool, ...[Num; N])");

        let tuple = Tuple {
            start: vec![ValueType::NUM, ValueType::BOOL],
            middle: Some(Slice::new(ValueType::NUM, TupleLen::from(2))),
            end: vec![],
        };
        assert_eq!(tuple.to_string(), "(Num, Bool, Num, Num)");

        let tuple = Tuple {
            start: vec![ValueType::NUM, ValueType::BOOL],
            middle: Some(Slice::new(ValueType::NUM, UnknownLen::Param(0))),
            end: vec![ValueType::Param(0)],
        };
        assert_eq!(tuple.to_string(), "(Num, Bool, ...[Num; N], T)");
    }

    #[test]
    fn equal_elements_static_two_simple_tuples() {
        let tuple = Tuple::from(vec![ValueType::NUM, ValueType::BOOL, ValueType::Var(0)]);
        let other_tuple = Tuple::from(vec![ValueType::Var(1), ValueType::BOOL, ValueType::Var(0)]);
        let equal_elements: Vec<_> = tuple.equal_elements_static(&other_tuple, 3).collect();

        assert_eq!(
            equal_elements,
            vec![
                (&ValueType::NUM, &ValueType::Var(1)),
                (&ValueType::BOOL, &ValueType::BOOL),
                (&ValueType::Var(0), &ValueType::Var(0)),
            ]
        );
    }

    #[test]
    fn equal_elements_static_simple_tuple_and_slice() {
        let tuple = Tuple::from(vec![ValueType::NUM, ValueType::BOOL, ValueType::Var(0)]);
        let slice = Tuple::from(Slice::new(ValueType::Var(1), UnknownLen::Var(0)));
        let equal_elements: Vec<_> = tuple.equal_elements_static(&slice, 3).collect();

        assert_eq!(
            equal_elements,
            vec![
                (&ValueType::NUM, &ValueType::Var(1)),
                (&ValueType::BOOL, &ValueType::Var(1)),
                (&ValueType::Var(0), &ValueType::Var(1)),
            ]
        );
    }

    #[test]
    fn equal_elements_static_slice_and_complex_tuple() {
        let slice = Tuple::from(Slice::new(ValueType::Var(1), UnknownLen::Var(0)));
        let tuple = Tuple {
            start: vec![ValueType::NUM],
            middle: Some(Slice::new(ValueType::Var(0), UnknownLen::Var(1))),
            end: vec![ValueType::BOOL, ValueType::Var(2)],
        };

        let mut expected_pairs = vec![
            (ValueType::Var(1), ValueType::NUM),
            (ValueType::Var(1), ValueType::BOOL),
            (ValueType::Var(1), ValueType::Var(2)),
        ];
        let equal_elements: Vec<_> = slice
            .equal_elements_static(&tuple, 3)
            .map(|(x, y)| (x.to_owned(), y.to_owned()))
            .collect();
        assert_eq!(equal_elements, expected_pairs);

        let equal_elements: Vec<_> = slice
            .equal_elements_static(&tuple, 4)
            .map(|(x, y)| (x.to_owned(), y.to_owned()))
            .collect();
        expected_pairs.insert(1, (ValueType::Var(1), ValueType::Var(0)));
        assert_eq!(equal_elements, expected_pairs);

        let equal_elements: Vec<_> = slice
            .equal_elements_static(&tuple, 5)
            .map(|(x, y)| (x.to_owned(), y.to_owned()))
            .collect();
        expected_pairs.insert(2, (ValueType::Var(1), ValueType::Var(0)));
        assert_eq!(equal_elements, expected_pairs);
    }

    fn create_test_tuples() -> (Tuple, Tuple) {
        let tuple = Tuple {
            start: vec![ValueType::NUM],
            middle: Some(Slice::new(ValueType::Var(0), UnknownLen::Var(1))),
            end: vec![ValueType::BOOL, ValueType::Var(2)],
        };
        let other_tuple = Tuple {
            start: vec![ValueType::NUM, ValueType::Var(3)],
            middle: Some(Slice::new(ValueType::BOOL, UnknownLen::Var(1))),
            end: vec![ValueType::Var(1)],
        };
        (tuple, other_tuple)
    }

    #[test]
    fn equal_elements_static_two_complex_tuples() {
        let (tuple, other_tuple) = create_test_tuples();

        let equal_elements: Vec<_> = tuple.equal_elements_static(&other_tuple, 3).collect();
        assert_eq!(
            equal_elements,
            vec![
                (&ValueType::NUM, &ValueType::NUM),
                (&ValueType::BOOL, &ValueType::Var(3)),
                (&ValueType::Var(2), &ValueType::Var(1)),
            ]
        );

        let equal_elements: Vec<_> = tuple.equal_elements_static(&other_tuple, 4).collect();
        assert_eq!(
            equal_elements,
            vec![
                (&ValueType::NUM, &ValueType::NUM),
                (&ValueType::Var(0), &ValueType::Var(3)),
                (&ValueType::BOOL, &ValueType::BOOL),
                (&ValueType::Var(2), &ValueType::Var(1)),
            ]
        );
    }

    #[test]
    fn equal_elements_dyn_two_slices() {
        let slice = Tuple::from(Slice::new(ValueType::Var(0), UnknownLen::Var(0)));
        let other_slice = Tuple::from(Slice::new(ValueType::NUM, UnknownLen::Var(1)));
        let equal_elements: Vec<_> = slice.equal_elements_dyn(&other_slice).collect();

        assert_eq!(equal_elements, vec![(&ValueType::Var(0), &ValueType::NUM)]);
    }

    #[test]
    fn equal_elements_dyn_two_complex_tuples() {
        let (tuple, other_tuple) = create_test_tuples();
        let equal_elements: Vec<_> = tuple.equal_elements_dyn(&other_tuple).collect();

        assert_eq!(
            equal_elements,
            vec![
                // Middle elements
                (&ValueType::Var(0), &ValueType::BOOL),
                // Borders
                (&ValueType::NUM, &ValueType::NUM),
                (&ValueType::Var(2), &ValueType::Var(1)),
                // Non-borders in first tuple.
                (&ValueType::Var(0), &ValueType::BOOL),
                // Non-borders in second tuple.
                (&ValueType::Var(0), &ValueType::Var(3)),
            ]
        );
    }
}
