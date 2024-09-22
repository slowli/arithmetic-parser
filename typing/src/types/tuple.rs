//! Tuple types.

use core::{cmp, fmt, iter, ops};

use crate::{
    alloc::{format, Box, Cow, Vec},
    arith::Num,
    PrimitiveType, Type,
};

/// Length variable.
///
/// A variable represents a certain unknown length. Variables can be either *free*
/// or *bound* to a [`Function`](crate::Function) (similar to const params in Rust, except lengths
/// always have the `usize` type).
/// Just as with [`TypeVar`](crate::TypeVar)s, types input to a [`TypeEnvironment`]
/// can only have bounded length variables (this is
/// verified in runtime), but types output by the inference process can contain both.
///
/// # Notation
///
/// - Bounded length variables are represented as `N`, `M`, `L`, etc.
/// - Free variables are represented as `_`.
///
/// [`TypeEnvironment`]: crate::TypeEnvironment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LengthVar {
    index: usize,
    is_free: bool,
}

impl fmt::Display for LengthVar {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_free {
            formatter.write_str("_")
        } else {
            formatter.write_str(Self::param_str(self.index).as_ref())
        }
    }
}

impl LengthVar {
    pub(crate) fn param_str(index: usize) -> Cow<'static, str> {
        const PARAM_NAMES: &str = "NMLKJI";
        PARAM_NAMES.get(index..=index).map_or_else(
            || Cow::from(format!("N{}", index - PARAM_NAMES.len())),
            Cow::from,
        )
    }

    /// Creates a bounded length variable that can be used to
    /// [build functions](crate::FunctionBuilder).
    pub const fn param(index: usize) -> Self {
        Self {
            index,
            is_free: false,
        }
    }

    /// Returns the 0-based index of this variable.
    pub fn index(self) -> usize {
        self.index
    }

    /// Is this variable free (not bounded in a function declaration)?
    pub fn is_free(self) -> bool {
        self.is_free
    }
}

/// Unknown / variable length, e.g., of a tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnknownLen {
    /// Length that can vary at runtime, similar to lengths of slices in Rust.
    Dynamic,
    /// Length variable.
    Var(LengthVar),
}

impl fmt::Display for UnknownLen {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dynamic => formatter.write_str("*"),
            Self::Var(var) => fmt::Display::fmt(var, formatter),
        }
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

impl UnknownLen {
    /// Creates a bounded type variable that can be used to [build functions](crate::FunctionBuilder).
    pub const fn param(index: usize) -> Self {
        Self::Var(LengthVar::param(index))
    }

    pub(crate) const fn free_var(index: usize) -> Self {
        Self::Var(LengthVar {
            index,
            is_free: true,
        })
    }
}

/// Generic tuple length.
///
/// A tuple length consists of the two components: an unknown / variable length,
/// such as [`UnknownLen::Var`], and a non-negative increment.
/// These components can be obtained via [`Self::components()`].
///
/// # Static lengths
///
/// Tuple lengths can be either *static* or *dynamic*. Dynamic lengths are lengths
/// that contain [`UnknownLen::Dynamic`].
///
/// Functions, [`TypeArithmetic`]s, etc. can specify constraints on lengths being static.
/// For example, this is a part of [`Ops`];
/// dynamically sized slices such as `[Num]` cannot be added / multiplied / etc.,
/// even if they are of the same type. This constraint is denoted as `len! N, M, ...`
/// in the function quantifier, e.g., `for<len! N> (['T; N]) -> 'T`.
///
/// If the constraint fails, an error will be raised with the [kind](crate::error::Error::kind)
/// set to [`ErrorKind::DynamicLen`].
///
/// [`TypeArithmetic`]: crate::arith::TypeArithmetic
/// [`Ops`]: crate::arith::Ops
/// [`ErrorKind::DynamicLen`]: crate::error::ErrorKind::DynamicLen
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TupleLen {
    var: Option<UnknownLen>,
    exact: usize,
}

impl fmt::Display for TupleLen {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.var, self.exact) {
            (Some(var), 0) => fmt::Display::fmt(var, formatter),
            (Some(var), exact) => write!(formatter, "{var} + {exact}"),
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
        !matches!(&self.var, Some(UnknownLen::Var(var)) if var.is_free())
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

/// Index of an element within a tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TupleIndex {
    /// 0-based index from the start of the tuple.
    Start(usize),
    /// Middle element.
    Middle,
    /// 0-based index from the end of the tuple.
    End(usize),
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
/// # Indexing
///
/// *Indexing* is accessing tuple elements via an expression like `xs.0`.
/// Tuple indexing is supported via [`FieldAccess`](arithmetic_parser::Expr::FieldAccess) expr,
/// where the field name is a decimal `usize` number.
///
/// The indexing support for type inference is quite limited.
/// For it to work, the receiver type must be known to be a tuple, and the index must be such
/// that the type of the corresponding element is decidable. Otherwise,
/// an [`UnsupportedIndex`] error will be raised.
///
/// | Tuple type | Index | Outcome |
/// |------------|-------|---------|
/// | `(Num, Bool)` | 0 | `Num` |
/// | `(Num, Bool)` | 1 | `Bool` |
/// | `(Num, Bool)` | 2 | Hard error; the index is out of bounds. |
/// | `Num` | 0 | Hard error; only tuples can be indexed. |
/// | `[Num; _]` | 0 | Error; the slice may be empty. |
/// | `[Num; _ + 1]` | 0 | `Num`; the slice is guaranteed to have 0th element. |
/// | `(Bool, ...[Num; _])` | 0 | `Bool` |
/// | `(Bool, ...[Num; _])` | 1 | Error; the slice part may be empty. |
/// | `(...[Num; _], Bool)` | 0 | Error; cannot decide if the result is `Num` or `Bool`. |
///
/// [`UnsupportedIndex`]: crate::error::ErrorKind::UnsupportedIndex
///
/// # Examples
///
/// Simple tuples can be created using the [`From`] trait. Complex tuples can be created
/// via [`Self::new()`].
///
/// ```
/// # use arithmetic_typing::{Slice, Tuple, UnknownLen, Type};
/// # use assert_matches::assert_matches;
/// let simple_tuple = Tuple::from(vec![Type::NUM, Type::BOOL]);
/// assert_matches!(simple_tuple.parts(), ([_, _], None, []));
/// assert!(simple_tuple.as_slice().is_none());
/// assert_eq!(simple_tuple.to_string(), "(Num, Bool)");
///
/// let slice_tuple = Tuple::from(
///    Type::NUM.repeat(UnknownLen::param(0)),
/// );
/// assert_matches!(slice_tuple.parts(), ([], Some(_), []));
/// assert!(slice_tuple.as_slice().is_some());
/// assert_eq!(slice_tuple.to_string(), "[Num; N]");
///
/// let complex_tuple = Tuple::new(
///     vec![Type::NUM],
///     Type::NUM.repeat(UnknownLen::param(0)),
///     vec![Type::BOOL, Type::param(0)],
/// );
/// assert_matches!(complex_tuple.parts(), ([_], Some(_), [_, _]));
/// assert_eq!(complex_tuple.to_string(), "(Num, ...[Num; N], Bool, 'T)");
/// ```
#[derive(Debug, Clone)]
pub struct Tuple<Prim: PrimitiveType = Num> {
    start: Vec<Type<Prim>>,
    middle: Option<Slice<Prim>>,
    end: Vec<Type<Prim>>,
}

impl<Prim: PrimitiveType> PartialEq for Tuple<Prim> {
    fn eq(&self, other: &Self) -> bool {
        let this_len = self.len();
        if this_len != other.len() {
            false
        } else if let (None, len) = this_len.components() {
            self.iter(len).zip(other.iter(len)).all(|(x, y)| x == y)
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
        start: Vec<Type<Prim>>,
        middle: Option<Slice<Prim>>,
        end: Vec<Type<Prim>>,
    ) -> Self {
        Self { start, middle, end }
    }

    /// Creates a new complex tuple.
    pub fn new(start: Vec<Type<Prim>>, middle: Slice<Prim>, end: Vec<Type<Prim>>) -> Self {
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
        self.start.iter().chain(&self.end).all(Type::is_concrete)
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

    /// Returns shared references to the parts comprising this tuple: start, middle, and end.
    #[allow(clippy::type_complexity)]
    pub fn parts(&self) -> (&[Type<Prim>], Option<&Slice<Prim>>, &[Type<Prim>]) {
        (&self.start, self.middle.as_ref(), &self.end)
    }

    /// Returns exclusive references to the parts comprising this tuple: start, middle, and end.
    #[allow(clippy::type_complexity)]
    pub fn parts_mut(
        &mut self,
    ) -> (
        &mut [Type<Prim>],
        Option<&mut Slice<Prim>>,
        &mut [Type<Prim>],
    ) {
        (&mut self.start, self.middle.as_mut(), &mut self.end)
    }

    /// Returns the length of this tuple.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arithmetic_typing::{Slice, Tuple, Type, UnknownLen, TupleLen};
    /// let tuple = Tuple::from(vec![Type::NUM, Type::BOOL]);
    /// assert_eq!(tuple.len(), TupleLen::from(2));
    ///
    /// let slice = Slice::new(Type::NUM, UnknownLen::param(0));
    /// let tuple = Tuple::from(slice.clone());
    /// assert_eq!(tuple.len(), TupleLen::from(UnknownLen::param(0)));
    ///
    /// let tuple = Tuple::new(vec![], slice, vec![Type::BOOL]);
    /// assert_eq!(tuple.len(), UnknownLen::param(0) + 1);
    /// ```
    pub fn len(&self) -> TupleLen {
        let increment = self.start.len() + self.end.len();
        self.resolved_middle_len() + increment
    }

    /// Returns `true` iff this tuple is guaranteed to be empty.
    pub fn is_empty(&self) -> bool {
        self.start.is_empty() && self.end.is_empty() && self.resolved_middle_len() == TupleLen::ZERO
    }

    pub(crate) fn push(&mut self, element: Type<Prim>) {
        if self.middle.is_some() {
            self.end.push(element);
        } else {
            self.start.push(element);
        }
    }

    pub(crate) fn set_middle(&mut self, element: Type<Prim>, len: TupleLen) {
        self.middle = Some(Slice::new(element, len));
    }

    /// Returns iterator over elements of this tuple assuming it has the given total length.
    pub(crate) fn iter(&self, total_len: usize) -> impl Iterator<Item = &Type<Prim>> + '_ {
        let middle_len = total_len - self.start.len() - self.end.len();
        let middle_element = self.middle.as_ref().map(Slice::element);

        self.start
            .iter()
            .chain(iter::repeat_with(move || middle_element.unwrap()).take(middle_len))
            .chain(&self.end)
    }

    /// Attempts to index into this tuple. `middle_len` specifies the resolved middle length.
    pub(crate) fn get_element(
        &self,
        index: usize,
        middle_len: TupleLen,
    ) -> Result<&Type<Prim>, IndexError> {
        if let Some(element) = self.start.get(index) {
            Ok(element)
        } else {
            self.middle
                .as_ref()
                .map_or(Err(IndexError::OutOfBounds), |middle| {
                    let middle_index = index - self.start.len();
                    if middle_index < middle_len.exact {
                        // The element is definitely in the middle.
                        Ok(middle.element.as_ref())
                    } else if middle_len.var.is_none() {
                        // The element is definitely in the end.
                        let end_index = middle_index - middle_len.exact;
                        self.end.get(end_index).ok_or(IndexError::OutOfBounds)
                    } else {
                        Err(IndexError::NoInfo)
                    }
                })
        }
    }

    /// Returns pairs of elements of this and `other` tuple that should be equal to each other.
    ///
    /// This method is specialized for the case when the length of middles is unknown.
    pub(crate) fn equal_elements_dyn<'a>(
        &'a self,
        other: &'a Self,
    ) -> impl Iterator<Item = (&'a Type<Prim>, &'a Type<Prim>)> + 'a {
        let middle_elem = self.middle.as_ref().unwrap().element.as_ref();
        let other_middle_elem = other.middle.as_ref().unwrap().element.as_ref();
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
    /// # use arithmetic_typing::{Slice, Tuple, TupleIndex, UnknownLen, Type};
    /// let complex_tuple = Tuple::new(
    ///     vec![Type::NUM],
    ///     Slice::new(Type::NUM, UnknownLen::param(0)),
    ///     vec![Type::BOOL, Type::param(0)],
    /// );
    /// let elements: Vec<_> = complex_tuple.element_types().collect();
    /// assert_eq!(elements, [
    ///     (TupleIndex::Start(0), &Type::NUM),
    ///     (TupleIndex::Middle, &Type::NUM),
    ///     (TupleIndex::End(0), &Type::BOOL),
    ///     (TupleIndex::End(1), &Type::param(0)),
    /// ]);
    /// ```
    pub fn element_types(&self) -> impl Iterator<Item = (TupleIndex, &Type<Prim>)> + '_ {
        let middle_element = self
            .middle
            .as_ref()
            .map(|slice| (TupleIndex::Middle, slice.element.as_ref()));
        let start = self
            .start
            .iter()
            .enumerate()
            .map(|(i, elem)| (TupleIndex::Start(i), elem));
        let end = self
            .end
            .iter()
            .enumerate()
            .map(|(i, elem)| (TupleIndex::End(i), elem));
        start.chain(middle_element).chain(end)
    }

    pub(crate) fn element_types_mut(&mut self) -> impl Iterator<Item = &mut Type<Prim>> + '_ {
        let middle_element = self.middle.as_mut().map(|slice| slice.element.as_mut());
        self.start
            .iter_mut()
            .chain(middle_element)
            .chain(&mut self.end)
    }
}

impl<Prim: PrimitiveType> From<Vec<Type<Prim>>> for Tuple<Prim> {
    fn from(elements: Vec<Type<Prim>>) -> Self {
        Self {
            start: elements,
            middle: None,
            end: Vec::new(),
        }
    }
}

/// Errors that can occur when indexing into a tuple.
#[derive(Debug)]
pub(crate) enum IndexError {
    /// Index is out of bounds.
    OutOfBounds,
    /// Not enough info to determine the type.
    NoInfo,
}

/// Slice type. Unlike in Rust, slices are a subset of tuples. If `length` is
/// exact (has no [`UnknownLen`] part), the slice is completely equivalent
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
/// use arithmetic_parser::grammars::{F32Grammar, Parse};
/// use arithmetic_typing::{Annotated, TupleLen, TypeEnvironment, Type};
///
/// # fn main() -> anyhow::Result<()> {
/// type Parser = Annotated<F32Grammar>;
/// let ast = Parser::parse_statements("xs: [Num; _] = (1, 2, 3);")?;
/// let mut env = TypeEnvironment::new();
/// env.process_statements(&ast)?;
/// // Slices with fixed length are equivalent to tuples.
/// assert_eq!(env["xs"].to_string(), "(Num, Num, Num)");
///
/// let code = "
///     xs: [Num] = (1, 2, 3);
///     ys = xs + 1; // works fine: despite `xs` having unknown length,
///                  // it's always possible to add a number to it
///     (_, _, z) = xs; // does not work: the tuple length is erased
/// ";
/// let ast = Parser::parse_statements(code)?;
/// let errors = env.process_statements(&ast).unwrap_err();
///
/// let err = errors.iter().next().unwrap();
/// assert_eq!(err.main_location().span(code), "(_, _, z)");
/// assert_eq!(env["ys"], env["xs"]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Slice<Prim: PrimitiveType = Num> {
    element: Box<Type<Prim>>,
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
    pub fn new(element: Type<Prim>, length: impl Into<TupleLen>) -> Self {
        Self {
            element: Box::new(element),
            length: length.into(),
        }
    }

    /// Returns the element type of this slice.
    pub fn element(&self) -> &Type<Prim> {
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
    use assert_matches::assert_matches;

    use super::*;
    use crate::alloc::{vec, ToString};

    #[test]
    fn tuple_length_display() {
        let len = TupleLen::from(3);
        assert_eq!(len.to_string(), "3");
        let len = UnknownLen::param(0) + 2;
        assert_eq!(len.to_string(), "N + 2");
    }

    #[test]
    fn slice_display() {
        let slice = Slice::new(Type::NUM, UnknownLen::param(0));
        assert_eq!(slice.to_string(), "[Num; N]");
        let slice = Slice::new(Type::NUM, UnknownLen::free_var(0));
        assert_eq!(slice.to_string(), "[Num; _]");
        let slice = Slice::new(Type::NUM, TupleLen::from(3));
        assert_eq!(slice.to_string(), "[Num; 3]");
    }

    #[test]
    fn tuple_display() {
        // Simple tuples.
        let tuple = Tuple::from(vec![Type::NUM, Type::BOOL]);
        assert_eq!(tuple.to_string(), "(Num, Bool)");
        let tuple = Tuple::from(Slice::new(Type::NUM, UnknownLen::param(0)));
        assert_eq!(tuple.to_string(), "[Num; N]");
        let tuple = Tuple::from(Slice::new(Type::NUM, TupleLen::from(3)));
        assert_eq!(tuple.to_string(), "(Num, Num, Num)");

        let tuple = Tuple {
            start: vec![Type::NUM, Type::BOOL],
            middle: Some(Slice::new(Type::NUM, UnknownLen::param(0))),
            end: vec![],
        };
        assert_eq!(tuple.to_string(), "(Num, Bool, ...[Num; N])");

        let tuple = Tuple {
            start: vec![Type::NUM, Type::BOOL],
            middle: Some(Slice::new(Type::NUM, TupleLen::from(2))),
            end: vec![],
        };
        assert_eq!(tuple.to_string(), "(Num, Bool, Num, Num)");

        let tuple = Tuple {
            start: vec![Type::NUM, Type::BOOL],
            middle: Some(Slice::new(Type::NUM, UnknownLen::param(0))),
            end: vec![Type::param(0)],
        };
        assert_eq!(tuple.to_string(), "(Num, Bool, ...[Num; N], 'T)");
    }

    #[test]
    fn equal_elements_static_two_simple_tuples() {
        let tuple = Tuple::from(vec![Type::NUM, Type::BOOL, Type::free_var(0)]);
        let other_tuple = <Tuple>::from(vec![Type::free_var(1), Type::BOOL, Type::free_var(0)]);
        let equal_elements: Vec<_> = tuple.iter(3).zip(other_tuple.iter(3)).collect();

        assert_eq!(
            equal_elements,
            vec![
                (&Type::NUM, &Type::free_var(1)),
                (&Type::BOOL, &Type::BOOL),
                (&Type::free_var(0), &Type::free_var(0)),
            ]
        );
    }

    #[test]
    fn equal_elements_static_simple_tuple_and_slice() {
        let tuple = Tuple::from(vec![Type::NUM, Type::BOOL, Type::free_var(0)]);
        let slice = <Tuple>::from(Slice::new(Type::free_var(1), UnknownLen::free_var(0)));
        let equal_elements: Vec<_> = tuple.iter(3).zip(slice.iter(3)).collect();

        assert_eq!(
            equal_elements,
            vec![
                (&Type::NUM, &Type::free_var(1)),
                (&Type::BOOL, &Type::free_var(1)),
                (&Type::free_var(0), &Type::free_var(1)),
            ]
        );
    }

    #[test]
    fn equal_elements_static_slice_and_complex_tuple() {
        let slice = <Tuple>::from(Slice::new(Type::free_var(1), UnknownLen::free_var(0)));
        let tuple = Tuple {
            start: vec![Type::NUM],
            middle: Some(Slice::new(Type::free_var(0), UnknownLen::free_var(1))),
            end: vec![Type::BOOL, Type::free_var(2)],
        };

        let mut expected_pairs = vec![
            (Type::free_var(1), Type::NUM),
            (Type::free_var(1), Type::BOOL),
            (Type::free_var(1), Type::free_var(2)),
        ];
        let equal_elements: Vec<_> = slice
            .iter(3)
            .zip(tuple.iter(3))
            .map(|(x, y)| (x.clone(), y.clone()))
            .collect();
        assert_eq!(equal_elements, expected_pairs);

        let equal_elements: Vec<_> = slice
            .iter(4)
            .zip(tuple.iter(4))
            .map(|(x, y)| (x.clone(), y.clone()))
            .collect();
        expected_pairs.insert(1, (Type::free_var(1), Type::free_var(0)));
        assert_eq!(equal_elements, expected_pairs);

        let equal_elements: Vec<_> = slice
            .iter(5)
            .zip(tuple.iter(5))
            .map(|(x, y)| (x.clone(), y.clone()))
            .collect();
        expected_pairs.insert(2, (Type::free_var(1), Type::free_var(0)));
        assert_eq!(equal_elements, expected_pairs);
    }

    fn create_test_tuples() -> (Tuple, Tuple) {
        let tuple = Tuple {
            start: vec![Type::NUM],
            middle: Some(Slice::new(Type::free_var(0), UnknownLen::free_var(1))),
            end: vec![Type::BOOL, Type::free_var(2)],
        };
        let other_tuple = Tuple {
            start: vec![Type::NUM, Type::free_var(3)],
            middle: Some(Slice::new(Type::BOOL, UnknownLen::free_var(1))),
            end: vec![Type::free_var(1)],
        };
        (tuple, other_tuple)
    }

    #[test]
    fn equal_elements_static_two_complex_tuples() {
        let (tuple, other_tuple) = create_test_tuples();

        let equal_elements: Vec<_> = tuple.iter(3).zip(other_tuple.iter(3)).collect();
        assert_eq!(
            equal_elements,
            vec![
                (&Type::NUM, &Type::NUM),
                (&Type::BOOL, &Type::free_var(3)),
                (&Type::free_var(2), &Type::free_var(1)),
            ]
        );

        let equal_elements: Vec<_> = tuple.iter(4).zip(other_tuple.iter(4)).collect();
        assert_eq!(
            equal_elements,
            vec![
                (&Type::NUM, &Type::NUM),
                (&Type::free_var(0), &Type::free_var(3)),
                (&Type::BOOL, &Type::BOOL),
                (&Type::free_var(2), &Type::free_var(1)),
            ]
        );
    }

    #[test]
    fn equal_elements_dyn_two_slices() {
        let slice = Tuple::from(Slice::new(Type::free_var(0), UnknownLen::free_var(0)));
        let other_slice = Tuple::from(Slice::new(Type::NUM, UnknownLen::free_var(1)));
        let equal_elements: Vec<_> = slice.equal_elements_dyn(&other_slice).collect();

        assert_eq!(equal_elements, vec![(&Type::free_var(0), &Type::NUM)]);
    }

    #[test]
    fn equal_elements_dyn_two_complex_tuples() {
        let (tuple, other_tuple) = create_test_tuples();
        let equal_elements: Vec<_> = tuple.equal_elements_dyn(&other_tuple).collect();

        assert_eq!(
            equal_elements,
            vec![
                // Middle elements
                (&Type::free_var(0), &Type::BOOL),
                // Borders
                (&Type::NUM, &Type::NUM),
                (&Type::free_var(2), &Type::free_var(1)),
                // Non-borders in first tuple.
                (&Type::free_var(0), &Type::BOOL),
                // Non-borders in second tuple.
                (&Type::free_var(0), &Type::free_var(3)),
            ]
        );
    }

    #[test]
    fn tuple_indexing() {
        // Ordinary tuple.
        let tuple = Tuple::from(vec![Type::NUM, Type::BOOL]);
        assert_eq!(*tuple.get_element(0, TupleLen::ZERO).unwrap(), Type::NUM,);
        assert_eq!(*tuple.get_element(1, TupleLen::ZERO).unwrap(), Type::BOOL,);
        assert_matches!(
            tuple.get_element(2, TupleLen::ZERO).unwrap_err(),
            IndexError::OutOfBounds
        );

        // Slice.
        let tuple = Tuple::from(Slice::new(Type::NUM, UnknownLen::param(0)));
        assert_eq!(*tuple.get_element(0, TupleLen::from(3)).unwrap(), Type::NUM);
        assert_matches!(
            tuple.get_element(3, TupleLen::from(3)).unwrap_err(),
            IndexError::OutOfBounds
        );
        assert_matches!(
            tuple
                .get_element(0, UnknownLen::free_var(0).into())
                .unwrap_err(),
            IndexError::NoInfo
        );
        assert_eq!(
            *tuple.get_element(0, UnknownLen::free_var(0) + 1).unwrap(),
            Type::NUM
        );

        // Tuple with all three components.
        let (tuple, _) = create_test_tuples();
        assert_eq!(
            *tuple
                .get_element(0, UnknownLen::free_var(0).into())
                .unwrap(),
            Type::NUM
        );
        assert_matches!(
            tuple
                .get_element(1, UnknownLen::free_var(0).into())
                .unwrap_err(),
            IndexError::NoInfo
        );

        assert_eq!(*tuple.get_element(1, 2.into()).unwrap(), Type::free_var(0));
        assert_eq!(*tuple.get_element(3, 2.into()).unwrap(), Type::BOOL);
        assert_matches!(
            tuple.get_element(5, 2.into()).unwrap_err(),
            IndexError::OutOfBounds
        );
    }
}
