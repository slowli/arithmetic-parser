//! Visitor traits allowing to traverse [`ValueType`] and related types.

use crate::{FnType, PrimitiveType, Tuple, TupleLen, TypeVar, ValueType};

/// Recursive traversal across the shared reference to a [`ValueType`].
///
/// Inspired by the [`Visit` trait from `syn`](https://docs.rs/syn/^1/syn/visit/trait.Visit.html).
///
/// # Examples
///
/// ```
/// use arithmetic_typing::{
///     visit::{self, Visit},
///     PrimitiveType, Slice, Tuple, UnknownLen, ValueType, TypeVar,
/// };
/// # use std::collections::HashMap;
///
/// /// Counts the number of mentions of type / length params in a type.
/// #[derive(Default)]
/// pub struct Mentions {
///     types: HashMap<usize, usize>,
///     lengths: HashMap<usize, usize>,
/// }
///
/// impl<'a, Prim: PrimitiveType> Visit<'a, Prim> for Mentions {
///     fn visit_var(&mut self, var: TypeVar) {
///         *self.types.entry(var.index()).or_default() += 1;
///     }
///
///     fn visit_tuple(&mut self, tuple: &'a Tuple<Prim>) {
///         let (_, middle, _) = tuple.parts();
///         let len = middle.and_then(|middle| middle.len().components().0);
///         if let Some(UnknownLen::Var(var)) = len {
///             *self.lengths.entry(var.index()).or_default() += 1;
///         }
///         visit::visit_tuple(self, tuple);
///     }
/// }
///
/// # fn main() -> anyhow::Result<()> {
/// let value_type: ValueType =
///     "(...['T; N], ('T) -> 'U) -> [('T, 'U); N]".parse()?;
/// let mut mentions = Mentions::default();
/// mentions.visit_type(&value_type);
/// assert_eq!(mentions.lengths[&0], 2); // `N` is mentioned twice
/// assert_eq!(mentions.types[&0], 3); // `T` is mentioned 3 times
/// assert_eq!(mentions.types[&1], 2); // `U` is mentioned twice
/// # Ok(())
/// # }
/// ```
#[allow(unused_variables)]
pub trait Visit<'ast, Prim: PrimitiveType> {
    /// Visits a generic type.
    ///
    /// The default implementation calls one of more specific methods corresponding to the `ty`
    /// variant.
    fn visit_type(&mut self, ty: &'ast ValueType<Prim>) {
        visit_type(self, ty)
    }

    /// Visits a type variable.
    ///
    /// The default implementation does nothing.
    fn visit_var(&mut self, var: TypeVar) {
        // Does nothing.
    }

    /// Visits a primitive type.
    ///
    /// The default implementation does nothing.
    fn visit_primitive(&mut self, primitive: &Prim) {
        // Does nothing.
    }

    /// Visits a tuple type.
    ///
    /// The default implementation calls [`Self::visit_type()`] for each tuple element,
    /// including the middle element if any.
    fn visit_tuple(&mut self, tuple: &'ast Tuple<Prim>) {
        visit_tuple(self, tuple);
    }

    /// Visits a functional type.
    ///
    /// The default implementation calls [`Self::visit_tuple()`] on arguments and then
    /// [`Self::visit_type()`] on the return value.
    fn visit_function(&mut self, function: &'ast FnType<Prim>) {
        visit_function(self, function);
    }
}

/// Default implementation of [`Visit::visit_type()`].
pub fn visit_type<'ast, Prim, V>(visitor: &mut V, ty: &'ast ValueType<Prim>)
where
    Prim: PrimitiveType,
    V: Visit<'ast, Prim> + ?Sized,
{
    match ty {
        ValueType::Some | ValueType::Any(_) => { /* Do nothing. */ }
        ValueType::Var(var) => visitor.visit_var(*var),
        ValueType::Prim(primitive) => visitor.visit_primitive(primitive),
        ValueType::Tuple(tuple) => visitor.visit_tuple(tuple),
        ValueType::Function(function) => visitor.visit_function(function.as_ref()),
    }
}

/// Default implementation of [`Visit::visit_type()`].
pub fn visit_tuple<'ast, Prim, V>(visitor: &mut V, tuple: &'ast Tuple<Prim>)
where
    Prim: PrimitiveType,
    V: Visit<'ast, Prim> + ?Sized,
{
    for ty in tuple.element_types() {
        visitor.visit_type(ty);
    }
}

/// Default implementation of [`Visit::visit_function()`].
pub fn visit_function<'ast, Prim, V>(visitor: &mut V, function: &'ast FnType<Prim>)
where
    Prim: PrimitiveType,
    V: Visit<'ast, Prim> + ?Sized,
{
    visitor.visit_tuple(&function.args);
    visitor.visit_type(&function.return_type);
}

/// Recursive traversal across the exclusive reference to a [`ValueType`].
///
/// Inspired by the [`VisitMut` trait from `syn`].
///
/// [`VisitMut` trait from `syn`]: https://docs.rs/syn/^1/syn/visit_mut/trait.VisitMut.html
///
/// # Examples
///
/// ```
/// use arithmetic_typing::{
///     visit::{self, VisitMut},
///     Num, ValueType,
/// };
///
/// /// Replaces all primitive types with `Num`.
/// struct Replacer;
///
/// impl VisitMut<Num> for Replacer {
///     fn visit_type_mut(&mut self, ty: &mut ValueType) {
///         match ty {
///             ValueType::Prim(_) => *ty = ValueType::NUM,
///             _ => visit::visit_type_mut(self, ty),
///         }
///     }
/// }
///
/// # fn main() -> anyhow::Result<()> {
/// let mut ty: ValueType =
///     "(Num, Bool, (Num) -> (Bool, Num))".parse()?;
/// Replacer.visit_type_mut(&mut ty);
/// assert_eq!(ty.to_string(), "(Num, Num, (Num) -> (Num, Num))");
/// # Ok(())
/// # }
/// ```
#[allow(unused_variables)]
pub trait VisitMut<Prim: PrimitiveType> {
    /// Visits a generic type.
    ///
    /// The default implementation calls one of more specific methods corresponding to the `ty`
    /// variant. For "simple" types (variables, params, primitive types) does nothing.
    fn visit_type_mut(&mut self, ty: &mut ValueType<Prim>) {
        visit_type_mut(self, ty)
    }

    /// Visits a tuple type.
    ///
    /// The default implementation calls [`Self::visit_middle_len_mut()`] for the middle length
    /// if the tuple has a middle. Then, [`Self::visit_type_mut()`] is called
    /// for each tuple element, including the middle element if any.
    fn visit_tuple_mut(&mut self, tuple: &mut Tuple<Prim>) {
        visit_tuple_mut(self, tuple);
    }

    /// Visits a middle length of a tuple.
    ///
    /// The default implementation does nothing.
    fn visit_middle_len_mut(&mut self, len: &mut TupleLen) {
        // Does nothing.
    }

    /// Visits a functional type.
    ///
    /// The default implementation calls [`Self::visit_tuple_mut()`] on arguments and then
    /// [`Self::visit_type_mut()`] on the return value.
    fn visit_function_mut(&mut self, function: &mut FnType<Prim>) {
        visit_function_mut(self, function);
    }
}

/// Default implementation of [`VisitMut::visit_type_mut()`].
pub fn visit_type_mut<Prim, V>(visitor: &mut V, ty: &mut ValueType<Prim>)
where
    Prim: PrimitiveType,
    V: VisitMut<Prim> + ?Sized,
{
    match ty {
        ValueType::Some | ValueType::Any(_) | ValueType::Var(_) | ValueType::Prim(_) => {}
        ValueType::Tuple(tuple) => visitor.visit_tuple_mut(tuple),
        ValueType::Function(function) => visitor.visit_function_mut(function.as_mut()),
    }
}

/// Default implementation of [`VisitMut::visit_tuple_mut()`].
pub fn visit_tuple_mut<Prim, V>(visitor: &mut V, tuple: &mut Tuple<Prim>)
where
    Prim: PrimitiveType,
    V: VisitMut<Prim> + ?Sized,
{
    if let Some(middle) = tuple.parts_mut().1 {
        visitor.visit_middle_len_mut(middle.len_mut());
    }
    for ty in tuple.element_types_mut() {
        visitor.visit_type_mut(ty);
    }
}

/// Default implementation of [`VisitMut::visit_function_mut()`].
pub fn visit_function_mut<Prim, V>(visitor: &mut V, function: &mut FnType<Prim>)
where
    Prim: PrimitiveType,
    V: VisitMut<Prim> + ?Sized,
{
    visitor.visit_tuple_mut(&mut function.args);
    visitor.visit_type_mut(&mut function.return_type);
}
