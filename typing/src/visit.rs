//! Visitor traits allowing to traverse [`Type`] and related types.

use crate::{FnType, Object, PrimitiveType, Tuple, TupleLen, Type, TypeVar};

/// Recursive traversal across the shared reference to a [`Type`].
///
/// Inspired by the [`Visit` trait from `syn`](https://docs.rs/syn/^1/syn/visit/trait.Visit.html).
///
/// # Examples
///
/// ```
/// use arithmetic_typing::{
///     ast::TypeAst, visit::{self, Visit},
///     PrimitiveType, Slice, Tuple, UnknownLen, Type, TypeVar,
/// };
/// # use std::{collections::HashMap, convert::TryFrom};
///
/// /// Counts the number of mentions of type / length params in a type.
/// #[derive(Default)]
/// pub struct Mentions {
///     types: HashMap<usize, usize>,
///     lengths: HashMap<usize, usize>,
/// }
///
/// impl<Prim: PrimitiveType> Visit<Prim> for Mentions {
///     fn visit_var(&mut self, var: TypeVar) {
///         *self.types.entry(var.index()).or_default() += 1;
///     }
///
///     fn visit_tuple(&mut self, tuple: &Tuple<Prim>) {
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
/// let ty = TypeAst::try_from("(...['T; N], ('T) -> 'U) -> [('T, 'U); N]")?;
/// let ty: Type = Type::try_from(&ty)?;
///
/// let mut mentions = Mentions::default();
/// mentions.visit_type(&ty);
/// assert_eq!(mentions.lengths[&0], 2); // `N` is mentioned twice
/// assert_eq!(mentions.types[&0], 3); // `T` is mentioned 3 times
/// assert_eq!(mentions.types[&1], 2); // `U` is mentioned twice
/// # Ok(())
/// # }
/// ```
#[allow(unused_variables)]
pub trait Visit<Prim: PrimitiveType> {
    /// Visits a generic type.
    ///
    /// The default implementation calls one of more specific methods corresponding to the `ty`
    /// variant.
    fn visit_type(&mut self, ty: &Type<Prim>) {
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
    fn visit_tuple(&mut self, tuple: &Tuple<Prim>) {
        visit_tuple(self, tuple);
    }

    /// Visits an object type.
    fn visit_object(&mut self, obj: &Object<Prim>) {
        visit_object(self, obj);
    }

    /// Visits a functional type.
    ///
    /// The default implementation calls [`Self::visit_tuple()`] on arguments and then
    /// [`Self::visit_type()`] on the return value.
    fn visit_function(&mut self, function: &FnType<Prim>) {
        visit_function(self, function);
    }
}

/// Default implementation of [`Visit::visit_type()`].
pub fn visit_type<Prim, V>(visitor: &mut V, ty: &Type<Prim>)
where
    Prim: PrimitiveType,
    V: Visit<Prim> + ?Sized,
{
    match ty {
        Type::Any => { /* Do nothing. */ }
        Type::Dyn(constraints) => {
            if let Some(object) = &constraints.inner.object {
                visitor.visit_object(object);
            }
        }
        Type::Var(var) => visitor.visit_var(*var),
        Type::Prim(primitive) => visitor.visit_primitive(primitive),
        Type::Tuple(tuple) => visitor.visit_tuple(tuple),
        Type::Object(obj) => visitor.visit_object(obj),
        Type::Function(function) => visitor.visit_function(function.as_ref()),
    }
}

/// Default implementation of [`Visit::visit_tuple()`].
pub fn visit_tuple<Prim, V>(visitor: &mut V, tuple: &Tuple<Prim>)
where
    Prim: PrimitiveType,
    V: Visit<Prim> + ?Sized,
{
    for (_, ty) in tuple.element_types() {
        visitor.visit_type(ty);
    }
}

/// Default implementation of [`Visit::visit_object()`].
pub fn visit_object<Prim, V>(visitor: &mut V, obj: &Object<Prim>)
where
    Prim: PrimitiveType,
    V: Visit<Prim> + ?Sized,
{
    for (_, ty) in obj.iter() {
        visitor.visit_type(ty);
    }
}

/// Default implementation of [`Visit::visit_function()`].
pub fn visit_function<Prim, V>(visitor: &mut V, function: &FnType<Prim>)
where
    Prim: PrimitiveType,
    V: Visit<Prim> + ?Sized,
{
    visitor.visit_tuple(&function.args);
    visitor.visit_type(&function.return_type);
}

/// Recursive traversal across the exclusive reference to a [`Type`].
///
/// Inspired by the [`VisitMut` trait from `syn`].
///
/// [`VisitMut` trait from `syn`]: https://docs.rs/syn/^1/syn/visit_mut/trait.VisitMut.html
///
/// # Examples
///
/// ```
/// use arithmetic_typing::{
///     ast::TypeAst, visit::{self, VisitMut}, Num, Type,
/// };
/// # use std::convert::TryFrom;
///
/// /// Replaces all primitive types with `Num`.
/// struct Replacer;
///
/// impl VisitMut<Num> for Replacer {
///     fn visit_type_mut(&mut self, ty: &mut Type) {
///         match ty {
///             Type::Prim(_) => *ty = Type::NUM,
///             _ => visit::visit_type_mut(self, ty),
///         }
///     }
/// }
///
/// # fn main() -> anyhow::Result<()> {
/// let ty = TypeAst::try_from("(Num, Bool, (Num) -> (Bool, Num))")?;
/// let mut ty = Type::try_from(&ty)?;
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
    fn visit_type_mut(&mut self, ty: &mut Type<Prim>) {
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

    /// Visits an object type.
    fn visit_object_mut(&mut self, obj: &mut Object<Prim>) {
        visit_object_mut(self, obj);
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
pub fn visit_type_mut<Prim, V>(visitor: &mut V, ty: &mut Type<Prim>)
where
    Prim: PrimitiveType,
    V: VisitMut<Prim> + ?Sized,
{
    match ty {
        Type::Any | Type::Var(_) | Type::Prim(_) => {}
        Type::Dyn(constraints) => {
            if let Some(object) = &mut constraints.inner.object {
                visitor.visit_object_mut(object);
            }
        }
        Type::Tuple(tuple) => visitor.visit_tuple_mut(tuple),
        Type::Object(obj) => visitor.visit_object_mut(obj),
        Type::Function(function) => visitor.visit_function_mut(function.as_mut()),
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

/// Default implementation of [`VisitMut::visit_object_mut()`].
pub fn visit_object_mut<Prim, V>(visitor: &mut V, obj: &mut Object<Prim>)
where
    Prim: PrimitiveType,
    V: VisitMut<Prim> + ?Sized,
{
    for (_, ty) in obj.iter_mut() {
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
