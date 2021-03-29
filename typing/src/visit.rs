//! Visitor traits.

#![allow(missing_docs)]

use crate::{FnType, PrimitiveType, Tuple, ValueType};

#[allow(unused_variables)]
pub trait Visit<'ast, Prim: PrimitiveType> {
    fn visit_type(&mut self, ty: &'ast ValueType<Prim>) {
        visit_type(self, ty)
    }

    fn visit_var(&mut self, index: usize) {
        // Does nothing.
    }

    fn visit_param(&mut self, index: usize) {
        // Does nothing.
    }

    fn visit_primitive(&mut self, primitive: &Prim) {
        // Does nothing.
    }

    fn visit_tuple(&mut self, tuple: &'ast Tuple<Prim>) {
        visit_tuple(self, tuple);
    }

    fn visit_function(&mut self, function: &'ast FnType<Prim>) {
        visit_function(self, function);
    }
}

pub fn visit_type<'ast, Prim, V>(visitor: &mut V, ty: &'ast ValueType<Prim>)
where
    Prim: PrimitiveType,
    V: Visit<'ast, Prim> + ?Sized,
{
    match ty {
        ValueType::Some => { /* Do nothing. */ }
        ValueType::Var(index) => visitor.visit_var(*index),
        ValueType::Param(index) => visitor.visit_param(*index),
        ValueType::Prim(primitive) => visitor.visit_primitive(primitive),
        ValueType::Tuple(tuple) => visitor.visit_tuple(tuple),
        ValueType::Function(function) => visitor.visit_function(function.as_ref()),
    }
}

pub fn visit_tuple<'ast, Prim, V>(visitor: &mut V, tuple: &'ast Tuple<Prim>)
where
    Prim: PrimitiveType,
    V: Visit<'ast, Prim> + ?Sized,
{
    for ty in tuple.element_types() {
        visitor.visit_type(ty);
    }
}

pub fn visit_function<'ast, Prim, V>(visitor: &mut V, function: &'ast FnType<Prim>)
where
    Prim: PrimitiveType,
    V: Visit<'ast, Prim> + ?Sized,
{
    visitor.visit_tuple(&function.args);
    visitor.visit_type(&function.return_type);
}

#[allow(unused_variables)]
pub trait VisitMut<Prim: PrimitiveType> {
    fn visit_type_mut(&mut self, ty: &mut ValueType<Prim>) {
        visit_type_mut(self, ty)
    }

    fn visit_tuple_mut(&mut self, tuple: &mut Tuple<Prim>) {
        visit_tuple_mut(self, tuple);
    }

    fn visit_function_mut(&mut self, function: &mut FnType<Prim>) {
        visit_function_mut(self, function);
    }
}

pub fn visit_type_mut<Prim, V>(visitor: &mut V, ty: &mut ValueType<Prim>)
where
    Prim: PrimitiveType,
    V: VisitMut<Prim> + ?Sized,
{
    match ty {
        ValueType::Some | ValueType::Var(_) | ValueType::Param(_) | ValueType::Prim(_) => {}
        ValueType::Tuple(tuple) => visitor.visit_tuple_mut(tuple),
        ValueType::Function(function) => visitor.visit_function_mut(function.as_mut()),
    }
}

pub fn visit_tuple_mut<Prim, V>(visitor: &mut V, tuple: &mut Tuple<Prim>)
where
    Prim: PrimitiveType,
    V: VisitMut<Prim> + ?Sized,
{
    for ty in tuple.element_types_mut() {
        visitor.visit_type_mut(ty);
    }
}

pub fn visit_function_mut<Prim, V>(visitor: &mut V, function: &mut FnType<Prim>)
where
    Prim: PrimitiveType,
    V: VisitMut<Prim> + ?Sized,
{
    visitor.visit_tuple_mut(&mut function.args);
    visitor.visit_type_mut(&mut function.return_type);
}
