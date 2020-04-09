//! Standard functions for the interpreter.

use anyhow::{ensure, format_err};

use super::{NativeFn, Value};
use crate::Grammar;

/// Assertion function.
#[derive(Debug, Clone, Copy)]
pub struct Assert;

impl<T: Grammar> NativeFn<T> for Assert {
    fn execute<'a>(&self, args: &[Value<'a, T>]) -> anyhow::Result<Value<'a, T>> {
        ensure!(
            args.len() == 1,
            "`assert` requires a single boolean argument"
        );
        match args[0] {
            Value::Bool(true) => Ok(Value::void()),
            Value::Bool(false) => Err(format_err!("Assertion failed")),
            _ => Err(format_err!("`assert` requires a single boolean argument")),
        }
    }
}

/// Unary function wrapper.
#[derive(Debug, Clone, Copy)]
pub struct UnaryFn<F> {
    function: F,
}

impl<F> UnaryFn<F> {
    /// Creates a new function.
    pub const fn new(function: F) -> Self {
        Self { function }
    }
}

impl<F, T> NativeFn<T> for UnaryFn<F>
where
    T: Grammar,
    F: Fn(T::Lit) -> T::Lit,
{
    fn execute<'a>(&self, args: &[Value<'a, T>]) -> anyhow::Result<Value<'a, T>> {
        match args {
            [Value::Simple(x)] => {
                let output = (self.function)(x.to_owned());
                Ok(Value::Simple(output))
            }
            _ => Err(format_err!(
                "Unary function requires one primitive argument",
            )),
        }
    }
}

/// Binary function wrapper.
#[derive(Debug, Clone, Copy)]
pub struct BinaryFn<F> {
    function: F,
}

impl<F> BinaryFn<F> {
    /// Creates a new function.
    pub const fn new(function: F) -> Self {
        Self { function }
    }
}

impl<F, T> NativeFn<T> for BinaryFn<F>
where
    T: Grammar,
    F: Fn(T::Lit, T::Lit) -> T::Lit,
{
    fn execute<'a>(&self, args: &[Value<'a, T>]) -> anyhow::Result<Value<'a, T>> {
        match args {
            [Value::Simple(x), Value::Simple(y)] => {
                let output = (self.function)(x.to_owned(), y.to_owned());
                Ok(Value::Simple(output))
            }
            _ => Err(format_err!(
                "Binary function requires two primitive arguments",
            )),
        }
    }
}
