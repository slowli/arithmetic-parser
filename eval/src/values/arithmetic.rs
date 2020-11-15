#![allow(missing_docs)]

use num_traits::{Num, Pow};

use core::{fmt, ops};

pub trait Arithmetic<T> {
    fn add(&self, x: T, y: T) -> anyhow::Result<T>;
    fn sub(&self, x: T, y: T) -> anyhow::Result<T>;
    fn mul(&self, x: T, y: T) -> anyhow::Result<T>;
    fn div(&self, x: T, y: T) -> anyhow::Result<T>;
    fn pow(&self, x: T, y: T) -> anyhow::Result<T>;

    fn neg(&self, x: T) -> anyhow::Result<T>;

    fn eq(&self, x: &T, y: &T) -> bool;
}

impl<T> fmt::Debug for dyn Arithmetic<T> + '_ {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("Arithmetic").finish()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StdArithmetic;

impl<T> Arithmetic<T> for StdArithmetic
where
    T: Copy + Num + PartialEq + ops::Neg<Output = T> + Pow<T, Output = T>,
{
    #[inline]
    fn add(&self, x: T, y: T) -> anyhow::Result<T> {
        Ok(x + y)
    }

    #[inline]
    fn sub(&self, x: T, y: T) -> anyhow::Result<T> {
        Ok(x - y)
    }

    #[inline]
    fn mul(&self, x: T, y: T) -> anyhow::Result<T> {
        Ok(x * y)
    }

    #[inline]
    fn div(&self, x: T, y: T) -> anyhow::Result<T> {
        Ok(x / y)
    }

    #[inline]
    fn pow(&self, x: T, y: T) -> anyhow::Result<T> {
        Ok(x.pow(y))
    }

    #[inline]
    fn neg(&self, x: T) -> anyhow::Result<T> {
        Ok(-x)
    }

    #[inline]
    fn eq(&self, x: &T, y: &T) -> bool {
        *x == *y
    }
}
