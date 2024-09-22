//! Generic arithmetics.

use core::{cmp::Ordering, convert::TryFrom, marker::PhantomData, ops};

use num_traits::{
    checked_pow, CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedSub, NumOps, One, Pow,
    Signed, Unsigned, WrappingAdd, WrappingMul, WrappingNeg, WrappingSub, Zero,
};

use crate::{
    arith::{Arithmetic, OrdArithmetic},
    error::ArithmeticError,
};

/// Arithmetic on a number type that implements all necessary operations natively.
///
/// As an example, this type implements [`Arithmetic`] for `f32`, `f64`, and the floating-point
/// complex numbers from the [`num-complex`] crate.
///
/// [`num-complex`]: https://crates.io/crates/num-complex/
#[derive(Debug, Clone, Copy, Default)]
pub struct StdArithmetic;

impl<T> Arithmetic<T> for StdArithmetic
where
    T: Clone + NumOps + PartialEq + ops::Neg<Output = T> + Pow<T, Output = T>,
{
    #[inline]
    fn add(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        Ok(x + y)
    }

    #[inline]
    fn sub(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        Ok(x - y)
    }

    #[inline]
    fn mul(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        Ok(x * y)
    }

    #[inline]
    fn div(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        Ok(x / y)
    }

    #[inline]
    fn pow(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        Ok(x.pow(y))
    }

    #[inline]
    fn neg(&self, x: T) -> Result<T, ArithmeticError> {
        Ok(-x)
    }

    #[inline]
    fn eq(&self, x: &T, y: &T) -> bool {
        *x == *y
    }
}

impl<T> OrdArithmetic<T> for StdArithmetic
where
    Self: Arithmetic<T>,
    T: PartialOrd,
{
    fn partial_cmp(&self, x: &T, y: &T) -> Option<Ordering> {
        x.partial_cmp(y)
    }
}

#[cfg(all(test, feature = "std"))]
static_assertions::assert_impl_all!(StdArithmetic: OrdArithmetic<f32>, OrdArithmetic<f64>);

#[cfg(all(test, feature = "complex"))]
static_assertions::assert_impl_all!(
    StdArithmetic: Arithmetic<num_complex::Complex32>,
    Arithmetic<num_complex::Complex64>
);

/// Helper trait for [`CheckedArithmetic`] describing how number negation should be implemented.
pub trait CheckedArithmeticKind<T> {
    /// Negates the provided `value`, or returns `None` if the value cannot be negated.
    fn checked_neg(value: T) -> Option<T>;
}

/// Arithmetic on an integer type (e.g., `i32`) that checks overflow and other failure
/// conditions for all operations.
///
/// As an example, this type implements [`Arithmetic`] for all built-in integer types
/// with a definite size (`u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`, `u128`, `i128`).
///
/// The type param defines how negation should be performed; it should be one of [`Checked`]
/// (default value), [`Unchecked`] or [`NegateOnlyZero`]. See the docs for these types for
/// more details.
#[derive(Debug)]
pub struct CheckedArithmetic<Kind = Checked>(PhantomData<Kind>);

impl<Kind> Clone for CheckedArithmetic<Kind> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Kind> Copy for CheckedArithmetic<Kind> {}

impl<Kind> Default for CheckedArithmetic<Kind> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Kind> CheckedArithmetic<Kind> {
    /// Creates a new arithmetic instance.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T, Kind> Arithmetic<T> for CheckedArithmetic<Kind>
where
    T: Clone + PartialEq + Zero + One + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv,
    Kind: CheckedArithmeticKind<T>,
    usize: TryFrom<T>,
{
    #[inline]
    fn add(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        x.checked_add(&y).ok_or(ArithmeticError::IntegerOverflow)
    }

    #[inline]
    fn sub(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        x.checked_sub(&y).ok_or(ArithmeticError::IntegerOverflow)
    }

    #[inline]
    fn mul(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        x.checked_mul(&y).ok_or(ArithmeticError::IntegerOverflow)
    }

    #[inline]
    fn div(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        x.checked_div(&y).ok_or(ArithmeticError::DivisionByZero)
    }

    #[inline]
    #[allow(clippy::map_err_ignore)]
    fn pow(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        let exp = usize::try_from(y).map_err(|_| ArithmeticError::InvalidExponent)?;
        checked_pow(x, exp).ok_or(ArithmeticError::IntegerOverflow)
    }

    #[inline]
    fn neg(&self, x: T) -> Result<T, ArithmeticError> {
        Kind::checked_neg(x).ok_or(ArithmeticError::IntegerOverflow)
    }

    #[inline]
    fn eq(&self, x: &T, y: &T) -> bool {
        *x == *y
    }
}

/// Marker for [`CheckedArithmetic`] signalling that negation should be inherited
/// from the [`CheckedNeg`] trait.
#[derive(Debug)]
pub struct Checked(());

impl<T: CheckedNeg> CheckedArithmeticKind<T> for Checked {
    fn checked_neg(value: T) -> Option<T> {
        value.checked_neg()
    }
}

/// Marker for [`CheckedArithmetic`] signalling that negation is only possible for zero.
#[derive(Debug)]
pub struct NegateOnlyZero(());

impl<T: Unsigned + Zero> CheckedArithmeticKind<T> for NegateOnlyZero {
    fn checked_neg(value: T) -> Option<T> {
        if value.is_zero() {
            Some(value)
        } else {
            None
        }
    }
}

/// Marker for [`CheckedArithmetic`] signalling that negation should be inherited from
/// the [`Neg`](ops::Neg) trait. This is appropriate if `Neg` never panics (e.g.,
/// for signed big integers).
#[derive(Debug)]
pub struct Unchecked(());

impl<T: Signed> CheckedArithmeticKind<T> for Unchecked {
    fn checked_neg(value: T) -> Option<T> {
        Some(-value)
    }
}

impl<T, Kind> OrdArithmetic<T> for CheckedArithmetic<Kind>
where
    Self: Arithmetic<T>,
    T: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, x: &T, y: &T) -> Option<Ordering> {
        x.partial_cmp(y)
    }
}

#[cfg(test)]
static_assertions::assert_impl_all!(
    CheckedArithmetic: OrdArithmetic<u8>,
    OrdArithmetic<i8>,
    OrdArithmetic<u16>,
    OrdArithmetic<i16>,
    OrdArithmetic<u32>,
    OrdArithmetic<i32>,
    OrdArithmetic<u64>,
    OrdArithmetic<i64>,
    OrdArithmetic<u128>,
    OrdArithmetic<i128>
);

/// Arithmetic on an integer type (e.g., `i32`), in which all operations have wrapping semantics.
///
/// As an example, this type implements [`Arithmetic`] for all built-in integer types
/// with a definite size (`u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`, `u128`, `i128`).
#[derive(Debug, Clone, Copy, Default)]
pub struct WrappingArithmetic;

impl<T> Arithmetic<T> for WrappingArithmetic
where
    T: Copy
        + PartialEq
        + Zero
        + One
        + WrappingAdd
        + WrappingSub
        + WrappingMul
        + WrappingNeg
        + ops::Div<T, Output = T>,
    usize: TryFrom<T>,
{
    #[inline]
    fn add(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        Ok(x.wrapping_add(&y))
    }

    #[inline]
    fn sub(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        Ok(x.wrapping_sub(&y))
    }

    #[inline]
    fn mul(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        Ok(x.wrapping_mul(&y))
    }

    #[inline]
    fn div(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        if y.is_zero() {
            Err(ArithmeticError::DivisionByZero)
        } else if y.wrapping_neg().is_one() {
            // Division by -1 is the only case when an overflow may occur. We just replace it
            // with `wrapping_neg`.
            Ok(x.wrapping_neg())
        } else {
            Ok(x / y)
        }
    }

    #[inline]
    #[allow(clippy::map_err_ignore)]
    fn pow(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        let exp = usize::try_from(y).map_err(|_| ArithmeticError::InvalidExponent)?;
        Ok(wrapping_exp(x, exp))
    }

    #[inline]
    fn neg(&self, x: T) -> Result<T, ArithmeticError> {
        Ok(x.wrapping_neg())
    }

    #[inline]
    fn eq(&self, x: &T, y: &T) -> bool {
        *x == *y
    }
}

impl<T> OrdArithmetic<T> for WrappingArithmetic
where
    Self: Arithmetic<T>,
    T: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, x: &T, y: &T) -> Option<Ordering> {
        x.partial_cmp(y)
    }
}

// Refactored from `num_traits::pow()`:
// https://docs.rs/num-traits/0.2.14/src/num_traits/pow.rs.html#189-211
fn wrapping_exp<T: Copy + One + WrappingMul>(mut base: T, mut exp: usize) -> T {
    if exp == 0 {
        return T::one();
    }

    while exp & 1 == 0 {
        base = base.wrapping_mul(&base);
        exp >>= 1;
    }
    if exp == 1 {
        return base;
    }

    let mut acc = base;
    while exp > 1 {
        exp >>= 1;
        base = base.wrapping_mul(&base);
        if exp & 1 == 1 {
            acc = acc.wrapping_mul(&base);
        }
    }
    acc
}

#[cfg(test)]
static_assertions::assert_impl_all!(
    WrappingArithmetic: OrdArithmetic<u8>,
    OrdArithmetic<i8>,
    OrdArithmetic<u16>,
    OrdArithmetic<i16>,
    OrdArithmetic<u32>,
    OrdArithmetic<i32>,
    OrdArithmetic<u64>,
    OrdArithmetic<i64>,
    OrdArithmetic<u128>,
    OrdArithmetic<i128>
);
