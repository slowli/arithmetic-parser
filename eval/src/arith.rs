//! `Arithmetic` trait and its implementations.
//!
//! # Traits
//!
//! An [`Arithmetic`] defines fallible arithmetic operations on literals
//! of an [`ExecutableModule`], namely, addition, subtraction, multiplication, division,
//! exponentiation (all binary ops), and negation (a unary op). Any module can be run
//! with any `Arithmetic` on its literals, although some modules are reasonably tied
//! to a particular arithmetic or class of arithmetics (e.g., arithmetics on finite fields).
//!
//! [`OrdArithmetic`] extends [`Arithmetic`] with a partial comparison operation
//! (i.e., an analogue to [`PartialOrd`]). This is motivated by the fact that comparisons
//! may be switched off during parsing, and some `Arithmetic`s do not have well-defined comparisons.
//!
//! [`ArithmeticExt`] helps converting an [`Arithmetic`] into an [`OrdArithmetic`].
//!
//! # Implementations
//!
//! This module defines the following kinds of arithmetics:
//!
//! - [`StdArithmetic`] takes all implementations from the corresponding [`ops`] traits. This
//!   means that it's safe to use *provided* the ops are infallible. As a counter-example,
//!   using [`StdArithmetic`] with built-in integer types (such as `u64`) is usually not a good
//!   idea since the corresponding ops have failure modes (e.g., division by zero or integer
//!   overflow).
//! - [`WrappingArithmetic`] is defined for integer types; it uses wrapping semantics for all ops.
//! - [`CheckedArithmetic`] is defined for integer types; it uses checked semantics for all ops.
//! - [`ModularArithmetic`] operates on integers modulo the specified number.
//!
//! All defined [`Arithmetic`]s strive to be as generic as possible.
//!
//! [`ExecutableModule`]: crate::ExecutableModule

#![allow(clippy::unknown_clippy_lints)] // `map_err_ignore` is newer than MSRV

use num_traits::{
    checked_pow, CheckedAdd, CheckedDiv, CheckedMul, CheckedNeg, CheckedSub, NumOps, One, Pow,
    Signed, Unsigned, WrappingAdd, WrappingMul, WrappingNeg, WrappingSub, Zero,
};

use core::{
    cmp::Ordering,
    convert::{TryFrom, TryInto},
    fmt,
    marker::PhantomData,
    mem, ops,
};

use crate::error::ArithmeticError;

#[cfg(feature = "bigint")]
mod bigint;

/// Encapsulates arithmetic operations on a certain number type.
///
/// Unlike operations on built-in integer types, arithmetic operations may be fallible.
/// Additionally, the arithmetic can have a state. This is used, for example, in
/// [`ModularArithmetic`], which stores the modulus in the state.
pub trait Arithmetic<T> {
    /// Adds two numbers.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is unsuccessful (e.g., on integer overflow).
    fn add(&self, x: T, y: T) -> Result<T, ArithmeticError>;

    /// Subtracts two numbers.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is unsuccessful (e.g., on integer underflow).
    fn sub(&self, x: T, y: T) -> Result<T, ArithmeticError>;

    /// Multiplies two numbers.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is unsuccessful (e.g., on integer overflow).
    fn mul(&self, x: T, y: T) -> Result<T, ArithmeticError>;

    /// Divides two numbers.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is unsuccessful (e.g., if `y` is zero or does
    /// not have a multiplicative inverse in the case of modular arithmetic).
    fn div(&self, x: T, y: T) -> Result<T, ArithmeticError>;

    /// Raises `x` to the power of `y`.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is unsuccessful (e.g., on integer overflow).
    fn pow(&self, x: T, y: T) -> Result<T, ArithmeticError>;

    /// Negates a number.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is unsuccessful (e.g., on integer overflow).
    fn neg(&self, x: T) -> Result<T, ArithmeticError>;

    /// Checks if two numbers are equal. Note that equality can be a non-trivial operation;
    /// e.g., different numbers may be equal as per modular arithmetic.
    fn eq(&self, x: &T, y: &T) -> bool;
}

/// Extends an [`Arithmetic`] with a comparison operation on numbers.
pub trait OrdArithmetic<T>: Arithmetic<T> {
    /// Compares two numbers. Returns `None` if the numbers are not comparable, or the comparison
    /// result otherwise.
    fn partial_cmp(&self, x: &T, y: &T) -> Option<Ordering>;
}

impl<T> fmt::Debug for dyn OrdArithmetic<T> + '_ {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("OrdArithmetic").finish()
    }
}

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
#[derive(Debug)]
pub struct CheckedArithmetic<Kind = Checked>(PhantomData<Kind>);

impl<Kind> Clone for CheckedArithmetic<Kind> {
    fn clone(&self) -> Self {
        Self(self.0)
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

/// Encapsulates extension of an unsigned integer type into signed and unsigned double-width types.
/// This allows performing certain operations (e.g., multiplication) without a possibility of
/// integer overflow.
pub trait DoubleWidth: Sized + Unsigned {
    /// Unsigned double-width extension type.
    type Wide: Copy + From<Self> + TryInto<Self> + NumOps + Unsigned;
    /// Signed double-width extension type.
    type SignedWide: Copy + From<Self> + TryInto<Self> + NumOps + Zero + One + Signed + PartialOrd;
}

impl DoubleWidth for u8 {
    type Wide = u16;
    type SignedWide = i16;
}

impl DoubleWidth for u16 {
    type Wide = u32;
    type SignedWide = i32;
}

impl DoubleWidth for u32 {
    type Wide = u64;
    type SignedWide = i64;
}

impl DoubleWidth for u64 {
    type Wide = u128;
    type SignedWide = i128;
}

/// Modular arithmetic on integers.
///
/// As an example, `ModularArithmetic<T>` implements `Arithmetic<T>` if `T` is one of unsigned
/// built-in integer types (`u8`, `u16`, `u32`, `u64`; `u128` **is excluded** because it cannot be
/// extended to double width).
#[derive(Debug, Clone, Copy)]
pub struct ModularArithmetic<T> {
    modulus: T,
}

impl<T> ModularArithmetic<T>
where
    T: Clone + PartialEq + NumOps + Zero + One,
{
    /// Creates a new arithmetic with the specified `modulus`.
    ///
    /// # Panics
    ///
    /// - Panics if modulus is 0 or 1.
    pub fn new(modulus: T) -> Self {
        assert!(!modulus.is_zero(), "Modulus cannot be 0");
        assert!(!modulus.is_one(), "Modulus cannot be 1");
        Self { modulus }
    }

    /// Returns the modulus for this arithmetic.
    pub fn modulus(&self) -> &T {
        &self.modulus
    }
}

impl<T> ModularArithmetic<T>
where
    T: Copy + PartialEq + NumOps + Zero + One + DoubleWidth,
{
    #[inline]
    fn mul_inner(self, x: T, y: T) -> T {
        let wide = (<T::Wide>::from(x) * <T::Wide>::from(y)) % <T::Wide>::from(self.modulus);
        wide.try_into().ok().unwrap() // `unwrap` is safe by construction
    }

    /// Computes the multiplicative inverse of `value` using the extended Euclid algorithm.
    /// Care is taken to not overflow anywhere.
    fn invert(self, value: T) -> Option<T> {
        let value = value % self.modulus; // Reduce value since this influences speed.
        let mut t = <T::SignedWide>::zero();
        let mut new_t = <T::SignedWide>::one();

        let modulus = <T::SignedWide>::from(self.modulus);
        let mut r = modulus;
        let mut new_r = <T::SignedWide>::from(value);

        while !new_r.is_zero() {
            let quotient = r / new_r;
            t = t - quotient * new_t;
            mem::swap(&mut new_t, &mut t);
            r = r - quotient * new_r;
            mem::swap(&mut new_r, &mut r);
        }

        if r > <T::SignedWide>::one() {
            None // r = gcd(self.modulus, value) > 1
        } else {
            if t.is_negative() {
                t = t + modulus;
            }
            Some(t.try_into().ok().unwrap())
            // ^-- `unwrap` is safe by construction
        }
    }

    fn modular_exp(self, base: T, mut exp: usize) -> T {
        if exp == 0 {
            return T::one();
        }

        let wide_modulus = <T::Wide>::from(self.modulus);
        let mut base = <T::Wide>::from(base % self.modulus);

        while exp & 1 == 0 {
            base = (base * base) % wide_modulus;
            exp >>= 1;
        }
        if exp == 1 {
            return base.try_into().ok().unwrap(); // `unwrap` is safe by construction
        }

        let mut acc = base;
        while exp > 1 {
            exp >>= 1;
            base = (base * base) % wide_modulus;
            if exp & 1 == 1 {
                acc = (acc * base) % wide_modulus;
            }
        }
        acc.try_into().ok().unwrap() // `unwrap` is safe by construction
    }
}

impl<T> Arithmetic<T> for ModularArithmetic<T>
where
    T: Copy + PartialEq + NumOps + Zero + One + DoubleWidth,
    usize: TryFrom<T>,
{
    #[inline]
    fn add(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        let wide = (<T::Wide>::from(x) + <T::Wide>::from(y)) % <T::Wide>::from(self.modulus);
        Ok(wide.try_into().ok().unwrap()) // `unwrap` is safe by construction
    }

    #[inline]
    fn sub(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        let y = y % self.modulus; // Prevent possible overflow in the following subtraction
        self.add(x, self.modulus - y)
    }

    #[inline]
    fn mul(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        Ok(self.mul_inner(x, y))
    }

    #[inline]
    fn div(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        if y.is_zero() {
            Err(ArithmeticError::DivisionByZero)
        } else {
            let y_inv = self.invert(y).ok_or(ArithmeticError::NoInverse)?;
            self.mul(x, y_inv)
        }
    }

    #[inline]
    #[allow(clippy::map_err_ignore)]
    fn pow(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        let exp = usize::try_from(y).map_err(|_| ArithmeticError::InvalidExponent)?;
        Ok(self.modular_exp(x, exp))
    }

    #[inline]
    fn neg(&self, x: T) -> Result<T, ArithmeticError> {
        let x = x % self.modulus; // Prevent possible overflow in the following subtraction
        Ok(self.modulus - x)
    }

    #[inline]
    fn eq(&self, x: &T, y: &T) -> bool {
        *x % self.modulus == *y % self.modulus
    }
}

#[cfg(test)]
static_assertions::assert_impl_all!(ModularArithmetic<u8>: Arithmetic<u8>);
#[cfg(test)]
static_assertions::assert_impl_all!(ModularArithmetic<u16>: Arithmetic<u16>);
#[cfg(test)]
static_assertions::assert_impl_all!(ModularArithmetic<u32>: Arithmetic<u32>);
#[cfg(test)]
static_assertions::assert_impl_all!(ModularArithmetic<u64>: Arithmetic<u64>);

/// Wrapper type allowing to extend an [`Arithmetic`] to an [`OrdArithmetic`] implementation.
///
/// # Examples
///
/// This type can only be constructed via [`ArithmeticExt`] trait. See it for the examples
/// of usage.
pub struct FullArithmetic<T, A> {
    base: A,
    comparison: fn(&T, &T) -> Option<Ordering>,
}

impl<T, A: Clone> Clone for FullArithmetic<T, A> {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            comparison: self.comparison,
        }
    }
}

impl<T, A: Copy> Copy for FullArithmetic<T, A> {}

impl<T, A: fmt::Debug> fmt::Debug for FullArithmetic<T, A> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FullArithmetic")
            .field("base", &self.base)
            .finish()
    }
}

impl<T, A> Arithmetic<T> for FullArithmetic<T, A>
where
    A: Arithmetic<T>,
{
    #[inline]
    fn add(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        self.base.add(x, y)
    }

    #[inline]
    fn sub(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        self.base.sub(x, y)
    }

    #[inline]
    fn mul(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        self.base.mul(x, y)
    }

    #[inline]
    fn div(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        self.base.div(x, y)
    }

    #[inline]
    fn pow(&self, x: T, y: T) -> Result<T, ArithmeticError> {
        self.base.pow(x, y)
    }

    #[inline]
    fn neg(&self, x: T) -> Result<T, ArithmeticError> {
        self.base.neg(x)
    }

    #[inline]
    fn eq(&self, x: &T, y: &T) -> bool {
        self.base.eq(x, y)
    }
}

impl<T, A> OrdArithmetic<T> for FullArithmetic<T, A>
where
    A: Arithmetic<T>,
{
    fn partial_cmp(&self, x: &T, y: &T) -> Option<Ordering> {
        (self.comparison)(x, y)
    }
}

/// Extension trait for [`Arithmetic`] allowing to combine the arithmetic with a number comparison.
///
/// # Examples
///
/// ```
/// use arithmetic_eval::arith::{ArithmeticExt, ModularArithmetic};
/// # use arithmetic_eval::{ExecutableModule, Value};
/// # use arithmetic_parser::grammars::{NumGrammar, Untyped, Parse};
///
/// # fn main() -> anyhow::Result<()> {
/// let base = ModularArithmetic::new(11);
///
/// // `ModularArithmetic` requires to define how numbers will be compared -
/// // and the simplest solution is to not compare them at all.
/// let program = Untyped::<NumGrammar<u32>>::parse_statements("1 < 3 || 1 >= 3")?;
/// let module = ExecutableModule::builder("test", &program)?.build();
/// assert_eq!(
///     module.with_arithmetic(&base.without_comparisons()).run()?,
///     Value::Bool(false)
/// );
///
/// // We can compare numbers by their integer value. This can lead
/// // to pretty confusing results, though.
/// let bogus_arithmetic = base.with_natural_comparison();
/// let program = Untyped::<NumGrammar<u32>>::parse_statements(r#"
///     (x, y, z) = (1, 12, 5);
///     x == y && x < z && y > z
/// "#)?;
/// let module = ExecutableModule::builder("test", &program)?.build();
/// assert_eq!(
///     module.with_arithmetic(&bogus_arithmetic).run()?,
///     Value::Bool(true)
/// );
///
/// // It's possible to fix the situation using a custom comparison function,
/// // which will compare numbers by their residual class.
/// let less_bogus_arithmetic = base.with_comparison(|&x: &u32, &y: &u32| {
///     (x % 11).partial_cmp(&(y % 11))
/// });
/// assert_eq!(
///     module.with_arithmetic(&less_bogus_arithmetic).run()?,
///     Value::Bool(false)
/// );
/// # Ok(())
/// # }
/// ```
pub trait ArithmeticExt<T>: Arithmetic<T> + Sized {
    /// Combines this arithmetic with a comparison function that assumes any two numbers are
    /// incomparable.
    fn without_comparisons(self) -> FullArithmetic<T, Self> {
        FullArithmetic {
            base: self,
            comparison: |_, _| None,
        }
    }

    /// Combines this arithmetic with a comparison function specified by the [`PartialOrd`]
    /// implementation for `T`.
    fn with_natural_comparison(self) -> FullArithmetic<T, Self>
    where
        T: PartialOrd,
    {
        FullArithmetic {
            base: self,
            comparison: |x, y| x.partial_cmp(y),
        }
    }

    /// Combines this arithmetic with the specified comparison function.
    fn with_comparison(
        self,
        comparison: fn(&T, &T) -> Option<Ordering>,
    ) -> FullArithmetic<T, Self> {
        FullArithmetic {
            base: self,
            comparison,
        }
    }
}

impl<T, A> ArithmeticExt<T> for A where A: Arithmetic<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn modular_arithmetic_basics() {
        let arithmetic = ModularArithmetic::new(11_u32);
        assert_eq!(arithmetic.add(1, 5).unwrap(), 6);
        assert_eq!(arithmetic.add(2, 9).unwrap(), 0);
        assert_eq!(arithmetic.add(5, 9).unwrap(), 3);
        assert_eq!(arithmetic.add(5, 20).unwrap(), 3);

        assert_eq!(arithmetic.sub(5, 9).unwrap(), 7);
        assert_eq!(arithmetic.sub(5, 20).unwrap(), 7);

        assert_eq!(arithmetic.mul(5, 4).unwrap(), 9);
        assert_eq!(arithmetic.mul(11, 4).unwrap(), 0);

        // Check overflows.
        assert_eq!(u32::max_value() % 11, 3);
        assert_eq!(
            arithmetic.mul(u32::max_value(), u32::max_value()).unwrap(),
            9
        );

        assert_eq!(arithmetic.div(1, 4).unwrap(), 3); // 4 * 3 = 12 = 1 (mod 11)
        assert_eq!(arithmetic.div(2, 4).unwrap(), 6);
        assert_eq!(arithmetic.div(1, 9).unwrap(), 5); // 9 * 5 = 45 = 1 (mod 11)

        assert_eq!(arithmetic.pow(2, 5).unwrap(), 10);
        assert_eq!(arithmetic.pow(3, 10).unwrap(), 1); // by Fermat theorem
        assert_eq!(arithmetic.pow(3, 4).unwrap(), 4);
        assert_eq!(arithmetic.pow(7, 3).unwrap(), 2);
    }

    #[test]
    fn modular_arithmetic_never_overflows() {
        const MODULUS: u8 = 241;

        let arithmetic = ModularArithmetic::new(MODULUS);
        for x in 0..=u8::max_value() {
            for y in 0..=u8::max_value() {
                let expected = (u16::from(x) + u16::from(y)) % u16::from(MODULUS);
                assert_eq!(u16::from(arithmetic.add(x, y).unwrap()), expected);

                let mut expected = (i16::from(x) - i16::from(y)) % i16::from(MODULUS);
                if expected < 0 {
                    expected += i16::from(MODULUS);
                }
                assert_eq!(i16::from(arithmetic.sub(x, y).unwrap()), expected);

                let expected = (u16::from(x) * u16::from(y)) % u16::from(MODULUS);
                assert_eq!(u16::from(arithmetic.mul(x, y).unwrap()), expected);
            }
        }

        for x in 0..=u8::max_value() {
            let inv = arithmetic.invert(x);
            if x % MODULUS == 0 {
                assert!(inv.is_none());
            } else {
                let inv = u16::from(inv.unwrap());
                assert_eq!((inv * u16::from(x)) % u16::from(MODULUS), 1);
            }
        }
    }

    // Takes ~1s in the debug mode.
    const SAMPLE_COUNT: usize = 25_000;

    fn mini_fuzz_for_prime_modulus(modulus: u64) {
        let arithmetic = ModularArithmetic::new(modulus);
        let unsigned_wide_mod = u128::from(modulus);
        let signed_wide_mod = i128::from(modulus);
        let mut rng = StdRng::seed_from_u64(modulus);

        for (x, y) in (0..SAMPLE_COUNT).map(|_| rng.gen::<(u64, u64)>()) {
            let expected = (u128::from(x) + u128::from(y)) % unsigned_wide_mod;
            assert_eq!(u128::from(arithmetic.add(x, y).unwrap()), expected);

            let mut expected = (i128::from(x) - i128::from(y)) % signed_wide_mod;
            if expected < 0 {
                expected += signed_wide_mod;
            }
            assert_eq!(i128::from(arithmetic.sub(x, y).unwrap()), expected);

            let expected = (u128::from(x) * u128::from(y)) % unsigned_wide_mod;
            assert_eq!(u128::from(arithmetic.mul(x, y).unwrap()), expected);
        }

        for x in (0..SAMPLE_COUNT).map(|_| rng.gen::<u64>()) {
            let inv = arithmetic.invert(x);
            if x % modulus == 0 {
                // Quite unlikely, but better be safe than sorry.
                assert!(inv.is_none());
            } else {
                let inv = u128::from(inv.unwrap());
                assert_eq!((inv * u128::from(x)) % unsigned_wide_mod, 1);
            }
        }

        for _ in 0..(SAMPLE_COUNT / 10) {
            let x = rng.gen::<u64>();
            let wide_x = u128::from(x);

            // Check a random small exponent.
            let exp = rng.gen_range(1_u64, 1_000);
            let expected_pow = (0..exp).fold(1_u128, |acc, _| (acc * wide_x) % unsigned_wide_mod);
            assert_eq!(u128::from(arithmetic.pow(x, exp).unwrap()), expected_pow);

            if x % modulus != 0 {
                // Check Fermat's little theorem.
                let pow = arithmetic.pow(x, modulus - 1).unwrap();
                assert_eq!(pow, 1);
            }
        }
    }

    #[test]
    fn mini_fuzz_for_small_modulus() {
        mini_fuzz_for_prime_modulus(3);
        mini_fuzz_for_prime_modulus(7);
        mini_fuzz_for_prime_modulus(23);
        mini_fuzz_for_prime_modulus(61);
    }

    #[test]
    fn mini_fuzz_for_u32_modulus() {
        // Primes taken from https://www.numberempire.com/primenumbers.php
        mini_fuzz_for_prime_modulus(3_000_000_019);
        mini_fuzz_for_prime_modulus(3_500_000_011);
        mini_fuzz_for_prime_modulus(4_000_000_007);
    }

    #[test]
    fn mini_fuzz_for_large_u64_modulus() {
        // Primes taken from https://bigprimes.org/
        mini_fuzz_for_prime_modulus(2_594_642_710_891_962_701);
        mini_fuzz_for_prime_modulus(5_647_618_287_156_850_721);
        mini_fuzz_for_prime_modulus(9_223_372_036_854_775_837);
        mini_fuzz_for_prime_modulus(10_902_486_311_044_492_273);
    }
}
