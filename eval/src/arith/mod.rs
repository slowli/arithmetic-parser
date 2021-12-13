//! `Arithmetic` trait and its implementations.
//!
//! # Traits
//!
//! An [`Arithmetic`] defines fallible arithmetic operations on primitive values
//! of an [`ExecutableModule`], namely, addition, subtraction, multiplication, division,
//! exponentiation (all binary ops), and negation (a unary op). Any module can be run
//! with any `Arithmetic` on its primitive values, although some modules are reasonably tied
//! to a particular arithmetic or a class of arithmetics (e.g., arithmetics on finite fields).
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
//! - [`StdArithmetic`] takes all implementations from the corresponding [`ops`](core::ops) traits.
//!   This means that it's safe to use *provided* the ops are infallible. As a counter-example,
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

#![allow(renamed_and_removed_lints, clippy::unknown_clippy_lints)]
// ^ `map_err_ignore` is newer than MSRV, and `clippy::unknown_clippy_lints` is removed
// since Rust 1.51.

use core::{cmp::Ordering, fmt};

use crate::error::ArithmeticError;

#[cfg(feature = "bigint")]
mod bigint;
mod generic;
mod modular;

pub use self::{
    generic::{
        Checked, CheckedArithmetic, CheckedArithmeticKind, NegateOnlyZero, StdArithmetic,
        Unchecked, WrappingArithmetic,
    },
    modular::{DoubleWidth, ModularArithmetic},
};

/// Encapsulates arithmetic operations on a certain primitive type (or an enum of primitive types).
///
/// Unlike operations on built-in integer types, arithmetic operations may be fallible.
/// Additionally, the arithmetic can have a state. This is used, for example, in
/// [`ModularArithmetic`], which stores the modulus in the state.
pub trait Arithmetic<T> {
    /// Adds two values.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is unsuccessful (e.g., on integer overflow).
    fn add(&self, x: T, y: T) -> Result<T, ArithmeticError>;

    /// Subtracts two values.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is unsuccessful (e.g., on integer underflow).
    fn sub(&self, x: T, y: T) -> Result<T, ArithmeticError>;

    /// Multiplies two values.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is unsuccessful (e.g., on integer overflow).
    fn mul(&self, x: T, y: T) -> Result<T, ArithmeticError>;

    /// Divides two values.
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

    /// Negates a value.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is unsuccessful (e.g., on integer overflow).
    fn neg(&self, x: T) -> Result<T, ArithmeticError>;

    /// Checks if two values are equal. Note that equality can be a non-trivial operation;
    /// e.g., different numbers may be equal as per modular arithmetic.
    fn eq(&self, x: &T, y: &T) -> bool;
}

/// Extends an [`Arithmetic`] with a comparison operation on values.
pub trait OrdArithmetic<T>: Arithmetic<T> {
    /// Compares two values. Returns `None` if the numbers are not comparable, or the comparison
    /// result otherwise.
    fn partial_cmp(&self, x: &T, y: &T) -> Option<Ordering>;
}

impl<T> fmt::Debug for dyn OrdArithmetic<T> + '_ {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("OrdArithmetic").finish()
    }
}

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

/// Extension trait for [`Arithmetic`] allowing to combine the arithmetic with comparisons.
///
/// # Examples
///
/// ```
/// use arithmetic_eval::arith::{ArithmeticExt, ModularArithmetic};
/// # use arithmetic_eval::{Environment, ExecutableModule, Value};
/// # use arithmetic_parser::grammars::{NumGrammar, Untyped, Parse};
///
/// # fn main() -> anyhow::Result<()> {
/// let base = ModularArithmetic::new(11);
///
/// // `ModularArithmetic` requires to define how numbers will be compared -
/// // and the simplest solution is to not compare them at all.
/// let program = Untyped::<NumGrammar<u32>>::parse_statements("1 < 3 || 1 >= 3")?;
/// let module = ExecutableModule::new("test", &program)?;
/// let env = Environment::with_arithmetic(base.without_comparisons());
/// assert_eq!(module.with_env(&env)?.run()?, Value::Bool(false));
///
/// // We can compare numbers by their integer value. This can lead
/// // to pretty confusing results, though.
/// let bogus_arithmetic = base.with_natural_comparison();
/// let program = Untyped::<NumGrammar<u32>>::parse_statements(r#"
///     (x, y, z) = (1, 12, 5);
///     x == y && x < z && y > z
/// "#)?;
/// let module = ExecutableModule::new("test", &program)?;
/// let env = Environment::with_arithmetic(bogus_arithmetic);
/// assert_eq!(module.with_env(&env)?.run()?, Value::Bool(true));
///
/// // It's possible to fix the situation using a custom comparison function,
/// // which will compare numbers by their residual class.
/// let less_bogus_arithmetic = base.with_comparison(|&x: &u32, &y: &u32| {
///     (x % 11).partial_cmp(&(y % 11))
/// });
/// let env = Environment::with_arithmetic(less_bogus_arithmetic);
/// assert_eq!(module.with_env(&env)?.run()?, Value::Bool(false));
/// # Ok(())
/// # }
/// ```
pub trait ArithmeticExt<T>: Arithmetic<T> + Sized {
    /// Combines this arithmetic with a comparison function that assumes any two values are
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
            comparison: T::partial_cmp,
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
