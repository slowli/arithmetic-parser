//! Standard grammars.

use num_traits::{Num, Pow};

use std::{fmt, marker::PhantomData, ops};

use crate::{Features, Grammar, NomResult, Span};

/// Single-type numeric grammar parameterized by the literal type.
#[derive(Debug)]
pub struct NumGrammar<T>(PhantomData<T>);

/// Type alias for a grammar on `f32` literals.
pub type F32Grammar = NumGrammar<f32>;
/// Type alias for a grammar on `f64` literals.
pub type F64Grammar = NumGrammar<f64>;

impl<T: NumLiteral> Grammar for NumGrammar<T> {
    type Lit = T;
    type Type = ();

    const FEATURES: Features = Features {
        type_annotations: false,
        ..Features::all()
    };

    fn parse_literal(input: Span<'_>) -> NomResult<'_, Self::Lit> {
        T::parse(input)
    }

    fn parse_type(_input: Span<'_>) -> NomResult<'_, Self::Type> {
        unimplemented!()
    }
}

/// Numeric literal used in `NumGrammar`s.
pub trait NumLiteral:
    'static + Copy + Num + fmt::Debug + ops::Neg<Output = Self> + Pow<Self, Output = Self>
{
    /// Tries to parse a literal.
    fn parse(input: Span<'_>) -> NomResult<'_, Self>;
}

impl NumLiteral for f32 {
    fn parse(input: Span<'_>) -> NomResult<'_, Self> {
        nom::number::complete::float(input)
    }
}

impl NumLiteral for f64 {
    fn parse(input: Span<'_>) -> NomResult<'_, Self> {
        nom::number::complete::double(input)
    }
}

#[cfg(feature = "num-complex")]
mod complex {
    use super::*;

    use nom::{
        character::complete::one_of,
        combinator::{map, opt},
        number::complete::{double, float},
        sequence::tuple,
    };
    use num_complex::Complex;

    fn complex_parser<'a, T: Num>(
        num_parser: impl Fn(Span<'a>) -> NomResult<'a, T>,
        input: Span<'a>,
    ) -> NomResult<'a, Complex<T>> {
        let parser = tuple((num_parser, opt(one_of("ij"))));
        map(parser, |(value, maybe_imag)| {
            if maybe_imag.is_some() {
                Complex::new(T::zero(), value)
            } else {
                Complex::new(value, T::zero())
            }
        })(input)
    }

    impl NumLiteral for num_complex::Complex32 {
        fn parse(input: Span<'_>) -> NomResult<'_, Self> {
            complex_parser(float, input)
        }
    }

    impl NumLiteral for num_complex::Complex64 {
        fn parse(input: Span<'_>) -> NomResult<'_, Self> {
            complex_parser(double, input)
        }
    }
}
