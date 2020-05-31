//! Standard grammars.

use nom::{
    bytes::complete::take_while_m_n,
    combinator::{not, peek},
    number::complete::{double, float},
    sequence::terminated,
};
use num_traits::{Num, Pow};

use core::{f32, f64, fmt, marker::PhantomData, ops};

use crate::{Features, Grammar, NomResult, Span};

// FIXME: `1.foo()` does not work because float parser consumes `1.`

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

/// Ensures that the child parser does not consume a part of a larger expression by rejecting
/// if the part following the input is an alphanumeric char or `_`.
///
/// For example, `float` parses `-Inf`, which can lead to parser failure if it's a part of
/// a larger expression (e.g., `-Infer(2, 3)`).
pub fn ensure_no_overlap<'a, T>(
    parser: impl Fn(Span<'a>) -> NomResult<'a, T>,
) -> impl Fn(Span<'a>) -> NomResult<'a, T> {
    terminated(
        parser,
        peek(not(take_while_m_n(1, 1, |c: char| {
            c.is_ascii_alphabetic() || c == '_'
        }))),
    )
}

impl NumLiteral for f32 {
    fn parse(input: Span<'_>) -> NomResult<'_, Self> {
        ensure_no_overlap(float)(input)
    }
}

impl NumLiteral for f64 {
    fn parse(input: Span<'_>) -> NomResult<'_, Self> {
        ensure_no_overlap(double)(input)
    }
}

#[cfg(feature = "num-complex")]
mod complex {
    use super::*;

    use nom::{
        character::complete::one_of,
        combinator::{map, opt},
        sequence::tuple,
    };
    use num_complex::Complex;

    fn complex_parser<'a, T: Num>(
        num_parser: impl Fn(Span<'a>) -> NomResult<'a, T>,
    ) -> impl Fn(Span<'a>) -> NomResult<'a, Complex<T>> {
        let parser = tuple((num_parser, opt(one_of("ij"))));
        map(parser, |(value, maybe_imag)| {
            if maybe_imag.is_some() {
                Complex::new(T::zero(), value)
            } else {
                Complex::new(value, T::zero())
            }
        })
    }

    impl NumLiteral for num_complex::Complex32 {
        fn parse(input: Span<'_>) -> NomResult<'_, Self> {
            ensure_no_overlap(complex_parser(float))(input)
        }
    }

    impl NumLiteral for num_complex::Complex64 {
        fn parse(input: Span<'_>) -> NomResult<'_, Self> {
            ensure_no_overlap(complex_parser(double))(input)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Expr, GrammarExt};

    use assert_matches::assert_matches;
    use core::f32::INFINITY;

    #[test]
    fn parsing_infinity() {
        let parsed = F32Grammar::parse_statements(Span::new("Inf")).unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Literal(lit) if lit == INFINITY);

        let parsed = F32Grammar::parse_statements(Span::new("-Inf")).unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Literal(lit) if lit == -INFINITY);

        let parsed = F32Grammar::parse_statements(Span::new("Infty")).unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Variable);

        let parsed = F32Grammar::parse_statements(Span::new("Infer(1)")).unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Function { .. });

        let parsed = F32Grammar::parse_statements(Span::new("-Infty")).unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Unary { .. });

        let parsed = F32Grammar::parse_statements(Span::new("-Infer(2, 3)")).unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Unary { .. });
    }
}