use anyhow::anyhow;
use bitflags::bitflags;
use nom::Err as NomErr;

use core::{fmt, marker::PhantomData};

use crate::{
    parser::{statements, streaming_statements},
    Block, Error, ErrorKind, InputSpan, NomResult, SpannedError,
};

bitflags! {
    /// Parsing features used to configure [`Parse`] implementations.
    pub struct Features: u64 {
        /// Enables parsing tuples.
        const TUPLES = 1;
        /// Enables parsing type annotations.
        const TYPE_ANNOTATIONS = 2;
        /// Enables parsing function definitions.
        const FN_DEFINITIONS = 4;
        /// Enables parsing blocks.
        const BLOCKS = 8;
        /// Enables parsing methods.
        const METHODS = 16;
        /// Enables parsing equality comparisons (`==`, `!=`), the `!` unary op and
        /// `&&`, `||` binary ops.
        const BOOLEAN_OPS_BASIC = 32;
        /// Enables parsing order comparisons (`>`, `<`, `>=`, `<=`).
        const ORDER_COMPARISONS = 64;
        /// Enables parsing objects.
        const OBJECTS = 128;

        /// Enables all Boolean operations.
        const BOOLEAN_OPS = Self::BOOLEAN_OPS_BASIC.bits | Self::ORDER_COMPARISONS.bits;
    }
}

impl Features {
    /// Creates a copy of these `Features` without any of the flags in `other`.
    pub const fn without(self, other: Self) -> Self {
        Self::from_bits_truncate(self.bits & !other.bits)
    }
}

/// Encapsulates parsing literals in a grammar.
///
/// # Examples
///
/// If your grammar does not need to support type annotations, you can define a `ParseLiteral` impl
/// and wrap it into [`Untyped`] to get a [`Grammar`] / [`Parse`]:
///
/// ```
/// use arithmetic_parser::{
///     grammars::{Features, ParseLiteral, Parse, Untyped},
///     ErrorKind, InputSpan, NomResult,
/// };
///
/// /// Grammar that parses `u64` numbers.
/// #[derive(Debug)]
/// struct IntegerGrammar;
///
/// impl ParseLiteral for IntegerGrammar {
///     type Lit = u64;
///
///     // We use the `nom` crate to construct necessary parsers.
///     fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
///         use nom::{character::complete::digit1, combinator::map_res};
///         let parser = |s: InputSpan<'_>| {
///             s.fragment().parse().map_err(ErrorKind::literal)
///         };
///         map_res(digit1, parser)(input)
///     }
/// }
///
/// // Here's how a grammar can be used.
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     x = 1 + 2 * 3 + sin(a^3 / b^2);
///     some_function = |a, b| (a + b, a - b);
///     other_function = |x| {
///         r = min(rand(), 0);
///         r * x
///     };
///     (y, z) = some_function({ x = x - 1; x }, x);
///     other_function(y - z)
/// "#;
/// let parsed = Untyped::<IntegerGrammar>::parse_statements(program)?;
/// println!("{:#?}", parsed);
/// # Ok(())
/// # }
/// ```
pub trait ParseLiteral: 'static {
    /// Type of the literal used in the grammar.
    type Lit: Clone + fmt::Debug;

    /// Attempts to parse a literal.
    ///
    /// Literals should follow these rules:
    ///
    /// - A literal must be distinguished from other constructs, in particular,
    ///   variable identifiers.
    /// - If a literal may end with `.` and methods are enabled in [`Parse::FEATURES`],
    ///   care should be taken for cases when `.` is a part of a call, rather than a part
    ///   of a literal. For example, a parser for real-valued literals should interpret `1.abs()`
    ///   as a call of the `abs` method on receiver `1`, rather than `1.` followed by
    ///   ineligible `abs()`.
    ///
    /// If a literal may start with `-` or `!` (in general, unary ops), these ops will be
    /// consumed as a part of the literal, rather than `Expr::Unary`, *unless* the literal
    /// is followed by an eligible higher-priority operation (i.e., a method call) *and*
    /// the literal without a preceding unary op is still eligible. That is, if `-1` and `1`
    /// are both valid literals, then `-1.abs()` will be parsed as negation applied to `1.abs()`.
    /// On the other hand, if `!foo!` is a valid literal, but `foo!` isn't, `!foo!.bar()` will
    /// be parsed as method `bar()` called on `!foo!`.
    ///
    /// # Return value
    ///
    /// The output should follow `nom` conventions on errors / failures.
    fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit>;
}

/// Extension of `ParseLiteral` that parses type annotations.
///
/// # Examples
///
/// Use a [`Typed`] wrapper to create a parser from the grammar, which will support all
/// parsing [`Features`]:
///
/// ```
/// # use arithmetic_parser::{
/// #     grammars::{Features, ParseLiteral, Grammar, Parse, Typed},
/// #     ErrorKind, InputSpan, NomResult,
/// # };
/// /// Grammar that parses `u64` numbers and has a single type annotation, `Num`.
/// #[derive(Debug)]
/// struct IntegerGrammar;
///
/// impl ParseLiteral for IntegerGrammar {
///     type Lit = u64;
///
///     fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
///         use nom::{character::complete::digit1, combinator::map_res};
///         let parser = |s: InputSpan<'_>| {
///             s.fragment().parse().map_err(ErrorKind::literal)
///         };
///         map_res(digit1, parser)(input)
///     }
/// }
///
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// struct Num;
///
/// impl Grammar<'_> for IntegerGrammar {
///     type Type = Num;
///
///     fn parse_type(input: InputSpan<'_>) -> NomResult<'_, Self::Type> {
///         use nom::{bytes::complete::tag, combinator::map};
///         map(tag("Num"), |_| Num)(input)
///     }
/// }
///
/// // Here's how a grammar can be used.
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     x = 1 + 2 * 3 + sin(a^3 / b^2);
///     some_function = |a, b: Num| (a + b, a - b);
///     other_function = |x| {
///         r = min(rand(), 0);
///         r * x
///     };
///     (y, z: Num) = some_function({ x = x - 1; x }, x);
///     other_function(y - z)
/// "#;
/// let parsed = Typed::<IntegerGrammar>::parse_statements(program)?;
/// println!("{:#?}", parsed);
/// # Ok(())
/// # }
/// ```
pub trait Grammar<'a>: ParseLiteral {
    /// Type of the type annotation used in the grammar.
    type Type: Clone + fmt::Debug;

    /// Attempts to parse a type annotation.
    ///
    /// # Return value
    ///
    /// The output should follow `nom` conventions on errors / failures.
    fn parse_type(input: InputSpan<'a>) -> NomResult<'a, Self::Type>;
}

/// Helper trait allowing `Parse` to accept multiple types as inputs.
pub trait IntoInputSpan<'a> {
    /// Converts input into a span.
    fn into_input_span(self) -> InputSpan<'a>;
}

impl<'a> IntoInputSpan<'a> for InputSpan<'a> {
    fn into_input_span(self) -> InputSpan<'a> {
        self
    }
}

impl<'a> IntoInputSpan<'a> for &'a str {
    fn into_input_span(self) -> InputSpan<'a> {
        InputSpan::new(self)
    }
}

/// Unites all necessary parsers to form a complete grammar definition.
///
/// The two extension points for a `Grammar` are *literals* and *type annotations*. They are
/// defined via [`Base`](Self::Base) type.
/// A `Grammar` also defines a set of supported [`Features`]. This allows to customize which
/// constructs are parsed.
///
/// Most common sets of `Features` are covered by [`Typed`] and [`Untyped`] wrappers;
/// these types allow to not declare `Parse` impl explicitly. It is still possible
/// to define custom `Parse` implementations, as shown in the example below.
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::{
/// #     grammars::{Features, ParseLiteral, Parse, Untyped},
/// #     ErrorKind, InputSpan, NomResult,
/// # };
/// #[derive(Debug)]
/// struct IntegerGrammar;
///
/// impl ParseLiteral for IntegerGrammar {
///     // Implementation skipped...
/// #   type Lit = u64;
/// #   fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
/// #       use nom::{character::complete::digit1, combinator::map_res};
/// #       let parser = |s: InputSpan<'_>| s.fragment().parse().map_err(ErrorKind::literal);
/// #       map_res(digit1, parser)(input)
/// #   }
/// }
///
/// impl Parse<'_> for IntegerGrammar {
///     type Base = Untyped<Self>;
///     const FEATURES: Features = Features::empty();
/// }
///
/// // Here's how a grammar can be used.
/// # fn main() -> anyhow::Result<()> {
/// let program = "x = 1 + 2 * 3 + sin(a^3 / b^2);";
/// let parsed = IntegerGrammar::parse_statements(program)?;
/// println!("{:#?}", parsed);
/// # Ok(())
/// # }
/// ```
pub trait Parse<'a> {
    /// Base for the grammar providing the literal and type annotation parsers.
    type Base: Grammar<'a>;
    /// Features supported by this grammar.
    const FEATURES: Features;

    /// Parses a list of statements.
    fn parse_statements<I>(input: I) -> Result<Block<'a, Self::Base>, Error<'a>>
    where
        I: IntoInputSpan<'a>,
        Self: Sized,
    {
        statements::<Self>(input.into_input_span())
    }

    /// Parses a potentially incomplete list of statements.
    fn parse_streaming_statements<I>(input: I) -> Result<Block<'a, Self::Base>, Error<'a>>
    where
        I: IntoInputSpan<'a>,
        Self: Sized,
    {
        streaming_statements::<Self>(input.into_input_span())
    }
}

/// Wrapper for [`ParseLiteral`] types that allows to use them as a [`Grammar`] or [`Parse`]r.
///
/// When used as a `Grammar`, type annotations are not supported; any use of an annotation will
/// lead to a parsing error. When used as a `Parse`r, all [`Features`] are on except for
/// type annotations.
///
/// # Examples
///
/// See [`ParseLiteral`] docs for an example of usage.
#[derive(Debug)]
pub struct Untyped<T>(PhantomData<T>);

impl<T: ParseLiteral> ParseLiteral for Untyped<T> {
    type Lit = T::Lit;

    #[inline]
    fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
        T::parse_literal(input)
    }
}

impl<T: ParseLiteral> Grammar<'_> for Untyped<T> {
    type Type = ();

    #[inline]
    fn parse_type(input: InputSpan<'_>) -> NomResult<'_, Self::Type> {
        let err = anyhow!("Type annotations are not supported by this parser");
        let err = SpannedError::new(input, ErrorKind::Type(err));
        Err(NomErr::Failure(err))
    }
}

impl<T: ParseLiteral> Parse<'_> for Untyped<T> {
    type Base = Self;

    const FEATURES: Features = Features::all().without(Features::TYPE_ANNOTATIONS);
}

/// Wrapper for [`Grammar`] types that allows to convert the type to a [`Parse`]r. The resulting
/// parser supports all [`Features`].
///
/// # Examples
///
/// See [`Grammar`] docs for an example of usage.
#[derive(Debug)]
// TODO: consider name change (`Parser`?)
pub struct Typed<T>(PhantomData<T>);

impl<'a, T: Grammar<'a>> Parse<'a> for Typed<T> {
    type Base = T;

    const FEATURES: Features = Features::all();
}
