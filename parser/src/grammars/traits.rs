use anyhow::anyhow;
use nom::Err as NomErr;

use core::{fmt, marker::PhantomData};

use crate::{
    parser::{statements, streaming_statements},
    Block, Error, ErrorKind, InputSpan, NomResult, SpannedError,
};

/// Level of support of Boolean operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BooleanOps {
    /// Do not support boolean operations at all.
    None,
    /// Basic operations (`==`, `!=`, `&&`, `||`).
    Basic,
    /// `Basic` + order comparison operations (`>`, `<`, `>=`, `<=`).
    OrderComparisons,
}

/// Parsing features for a `Grammar`.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // flags are independent
#[non_exhaustive]
pub struct Features {
    /// Parse tuple types?
    pub tuples: bool,
    /// Parse type annotations?
    pub type_annotations: bool,
    /// Parse function definitions?
    pub fn_definitions: bool,
    /// Parse blocks?
    pub blocks: bool,
    /// Parse methods?
    pub methods: bool,
    /// Level of support of Boolean operations.
    pub boolean_ops: BooleanOps,
}

impl Features {
    /// Returns the set of all available features.
    pub const fn all() -> Self {
        Self {
            tuples: true,
            type_annotations: true,
            fn_definitions: true,
            blocks: true,
            methods: true,
            boolean_ops: BooleanOps::OrderComparisons,
        }
    }

    /// Returns the set with all features disabled.
    pub const fn none() -> Self {
        Self {
            tuples: false,
            type_annotations: false,
            fn_definitions: false,
            blocks: false,
            methods: false,
            boolean_ops: BooleanOps::None,
        }
    }
}

/// Encapsulates parsing literals in a grammar.
///
/// # Examples
///
/// If your grammar does not need to support type annotations, you can define a `ParseLiteral` impl
/// and wrap it into [`Untyped`] to get a [`Grammar`] / [`Parse`]:
///
/// [`Untyped`]: struct.Untyped.html
/// [`Grammar`]: trait.Grammar.html
/// [`Parse`]: trait.Parse.html
///
/// ```
/// use arithmetic_parser::{
///     grammars::{Features, ParseLiteral, Parse, Untyped},
///     InputSpan, NomResult,
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
///         map_res(digit1, |s: InputSpan<'_>| s.fragment().parse())(input)
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
    ///
    /// [`Parse::FEATURES`]: trait.Parse.html#associatedconstant.FEATURES
    fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit>;
}

/// Extension of [`ParseLiteral`] that parses type annotations.
///
/// # Examples
///
/// Use a [`Typed`] wrapper to create a parser from the grammar, which will support all features:
///
/// [`ParseLiteral`]: trait.ParseLiteral.html
///
/// ```
/// # use arithmetic_parser::{
/// #     grammars::{Features, ParseLiteral, Grammar, Parse, Typed},
/// #     InputSpan, NomResult,
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
///         map_res(digit1, |s: InputSpan<'_>| s.fragment().parse())(input)
///     }
/// }
///
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// struct Num;
///
/// impl Grammar for IntegerGrammar {
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
pub trait Grammar: ParseLiteral {
    /// Type of the type annotation used in the grammar.
    type Type: Clone + fmt::Debug;

    /// Attempts to parse a type hint.
    ///
    /// # Return value
    ///
    /// The output should follow `nom` conventions on errors / failures.
    fn parse_type(input: InputSpan<'_>) -> NomResult<'_, Self::Type>;
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
/// defined via [`Base`] type.
/// A `Grammar` also defines a set of supported [`Features`]. This allows to customize which
/// constructs are parsed.
///
/// Most common sets of `Features` are covered by [`Typed`] and [`Untyped`] wrappers, but it
/// is possible to define custom [`Parse`] implementations as well.
///
/// [`Base`]: #associatedtype.Base
/// [`Features`]: struct.Features.html
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::{
/// #     grammars::{Features, ParseLiteral, Parse, Untyped},
/// #     InputSpan, NomResult,
/// # };
/// #[derive(Debug)]
/// struct IntegerGrammar;
///
/// impl ParseLiteral for IntegerGrammar {
///     // Implementation skipped...
/// #   type Lit = u64;
/// #   fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
/// #       use nom::{character::complete::digit1, combinator::map_res};
/// #       map_res(digit1, |s: InputSpan<'_>| s.fragment().parse())(input)
/// #   }
/// }
///
/// impl Parse for IntegerGrammar {
///     type Base = Untyped<Self>;
///     const FEATURES: Features = Features::none();
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
pub trait Parse {
    /// Base for the grammar providing the literal parser.
    type Base: Grammar;
    /// Features supported by this grammar.
    const FEATURES: Features;

    /// Parses a list of statements.
    fn parse_statements<'a, I>(input: I) -> Result<Block<'a, Self::Base>, Error<'a>>
    where
        I: IntoInputSpan<'a>,
        Self: Sized,
    {
        statements::<Self>(input.into_input_span())
    }

    /// Parses a potentially incomplete list of statements.
    fn parse_streaming_statements<'a, I>(input: I) -> Result<Block<'a, Self::Base>, Error<'a>>
    where
        I: IntoInputSpan<'a>,
        Self: Sized,
    {
        streaming_statements::<Self>(input.into_input_span())
    }
}

/// Wrapper for [`ParseLiteral`] types that allows to use them as a [`Grammar`] or [`Parse`].
///
/// # Examples
///
/// See [`ParseLiteral`] docs for an example of usage.
///
/// [`ParseLiteral`]: trait.ParseLiteral.html
/// [`Grammar`]: trait.Grammar.html
/// [`Parse`]: trait.Parse.html
#[derive(Debug, Clone, Copy, Default)]
pub struct Untyped<T>(PhantomData<T>);

impl<T: ParseLiteral> ParseLiteral for Untyped<T> {
    type Lit = T::Lit;

    #[inline]
    fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
        T::parse_literal(input)
    }
}

impl<T: ParseLiteral> Grammar for Untyped<T> {
    type Type = ();

    #[inline]
    fn parse_type(input: InputSpan<'_>) -> NomResult<'_, Self::Type> {
        let err = anyhow!("Type annotations are not supported by this parser");
        let err = SpannedError::new(input, ErrorKind::Type(err));
        Err(NomErr::Failure(err))
    }
}

impl<T: ParseLiteral> Parse for Untyped<T> {
    type Base = Self;

    const FEATURES: Features = Features {
        type_annotations: false,
        ..Features::all()
    };
}

/// Wrapper for [`Grammar`] types that allows to convert the type to a [`Parse`]r.
///
/// # Examples
///
/// See [`Grammar`] docs for an example of usage.
///
/// [`Grammar`]: trait.Grammar.html
/// [`Parse`]: trait.Parse.html
#[derive(Debug, Clone, Copy, Default)]
pub struct Typed<T>(PhantomData<T>);

impl<T: Grammar> Parse for Typed<T> {
    type Base = T;

    const FEATURES: Features = Features::all();
}
