use crate::{
    parser::{statements, streaming_statements},
    Block, Error, InputSpan, NomResult,
};

use core::fmt;

/// Level of support of Boolean operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BooleanOps {
    /// Do not support boolean operations at all.
    None,
    /// Basic operations (`==`, `!=`, `&&`, `||`).
    Basic,
    /// `Basic` + order comparison operations (`>`, `<`, `>=`, `<=`)?
    Full,
}

/// Parsing features for a `Grammar`.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // flags are independent
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
            boolean_ops: BooleanOps::Full,
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

/// Unites all necessary parsers to form a complete grammar definition.
///
/// The two extension points for a `Grammar` are *literals* and *type annotations*. Each of them
/// have a corresponding associated type and a parser method ([`Lit`] and [`parse_literal`]
/// for literals; [`Type`] and [`parse_type`] for annotations).
///
/// A `Grammar` also defines a set of supported [`Features`]. This allows to customize which
/// constructs are parsed.
///
/// [`Lit`]: #associatedtype.Lit
/// [`parse_literal`]: #tymethod.parse_literal
/// [`Type`]: #associatedtype.Type
/// [`parse_type`]: #tymethod.parse_type
/// [`Features`]: struct.Features.html
///
/// # Examples
///
/// ```
/// use arithmetic_parser::{Features, Grammar, GrammarExt, InputSpan, NomResult};
///
/// /// Grammar that parses `u64` numbers and has a single type annotation, `Num`.
/// #[derive(Debug)]
/// struct IntegerGrammar;
///
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// struct Num;
///
/// impl Grammar for IntegerGrammar {
///     type Lit = u64;
///     type Type = Num;
///     const FEATURES: Features = Features::all();
///
///     // We use the `nom` crate to construct necessary parsers.
///     fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
///         use nom::{character::complete::digit1, combinator::map_res};
///         map_res(digit1, |s: InputSpan<'_>| s.fragment().parse())(input)
///     }
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
/// let parsed = IntegerGrammar::parse_statements(program)?;
/// println!("{:#?}", parsed);
/// # Ok(())
/// # }
/// ```
pub trait Grammar: 'static {
    /// Type of the literal used in the grammar.
    type Lit: Clone + fmt::Debug;
    /// Type of the type annotation used in the grammar.
    type Type: Clone + fmt::Debug;

    /// Features supported by this grammar.
    const FEATURES: Features;

    /// Attempts to parse a literal.
    ///
    /// Literals should follow these rules:
    ///
    /// - A literal must be distinguished from other constructs, in particular,
    ///   variable identifiers.
    /// - If a literal may end with `.` and methods are enabled in [`FEATURES`], care should be
    ///   taken for cases when `.` is a part of a call, rather than a part of a literal.
    ///   For example, a parser for real-valued literals should interpret `1.abs()`
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
    /// [`FEATURES`]: #associatedconstant.FEATURES
    fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit>;

    /// Attempts to parse a type hint.
    ///
    /// # Return value
    ///
    /// The output should follow `nom` conventions on errors / failures.
    fn parse_type(input: InputSpan<'_>) -> NomResult<'_, Self::Type>;
}

/// Helper trait allowing `GrammarExt` to accept multiple types as inputs.
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

/// Extension trait for `Grammar` used by the client applications.
pub trait GrammarExt: Grammar {
    /// Parses a list of statements.
    fn parse_statements<'a, I>(input: I) -> Result<Block<'a, Self>, Error<'a>>
    where
        I: IntoInputSpan<'a>,
        Self: Sized;

    /// Parses a potentially incomplete list of statements.
    fn parse_streaming_statements<'a, I>(input: I) -> Result<Block<'a, Self>, Error<'a>>
    where
        I: IntoInputSpan<'a>,
        Self: Sized;
}

impl<T: Grammar> GrammarExt for T {
    fn parse_statements<'a, I>(input: I) -> Result<Block<'a, Self>, Error<'a>>
    where
        I: IntoInputSpan<'a>,
        Self: Sized,
    {
        statements(input.into_input_span())
    }

    fn parse_streaming_statements<'a, I>(input: I) -> Result<Block<'a, Self>, Error<'a>>
    where
        I: IntoInputSpan<'a>,
        Self: Sized,
    {
        streaming_statements(input.into_input_span())
    }
}
