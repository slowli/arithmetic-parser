use crate::{
    parser::{statements, streaming_statements},
    Block, Error, InputSpan, NomResult,
};

use core::fmt;

/// Parsing features for a `Grammar`.
// TODO: make boolean expressions optional, too.
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
    /// Parse order comparison operations (`>`, `<`, `>=`, `<=`)?
    pub order_comparisons: bool,
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
            order_comparisons: true,
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
            order_comparisons: false,
        }
    }
}

impl Default for Features {
    fn default() -> Self {
        Self::all()
    }
}

/// Unites all necessary parsers to form a complete grammar definition.
pub trait Grammar: 'static {
    /// Type of the literal used in the grammar.
    type Lit: Clone + fmt::Debug;
    /// Type of the type declaration used in the grammar.
    type Type: Clone + fmt::Debug;

    /// Features supported by this grammar.
    const FEATURES: Features;

    /// Attempts to parse a literal.
    ///
    /// # Return value
    ///
    /// The output should follow `nom` conventions on errors / failures.
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
