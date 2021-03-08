//! Logic for converting `Parsed*` types into their "main" counterparts.

use nom::Err as NomErr;

use std::{collections::HashMap, convert::TryFrom, fmt};

use crate::{
    parser::{ParsedFnType, ParsedTupleLength, ParsedValueType},
    ConstParamDescription, FnArgs, FnType, TupleLength, TypeParamDescription, ValueType,
};
use arithmetic_parser::{ErrorKind as ParseErrorKind, InputSpan, LocatedSpan, NomResult};

/// Kinds of errors that can occur when converting `Parsed*` types into their "main" counterparts.
#[derive(Debug)]
#[non_exhaustive]
pub enum ConversionErrorKind {
    /// Duplicate const param definition.
    DuplicateConst {
        /// Offending definition.
        name: String,
        /// Previous definition of a param with the same name.
        previous: LocatedSpan<usize>,
    },
    /// Duplicate type param definition.
    DuplicateTypeParam {
        /// Offending definition.
        name: String,
        /// Previous definition of a param with the same name.
        previous: LocatedSpan<usize>,
    },
    /// Undefined const param.
    UndefinedConst(String),
    /// Undefined type param.
    UndefinedTypeParam(String),
}

impl ConversionErrorKind {
    fn with_span(self, span: InputSpan<'_>) -> ConversionError<&str> {
        ConversionError {
            inner: LocatedSpan::from(span).copy_with_extra(self),
        }
    }
}

impl fmt::Display for ConversionErrorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateConst { name, .. } => {
                write!(formatter, "Duplicate const param definition: `{}`", name)
            }
            Self::DuplicateTypeParam { name, .. } => {
                write!(formatter, "Duplicate type param definition: `{}`", name)
            }
            Self::UndefinedConst(name) => {
                write!(formatter, "Undefined const param `{}`", name)
            }
            Self::UndefinedTypeParam(name) => {
                write!(formatter, "Undefined type param `{}`", name)
            }
        }
    }
}

impl std::error::Error for ConversionErrorKind {}

#[derive(Debug)]
pub struct ConversionError<Span> {
    inner: LocatedSpan<Span, ConversionErrorKind>,
}

impl<Span> ConversionError<Span> {
    /// Returns the kind of this error.
    pub fn kind(&self) -> &ConversionErrorKind {
        &self.inner.extra
    }
}

impl<Span: Copy> ConversionError<Span> {
    /// Returns the main span on which the error has occurred.
    pub fn main_span(&self) -> LocatedSpan<Span> {
        self.inner.with_no_extra()
    }
}

impl<Span> fmt::Display for ConversionError<Span> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}:{}: {}",
            self.inner.location_line(),
            self.inner.location_offset(),
            self.kind()
        )
    }
}

impl<Span: fmt::Debug> std::error::Error for ConversionError<Span> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.inner.extra)
    }
}

/// Intermediate conversion state.
#[derive(Debug, Default, Clone)]
struct ConversionState<'a> {
    const_params: HashMap<&'a str, (InputSpan<'a>, usize)>,
    type_params: HashMap<&'a str, (InputSpan<'a>, usize)>,
}

impl<'a> ConversionState<'a> {
    fn insert_const_param(&mut self, param: InputSpan<'a>) -> Result<(), ConversionError<&'a str>> {
        let new_idx = self.const_params.len();
        let name = *param.fragment();
        if let Some((previous, _)) = self.const_params.insert(name, (param, new_idx)) {
            let err = ConversionErrorKind::DuplicateConst {
                name: name.to_owned(),
                previous: LocatedSpan::from(previous).map_fragment(str::len),
            };
            Err(err.with_span(param))
        } else {
            Ok(())
        }
    }

    fn insert_type_param(&mut self, param: InputSpan<'a>) -> Result<(), ConversionError<&'a str>> {
        let new_idx = self.type_params.len();
        let name = *param.fragment();
        if let Some((previous, _)) = self.type_params.insert(name, (param, new_idx)) {
            let err = ConversionErrorKind::DuplicateTypeParam {
                name: name.to_owned(),
                previous: LocatedSpan::from(previous).map_fragment(str::len),
            };
            Err(err.with_span(param))
        } else {
            Ok(())
        }
    }

    fn type_param_idx(&self, param_name: &str) -> Option<usize> {
        self.type_params.get(param_name).map(|(_, idx)| *idx)
    }

    fn const_param_idx(&self, param_name: &str) -> Option<usize> {
        self.const_params.get(param_name).map(|(_, idx)| *idx)
    }
}

impl<'a> ParsedValueType<'a> {
    fn try_convert(
        &self,
        state: &ConversionState<'a>,
    ) -> Result<ValueType, ConversionError<&'a str>> {
        Ok(match self {
            Self::Any => ValueType::Any,
            Self::Bool => ValueType::Bool,
            Self::Number => ValueType::Number,

            Self::Ident(ident) => {
                let name = *ident.fragment();
                let idx = state.type_param_idx(name).ok_or_else(|| {
                    ConversionErrorKind::UndefinedTypeParam(name.to_owned()).with_span(*ident)
                })?;
                ValueType::TypeParam(idx)
            }

            Self::Function(fn_type) => {
                let converted_fn = fn_type.try_convert(state.clone())?;
                ValueType::Function(Box::new(converted_fn))
            }

            Self::Tuple(elements) => {
                let converted_elements: Result<Vec<_>, _> =
                    elements.iter().map(|elt| elt.try_convert(state)).collect();
                ValueType::Tuple(converted_elements?)
            }

            Self::Slice { element, length } => {
                let converted_length = match length {
                    ParsedTupleLength::Ident(ident) => {
                        let name = *ident.fragment();
                        let const_param = state.const_param_idx(name).ok_or_else(|| {
                            ConversionErrorKind::UndefinedConst(name.to_owned()).with_span(*ident)
                        })?;
                        TupleLength::Param(const_param)
                    }
                    ParsedTupleLength::Any => TupleLength::Any,
                    ParsedTupleLength::Dynamic => TupleLength::Dynamic,
                };
                ValueType::Slice {
                    element: Box::new(element.try_convert(state)?),
                    length: converted_length,
                }
            }
        })
    }
}

impl<'a> TryFrom<ParsedValueType<'a>> for ValueType {
    type Error = ConversionError<&'a str>;

    fn try_from(value: ParsedValueType<'a>) -> Result<Self, Self::Error> {
        value.try_convert(&ConversionState::default())
    }
}

impl ValueType {
    pub fn parse(input: InputSpan<'_>) -> NomResult<'_, Self> {
        let (rest, parsed) = ParsedValueType::parse(input)?;
        let ty = ValueType::try_from(parsed).map_err(|err| {
            let err_span = err.main_span();
            let err = ParseErrorKind::Type(err.inner.extra.into()).with_span(&err_span);
            NomErr::Failure(err)
        })?;
        Ok((rest, ty))
    }
}

impl<'a> ParsedFnType<'a> {
    fn try_convert(
        &self,
        mut state: ConversionState<'a>,
    ) -> Result<FnType, ConversionError<&'a str>> {
        // Check params for consistency.
        for param in &self.const_params {
            state.insert_const_param(*param)?;
        }
        for (param, _) in &self.type_params {
            state.insert_type_param(*param)?;
        }

        let args: Result<Vec<_>, _> = self
            .args
            .iter()
            .map(|arg| arg.try_convert(&state))
            .collect();
        Ok(FnType {
            args: FnArgs::List(args?),
            return_type: self.return_type.try_convert(&state)?,

            type_params: self
                .type_params
                .iter()
                .map(|(name, bounds)| {
                    (
                        state.type_param_idx(name.fragment()).unwrap(),
                        TypeParamDescription {
                            maybe_non_linear: bounds.maybe_non_linear,
                        },
                    )
                })
                .collect(),

            const_params: self
                .const_params
                .iter()
                .map(|name| {
                    (
                        state.const_param_idx(name.fragment()).unwrap(),
                        ConstParamDescription,
                    )
                })
                .collect(),
        })
    }
}

impl<'a> TryFrom<ParsedFnType<'a>> for FnType {
    type Error = ConversionError<&'a str>;

    fn try_from(value: ParsedFnType<'a>) -> Result<Self, Self::Error> {
        value.try_convert(ConversionState::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use assert_matches::assert_matches;

    #[test]
    fn converting_raw_fn_type() {
        let input = InputSpan::new("fn<const N; T>([T; N], fn(T) -> Bool) -> Bool");
        let (_, fn_type) = ParsedFnType::parse(input).unwrap();
        let fn_type = FnType::try_from(fn_type).unwrap();

        assert_eq!(fn_type.to_string(), *input.fragment());
    }

    #[test]
    fn converting_raw_fn_type_duplicate_type() {
        let input = InputSpan::new("fn<T, T>([T; N], fn(T) -> Bool) -> Bool");
        let (_, fn_type) = ParsedFnType::parse(input).unwrap();
        let err = FnType::try_from(fn_type).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 6);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::DuplicateTypeParam { name, previous }
                if name == "T" && previous.location_offset() == 3
        );
    }

    #[test]
    fn converting_raw_fn_type_duplicate_type_in_embedded_fn() {
        let input = InputSpan::new("fn<const N; T>([T; N], fn<T>(T) -> Bool) -> Bool");
        let (_, fn_type) = ParsedFnType::parse(input).unwrap();
        let err = FnType::try_from(fn_type).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 26);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::DuplicateTypeParam { name, previous }
                if name == "T" && previous.location_offset() == 12
        );
    }

    #[test]
    fn converting_raw_fn_type_duplicate_const() {
        let input = InputSpan::new("fn<const N, N; T>([T; N], fn(T) -> Bool) -> Bool");
        let (_, fn_type) = ParsedFnType::parse(input).unwrap();
        let err = FnType::try_from(fn_type).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 12);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::DuplicateConst { name, previous }
                if name == "N" && previous.location_offset() == 9
        );
    }

    #[test]
    fn converting_raw_fn_type_duplicate_const_in_embedded_fn() {
        let input = InputSpan::new("fn<const N; T>([T; N], fn<const N>(T) -> Bool) -> Bool");
        let (_, fn_type) = ParsedFnType::parse(input).unwrap();
        let err = FnType::try_from(fn_type).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 32);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::DuplicateConst { name, previous }
                if name == "N" && previous.location_offset() == 9
        );
    }

    #[test]
    fn converting_raw_fn_type_undefined_type() {
        let input = InputSpan::new("fn<const N>([T; N], fn(T) -> Bool) -> Bool");
        let (_, fn_type) = ParsedFnType::parse(input).unwrap();
        let err = FnType::try_from(fn_type).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 13);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::UndefinedTypeParam(name) if name == "T"
        );
    }

    #[test]
    fn converting_raw_fn_type_undefined_const() {
        let input = InputSpan::new("fn<T>([T; N], fn(T) -> Bool) -> Bool");
        let (_, fn_type) = ParsedFnType::parse(input).unwrap();
        let err = FnType::try_from(fn_type).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 10);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::UndefinedConst(name) if name == "N"
        );
    }
}
