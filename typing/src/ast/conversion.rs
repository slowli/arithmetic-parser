//! Logic for converting `*Ast` types into their "main" counterparts.

use nom::Err as NomErr;

use std::{collections::HashMap, convert::TryFrom, fmt, str::FromStr};

use crate::{
    ast::{FnTypeAst, TupleLengthAst, ValueTypeAst},
    ConstParamDescription, FnArgs, FnType, TupleLength, TypeParamDescription, ValueType,
};
use arithmetic_parser::{
    ErrorKind as ParseErrorKind, InputSpan, LocatedSpan, NomResult, SpannedError, StripCode,
};

/// Kinds of errors that can occur when converting `*Ast` types into their "main" counterparts.
///
/// During parsing, errors of this type are wrapped into the [`Type`](ParseErrorKind::Type)
/// variant of parsing errors.
///
/// # Examples
///
/// ```
/// use arithmetic_parser::{grammars::{Parse, Typed}, ErrorKind};
/// use arithmetic_typing::{ast::ConversionErrorKind, NumGrammar};
/// # use assert_matches::assert_matches;
///
/// let code = "bogus_slice: [T; _] = (1, 2, 3);";
/// let err = Typed::<NumGrammar<f32>>::parse_statements(code).unwrap_err();
///
/// assert_eq!(*err.span().fragment(), "T");
/// let err = match err.kind() {
///     ErrorKind::Type(type_err) => type_err
///         .downcast_ref::<ConversionErrorKind>()
///         .unwrap(),
///     _ => unreachable!(),
/// };
/// assert_matches!(
///     err,
///     ConversionErrorKind::UndefinedTypeParam(t) if t == "T"
/// );
/// ```
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
    // TODO: unused params?
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

/// Errors that can occur when converting `*Ast` types into their "main" counterparts, together
/// with an error span.
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

impl<'a> ValueTypeAst<'a> {
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
                    TupleLengthAst::Ident(ident) => {
                        let name = *ident.fragment();
                        let const_param = state.const_param_idx(name).ok_or_else(|| {
                            ConversionErrorKind::UndefinedConst(name.to_owned()).with_span(*ident)
                        })?;
                        TupleLength::Param(const_param)
                    }
                    TupleLengthAst::Any => TupleLength::Any,
                    TupleLengthAst::Dynamic => TupleLength::Dynamic,
                };
                ValueType::Slice {
                    element: Box::new(element.try_convert(state)?),
                    length: converted_length,
                }
            }
        })
    }
}

impl<'a> TryFrom<ValueTypeAst<'a>> for ValueType {
    type Error = ConversionError<&'a str>;

    fn try_from(value: ValueTypeAst<'a>) -> Result<Self, Self::Error> {
        value.try_convert(&ConversionState::default())
    }
}

impl ValueType {
    /// Parses type from `input`.
    pub fn parse(input: InputSpan<'_>) -> NomResult<'_, Self> {
        Self::parse_inner(input, false)
    }

    fn parse_inner(input: InputSpan<'_>, consume_all_input: bool) -> NomResult<'_, Self> {
        let (rest, parsed) = ValueTypeAst::parse(input)?;
        if consume_all_input && !rest.fragment().is_empty() {
            let err = ParseErrorKind::Leftovers.with_span(&rest.into());
            return Err(NomErr::Failure(err));
        }

        let ty = ValueType::try_from(parsed).map_err(|err| {
            let err_span = err.main_span();
            let err = ParseErrorKind::Type(err.inner.extra.into()).with_span(&err_span);
            NomErr::Failure(err)
        })?;
        Ok((rest, ty))
    }
}

impl FromStr for ValueType {
    type Err = SpannedError<usize>;

    fn from_str(def: &str) -> Result<Self, Self::Err> {
        let input = InputSpan::new(def);
        let (_, ty) = Self::parse_inner(input, true).map_err(|err| match err {
            NomErr::Incomplete(_) => ParseErrorKind::Incomplete
                .with_span(&input.into())
                .strip_code(),
            NomErr::Error(e) | NomErr::Failure(e) => e.strip_code(),
        })?;

        Ok(ty)
    }
}

impl<'a> FnTypeAst<'a> {
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

impl<'a> TryFrom<FnTypeAst<'a>> for FnType {
    type Error = ConversionError<&'a str>;

    fn try_from(value: FnTypeAst<'a>) -> Result<Self, Self::Error> {
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
        let (_, fn_type) = FnTypeAst::parse(input).unwrap();
        let fn_type = FnType::try_from(fn_type).unwrap();

        assert_eq!(fn_type.to_string(), *input.fragment());
    }

    #[test]
    fn converting_raw_fn_type_duplicate_type() {
        let input = InputSpan::new("fn<T, T>([T; N], fn(T) -> Bool) -> Bool");
        let (_, fn_type) = FnTypeAst::parse(input).unwrap();
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
        let (_, fn_type) = FnTypeAst::parse(input).unwrap();
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
        let (_, fn_type) = FnTypeAst::parse(input).unwrap();
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
        let (_, fn_type) = FnTypeAst::parse(input).unwrap();
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
        let (_, fn_type) = FnTypeAst::parse(input).unwrap();
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
        let (_, fn_type) = FnTypeAst::parse(input).unwrap();
        let err = FnType::try_from(fn_type).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 10);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::UndefinedConst(name) if name == "N"
        );
    }

    #[test]
    fn parsing_basic_value_types() {
        let num_type: ValueType = "Num".parse().unwrap();
        assert_eq!(num_type, ValueType::Number);

        let bool_type: ValueType = "Bool".parse().unwrap();
        assert_eq!(bool_type, ValueType::Bool);

        let tuple_type: ValueType = "(Num, (Bool, _))".parse().unwrap();
        assert_eq!(
            tuple_type,
            ValueType::Tuple(vec![
                ValueType::Number,
                ValueType::Tuple(vec![ValueType::Bool, ValueType::Any]),
            ])
        );

        let slice_type: ValueType = "[(Num, _); _]".parse().unwrap();
        let (element_ty, length) = match slice_type {
            ValueType::Slice { element, length } => (*element, length),
            _ => panic!("Unexpected type: {:?}", slice_type),
        };
        assert_eq!(
            element_ty,
            ValueType::Tuple(vec![ValueType::Number, ValueType::Any])
        );
        assert_matches!(length, TupleLength::Any);
    }

    #[test]
    fn parsing_functional_value_type() {
        let ty: ValueType = "fn<const N; T, U>([T; N], fn(T) -> U) -> U"
            .parse()
            .unwrap();
        let ty = match ty {
            ValueType::Function(fn_type) => *fn_type,
            _ => panic!("Unexpected type: {:?}", ty),
        };
        assert_eq!(ty.const_params.len(), 1);
        assert_eq!(ty.type_params.len(), 2);
        assert_eq!(ty.return_type, ValueType::TypeParam(1));
    }

    #[test]
    fn parsing_incomplete_value_type() {
        const INCOMPLETE_TYPES: &[&str] = &[
            "fn<",
            "fn<co",
            "fn<const N;",
            "fn<const N; T",
            "fn<const N; T,",
            "fn<const N; T, U>",
            "fn<const N; T, U>(",
            "fn<const N; T, U>([T; ",
            "fn<const N; T, U>([T; N], fn(",
            "fn<const N; T, U>([T; N], fn(T)",
            "fn<const N; T, U>([T; N], fn(T)",
            "fn<const N; T, U>([T; N], fn(T)) -",
            "fn<const N; T, U>([T; N], fn(T)) ->",
        ];

        for &input in INCOMPLETE_TYPES {
            // TODO: some of reported errors are difficult to interpret; should clarify.
            input.parse::<ValueType>().unwrap_err();
        }
    }

    #[test]
    fn parsing_value_type_with_conversion_error() {
        let input = "[T; _]";
        let err = input.parse::<ValueType>().unwrap_err();
        assert_eq!(err.span().location_offset(), 1);
        let err = match err.kind() {
            ParseErrorKind::Type(err) => err.downcast_ref::<ConversionErrorKind>().unwrap(),
            _ => panic!("Unexpected error type: {:?}", err),
        };
        assert_matches!(err, ConversionErrorKind::UndefinedTypeParam(name) if name == "T");
    }
}
