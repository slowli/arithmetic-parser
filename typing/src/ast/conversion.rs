//! Logic for converting `*Ast` types into their "main" counterparts.

use nom::Err as NomErr;

use std::{collections::HashMap, convert::TryFrom, fmt, str::FromStr};

use crate::{
    ast::{FnTypeAst, SliceAst, TupleAst, TupleLengthAst, TypeConstraintsAst, ValueTypeAst},
    types::TypeParamDescription,
    FnType, PrimitiveType, Slice, Tuple, TupleLength, ValueType,
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
/// use arithmetic_parser::{grammars::{Parse, NumGrammar, Typed}, ErrorKind};
/// use arithmetic_typing::{ast::ConversionErrorKind, Annotated};
/// # use assert_matches::assert_matches;
///
/// type Parser = Typed<Annotated<NumGrammar<f32>>>;
///
/// let code = "bogus_slice: [T; _] = (1, 2, 3);";
/// let err = Parser::parse_statements(code).unwrap_err();
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
    /// Invalid type constraint.
    InvalidConstraint(String),
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
            Self::InvalidConstraint(name) => {
                write!(formatter, "Invalid type constraint: `{}`", name)
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

impl<'a> TypeConstraintsAst<'a> {
    fn try_convert<Prim>(&self) -> Result<Prim::Constraints, ConversionError<&'a str>>
    where
        Prim: PrimitiveType,
    {
        self.constraints
            .iter()
            .try_fold(Prim::Constraints::default(), |mut acc, input| {
                let input_str = *input.fragment();
                let partial = Prim::Constraints::from_str(input_str).map_err(|_| {
                    ConversionErrorKind::InvalidConstraint(input_str.to_owned()).with_span(*input)
                })?;
                acc |= &partial;
                Ok(acc)
            })
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

impl<'a, Prim: PrimitiveType> TupleAst<'a, Prim> {
    fn try_convert(
        &self,
        state: &ConversionState<'a>,
    ) -> Result<Tuple<Prim>, ConversionError<&'a str>> {
        let start = self
            .start
            .iter()
            .map(|element| element.try_convert(state))
            .collect::<Result<Vec<_>, _>>()?;
        let middle = self
            .middle
            .as_ref()
            .map(|middle| middle.try_convert(state))
            .transpose()?;
        let end = self
            .end
            .iter()
            .map(|element| element.try_convert(state))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Tuple::new(start, middle, end))
    }
}

impl<'a, Prim: PrimitiveType> SliceAst<'a, Prim> {
    fn try_convert(
        &self,
        state: &ConversionState<'a>,
    ) -> Result<Slice<Prim>, ConversionError<&'a str>> {
        let element = self.element.try_convert(state)?;
        let converted_length = match &self.length {
            TupleLengthAst::Ident(ident) => {
                let name = *ident.fragment();
                let const_param = state.const_param_idx(name).ok_or_else(|| {
                    ConversionErrorKind::UndefinedConst(name.to_owned()).with_span(*ident)
                })?;
                TupleLength::Param(const_param)
            }
            TupleLengthAst::Any => TupleLength::Some { is_dynamic: false },
            TupleLengthAst::Dynamic => TupleLength::Some { is_dynamic: true },
        };

        Ok(Slice::new(element, converted_length))
    }
}

impl<'a, Prim: PrimitiveType> ValueTypeAst<'a, Prim> {
    fn try_convert(
        &self,
        state: &ConversionState<'a>,
    ) -> Result<ValueType<Prim>, ConversionError<&'a str>> {
        Ok(match self {
            Self::Any => ValueType::Some,
            Self::Prim(prim) => ValueType::Prim(prim.to_owned()),

            Self::Ident(ident) => {
                let name = *ident.fragment();
                let idx = state.type_param_idx(name).ok_or_else(|| {
                    ConversionErrorKind::UndefinedTypeParam(name.to_owned()).with_span(*ident)
                })?;
                ValueType::Param(idx)
            }

            Self::Function(fn_type) => {
                let converted_fn = fn_type.try_convert(state.clone())?;
                ValueType::Function(Box::new(converted_fn))
            }

            Self::Tuple(tuple) => tuple.try_convert(state)?.into(),
            Self::Slice(slice) => slice.try_convert(state)?.into(),
        })
    }
}

impl<'a, Prim: PrimitiveType> TryFrom<ValueTypeAst<'a, Prim>> for ValueType<Prim> {
    type Error = ConversionError<&'a str>;

    fn try_from(value: ValueTypeAst<'a, Prim>) -> Result<Self, Self::Error> {
        value.try_convert(&ConversionState::default())
    }
}

impl<Prim: PrimitiveType> ValueType<Prim> {
    /// Parses type from `input`.
    pub fn parse(input: InputSpan<'_>) -> NomResult<'_, Self> {
        parse_inner(ValueTypeAst::parse, input, false)
    }
}

/// Shared parsing code for `ValueType` and `FnType`.
fn parse_inner<'a, Ast, T>(
    parser: fn(InputSpan<'a>) -> NomResult<'a, Ast>,
    input: InputSpan<'a>,
    consume_all_input: bool,
) -> NomResult<'a, T>
where
    T: TryFrom<Ast, Error = ConversionError<&'a str>>,
{
    let (rest, parsed) = parser(input)?;
    if consume_all_input && !rest.fragment().is_empty() {
        let err = ParseErrorKind::Leftovers.with_span(&rest.into());
        return Err(NomErr::Failure(err));
    }

    let ty = T::try_from(parsed).map_err(|err| {
        let err_span = err.main_span();
        let err = ParseErrorKind::Type(err.inner.extra.into()).with_span(&err_span);
        NomErr::Failure(err)
    })?;
    Ok((rest, ty))
}

/// Shared `FromStr` logic for `ValueType` and `FnType`.
fn from_str<'a, Ast, T>(
    parser: fn(InputSpan<'a>) -> NomResult<'a, Ast>,
    def: &'a str,
) -> Result<T, SpannedError<usize>>
where
    T: TryFrom<Ast, Error = ConversionError<&'a str>>,
{
    let input = InputSpan::new(def);
    let (_, ty) = parse_inner(parser, input, true).map_err(|err| match err {
        NomErr::Incomplete(_) => ParseErrorKind::Incomplete
            .with_span(&input.into())
            .strip_code(),
        NomErr::Error(e) | NomErr::Failure(e) => e.strip_code(),
    })?;

    Ok(ty)
}

impl<Prim: PrimitiveType> FromStr for ValueType<Prim> {
    type Err = SpannedError<usize>;

    fn from_str(def: &str) -> Result<Self, Self::Err> {
        from_str(ValueTypeAst::parse, def)
    }
}

impl<'a, Prim: PrimitiveType> FnTypeAst<'a, Prim> {
    fn try_convert(
        &self,
        mut state: ConversionState<'a>,
    ) -> Result<FnType<Prim>, ConversionError<&'a str>> {
        // Check params for consistency.
        for (param, _) in &self.len_params {
            state.insert_const_param(*param)?;
        }
        for (param, _) in &self.type_params {
            state.insert_type_param(*param)?;
        }

        let args = self.args.try_convert(&state)?;

        let const_params = self.len_params.iter().map(|(name, ty)| {
            (
                state.const_param_idx(name.fragment()).unwrap(),
                (*ty).into(),
            )
        });

        let type_params = self.type_params.iter().map(|(name, constraints)| {
            let constraints = constraints.try_convert::<Prim>()?;
            Ok((
                state.type_param_idx(name.fragment()).unwrap(),
                TypeParamDescription::new(constraints),
            ))
        });

        let fn_type = FnType::new(args, self.return_type.try_convert(&state)?)
            .with_len_params(const_params.collect())
            .with_type_params(type_params.collect::<Result<Vec<_>, _>>()?);
        Ok(fn_type)
    }
}

impl<'a, Prim: PrimitiveType> TryFrom<FnTypeAst<'a, Prim>> for FnType<Prim> {
    type Error = ConversionError<&'a str>;

    fn try_from(value: FnTypeAst<'a, Prim>) -> Result<Self, Self::Error> {
        value.try_convert(ConversionState::default())
    }
}

impl<Prim: PrimitiveType> FnType<Prim> {
    /// Parses a functional type from `input`.
    pub fn parse(input: InputSpan<'_>) -> NomResult<'_, Self> {
        parse_inner(FnTypeAst::parse, input, false)
    }
}

impl<Prim: PrimitiveType> FromStr for FnType<Prim> {
    type Err = SpannedError<usize>;

    fn from_str(def: &str) -> Result<Self, Self::Err> {
        from_str(FnTypeAst::parse, def)
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;

    use super::*;

    #[test]
    fn converting_raw_fn_type() {
        let input = InputSpan::new("fn<len N; T>([T; N], fn(T) -> Bool) -> Bool");
        let (_, fn_type) = <FnTypeAst>::parse(input).unwrap();
        let fn_type = FnType::try_from(fn_type).unwrap();

        assert_eq!(fn_type.to_string(), *input.fragment());
    }

    #[test]
    fn converting_raw_fn_type_with_constraint() {
        let input = InputSpan::new("fn<len N; T: Lin>([T; N], fn(T) -> Bool) -> Bool");
        let (_, fn_type) = <FnTypeAst>::parse(input).unwrap();
        let fn_type = FnType::try_from(fn_type).unwrap();

        assert_eq!(fn_type.to_string(), *input.fragment());
    }

    #[test]
    fn converting_raw_fn_type_duplicate_type() {
        let input = InputSpan::new("fn<T, T>([T; N], fn(T) -> Bool) -> Bool");
        let (_, fn_type) = <FnTypeAst>::parse(input).unwrap();
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
        let input = InputSpan::new("fn<len N; T>([T; N], fn<T>(T) -> Bool) -> Bool");
        let (_, fn_type) = <FnTypeAst>::parse(input).unwrap();
        let err = FnType::try_from(fn_type).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 24);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::DuplicateTypeParam { name, previous }
                if name == "T" && previous.location_offset() == 10
        );
    }

    #[test]
    fn converting_raw_fn_type_duplicate_const() {
        let input = InputSpan::new("fn<len N, N; T>([T; N], fn(T) -> Bool) -> Bool");
        let (_, fn_type) = <FnTypeAst>::parse(input).unwrap();
        let err = FnType::try_from(fn_type).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 10);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::DuplicateConst { name, previous }
                if name == "N" && previous.location_offset() == 7
        );
    }

    #[test]
    fn converting_raw_fn_type_duplicate_const_in_embedded_fn() {
        let input = InputSpan::new("fn<len N; T>([T; N], fn<len N>(T) -> Bool) -> Bool");
        let (_, fn_type) = <FnTypeAst>::parse(input).unwrap();
        let err = FnType::try_from(fn_type).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 28);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::DuplicateConst { name, previous }
                if name == "N" && previous.location_offset() == 7
        );
    }

    #[test]
    fn converting_raw_fn_type_undefined_type() {
        let input = InputSpan::new("fn<len N>([T; N], fn(T) -> Bool) -> Bool");
        let (_, fn_type) = <FnTypeAst>::parse(input).unwrap();
        let err = FnType::try_from(fn_type).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 11);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::UndefinedTypeParam(name) if name == "T"
        );
    }

    #[test]
    fn converting_raw_fn_type_undefined_const() {
        let input = InputSpan::new("fn<T>([T; N], fn(T) -> Bool) -> Bool");
        let (_, fn_type) = <FnTypeAst>::parse(input).unwrap();
        let err = FnType::try_from(fn_type).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 10);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::UndefinedConst(name) if name == "N"
        );
    }

    #[test]
    fn converting_raw_fn_type_invalid_constraint() {
        let input = InputSpan::new("fn<T: Bug>([T; _]) -> Bool");
        let (_, fn_type) = <FnTypeAst>::parse(input).unwrap();
        let err = FnType::try_from(fn_type).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 6);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::InvalidConstraint(name) if name == "Bug"
        );
    }

    #[test]
    fn parsing_basic_value_types() {
        let num_type: ValueType = "Num".parse().unwrap();
        assert_eq!(num_type, ValueType::NUM);

        let bool_type: ValueType = "Bool".parse().unwrap();
        assert_eq!(bool_type, ValueType::BOOL);

        let tuple_type: ValueType = "(Num, (Bool, _))".parse().unwrap();
        assert_eq!(
            tuple_type,
            ValueType::from((
                ValueType::NUM,
                ValueType::Tuple(vec![ValueType::BOOL, ValueType::Some].into()),
            ))
        );

        let slice_type: ValueType = "[(Num, _); _]".parse().unwrap();
        let slice_type = match &slice_type {
            ValueType::Tuple(tuple) => tuple.as_slice().unwrap(),
            _ => panic!("Unexpected type: {:?}", slice_type),
        };
        assert_eq!(
            *slice_type.element(),
            ValueType::from((ValueType::NUM, ValueType::Some))
        );
        assert_matches!(slice_type.len(), TupleLength::Some { is_dynamic: false });
    }

    #[test]
    fn parsing_functional_value_type() {
        let ty: ValueType = "fn<len N; T, U>([T; N], fn(T) -> U) -> U".parse().unwrap();
        let ty = match ty {
            ValueType::Function(fn_type) => *fn_type,
            _ => panic!("Unexpected type: {:?}", ty),
        };
        assert_eq!(ty.len_params.len(), 1);
        assert_eq!(ty.type_params.len(), 2);
        assert_eq!(ty.return_type, ValueType::Param(1));
    }

    #[test]
    fn parsing_incomplete_value_type() {
        const INCOMPLETE_TYPES: &[&str] = &[
            "fn<",
            "fn<co",
            "fn<len N;",
            "fn<len N; T",
            "fn<len N; T,",
            "fn<len N; T, U>",
            "fn<len N; T, U>(",
            "fn<len N; T, U>([T; ",
            "fn<len N; T, U>([T; N], fn(",
            "fn<len N; T, U>([T; N], fn(T)",
            "fn<len N; T, U>([T; N], fn(T)",
            "fn<len N; T, U>([T; N], fn(T)) -",
            "fn<len N; T, U>([T; N], fn(T)) ->",
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
