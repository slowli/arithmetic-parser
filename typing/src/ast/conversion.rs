//! Logic for converting `*Ast` types into their "main" counterparts.

use nom::Err as NomErr;

use std::{
    collections::{HashMap, HashSet},
    convert::TryFrom,
    fmt,
    str::FromStr,
};

use crate::{
    ast::{ConstraintsAst, FnTypeAst, SliceAst, TupleAst, TupleLenAst, ValueTypeAst},
    types::{ParamConstraints, ParamQuantifier},
    FnType, PrimitiveType, Slice, Tuple, UnknownLen, ValueType,
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
    /// Embedded param quantifiers.
    EmbeddedQuantifier,
    /// Length param not scoped by a function.
    FreeLengthVar(String),
    /// Type param not scoped by a function.
    FreeTypeVar(String),
    /// Undefined const param.
    UnusedLength(String),
    /// Undefined type param.
    UnusedTypeParam(String),
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
            Self::EmbeddedQuantifier => {
                formatter.write_str("`for` quantifier within the scope of another `for` quantifier")
            }
            Self::FreeLengthVar(name) => {
                write!(
                    formatter,
                    "Length param `{}` is not scoped by function definition",
                    name
                )
            }
            Self::FreeTypeVar(name) => {
                write!(
                    formatter,
                    "Type param `{}` is not scoped by function definition",
                    name
                )
            }
            Self::UnusedLength(name) => {
                write!(formatter, "Unused length param `{}`", name)
            }
            Self::UnusedTypeParam(name) => {
                write!(formatter, "Unused type param `{}`", name)
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
    len_params: HashMap<&'a str, usize>,
    type_params: HashMap<&'a str, usize>,
}

impl<'a> ConversionState<'a> {
    fn type_param_idx(&mut self, param_name: &'a str) -> usize {
        let type_param_count = self.type_params.len();
        *self
            .type_params
            .entry(param_name)
            .or_insert(type_param_count)
    }

    fn len_param_idx(&mut self, param_name: &'a str) -> usize {
        let len_param_count = self.len_params.len();
        *self.len_params.entry(param_name).or_insert(len_param_count)
    }
}

impl<'a, Prim: PrimitiveType> ConstraintsAst<'a, Prim> {
    fn try_convert(
        &self,
        state: &ConversionState<'a>,
    ) -> Result<ParamConstraints<Prim>, ConversionError<&'a str>> {
        let mut dyn_lengths = HashSet::with_capacity(self.dyn_lengths.len());
        for dyn_length in &self.dyn_lengths {
            let name = *dyn_length.fragment();
            if let Some(index) = state.len_params.get(name) {
                dyn_lengths.insert(*index);
            } else {
                let err = ConversionErrorKind::UnusedLength(name.to_owned()).with_span(*dyn_length);
                return Err(err);
            }
        }

        let mut type_params = HashMap::with_capacity(self.type_params.len());
        for (param, constraints) in &self.type_params {
            let name = *param.fragment();
            if let Some(index) = state.type_params.get(name) {
                type_params.insert(*index, constraints.computed.clone());
            } else {
                let err = ConversionErrorKind::UnusedTypeParam(name.to_owned()).with_span(*param);
                return Err(err);
            }
        }

        Ok(ParamConstraints {
            dyn_lengths,
            type_params,
        })
    }
}

impl<'a, Prim: PrimitiveType> TupleAst<'a, Prim> {
    fn try_convert(
        &self,
        mut state: Option<&mut ConversionState<'a>>,
    ) -> Result<Tuple<Prim>, ConversionError<&'a str>> {
        let start = self
            .start
            .iter()
            .map(|element| element.try_convert(state.as_deref_mut()))
            .collect::<Result<Vec<_>, _>>()?;
        let middle = self
            .middle
            .as_ref()
            .map(|middle| middle.try_convert(state.as_deref_mut()))
            .transpose()?;
        let end = self
            .end
            .iter()
            .map(|element| element.try_convert(state.as_deref_mut()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Tuple::from_parts(start, middle, end))
    }
}

impl<'a, Prim: PrimitiveType> SliceAst<'a, Prim> {
    fn try_convert(
        &self,
        mut state: Option<&mut ConversionState<'a>>,
    ) -> Result<Slice<Prim>, ConversionError<&'a str>> {
        let element = self.element.try_convert(state.as_deref_mut())?;
        let converted_length = match &self.length {
            TupleLenAst::Ident(ident) => {
                let name = *ident.fragment();
                let const_param = if let Some(state) = state {
                    state.len_param_idx(name)
                } else {
                    let err = ConversionErrorKind::FreeLengthVar(name.to_owned()).with_span(*ident);
                    return Err(err);
                };
                UnknownLen::param(const_param)
            }
            TupleLenAst::Some => UnknownLen::Some,
            TupleLenAst::Dynamic => UnknownLen::Dynamic,
        };

        Ok(Slice::new(element, converted_length))
    }
}

impl<'a, Prim: PrimitiveType> ValueTypeAst<'a, Prim> {
    fn try_convert(
        &self,
        state: Option<&mut ConversionState<'a>>,
    ) -> Result<ValueType<Prim>, ConversionError<&'a str>> {
        Ok(match self {
            Self::Any => ValueType::Some,
            Self::Prim(prim) => ValueType::Prim(prim.to_owned()),

            Self::Param(ident) => {
                let name = *ident.fragment();
                let idx = if let Some(state) = state {
                    state.type_param_idx(name)
                } else {
                    let err = ConversionErrorKind::FreeTypeVar(name.to_owned()).with_span(*ident);
                    return Err(err);
                };
                ValueType::param(idx)
            }

            Self::Function {
                constraints,
                function,
            } => {
                if let Some(state) = state {
                    if let Some(constraints) = constraints {
                        let err = ConversionErrorKind::EmbeddedQuantifier
                            .with_span(constraints.for_keyword);
                        return Err(err);
                    }
                    function.try_convert(state)?.into()
                } else {
                    let mut state = ConversionState::default();
                    let mut converted_fn = function.try_convert(&mut state)?;

                    let constraints = if let Some(constraints) = constraints {
                        constraints.try_convert(&state)?
                    } else {
                        ParamConstraints::default()
                    };
                    ParamQuantifier::set_params(&mut converted_fn, constraints);
                    converted_fn.into()
                }
            }

            Self::Tuple(tuple) => tuple.try_convert(state)?.into(),
            Self::Slice(slice) => slice.try_convert(state)?.into(),
        })
    }
}

impl<'a, Prim: PrimitiveType> TryFrom<ValueTypeAst<'a, Prim>> for ValueType<Prim> {
    type Error = ConversionError<&'a str>;

    fn try_from(value: ValueTypeAst<'a, Prim>) -> Result<Self, Self::Error> {
        value.try_convert(None)
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
        state: &mut ConversionState<'a>,
    ) -> Result<FnType<Prim>, ConversionError<&'a str>> {
        let args = self.args.try_convert(Some(state))?;
        let return_type = self.return_type.try_convert(Some(state))?;
        Ok(FnType::new(args, return_type))
    }
}

impl<'a, Prim: PrimitiveType> TryFrom<FnTypeAst<'a, Prim>> for FnType<Prim> {
    type Error = ConversionError<&'a str>;

    fn try_from(value: FnTypeAst<'a, Prim>) -> Result<Self, Self::Error> {
        value.try_convert(&mut ConversionState::default())
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
    use crate::TupleLen;

    #[test]
    fn converting_raw_fn_type() {
        let input = InputSpan::new("fn(['T; N], fn('T) -> Bool) -> Bool");
        let (_, fn_type) = <FnTypeAst>::parse(input).unwrap();
        let fn_type = FnType::try_from(fn_type).unwrap();

        assert_eq!(fn_type.to_string(), *input.fragment());
    }

    #[test]
    fn converting_fn_type_with_constraint() {
        let input = InputSpan::new("for<'T: Lin> fn(['T; N], fn('T) -> Bool) -> Bool");
        let (_, ast) = <ValueTypeAst>::parse(input).unwrap();
        let fn_type = ValueType::try_from(ast).unwrap();

        assert_eq!(
            fn_type.to_string(),
            "fn<len N; 'T: Lin>(['T; N], fn('T) -> Bool) -> Bool"
        );
    }

    #[test]
    fn converting_fn_type_unused_type() {
        let input = InputSpan::new("for<'T: Lin> fn(Num) -> Bool");
        let (_, ast) = <ValueTypeAst>::parse(input).unwrap();
        let err = ValueType::try_from(ast).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 5);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::UnusedTypeParam(name) if name == "T"
        );
    }

    #[test]
    fn converting_fn_type_unused_length() {
        let input = InputSpan::new("for<len N*> fn(Num) -> Bool");
        let (_, ast) = <ValueTypeAst>::parse(input).unwrap();
        let err = ValueType::try_from(ast).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 8);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::UnusedLength(name) if name == "N"
        );
    }

    #[test]
    fn converting_fn_type_free_type_param() {
        let input = InputSpan::new("(Num, 'T)");
        let (_, ast) = <ValueTypeAst>::parse(input).unwrap();
        let err = ValueType::try_from(ast).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 7);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::FreeTypeVar(name) if name == "T"
        );
    }

    #[test]
    fn converting_fn_type_free_length() {
        let input = InputSpan::new("[Num; N]");
        let (_, ast) = <ValueTypeAst>::parse(input).unwrap();
        let err = ValueType::try_from(ast).unwrap_err();

        assert_eq!(err.main_span().location_offset(), 6);
        assert_matches!(
            err.kind(),
            ConversionErrorKind::FreeLengthVar(name) if name == "N"
        );
    }

    #[test]
    fn converting_fn_type_invalid_constraint() {
        let input = InputSpan::new("for<'T: Bug> fn(['T; _]) -> Bool");
        let err = match <ValueTypeAst>::parse(input).unwrap_err() {
            NomErr::Failure(err) => err,
            other => panic!("Unexpected error type: {:?}", other),
        };

        assert_eq!(*err.span().fragment(), "Bug");
        assert_matches!(
            err.kind(),
            ParseErrorKind::Type(err) if err.to_string() == "Cannot parse type constraint"
        );
    }

    #[test]
    fn embedded_type_with_constraints() {
        let input = InputSpan::new("fn('T, for<'U: Lin> fn('U) -> 'U)");
        let (_, ast) = <ValueTypeAst>::parse(input).unwrap();
        let err = ValueType::try_from(ast).unwrap_err();

        assert_eq!(*err.main_span().fragment(), "for");
        assert_eq!(err.main_span().location_offset(), 7);
        assert_matches!(err.kind(), ConversionErrorKind::EmbeddedQuantifier);
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
        assert_eq!(slice_type.len(), TupleLen::from(UnknownLen::Some));
    }

    #[test]
    fn parsing_functional_value_type() {
        let ty: ValueType = "fn(['T; N], fn('T) -> 'U) -> 'U".parse().unwrap();
        let ty = match ty {
            ValueType::Function(fn_type) => *fn_type,
            _ => panic!("Unexpected type: {:?}", ty),
        };

        assert_eq!(ty.params.as_ref().unwrap().len_params.len(), 1);
        assert_eq!(ty.params.as_ref().unwrap().type_params.len(), 2);
        assert_eq!(ty.return_type, ValueType::param(1));
    }

    #[test]
    fn parsing_functional_type_with_varargs() {
        let ty: ValueType = "fn(...[Num; N]) -> Num".parse().unwrap();
        let ty = match ty {
            ValueType::Function(fn_type) => *fn_type,
            _ => panic!("Unexpected type: {:?}", ty),
        };

        assert_eq!(ty.params.as_ref().unwrap().len_params.len(), 1);
        assert!(ty.params.as_ref().unwrap().type_params.is_empty());
        let args_slice = ty.args.as_slice().unwrap();
        assert_eq!(*args_slice.element(), ValueType::NUM);
        assert_eq!(args_slice.len(), UnknownLen::param(0).into());
    }

    #[test]
    fn parsing_incomplete_value_type() {
        const INCOMPLETE_TYPES: &[&str] = &[
            "fn(",
            "fn(['T; ",
            "fn(['T; N], fn(",
            "fn(['T; N], fn('T)",
            "fn(['T; N], fn('T)) -",
            "fn(['T; N], fn('T)) ->",
        ];

        for &input in INCOMPLETE_TYPES {
            // TODO: some of reported errors are difficult to interpret; should clarify.
            input.parse::<ValueType>().unwrap_err();
        }
    }

    #[test]
    fn parsing_value_type_with_conversion_error() {
        let input = "['T; _]";
        let err = input.parse::<ValueType>().unwrap_err();
        assert_eq!(err.span().location_offset(), 2);
        let err = match err.kind() {
            ParseErrorKind::Type(err) => err.downcast_ref::<ConversionErrorKind>().unwrap(),
            _ => panic!("Unexpected error type: {:?}", err),
        };
        assert_matches!(err, ConversionErrorKind::FreeTypeVar(name) if name == "T");
    }
}
