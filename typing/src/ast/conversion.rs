//! Logic for converting `*Ast` types into their "main" counterparts.

use nom::Err as NomErr;

use std::{
    collections::{HashMap, HashSet},
    convert::TryFrom,
    fmt,
    str::FromStr,
};

use crate::ast::TypeConstraintsAst;
use crate::{
    ast::{ConstraintsAst, FnTypeAst, SliceAst, TupleAst, TupleLenAst, ValueTypeAst},
    error::{TypeError, TypeErrors},
    types::{ParamConstraints, ParamQuantifier},
    FnType, PrimitiveType, Slice, Tuple, TypeEnvironment, UnknownLen, ValueType,
};
use arithmetic_parser::{ErrorKind as ParseErrorKind, InputSpan, NomResult, SpannedError};

/// Kinds of errors that can occur when converting `*Ast` types into their "main" counterparts.
///
/// During parsing, errors of this type are wrapped into the [`Type`](ParseErrorKind::Type)
/// variant of parsing errors.
///
/// # Examples
///
/// ```
/// use arithmetic_parser::{grammars::{Parse, NumGrammar, Typed}, ErrorKind};
/// use arithmetic_typing::{ast::ConversionError, Annotated};
/// # use assert_matches::assert_matches;
///
/// type Parser = Typed<Annotated<NumGrammar<f32>>>;
///
/// let code = "bogus_slice: ['T; _] = (1, 2, 3);";
/// let err = Parser::parse_statements(code).unwrap_err();
///
/// assert_eq!(*err.span().fragment(), "T");
/// let err = match err.kind() {
///     ErrorKind::Type(type_err) => type_err
///         .downcast_ref::<ConversionError>()
///         .unwrap(),
///     _ => unreachable!(),
/// };
/// assert_matches!(
///     err,
///     ConversionError::FreeTypeVar(t) if t == "T"
/// );
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ConversionError {
    /// Embedded param quantifiers.
    EmbeddedQuantifier,
    /// Length param not scoped by a function.
    FreeLengthVar(String),
    /// Type param not scoped by a function.
    FreeTypeVar(String),
    /// Unused length param.
    UnusedLength(String),
    /// Unused length param.
    UnusedTypeParam(String),
    /// Unknown type name.
    UnknownType(String),
    /// Unknown constraint.
    UnknownConstraint(String),
}

impl fmt::Display for ConversionError {
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
            Self::UnknownType(name) => {
                write!(formatter, "Unknown type `{}`", name)
            }
            Self::UnknownConstraint(name) => {
                write!(formatter, "Unknown constraint `{}`", name)
            }
        }
    }
}

/// Intermediate conversion state.
#[derive(Debug)]
pub(crate) struct ConversionState<'r, 'a, Prim: PrimitiveType> {
    env: &'r mut TypeEnvironment<Prim>,
    errors: &'r mut TypeErrors<'a, Prim>,
    len_params: HashMap<&'a str, usize>,
    type_params: HashMap<&'a str, usize>,
    is_in_function: bool,
}

impl<'r, 'a, Prim: PrimitiveType> ConversionState<'r, 'a, Prim> {
    pub fn new(env: &'r mut TypeEnvironment<Prim>, errors: &'r mut TypeErrors<'a, Prim>) -> Self {
        Self {
            env,
            errors,
            len_params: HashMap::new(),
            type_params: HashMap::new(),
            is_in_function: false,
        }
    }

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

impl<'a> TypeConstraintsAst<'a> {
    fn convert<Prim: PrimitiveType>(
        &self,
        state: &mut ConversionState<'_, 'a, Prim>,
    ) -> Prim::Constraints {
        self.terms
            .iter()
            .fold(Prim::Constraints::default(), |mut acc, &input| {
                let input_str = *input.fragment();
                let partial = Prim::Constraints::from_str(input_str)
                    .map_err(|_| {
                        let err = ConversionError::UnknownConstraint(input_str.to_owned());
                        state.errors.push(TypeError::conversion(err, input));
                    })
                    .unwrap_or_default();
                acc |= &partial;
                acc
            })
    }
}

impl<'a> ConstraintsAst<'a> {
    fn convert<Prim: PrimitiveType>(
        &self,
        state: &mut ConversionState<'_, 'a, Prim>,
    ) -> ParamConstraints<Prim> {
        let mut static_lengths = HashSet::with_capacity(self.static_lengths.len());
        for dyn_length in &self.static_lengths {
            let name = *dyn_length.fragment();
            if let Some(index) = state.len_params.get(name) {
                static_lengths.insert(*index);
            } else {
                let err = ConversionError::UnusedLength(name.to_owned());
                state.errors.push(TypeError::conversion(err, *dyn_length));
            }
        }

        let mut type_params = HashMap::with_capacity(self.type_params.len());
        for (param, constraints) in &self.type_params {
            let name = *param.fragment();
            if let Some(index) = state.type_params.get(name) {
                type_params.insert(*index, constraints.convert(state));
            } else {
                let err = ConversionError::UnusedTypeParam(name.to_owned());
                state.errors.push(TypeError::conversion(err, *param));
            }
        }

        ParamConstraints {
            static_lengths,
            type_params,
        }
    }
}

impl<'a> TupleAst<'a> {
    fn convert<Prim: PrimitiveType>(
        &self,
        state: &mut ConversionState<'_, 'a, Prim>,
    ) -> Tuple<Prim> {
        let start = self
            .start
            .iter()
            .map(|element| element.convert(state))
            .collect();
        let middle = self.middle.as_ref().map(|middle| middle.convert(state));
        let end = self
            .end
            .iter()
            .map(|element| element.convert(state))
            .collect();
        Tuple::from_parts(start, middle, end)
    }
}

impl<'a> SliceAst<'a> {
    fn convert<Prim: PrimitiveType>(
        &self,
        state: &mut ConversionState<'_, 'a, Prim>,
    ) -> Slice<Prim> {
        let element = self.element.convert(state);

        let converted_length = match &self.length {
            TupleLenAst::Ident(ident) => {
                let name = *ident.fragment();
                if state.is_in_function {
                    let const_param = state.len_param_idx(name);
                    UnknownLen::param(const_param)
                } else {
                    let err = ConversionError::FreeLengthVar(name.to_owned());
                    state.errors.push(TypeError::conversion(err, *ident));
                    UnknownLen::Some
                }
            }
            TupleLenAst::Some => UnknownLen::Some,
            TupleLenAst::Dynamic => UnknownLen::Dynamic,
        };

        Slice::new(element, converted_length)
    }
}

impl<'a> ValueTypeAst<'a> {
    pub(crate) fn convert<Prim: PrimitiveType>(
        &self,
        state: &mut ConversionState<'_, 'a, Prim>,
    ) -> ValueType<Prim> {
        match self {
            Self::Some => ValueType::Some,
            Self::Any(constraints) => ValueType::Any(constraints.convert(state)),
            Self::Ident(prim) => {
                let ident = *prim.fragment();
                if let Ok(prim_type) = Prim::from_str(ident) {
                    ValueType::Prim(prim_type)
                } else {
                    let err = ConversionError::UnknownType(ident.to_owned());
                    state.errors.push(TypeError::conversion(err, *prim));
                    ValueType::Some
                }
            }

            Self::Param(ident) => {
                let name = *ident.fragment();
                if state.is_in_function {
                    let idx = state.type_param_idx(name);
                    ValueType::param(idx)
                } else {
                    let err = ConversionError::FreeTypeVar(name.to_owned());
                    state.errors.push(TypeError::conversion(err, *ident));
                    ValueType::Some
                }
            }

            Self::Function {
                constraints,
                function,
            } => {
                if state.is_in_function {
                    if let Some(constraints) = constraints {
                        let err = ConversionError::EmbeddedQuantifier;
                        state
                            .errors
                            .push(TypeError::conversion(err, constraints.for_keyword));
                    }
                    function.convert(state).into()
                } else {
                    state.is_in_function = true;
                    let mut converted_fn = function.convert(state);
                    state.is_in_function = false;

                    let constraints = constraints
                        .as_ref()
                        .map_or_else(ParamConstraints::default, |c| c.convert(state));
                    ParamQuantifier::set_params(&mut converted_fn, constraints);
                    converted_fn.into()
                }
            }

            Self::Tuple(tuple) => tuple.convert(state).into(),
            Self::Slice(slice) => slice.convert(state).into(),
        }
    }

    /// Tries to convert this type into [`ValueType`].
    pub fn try_convert<Prim>(&self) -> Result<ValueType<Prim>, TypeErrors<'a, Prim>>
    where
        Prim: PrimitiveType,
    {
        let mut env = TypeEnvironment::new();
        let mut errors = TypeErrors::new();
        let mut state = ConversionState::new(&mut env, &mut errors);

        let output = self.convert(&mut state);
        if errors.is_empty() {
            Ok(output)
        } else {
            Err(errors)
        }
    }
}

impl<'a> FnTypeAst<'a> {
    fn convert<Prim: PrimitiveType>(
        &self,
        state: &mut ConversionState<'_, 'a, Prim>,
    ) -> FnType<Prim> {
        let args = self.args.convert(state);
        let return_type = self.return_type.convert(state);
        FnType::new(args, return_type)
    }

    /// Tries to convert this type into [`ValueType`].
    pub fn try_convert<Prim>(&self) -> Result<FnType<Prim>, TypeErrors<'a, Prim>>
    where
        Prim: PrimitiveType,
    {
        let mut env = TypeEnvironment::new();
        let mut errors = TypeErrors::new();
        let mut state = ConversionState::new(&mut env, &mut errors);
        state.is_in_function = true;

        let output = self.convert(&mut state);
        if errors.is_empty() {
            Ok(output)
        } else {
            Err(errors)
        }
    }
}

/// Shared parsing code for `ValueTypeAst` and `FnTypeAst`.
fn parse_inner<'a, Ast>(
    parser: fn(InputSpan<'a>) -> NomResult<'a, Ast>,
    input: InputSpan<'a>,
) -> NomResult<'a, Ast> {
    let (rest, ast) = parser(input)?;
    if !rest.fragment().is_empty() {
        let err = ParseErrorKind::Leftovers.with_span(&rest.into());
        return Err(NomErr::Failure(err));
    }
    Ok((rest, ast))
}

/// Shared `TryFrom<&str>` logic for `ValueTypeAst` and `FnTypeAst`.
fn from_str<'a, Ast>(
    parser: fn(InputSpan<'a>) -> NomResult<'a, Ast>,
    def: &'a str,
) -> Result<Ast, SpannedError<&'a str>> {
    let input = InputSpan::new(def);
    let (_, ast) = parse_inner(parser, input).map_err(|err| match err {
        NomErr::Incomplete(_) => ParseErrorKind::Incomplete.with_span(&input.into()),
        NomErr::Error(e) | NomErr::Failure(e) => e,
    })?;
    Ok(ast)
}

impl<'a> TryFrom<&'a str> for ValueTypeAst<'a> {
    type Error = SpannedError<&'a str>;

    fn try_from(def: &'a str) -> Result<Self, Self::Error> {
        from_str(ValueTypeAst::parse, def)
    }
}

impl<'a> TryFrom<&'a str> for FnTypeAst<'a> {
    type Error = SpannedError<&'a str>;

    fn try_from(def: &'a str) -> Result<Self, Self::Error> {
        from_str(FnTypeAst::parse, def)
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;

    use super::*;
    use crate::{error::TypeErrorKind, Num, TupleLen};

    #[test]
    fn converting_raw_fn_type() {
        let input = InputSpan::new("(['T; N], ('T) -> Bool) -> Bool");
        let (_, fn_type) = FnTypeAst::parse(input).unwrap();
        let fn_type = fn_type.try_convert::<Num>().unwrap();

        assert_eq!(fn_type.to_string(), *input.fragment());
    }

    #[test]
    fn converting_fn_type_with_constraint() {
        let input = InputSpan::new("for<'T: Lin> (['T; N], ('T) -> Bool) -> Bool");
        let (_, ast) = ValueTypeAst::parse(input).unwrap();
        let fn_type = ast.try_convert::<Num>().unwrap();

        assert_eq!(fn_type.to_string(), *input.fragment());
    }

    #[test]
    fn converting_fn_type_unused_type() {
        let input = InputSpan::new("for<'T: Lin> (Num) -> Bool");
        let (_, ast) = ValueTypeAst::parse(input).unwrap();
        let err = ast.try_convert::<Num>().unwrap_err().single();

        assert_eq!(err.span().location_offset(), 5);
        assert_matches!(
            err.kind(),
            TypeErrorKind::Conversion(ConversionError::UnusedTypeParam(name)) if name == "T"
        );
    }

    #[test]
    fn converting_fn_type_unused_length() {
        let input = InputSpan::new("for<len! N> (Num) -> Bool");
        let (_, ast) = <ValueTypeAst>::parse(input).unwrap();
        let err = ast.try_convert::<Num>().unwrap_err().single();

        assert_eq!(err.span().location_offset(), 9);
        assert_matches!(
            err.kind(),
            TypeErrorKind::Conversion(ConversionError::UnusedLength(name)) if name == "N"
        );
    }

    #[test]
    fn converting_fn_type_free_type_param() {
        let input = InputSpan::new("(Num, 'T)");
        let (_, ast) = <ValueTypeAst>::parse(input).unwrap();
        let err = ast.try_convert::<Num>().unwrap_err().single();

        assert_eq!(err.span().location_offset(), 7);
        assert_matches!(
            err.kind(),
            TypeErrorKind::Conversion(ConversionError::FreeTypeVar(name)) if name == "T"
        );
    }

    #[test]
    fn converting_fn_type_free_length() {
        let input = InputSpan::new("[Num; N]");
        let (_, ast) = ValueTypeAst::parse(input).unwrap();
        let err = ast.try_convert::<Num>().unwrap_err().single();

        assert_eq!(err.span().location_offset(), 6);
        assert_matches!(
            err.kind(),
            TypeErrorKind::Conversion(ConversionError::FreeLengthVar(name)) if name == "N"
        );
    }

    #[test]
    fn converting_fn_type_invalid_constraint() {
        let input = InputSpan::new("for<'T: Bug> (['T; _]) -> Bool");
        let (_, ast) = ValueTypeAst::parse(input).unwrap();
        let err = ast.try_convert::<Num>().unwrap_err().single();

        assert_eq!(*err.span().fragment(), "Bug");
        assert_matches!(
            err.kind(),
            TypeErrorKind::Conversion(ConversionError::UnknownConstraint(id))
                if id == "Bug"
        );
    }

    #[test]
    fn embedded_type_with_constraints() {
        let input = InputSpan::new("('T, for<'U: Lin> ('U) -> 'U) -> ()");
        let (_, ast) = <ValueTypeAst>::parse(input).unwrap();
        let err = ast.try_convert::<Num>().unwrap_err().single();

        assert_eq!(*err.span().fragment(), "for");
        assert_eq!(err.span().location_offset(), 5);
        assert_matches!(
            err.kind(),
            TypeErrorKind::Conversion(ConversionError::EmbeddedQuantifier)
        );
    }

    #[test]
    fn parsing_basic_value_types() -> anyhow::Result<()> {
        let num_type = ValueTypeAst::try_from("Num")?.try_convert::<Num>()?;
        assert_eq!(num_type, ValueType::NUM);

        let bool_type = ValueTypeAst::try_from("Bool")?.try_convert::<Num>()?;
        assert_eq!(bool_type, ValueType::BOOL);

        let tuple_type = ValueTypeAst::try_from("(Num, (Bool, _))")?.try_convert::<Num>()?;
        assert_eq!(
            tuple_type,
            ValueType::from((
                ValueType::NUM,
                ValueType::Tuple(vec![ValueType::BOOL, ValueType::Some].into()),
            ))
        );

        let slice_type = ValueTypeAst::try_from("[(Num, _); _]")?.try_convert::<Num>()?;
        let slice_type = match &slice_type {
            ValueType::Tuple(tuple) => tuple.as_slice().unwrap(),
            _ => panic!("Unexpected type: {:?}", slice_type),
        };

        assert_eq!(
            *slice_type.element(),
            ValueType::from((ValueType::NUM, ValueType::Some))
        );
        assert_eq!(slice_type.len(), TupleLen::from(UnknownLen::Some));
        Ok(())
    }

    #[test]
    fn parsing_functional_value_type() -> anyhow::Result<()> {
        let ty = ValueTypeAst::try_from("(['T; N], ('T) -> 'U) -> 'U")?.try_convert::<Num>()?;
        let ty = match ty {
            ValueType::Function(fn_type) => *fn_type,
            _ => panic!("Unexpected type: {:?}", ty),
        };

        assert_eq!(ty.params.as_ref().unwrap().len_params.len(), 1);
        assert_eq!(ty.params.as_ref().unwrap().type_params.len(), 2);
        assert_eq!(ty.return_type, ValueType::param(1));
        Ok(())
    }

    #[test]
    fn parsing_functional_type_with_varargs() -> anyhow::Result<()> {
        let ty: ValueType = ValueTypeAst::try_from("(...[Num; N]) -> Num")?.try_convert::<Num>()?;
        let ty = match ty {
            ValueType::Function(fn_type) => *fn_type,
            _ => panic!("Unexpected type: {:?}", ty),
        };

        assert_eq!(ty.params.as_ref().unwrap().len_params.len(), 1);
        assert!(ty.params.as_ref().unwrap().type_params.is_empty());
        let args_slice = ty.args.as_slice().unwrap();
        assert_eq!(*args_slice.element(), ValueType::NUM);
        assert_eq!(args_slice.len(), UnknownLen::param(0).into());
        Ok(())
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
            ValueTypeAst::try_from(input).unwrap_err();
        }
    }
}
