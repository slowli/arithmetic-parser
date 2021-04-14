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
    ast::{ConstraintsAst, FnTypeAst, SliceAst, SpannedTypeAst, TupleAst, TupleLenAst, TypeAst},
    error::{Error, Errors},
    types::{ParamConstraints, ParamQuantifier},
    FnType, PrimitiveType, Slice, Tuple, Type, TypeEnvironment, UnknownLen,
};
use arithmetic_parser::{ErrorKind as ParseErrorKind, InputSpan, NomResult, Spanned, SpannedError};

/// Kinds of errors that can occur when converting `*Ast` types into their "main" counterparts.
///
/// During type inference, errors of this type are wrapped into the [`AstConversion`]
/// variant of typing errors.
///
/// [`AstConversion`]: crate::ErrorKind::AstConversion
///
/// # Examples
///
/// ```
/// use arithmetic_parser::grammars::{Parse, NumGrammar, Typed};
/// use arithmetic_typing::{
///     ast::AstConversionError, ErrorKind, Annotated, TypeEnvironment,
/// };
/// # use assert_matches::assert_matches;
///
/// type Parser = Typed<Annotated<NumGrammar<f32>>>;
///
/// # fn main() -> anyhow::Result<()> {
/// let code = "bogus_slice: ['T; _] = (1, 2, 3);";
/// let code = Parser::parse_statements(code)?;
///
/// let errors = TypeEnvironment::new().process_statements(&code).unwrap_err();
/// let err = errors.into_iter().next().unwrap();
/// assert_eq!(*err.span().fragment(), "'T");
/// assert_matches!(
///     err.kind(),
///     ErrorKind::AstConversion(AstConversionError::FreeTypeVar(id))
///         if id == "T"
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AstConversionError {
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

impl fmt::Display for AstConversionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmbeddedQuantifier => {
                formatter.write_str("`for` quantifier for a function that is not top-level")
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

impl std::error::Error for AstConversionError {}

/// Intermediate conversion state.
#[derive(Debug)]
pub(crate) struct AstConversionState<'r, 'a, Prim: PrimitiveType> {
    env: &'r mut TypeEnvironment<Prim>,
    errors: &'r mut Errors<'a, Prim>,
    len_params: HashMap<&'a str, usize>,
    type_params: HashMap<&'a str, usize>,
    is_in_function: bool,
}

impl<'r, 'a, Prim: PrimitiveType> AstConversionState<'r, 'a, Prim> {
    pub fn new(env: &'r mut TypeEnvironment<Prim>, errors: &'r mut Errors<'a, Prim>) -> Self {
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

    fn new_type(&mut self) -> Type<Prim> {
        self.env.substitutions.new_type_var()
    }

    fn new_len(&mut self) -> UnknownLen {
        self.env.substitutions.new_len_var()
    }

    pub(crate) fn convert_type(&mut self, ty: &SpannedTypeAst<'a>) -> Type<Prim> {
        match &ty.extra {
            TypeAst::Some => self.new_type(),
            TypeAst::Any(constraints) => Type::Any(constraints.convert(self)),
            TypeAst::Ident => {
                let ident = *ty.fragment();
                if let Ok(prim_type) = Prim::from_str(ident) {
                    Type::Prim(prim_type)
                } else {
                    let err = AstConversionError::UnknownType(ident.to_owned());
                    self.errors.push(Error::conversion(err, ty));
                    self.new_type()
                }
            }

            TypeAst::Param => {
                let name = &ty.fragment()[1..];
                if self.is_in_function {
                    let idx = self.type_param_idx(name);
                    Type::param(idx)
                } else {
                    let err = AstConversionError::FreeTypeVar(name.to_owned());
                    self.errors.push(Error::conversion(err, ty));
                    self.new_type()
                }
            }

            TypeAst::Function(function) => self.convert_fn(function, None),
            TypeAst::FunctionWithConstraints {
                function,
                constraints,
            } => self.convert_fn(&function.extra, Some(constraints)),

            TypeAst::Tuple(tuple) => tuple.convert(self).into(),
            TypeAst::Slice(slice) => slice.convert(self).into(),
        }
    }

    fn convert_fn(
        &mut self,
        function: &FnTypeAst<'a>,
        constraints: Option<&Spanned<'a, ConstraintsAst<'a>>>,
    ) -> Type<Prim> {
        if self.is_in_function {
            if let Some(constraints) = constraints {
                let err = AstConversionError::EmbeddedQuantifier;
                self.errors.push(Error::conversion(err, constraints));
            }
            function.convert(self).into()
        } else {
            self.is_in_function = true;
            let mut converted_fn = function.convert(self);
            let constraints =
                constraints.map_or_else(ParamConstraints::default, |c| c.extra.convert(self));
            ParamQuantifier::set_params(&mut converted_fn, constraints);

            self.is_in_function = false;
            self.type_params.clear();
            self.len_params.clear();
            converted_fn.into()
        }
    }
}

impl<'a> TypeConstraintsAst<'a> {
    fn convert<Prim: PrimitiveType>(
        &self,
        state: &mut AstConversionState<'_, 'a, Prim>,
    ) -> Prim::Constraints {
        self.terms
            .iter()
            .fold(Prim::Constraints::default(), |mut acc, input| {
                let input_str = *input.fragment();
                let partial = Prim::Constraints::from_str(input_str)
                    .map_err(|_| {
                        let err = AstConversionError::UnknownConstraint(input_str.to_owned());
                        state.errors.push(Error::conversion(err, input));
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
        state: &mut AstConversionState<'_, 'a, Prim>,
    ) -> ParamConstraints<Prim> {
        let mut static_lengths = HashSet::with_capacity(self.static_lengths.len());
        for dyn_length in &self.static_lengths {
            let name = *dyn_length.fragment();
            if let Some(index) = state.len_params.get(name) {
                static_lengths.insert(*index);
            } else {
                let err = AstConversionError::UnusedLength(name.to_owned());
                state.errors.push(Error::conversion(err, dyn_length));
            }
        }

        let mut type_params = HashMap::with_capacity(self.type_params.len());
        for (param, constraints) in &self.type_params {
            let name = *param.fragment();
            if let Some(index) = state.type_params.get(name) {
                type_params.insert(*index, constraints.convert(state));
            } else {
                let err = AstConversionError::UnusedTypeParam(name.to_owned());
                state.errors.push(Error::conversion(err, param));
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
        state: &mut AstConversionState<'_, 'a, Prim>,
    ) -> Tuple<Prim> {
        let start = self
            .start
            .iter()
            .map(|element| state.convert_type(element))
            .collect();
        let middle = self
            .middle
            .as_ref()
            .map(|middle| middle.extra.convert(state));
        let end = self
            .end
            .iter()
            .map(|element| state.convert_type(element))
            .collect();
        Tuple::from_parts(start, middle, end)
    }
}

impl<'a> SliceAst<'a> {
    fn convert<Prim: PrimitiveType>(
        &self,
        state: &mut AstConversionState<'_, 'a, Prim>,
    ) -> Slice<Prim> {
        let element = state.convert_type(&self.element);

        let converted_length = match &self.length {
            TupleLenAst::Ident(ident) => {
                let name = *ident.fragment();
                if state.is_in_function {
                    let const_param = state.len_param_idx(name);
                    UnknownLen::param(const_param)
                } else {
                    let err = AstConversionError::FreeLengthVar(name.to_owned());
                    state.errors.push(Error::conversion(err, ident));
                    state.new_len()
                }
            }
            TupleLenAst::Some => state.new_len(),
            TupleLenAst::Dynamic => UnknownLen::Dynamic,
        };

        Slice::new(element, converted_length)
    }
}

impl<'a> FnTypeAst<'a> {
    fn convert<Prim: PrimitiveType>(
        &self,
        state: &mut AstConversionState<'_, 'a, Prim>,
    ) -> FnType<Prim> {
        let args = self.args.extra.convert(state);
        let return_type = state.convert_type(&self.return_type);
        FnType::new(args, return_type)
    }

    /// Tries to convert this type into [`FnType`].
    pub fn try_convert<Prim>(&self) -> Result<FnType<Prim>, Errors<'a, Prim>>
    where
        Prim: PrimitiveType,
    {
        let mut env = TypeEnvironment::new();
        let mut errors = Errors::new();
        let mut state = AstConversionState::new(&mut env, &mut errors);
        state.is_in_function = true;

        let output = self.convert(&mut state);
        if errors.is_empty() {
            Ok(output)
        } else {
            Err(errors)
        }
    }
}

/// Shared parsing code for `TypeAst` and `FnTypeAst`.
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

/// Shared `TryFrom<&str>` logic for `TypeAst` and `FnTypeAst`.
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

impl<'a> TypeAst<'a> {
    /// Parses type AST from a string.
    pub fn try_from(def: &'a str) -> Result<SpannedTypeAst<'a>, SpannedError<&'a str>> {
        from_str(TypeAst::parse, def)
    }
}

impl<'a, Prim: PrimitiveType> TryFrom<&SpannedTypeAst<'a>> for Type<Prim> {
    type Error = Errors<'a, Prim>;

    fn try_from(ast: &SpannedTypeAst<'a>) -> Result<Self, Self::Error> {
        let mut env = TypeEnvironment::new();
        let mut errors = Errors::new();
        let mut state = AstConversionState::new(&mut env, &mut errors);

        let output = state.convert_type(ast);
        if errors.is_empty() {
            Ok(output)
        } else {
            Err(errors)
        }
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
    use crate::{ErrorKind, Num};

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
        let (_, ast) = TypeAst::parse(input).unwrap();
        let fn_type = <Type>::try_from(&ast).unwrap();

        assert_eq!(fn_type.to_string(), *input.fragment());
    }

    #[test]
    fn converting_fn_type_unused_type() {
        let input = InputSpan::new("for<'T: Lin> (Num) -> Bool");
        let (_, ast) = TypeAst::parse(input).unwrap();
        let err = <Type>::try_from(&ast).unwrap_err().single();

        assert_eq!(err.span().location_offset(), 5);
        assert_matches!(
            err.kind(),
            ErrorKind::AstConversion(AstConversionError::UnusedTypeParam(name)) if name == "T"
        );
    }

    #[test]
    fn converting_fn_type_unused_length() {
        let input = InputSpan::new("for<len! N> (Num) -> Bool");
        let (_, ast) = TypeAst::parse(input).unwrap();
        let err = <Type>::try_from(&ast).unwrap_err().single();

        assert_eq!(err.span().location_offset(), 9);
        assert_matches!(
            err.kind(),
            ErrorKind::AstConversion(AstConversionError::UnusedLength(name)) if name == "N"
        );
    }

    #[test]
    fn converting_fn_type_free_type_param() {
        let input = InputSpan::new("(Num, 'T)");
        let (_, ast) = TypeAst::parse(input).unwrap();
        let err = <Type>::try_from(&ast).unwrap_err().single();

        assert_eq!(err.span().location_offset(), 6);
        assert_matches!(
            err.kind(),
            ErrorKind::AstConversion(AstConversionError::FreeTypeVar(name)) if name == "T"
        );
    }

    #[test]
    fn converting_fn_type_free_length() {
        let input = InputSpan::new("[Num; N]");
        let (_, ast) = TypeAst::parse(input).unwrap();
        let err = <Type>::try_from(&ast).unwrap_err().single();

        assert_eq!(err.span().location_offset(), 6);
        assert_matches!(
            err.kind(),
            ErrorKind::AstConversion(AstConversionError::FreeLengthVar(name)) if name == "N"
        );
    }

    #[test]
    fn converting_fn_type_invalid_constraint() {
        let input = InputSpan::new("for<'T: Bug> (['T; _]) -> Bool");
        let (_, ast) = TypeAst::parse(input).unwrap();
        let err = <Type>::try_from(&ast).unwrap_err().single();

        assert_eq!(*err.span().fragment(), "Bug");
        assert_matches!(
            err.kind(),
            ErrorKind::AstConversion(AstConversionError::UnknownConstraint(id))
                if id == "Bug"
        );
    }

    #[test]
    fn embedded_type_with_constraints() {
        let input = InputSpan::new("('T, for<'U: Lin> ('U) -> 'U) -> ()");
        let (_, ast) = TypeAst::parse(input).unwrap();
        let err = <Type>::try_from(&ast).unwrap_err().single();

        assert_eq!(*err.span().fragment(), "for<'U: Lin>");
        assert_eq!(err.span().location_offset(), 5);
        assert_matches!(
            err.kind(),
            ErrorKind::AstConversion(AstConversionError::EmbeddedQuantifier)
        );
    }

    #[test]
    fn parsing_basic_types() -> anyhow::Result<()> {
        let num_type = <Type>::try_from(&TypeAst::try_from("Num")?)?;
        assert_eq!(num_type, Type::NUM);

        let bool_type = <Type>::try_from(&TypeAst::try_from("Bool")?)?;
        assert_eq!(bool_type, Type::BOOL);

        let tuple_type = <Type>::try_from(&TypeAst::try_from("(Num, (Bool, _))")?)?;
        assert_eq!(
            tuple_type,
            Type::from((
                Type::NUM,
                Type::Tuple(vec![Type::BOOL, Type::free_var(0)].into()),
            ))
        );

        let slice_type = <Type>::try_from(&TypeAst::try_from("[(Num, _); _]")?)?;
        let slice_type = match &slice_type {
            Type::Tuple(tuple) => tuple.as_slice().unwrap(),
            _ => panic!("Unexpected type: {:?}", slice_type),
        };

        assert_eq!(
            *slice_type.element(),
            Type::from((Type::NUM, Type::free_var(0)))
        );
        assert_matches!(
            slice_type.len().components(),
            (Some(UnknownLen::Var(var)), 0) if var.is_free()
        );
        Ok(())
    }

    #[test]
    fn parsing_functional_type() -> anyhow::Result<()> {
        let ty = <Type>::try_from(&TypeAst::try_from("(['T; N], ('T) -> 'U) -> 'U")?)?;
        let ty = match ty {
            Type::Function(fn_type) => *fn_type,
            _ => panic!("Unexpected type: {:?}", ty),
        };

        assert_eq!(ty.params.as_ref().unwrap().len_params.len(), 1);
        assert_eq!(ty.params.as_ref().unwrap().type_params.len(), 2);
        assert_eq!(ty.return_type, Type::param(1));
        Ok(())
    }

    #[test]
    fn parsing_functional_type_with_varargs() -> anyhow::Result<()> {
        let ty = <Type>::try_from(&TypeAst::try_from("(...[Num; N]) -> Num")?)?;
        let ty = match ty {
            Type::Function(fn_type) => *fn_type,
            _ => panic!("Unexpected type: {:?}", ty),
        };

        assert_eq!(ty.params.as_ref().unwrap().len_params.len(), 1);
        assert!(ty.params.as_ref().unwrap().type_params.is_empty());
        let args_slice = ty.args.as_slice().unwrap();
        assert_eq!(*args_slice.element(), Type::NUM);
        assert_eq!(args_slice.len(), UnknownLen::param(0).into());
        Ok(())
    }

    #[test]
    fn parsing_incomplete_type() {
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
            TypeAst::try_from(input).unwrap_err();
        }
    }
}
