//! Logic for converting `*Ast` types into their "main" counterparts.

use nom::Err as NomErr;

use std::{
    collections::{HashMap, HashSet},
    convert::TryFrom,
    fmt,
};

use crate::{
    arith::{CompleteConstraints, Constraint, ConstraintSet},
    ast::{
        ConstraintsAst, FunctionAst, ObjectAst, SliceAst, SpannedTypeAst, TupleAst, TupleLenAst,
        TypeAst, TypeConstraintsAst,
    },
    error::{Error, Errors},
    types::{ParamConstraints, ParamQuantifier},
    DynConstraints, Function, Object, PrimitiveType, Slice, Tuple, Type, TypeEnvironment,
    UnknownLen,
};
use arithmetic_parser::{ErrorKind as ParseErrorKind, InputSpan, NomResult, Spanned, SpannedError};

/// Kinds of errors that can occur when converting `*Ast` types into their "main" counterparts.
///
/// During type inference, errors of this type are wrapped into the [`AstConversion`]
/// variant of typing errors.
///
/// [`AstConversion`]: crate::error::ErrorKind::AstConversion
///
/// # Examples
///
/// ```
/// use arithmetic_parser::grammars::{Parse, F32Grammar};
/// use arithmetic_typing::{
///     ast::AstConversionError, error::ErrorKind, Annotated, TypeEnvironment,
/// };
/// # use assert_matches::assert_matches;
///
/// # fn main() -> anyhow::Result<()> {
/// let code = "bogus_slice: ['T; _] = (1, 2, 3);";
/// let code = Annotated::<F32Grammar>::parse_statements(code)?;
///
/// let errors = TypeEnvironment::new().process_statements(&code).unwrap_err();
/// let err = errors.into_iter().next().unwrap();
/// assert_eq!(*err.main_span().fragment(), "'T");
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
    /// Some type (`_`) encountered when parsing a standalone type.
    ///
    /// `_` types are only allowed in the context of a [`TypeEnvironment`]. It is a logical
    /// error to use them when parsing standalone types.
    InvalidSomeType,
    /// Some length (`_`) encountered when parsing a standalone type.
    ///
    /// `_` lengths are only allowed in the context of a [`TypeEnvironment`]. It is a logical
    /// error to use them when parsing standalone types.
    InvalidSomeLength,
    /// Field with the same name is defined multiple times in an object type.
    DuplicateField(String),
    /// Constraint is not object-safe.
    NotObjectSafe(String),
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

            Self::InvalidSomeType => {
                formatter.write_str("`_` type is disallowed when parsing standalone type")
            }
            Self::InvalidSomeLength => {
                formatter.write_str("`_` length is disallowed when parsing standalone type")
            }

            Self::DuplicateField(name) => {
                write!(formatter, "Duplicate field `{}` in object type", name)
            }

            Self::NotObjectSafe(name) => {
                write!(formatter, "Constraint `{}` is not object-safe", name)
            }
        }
    }
}

impl std::error::Error for AstConversionError {}

/// Intermediate conversion state.
#[derive(Debug)]
pub(crate) struct AstConversionState<'r, 'a, Prim: PrimitiveType> {
    env: Option<&'r mut TypeEnvironment<Prim>>,
    known_constraints: ConstraintSet<Prim>,
    errors: &'r mut Errors<'a, Prim>,
    len_params: HashMap<&'a str, usize>,
    type_params: HashMap<&'a str, usize>,
    is_in_function: bool,
}

impl<'r, 'a, Prim: PrimitiveType> AstConversionState<'r, 'a, Prim> {
    pub fn new(env: &'r mut TypeEnvironment<Prim>, errors: &'r mut Errors<'a, Prim>) -> Self {
        let known_constraints = env.known_constraints.clone();
        Self {
            env: Some(env),
            known_constraints,
            errors,
            len_params: HashMap::new(),
            type_params: HashMap::new(),
            is_in_function: false,
        }
    }

    fn without_env(errors: &'r mut Errors<'a, Prim>) -> Self {
        Self {
            env: None,
            known_constraints: Prim::well_known_constraints(),
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

    fn new_type(&mut self, span: Option<&SpannedTypeAst<'a>>) -> Type<Prim> {
        let errors = &mut *self.errors;
        self.env.as_mut().map_or_else(
            || {
                if let Some(span) = span {
                    let err = AstConversionError::InvalidSomeType;
                    errors.push(Error::conversion(err, span));
                }
                // We don't particularly care about the returned value; the enclosing type
                // will be discarded anyway.
                Type::free_var(0)
            },
            |env| env.substitutions.new_type_var(),
        )
    }

    fn new_len(&mut self, span: Option<&Spanned<'a, TupleLenAst>>) -> UnknownLen {
        let errors = &mut *self.errors;
        self.env.as_mut().map_or_else(
            || {
                if let Some(span) = span {
                    let err = AstConversionError::InvalidSomeLength;
                    errors.push(Error::conversion(err, span));
                }
                // We don't particularly care about the returned value; the enclosing type
                // will be discarded anyway.
                UnknownLen::free_var(0)
            },
            |env| env.substitutions.new_len_var(),
        )
    }

    fn resolve_constraint(&self, name: &str) -> Option<(Box<dyn Constraint<Prim>>, bool)> {
        self.known_constraints
            .get_by_name(name)
            .map(|(constraint, is_object_safe)| (constraint.clone_boxed(), is_object_safe))
    }

    pub(crate) fn convert_type(&mut self, ty: &SpannedTypeAst<'a>) -> Type<Prim> {
        match &ty.extra {
            TypeAst::Some => self.new_type(Some(ty)),
            TypeAst::Any => Type::Any,
            TypeAst::Dyn(constraints) => Type::Dyn(constraints.convert_dyn(self)),
            TypeAst::Ident => {
                let ident = *ty.fragment();
                if let Ok(prim_type) = Prim::from_str(ident) {
                    Type::Prim(prim_type)
                } else {
                    let err = AstConversionError::UnknownType(ident.to_owned());
                    self.errors.push(Error::conversion(err, ty));
                    self.new_type(None)
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
                    self.new_type(None)
                }
            }

            TypeAst::Function(function) => self.convert_fn(function, None),
            TypeAst::FunctionWithConstraints {
                function,
                constraints,
            } => self.convert_fn(&function.extra, Some(constraints)),

            TypeAst::Tuple(tuple) => tuple.convert(self).into(),
            TypeAst::Slice(slice) => slice.convert(self).into(),
            TypeAst::Object(object) => object.convert(self).into(),
        }
    }

    fn convert_fn(
        &mut self,
        function: &FunctionAst<'a>,
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
    ) -> CompleteConstraints<Prim> {
        self.do_convert(state, false)
    }

    fn convert_dyn<Prim: PrimitiveType>(
        &self,
        state: &mut AstConversionState<'_, 'a, Prim>,
    ) -> DynConstraints<Prim> {
        DynConstraints {
            inner: self.do_convert(state, true),
        }
    }

    fn do_convert<Prim: PrimitiveType>(
        &self,
        state: &mut AstConversionState<'_, 'a, Prim>,
        require_object_safety: bool,
    ) -> CompleteConstraints<Prim> {
        let mut constraints = CompleteConstraints::default();
        if let Some(object) = &self.object {
            constraints.object = Some(object.convert(state));
        }

        self.terms.iter().fold(constraints, |mut acc, input| {
            let input_str = *input.fragment();
            if let Some((constraint, is_object_safe)) = state.resolve_constraint(input_str) {
                if require_object_safety && !is_object_safe {
                    let err = AstConversionError::NotObjectSafe(input_str.to_owned());
                    state.errors.push(Error::conversion(err, input));
                } else {
                    acc.simple.insert_boxed(constraint);
                }
            } else {
                let err = AstConversionError::UnknownConstraint(input_str.to_owned());
                state.errors.push(Error::conversion(err, input));
            }
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
            type_params,
            static_lengths,
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

        let converted_length = match &self.length.extra {
            TupleLenAst::Ident => {
                let name = *self.length.fragment();
                if state.is_in_function {
                    let const_param = state.len_param_idx(name);
                    UnknownLen::param(const_param)
                } else {
                    let err = AstConversionError::FreeLengthVar(name.to_owned());
                    state.errors.push(Error::conversion(err, &self.length));
                    state.new_len(None)
                }
            }
            TupleLenAst::Some => state.new_len(Some(&self.length)),
            TupleLenAst::Dynamic => UnknownLen::Dynamic,
        };

        Slice::new(element, converted_length)
    }
}

impl<'a> ObjectAst<'a> {
    fn convert<Prim: PrimitiveType>(
        &self,
        state: &mut AstConversionState<'_, 'a, Prim>,
    ) -> Object<Prim> {
        let mut fields = HashMap::new();
        for (field_name, ty) in &self.fields {
            let field_name_str = *field_name.fragment();
            if fields.contains_key(field_name_str) {
                let err = AstConversionError::DuplicateField(field_name_str.to_owned());
                state.errors.push(Error::conversion(err, field_name));
            } else {
                fields.insert(field_name_str.to_owned(), state.convert_type(ty));
            }
        }
        Object::from_map(fields)
    }
}

impl<'a> FunctionAst<'a> {
    fn convert<Prim: PrimitiveType>(
        &self,
        state: &mut AstConversionState<'_, 'a, Prim>,
    ) -> Function<Prim> {
        let args = self.args.extra.convert(state);
        let return_type = state.convert_type(&self.return_type);
        Function::new(args, return_type)
    }

    /// Tries to convert this type into a [`Function`].
    pub fn try_convert<Prim>(&self) -> Result<Function<Prim>, Errors<'a, Prim>>
    where
        Prim: PrimitiveType,
    {
        let mut errors = Errors::new();
        let mut state = AstConversionState::without_env(&mut errors);
        state.is_in_function = true;

        let output = self.convert(&mut state);
        if errors.is_empty() {
            Ok(output)
        } else {
            Err(errors)
        }
    }
}

/// Shared parsing code for `TypeAst` and `FunctionAst`.
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

/// Shared `TryFrom<&str>` logic for `TypeAst` and `FunctionAst`.
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
        let mut errors = Errors::new();
        let mut state = AstConversionState::without_env(&mut errors);

        let output = state.convert_type(ast);
        if errors.is_empty() {
            Ok(output)
        } else {
            Err(errors)
        }
    }
}

impl<'a> TryFrom<&'a str> for FunctionAst<'a> {
    type Error = SpannedError<&'a str>;

    fn try_from(def: &'a str) -> Result<Self, Self::Error> {
        from_str(FunctionAst::parse, def)
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;

    use super::*;
    use crate::arith::Num;

    #[test]
    fn converting_raw_fn_type() {
        let input = InputSpan::new("(['T; N], ('T) -> Bool) -> Bool");
        let (_, fn_type) = FunctionAst::parse(input).unwrap();
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
    fn parsing_basic_types() -> anyhow::Result<()> {
        let num_type = <Type>::try_from(&TypeAst::try_from("Num")?)?;
        assert_eq!(num_type, Type::NUM);

        let bool_type = <Type>::try_from(&TypeAst::try_from("Bool")?)?;
        assert_eq!(bool_type, Type::BOOL);

        let tuple_type = <Type>::try_from(&TypeAst::try_from("(Num, (Bool, Bool))")?)?;
        assert_eq!(
            tuple_type,
            Type::from((Type::NUM, Type::Tuple(vec![Type::BOOL; 2].into()),))
        );

        let slice_type = <Type>::try_from(&TypeAst::try_from("[(Num, Bool)]")?)?;
        let slice_type = match &slice_type {
            Type::Tuple(tuple) => tuple.as_slice().unwrap(),
            _ => panic!("Unexpected type: {:?}", slice_type),
        };

        assert_eq!(*slice_type.element(), Type::from((Type::NUM, Type::BOOL)));
        assert_matches!(
            slice_type.len().components(),
            (Some(UnknownLen::Dynamic), 0)
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

    #[test]
    fn parsing_type_with_object_constraint() -> anyhow::Result<()> {
        let type_def = "for<'T: { x: Num } + Lin> ('T) -> Bool";
        let ty = TypeAst::try_from(type_def)?;
        let ty = <Type>::try_from(&ty)?;
        let ty = match ty {
            Type::Function(fn_type) => *fn_type,
            _ => panic!("Unexpected type: {:?}", ty),
        };

        let type_params = &ty.params.as_ref().unwrap().type_params;
        assert_eq!(type_params.len(), 1);
        let (_, type_params) = &type_params[0];
        assert!(type_params.object.is_some());
        assert!(type_params.simple.get_by_name("Lin").is_some());

        assert_eq!(ty.to_string(), type_def);
        Ok(())
    }
}
