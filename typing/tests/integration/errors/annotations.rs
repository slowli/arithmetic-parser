//! Errors with explicit type annotations.

use assert_matches::assert_matches;

use std::convert::TryFrom;

use arithmetic_parser::{grammars::Parse, InputSpan};
use arithmetic_typing::{
    ast::{AstConversionError, TypeAst},
    error::{ErrorContext, ErrorKind, TupleContext},
    Prelude, TupleLen, Type, TypeEnvironment,
};

use crate::{assert_incompatible_types, errors::tuple_element, zip_fn_type, ErrorsExt, F32Grammar};

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
    let input = InputSpan::new("for<'T: Bug> (['T; N]) -> Bool");
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
fn error_when_parsing_standalone_some_type() {
    let errors = <Type>::try_from(&TypeAst::try_from("(_) -> Num").unwrap()).unwrap_err();
    let err = errors.single();

    assert_eq!(*err.span().fragment(), "_");
    assert_matches!(
        err.kind(),
        ErrorKind::AstConversion(AstConversionError::InvalidSomeType)
    );
}

#[test]
fn error_when_parsing_standalone_some_length() {
    let errors = <Type>::try_from(&TypeAst::try_from("[Num; _]").unwrap()).unwrap_err();
    let err = errors.single();

    assert_eq!(*err.span().fragment(), "_");
    assert_matches!(
        err.kind(),
        ErrorKind::AstConversion(AstConversionError::InvalidSomeLength)
    );
}

#[test]
fn contradicting_type_hint() {
    let code = "x: (Num, _) = (1, 2, 3);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch {
            lhs,
            rhs,
            context: TupleContext::Generic,
        } if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(3)
    );
}

#[test]
fn contradicting_type_hint_with_slice() {
    let code = "x: [Num; _] = (1, 2 == 3);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_incompatible_types(&err.kind(), &Type::NUM, &Type::BOOL);
}

#[test]
fn invalid_type_hint_with_fn_arg() {
    let code = "foo = |xs, map_fn: (_, _) -> _| xs.map(map_fn);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch {
            lhs,
            rhs,
            context: TupleContext::FnArgs,
        } if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(1)
    );
}

#[test]
fn invalid_type_hint_with_fn_declaration() {
    let code = "foo: (_) -> Bool = |x| x + 3;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_incompatible_types(&err.kind(), &Type::NUM, &Type::BOOL);
}

#[test]
fn unsupported_type_param_in_generic_fn() {
    let code = "identity: (('Arg,)) -> ('Arg,) = |x| x;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert!(err.location().is_empty());
    assert_matches!(err.context(), ErrorContext::Assignment { .. });
    assert_eq!(*err.span().fragment(), "(('Arg,)) -> ('Arg,)");
    assert_matches!(err.kind(), ErrorKind::UnsupportedParam);
}

#[test]
fn unsupported_type_param_location() {
    let code_samples = &[
        "identity: (_, (('Arg,)) -> ('Arg,)) = (3, |x| x);",
        "(_, identity: (('Arg,)) -> ('Arg,)) = (3, |x| x);",
    ];

    for &code in code_samples {
        let block = F32Grammar::parse_statements(code).unwrap();
        let mut type_env = TypeEnvironment::new();
        let err = type_env.process_statements(&block).unwrap_err().single();

        assert_eq!(err.location(), [tuple_element(1)]);
        assert_matches!(err.context(), ErrorContext::Assignment { .. });
        assert_eq!(*err.span().fragment(), "(('Arg,)) -> ('Arg,)");
        assert_matches!(err.kind(), ErrorKind::UnsupportedParam);
    }
}

#[test]
fn unsupported_const_param_in_generic_fn() {
    let code = "identity: ([Num; N]) -> [Num; N] = |x| x;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "([Num; N]) -> [Num; N]");
    assert_matches!(err.kind(), ErrorKind::UnsupportedParam);
}

#[test]
fn adding_dynamically_typed_slices() {
    let code = r#"
        x: [Num] = (1, 2);
        y: [Num] = (3, 4, 5);
        x + y
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let errors: Vec<_> = type_env
        .process_statements(&block)
        .unwrap_err()
        .into_iter()
        .collect();

    assert_eq!(errors.len(), 2);
    assert_eq!(*errors[0].span().fragment(), "x");
    assert_matches!(errors[0].kind(), ErrorKind::DynamicLen(_));
    assert_eq!(*errors[1].span().fragment(), "y");
    assert_matches!(errors[1].kind(), ErrorKind::DynamicLen(_));
}

#[test]
fn unifying_dynamic_slices_error() {
    let code = r#"
        x: [Num] = (1, 2);
        y: [Num] = (3, 4, 5);
        x.zip_with(y)
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("zip_with", zip_fn_type());
    let errors: Vec<_> = type_env
        .process_statements(&block)
        .unwrap_err()
        .into_iter()
        .collect();

    assert_eq!(errors.len(), 2);
    assert_matches!(errors[0].kind(), ErrorKind::DynamicLen(_));
    assert_matches!(errors[1].kind(), ErrorKind::DynamicLen(_));
}

#[test]
fn bogus_annotation_in_fn_definition() {
    let code = "|x: Bogus| x + 1";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "Bogus");
    assert_matches!(
        err.kind(),
        ErrorKind::AstConversion(AstConversionError::UnknownType(ty))
            if ty == "Bogus"
    );
}

#[test]
fn custom_constraint_if_not_added_to_env() {
    let code = "x: any Hash = (1, 2);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.span().fragment(), "Hash");
    assert_matches!(
        err.kind(),
        ErrorKind::AstConversion(AstConversionError::UnknownConstraint(c))
            if c == "Hash"
    );
}

#[test]
fn type_cast_basic_error() {
    let code = "x = (1, 2) as Num;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "(1, 2)");
    assert_eq!(*err.root_span().fragment(), "(1, 2) as Num");
    assert_matches!(
        err.kind(),
        ErrorKind::TypeMismatch(lhs, rhs) if *lhs == Type::NUM && rhs.to_string() == "(Num, Num)"
    );
    assert_matches!(
        err.context(),
        ErrorContext::TypeCast { source, target }
            if *target == Type::NUM && source.to_string() == "(Num, Num)"
    );
}

#[test]
fn type_cast_error_in_subtype() {
    let code = "x = (1, |x: Num| x + 3) as any Lin;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_matches!(
        err.kind(),
        ErrorKind::FailedConstraint { constraint, .. } if constraint.to_string() == "Lin"
    );
    assert_matches!(err.context(), ErrorContext::TypeCast { .. });
    assert_eq!(err.location(), [tuple_element(1)]);
    assert_eq!(*err.span().fragment(), "|x: Num| x + 3");
}

#[test]
fn insufficient_info_when_indexing_tuple() {
    let code = "xs = (1, 2) as [_]; xs.0";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "xs.0");
    assert_matches!(err.kind(), ErrorKind::UnsupportedIndex);
    assert_matches!(err.context(), ErrorContext::TupleIndex { ty } if ty.to_string() == "[Num]");
}
