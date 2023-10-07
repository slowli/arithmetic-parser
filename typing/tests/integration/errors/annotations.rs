//! Errors with explicit type annotations.

use assert_matches::assert_matches;

use std::convert::TryFrom;

use arithmetic_parser::{grammars::Parse, InputSpan};
use arithmetic_typing::{
    ast::{AstConversionError, TypeAst},
    defs::Prelude,
    error::{ErrorContext, ErrorKind, TupleContext},
    TupleLen, Type, TypeEnvironment,
};

use crate::{
    assert_incompatible_types,
    errors::{fn_arg, tuple_element},
    hash_fn_type, zip_fn_type, ErrorsExt, F32Grammar, Hashed,
};

#[test]
fn converting_fn_type_unused_type() {
    let input = InputSpan::new("for<'T: Lin> (Num) -> Bool");
    let (_, ast) = TypeAst::parse(input).unwrap();
    let err = <Type>::try_from(&ast).unwrap_err().single();

    assert_eq!(err.main_location().location_offset(), 5);
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

    assert_eq!(err.main_location().location_offset(), 9);
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

    assert_eq!(err.main_location().location_offset(), 6);
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

    assert_eq!(err.main_location().location_offset(), 6);
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

    assert_eq!(err.main_location().span(&input), "Bug");
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

    assert_eq!(err.main_location().span(&input), "for<'U: Lin>");
    assert_eq!(err.main_location().location_offset(), 5);
    assert_matches!(
        err.kind(),
        ErrorKind::AstConversion(AstConversionError::EmbeddedQuantifier)
    );
}

#[test]
fn object_type_with_duplicate_fields() {
    let input = InputSpan::new("{ x: Num, x: (Num,) }");
    let (_, ast) = TypeAst::parse(input).unwrap();
    let err = <Type>::try_from(&ast).unwrap_err().single();

    assert_eq!(err.main_location().span(&input), "x");
    assert_eq!(err.main_location().location_offset(), 10);
    assert_matches!(
        err.kind(),
        ErrorKind::AstConversion(AstConversionError::DuplicateField(field))
            if field == "x"
    );
}

#[test]
fn error_when_parsing_standalone_some_type() {
    let code = "(_) -> Num";
    let errors = <Type>::try_from(&TypeAst::try_from(code).unwrap()).unwrap_err();
    let err = errors.single();

    assert_eq!(err.main_location().span(code), "_");
    assert_matches!(
        err.kind(),
        ErrorKind::AstConversion(AstConversionError::InvalidSomeType)
    );
}

#[test]
fn error_when_parsing_standalone_some_length() {
    let code = "[Num; _]";
    let errors = <Type>::try_from(&TypeAst::try_from(code).unwrap()).unwrap_err();
    let err = errors.single();

    assert_eq!(err.main_location().span(code), "_");
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

    assert_incompatible_types(err.kind(), &Type::NUM, &Type::BOOL);
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

    assert_incompatible_types(err.kind(), &Type::NUM, &Type::BOOL);
}

#[test]
fn unsupported_type_param_in_generic_fn() {
    let code = "identity: (('Arg,)) -> ('Arg,) = |x| x;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert!(err.location().is_empty());
    assert_matches!(err.context(), ErrorContext::Assignment { .. });
    assert_eq!(err.main_location().span(code), "(('Arg,)) -> ('Arg,)");
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
        assert_eq!(err.main_location().span(code), "(('Arg,)) -> ('Arg,)");
        assert_matches!(err.kind(), ErrorKind::UnsupportedParam);
    }
}

#[test]
fn unsupported_const_param_in_generic_fn() {
    let code = "identity: ([Num; N]) -> [Num; N] = |x| x;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(err.main_location().span(code), "([Num; N]) -> [Num; N]");
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
    assert_eq!(errors[0].main_location().span(code), "x");
    assert_matches!(errors[0].kind(), ErrorKind::DynamicLen(_));
    assert_eq!(errors[1].main_location().span(code), "y");
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

    assert_eq!(err.main_location().span(code), "Bogus");
    assert_matches!(
        err.kind(),
        ErrorKind::AstConversion(AstConversionError::UnknownType(ty))
            if ty == "Bogus"
    );
}

#[test]
fn custom_constraint_if_not_added_to_env() {
    let code = "x: dyn Hash = (1, 2);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(err.main_location().span(code), "Hash");
    assert_matches!(
        err.kind(),
        ErrorKind::AstConversion(AstConversionError::UnknownConstraint(c))
            if c == "Hash"
    );
}

#[test]
fn custom_constraint_if_incorrectly_added_to_env() {
    let code = "x: dyn Hash = (1, 2);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .insert_constraint(Hashed)
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(err.main_location().span(code), "Hash");
    assert_matches!(
        err.kind(),
        ErrorKind::AstConversion(AstConversionError::NotObjectSafe(c))
            if c == "Hash"
    );
}

#[test]
fn type_cast_basic_error() {
    let code = "x = (1, 2) as Num;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(err.main_location().span(code), "(1, 2)");
    assert_eq!(err.root_location().span(code), "(1, 2) as Num");
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
    let code = "x = (1, |x: Num| x + 3) as dyn Lin;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_matches!(
        err.kind(),
        ErrorKind::FailedConstraint { constraint, .. } if constraint.to_string() == "Lin"
    );
    assert_matches!(err.context(), ErrorContext::TypeCast { .. });
    assert_eq!(err.location(), [tuple_element(1)]);
    assert_eq!(err.main_location().span(code), "|x: Num| x + 3");
}

#[test]
fn insufficient_info_when_indexing_tuple() {
    let code = "xs = (1, 2) as [_]; xs.0";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(err.main_location().span(code), "xs.0");
    assert_matches!(err.kind(), ErrorKind::UnsupportedIndex);
    assert_matches!(err.context(), ErrorContext::TupleIndex { ty } if ty.to_string() == "[Num]");
}

#[test]
fn object_annotation_mismatch() {
    let code = "obj: { x: _ } = #{ x: 1, y: 2 };";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_matches!(err.kind(), ErrorKind::FieldsMismatch { .. });
}

#[test]
fn missing_field_after_object_annotation() {
    let code = "|obj: { x: _ }| obj.x == obj.y";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_matches!(
        err.kind(),
        ErrorKind::MissingFields { fields, .. } if fields.len() == 1 && fields.contains("y")
    );
}

#[test]
fn dyn_constraint_non_object() {
    let code = "#{ x: 1 } as dyn Lin as dyn { x: Num } + Lin";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_matches!(err.kind(), ErrorKind::CannotAccessFields);
}

#[test]
fn dyn_constraint_missing_additional_constraint() {
    let code = "#{ x: 1 } as dyn { x: Num } as dyn { x: Num } + Lin";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_matches!(
        err.kind(),
        ErrorKind::FailedConstraint { ty, constraint }
            if ty.to_string() == "dyn { x: Num }" && constraint.to_string() == "Lin"
    );
}

#[test]
fn dyn_constraint_failing_additional_constraint() {
    let code = "#{ x: 1, fun: || 1 } as dyn { x: Num } + Lin";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_matches!(
        err.kind(),
        ErrorKind::FailedConstraint { constraint, .. } if constraint.to_string() == "Lin"
    );
}

#[test]
fn contradicting_dyn_constraint_via_field_access() {
    let code = "|obj| { _: dyn Lin = obj; !obj.x; }";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(err.main_location().span(code), "!obj.x");
    assert_matches!(err.context(), ErrorContext::UnaryOp(_));
    assert_matches!(err.kind(), ErrorKind::FailedConstraint { ty, .. } if *ty == Type::BOOL);
}

#[test]
fn contradicting_field_types_via_annotations() {
    let code = r#"
       |obj| {
            { x -> _: Num } = obj; !obj.x
       }
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(err.main_location().span(code), "!obj.x");
    assert_matches!(err.context(), ErrorContext::UnaryOp(_));
    assert_matches!(
        err.kind(),
        ErrorKind::TypeMismatch(lhs, rhs) if *lhs == Type::BOOL && *rhs == Type::NUM
    );
}

#[test]
fn contradicting_constraint_with_dyn_object() {
    let code = "hash(#{ x: 1 } as dyn { x: Num })";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env
        .insert("hash", hash_fn_type())
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(err.main_location().span(code), "#{ x: 1 }");
    assert_eq!(err.location(), [fn_arg(0)]);
    assert_matches!(
        err.kind(),
        ErrorKind::FailedConstraint { ty: Type::Dyn(_), constraint }
            if constraint.to_string() == "Hash"
    );
}

#[test]
fn extra_fields_in_dyn_fn_arg() {
    let code = "|objs: [dyn { x: _ }; _]| objs.map(|obj| obj.x + obj.y)";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env
        .insert("map", Prelude::Map)
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(err.main_location().span(code), "|obj| obj.x + obj.y");
    assert_eq!(err.location(), [fn_arg(1), fn_arg(0)]);
    assert_matches!(
        err.kind(),
        ErrorKind::MissingFields { fields, .. } if fields.contains("y")
    );
}
