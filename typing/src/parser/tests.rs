use super::*;

use assert_matches::assert_matches;

#[test]
fn fn_const_params() {
    let input = InputSpan::new("<const N>");
    let (rest, (const_params, type_params)) = fn_params(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(type_params.is_empty());
    assert_eq!(const_params.len(), 1);
    assert_eq!(*const_params[0].fragment(), "N");
}

#[test]
fn fn_type_params() {
    let input = InputSpan::new("<T, U>");
    let (rest, (const_params, type_params)) = fn_params(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(const_params.is_empty());
    assert_eq!(
        type_params
            .iter()
            .map(|(param, _)| *param.fragment())
            .collect::<Vec<_>>(),
        vec!["T", "U"]
    );
}

#[test]
fn fn_type_params_with_bounds() {
    let input = InputSpan::new("<T: ?Lin, U>");
    let (rest, (const_params, type_params)) = fn_params(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(const_params.is_empty());
    assert_eq!(
        type_params
            .iter()
            .map(|(param, bounds)| (*param.fragment(), bounds.maybe_non_linear))
            .collect::<Vec<_>>(),
        vec![("T", true), ("U", false)]
    );
}

#[test]
fn fn_params_mixed() {
    let input = InputSpan::new("<const N; T, U>");
    let (rest, (const_params, type_params)) = fn_params(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(const_params.len(), 1);
    assert_eq!(*const_params[0].fragment(), "N");
    assert_eq!(
        type_params
            .iter()
            .map(|(param, _)| *param.fragment())
            .collect::<Vec<_>>(),
        vec!["T", "U"]
    );
}

#[test]
fn simple_tuple() {
    let input = InputSpan::new("(Num, _, Bool, _)");
    let (rest, elements) = tuple_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(
        elements,
        vec![
            RawValueType::Number,
            RawValueType::Any,
            RawValueType::Bool,
            RawValueType::Any,
        ]
    );
}

#[test]
fn simple_slice_with_length() {
    let input = InputSpan::new("[Num; N]");
    let (rest, (element, len)) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(element, RawValueType::Number);
    assert_matches!(len, RawTupleLength::Ident(ident) if *ident.fragment() == "N");
}

#[test]
fn simple_slice_without_length() {
    let input = InputSpan::new("[T]");
    let (rest, (element, len)) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(element, RawValueType::Ident(ident) if *ident.fragment() == "T");
    assert_eq!(len, RawTupleLength::Dynamic);
}

#[test]
fn complex_slice_type() {
    let input = InputSpan::new("[(Num, Bool); M]");
    let (rest, (element, len)) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(
        element,
        RawValueType::Tuple(vec![RawValueType::Number, RawValueType::Bool])
    );
    assert_matches!(len, RawTupleLength::Ident(ident) if *ident.fragment() == "M");
}

#[test]
fn embedded_slice_type() {
    let input = InputSpan::new("[(Num, [Bool]); M]");
    let (rest, (element, len)) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(len, RawTupleLength::Ident(ident) if *ident.fragment() == "M");
    let first_element = match &element {
        RawValueType::Tuple(elements) => &elements[1],
        _ => panic!("Unexpected slice element: {:?}", element),
    };
    assert_matches!(
        first_element,
        RawValueType::Slice { element, length: RawTupleLength::Dynamic }
            if **element == RawValueType::Bool
    );
}

#[test]
fn simple_fn_type() {
    let input = InputSpan::new("fn() -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(fn_type.const_params.is_empty());
    assert!(fn_type.type_params.is_empty());
    assert!(fn_type.args.is_empty());
    assert_eq!(fn_type.return_type, RawValueType::Number);
}

#[test]
fn simple_fn_type_with_args() {
    let input = InputSpan::new("fn((Num, Num), Bool) -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(fn_type.const_params.is_empty());
    assert!(fn_type.type_params.is_empty());
    assert_eq!(fn_type.args.len(), 2);
    assert_eq!(
        fn_type.args[0],
        RawValueType::Tuple(vec![RawValueType::Number; 2])
    );
    assert_eq!(fn_type.args[1], RawValueType::Bool);
    assert_eq!(fn_type.return_type, RawValueType::Number);
}

#[test]
fn fn_type_with_type_params() {
    let input = InputSpan::new("fn<T>(Bool, T, T) -> T");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(fn_type.const_params.is_empty());
    assert_eq!(fn_type.type_params.len(), 1);
    assert_eq!(*fn_type.type_params[0].0.fragment(), "T");
    assert_eq!(fn_type.args.len(), 3);
    assert_matches!(fn_type.args[1], RawValueType::Ident(ident) if *ident.fragment() == "T");
    assert_matches!(fn_type.return_type, RawValueType::Ident(ident) if *ident.fragment() == "T");
}

#[test]
fn fn_type_accepting_fn_arg() {
    let input = InputSpan::new("fn<const N; T>([T; N], fn(T) -> Bool) -> Bool");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(fn_type.const_params.len(), 1);
    assert_eq!(fn_type.type_params.len(), 1);
    assert_eq!(fn_type.args.len(), 2);
    assert_eq!(fn_type.return_type, RawValueType::Bool);

    let inner_fn = match &fn_type.args[1] {
        RawValueType::Function(inner_fn) => inner_fn.as_ref(),
        ty => panic!("Unexpected arg type: {:?}", ty),
    };
    assert!(inner_fn.const_params.is_empty());
    assert!(inner_fn.type_params.is_empty());
    assert_matches!(
        inner_fn.args.as_slice(),
        [RawValueType::Ident(ident)] if *ident.fragment() == "T"
    );
    assert_eq!(inner_fn.return_type, RawValueType::Bool);
}

#[test]
fn fn_type_returning_fn_arg() {
    let input = InputSpan::new("fn(Num) -> fn(Bool) -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(fn_type.const_params.is_empty());
    assert!(fn_type.type_params.is_empty());
    assert_eq!(fn_type.args, vec![RawValueType::Number]);

    let returned_fn = match fn_type.return_type {
        RawValueType::Function(returned_fn) => *returned_fn,
        ty => panic!("Unexpected return type: {:?}", ty),
    };
    assert!(returned_fn.const_params.is_empty());
    assert!(returned_fn.type_params.is_empty());
    assert_eq!(returned_fn.args, vec![RawValueType::Bool]);
    assert_eq!(returned_fn.return_type, RawValueType::Number);
}

#[test]
fn converting_raw_fn_type() {
    let input = InputSpan::new("fn<const N; T>([T; N], fn(T) -> Bool) -> Bool");
    let (_, fn_type) = fn_definition(input).unwrap();
    let fn_type = fn_type.try_convert().unwrap();

    assert_eq!(fn_type.to_string(), *input.fragment());
}

#[test]
fn converting_raw_fn_type_duplicate_type() {
    let input = InputSpan::new("fn<T, T>([T; N], fn(T) -> Bool) -> Bool");
    let (_, fn_type) = fn_definition(input).unwrap();
    let err = fn_type.try_convert().unwrap_err();
    assert_matches!(
        err,
        ConversionError::DuplicateTypeParam { definition, previous }
            if definition.location_offset() == 6 && previous.location_offset() == 3
    );
}

#[test]
fn converting_raw_fn_type_duplicate_type_in_embedded_fn() {
    let input = InputSpan::new("fn<const N; T>([T; N], fn<T>(T) -> Bool) -> Bool");
    let (_, fn_type) = fn_definition(input).unwrap();
    let err = fn_type.try_convert().unwrap_err();
    assert_matches!(
        err,
        ConversionError::DuplicateTypeParam { definition, previous }
            if definition.location_offset() == 26 && previous.location_offset() == 12
    );
}

#[test]
fn converting_raw_fn_type_duplicate_const() {
    let input = InputSpan::new("fn<const N, N; T>([T; N], fn(T) -> Bool) -> Bool");
    let (_, fn_type) = fn_definition(input).unwrap();
    let err = fn_type.try_convert().unwrap_err();
    assert_matches!(
        err,
        ConversionError::DuplicateConst { definition, previous }
            if definition.location_offset() == 12 && previous.location_offset() == 9
    );
}

#[test]
fn converting_raw_fn_type_duplicate_const_in_embedded_fn() {
    let input = InputSpan::new("fn<const N; T>([T; N], fn<const N>(T) -> Bool) -> Bool");
    let (_, fn_type) = fn_definition(input).unwrap();
    let err = fn_type.try_convert().unwrap_err();
    assert_matches!(
        err,
        ConversionError::DuplicateConst { definition, previous }
            if definition.location_offset() == 32 && previous.location_offset() == 9
    );
}

#[test]
fn converting_raw_fn_type_undefined_type() {
    let input = InputSpan::new("fn<const N>([T; N], fn(T) -> Bool) -> Bool");
    let (_, fn_type) = fn_definition(input).unwrap();
    let err = fn_type.try_convert().unwrap_err();
    assert_matches!(
        err,
        ConversionError::UndefinedTypeParam(definition) if definition.location_offset() == 13
    );
}

#[test]
fn converting_raw_fn_type_undefined_const() {
    let input = InputSpan::new("fn<T>([T; N], fn(T) -> Bool) -> Bool");
    let (_, fn_type) = fn_definition(input).unwrap();
    let err = fn_type.try_convert().unwrap_err();
    assert_matches!(
        err,
        ConversionError::UndefinedConst(definition) if definition.location_offset() == 10
    );
}
