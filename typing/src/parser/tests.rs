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
            ParsedValueType::Number,
            ParsedValueType::Any,
            ParsedValueType::Bool,
            ParsedValueType::Any,
        ]
    );
}

#[test]
fn simple_slice_with_length() {
    let input = InputSpan::new("[Num; N]");
    let (rest, (element, len)) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(element, ParsedValueType::Number);
    assert_matches!(len, ParsedTupleLength::Ident(ident) if *ident.fragment() == "N");
}

#[test]
fn simple_slice_without_length() {
    let input = InputSpan::new("[T]");
    let (rest, (element, len)) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(element, ParsedValueType::Ident(ident) if *ident.fragment() == "T");
    assert_eq!(len, ParsedTupleLength::Dynamic);
}

#[test]
fn complex_slice_type() {
    let input = InputSpan::new("[(Num, Bool); M]");
    let (rest, (element, len)) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(
        element,
        ParsedValueType::Tuple(vec![ParsedValueType::Number, ParsedValueType::Bool])
    );
    assert_matches!(len, ParsedTupleLength::Ident(ident) if *ident.fragment() == "M");
}

#[test]
fn embedded_slice_type() {
    let input = InputSpan::new("[(Num, [Bool]); M]");
    let (rest, (element, len)) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(len, ParsedTupleLength::Ident(ident) if *ident.fragment() == "M");
    let first_element = match &element {
        ParsedValueType::Tuple(elements) => &elements[1],
        _ => panic!("Unexpected slice element: {:?}", element),
    };
    assert_matches!(
        first_element,
        ParsedValueType::Slice { element, length: ParsedTupleLength::Dynamic }
            if **element == ParsedValueType::Bool
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
    assert_eq!(fn_type.return_type, ParsedValueType::Number);
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
        ParsedValueType::Tuple(vec![ParsedValueType::Number; 2])
    );
    assert_eq!(fn_type.args[1], ParsedValueType::Bool);
    assert_eq!(fn_type.return_type, ParsedValueType::Number);
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
    assert_matches!(fn_type.args[1], ParsedValueType::Ident(ident) if *ident.fragment() == "T");
    assert_matches!(fn_type.return_type, ParsedValueType::Ident(ident) if *ident.fragment() == "T");
}

#[test]
fn fn_type_accepting_fn_arg() {
    let input = InputSpan::new("fn<const N; T>([T; N], fn(T) -> Bool) -> Bool");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(fn_type.const_params.len(), 1);
    assert_eq!(fn_type.type_params.len(), 1);
    assert_eq!(fn_type.args.len(), 2);
    assert_eq!(fn_type.return_type, ParsedValueType::Bool);

    let inner_fn = match &fn_type.args[1] {
        ParsedValueType::Function(inner_fn) => inner_fn.as_ref(),
        ty => panic!("Unexpected arg type: {:?}", ty),
    };
    assert!(inner_fn.const_params.is_empty());
    assert!(inner_fn.type_params.is_empty());
    assert_matches!(
        inner_fn.args.as_slice(),
        [ParsedValueType::Ident(ident)] if *ident.fragment() == "T"
    );
    assert_eq!(inner_fn.return_type, ParsedValueType::Bool);
}

#[test]
fn fn_type_returning_fn_arg() {
    let input = InputSpan::new("fn(Num) -> fn(Bool) -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(fn_type.const_params.is_empty());
    assert!(fn_type.type_params.is_empty());
    assert_eq!(fn_type.args, vec![ParsedValueType::Number]);

    let returned_fn = match fn_type.return_type {
        ParsedValueType::Function(returned_fn) => *returned_fn,
        ty => panic!("Unexpected return type: {:?}", ty),
    };
    assert!(returned_fn.const_params.is_empty());
    assert!(returned_fn.type_params.is_empty());
    assert_eq!(returned_fn.args, vec![ParsedValueType::Bool]);
    assert_eq!(returned_fn.return_type, ParsedValueType::Number);
}
