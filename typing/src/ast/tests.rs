use assert_matches::assert_matches;

use super::*;
use crate::Num;

#[test]
fn fn_const_params() {
    let input = InputSpan::new("<len N>");
    let (rest, (const_params, type_params)) = fn_params(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(type_params.is_empty());
    assert_eq!(const_params.len(), 1);
    assert_eq!(*const_params[0].0.fragment(), "N");
    assert_eq!(const_params[0].1, LengthKind::Static);
}

#[test]
fn fn_const_dyn_params() {
    let input = InputSpan::new("<len N, M*>");
    let (rest, (const_params, type_params)) = fn_params(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(type_params.is_empty());
    assert_eq!(const_params.len(), 2);
    assert_eq!(*const_params[1].0.fragment(), "M");
    assert_eq!(const_params[1].1, LengthKind::Dynamic);
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
    let input = InputSpan::new("<T: Lin, U>");
    let (rest, (const_params, type_params)) = fn_params(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(const_params.is_empty());
    assert_eq!(
        type_params
            .iter()
            .map(|(param, constraints)| (*param.fragment(), constraints.constraints.is_empty()))
            .collect::<Vec<_>>(),
        vec![("T", false), ("U", true)]
    );
}

#[test]
fn fn_params_mixed() {
    let input = InputSpan::new("<len N; T, U>");
    let (rest, (const_params, type_params)) = fn_params(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(const_params.len(), 1);
    assert_eq!(*const_params[0].0.fragment(), "N");
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
        elements.start,
        vec![
            ValueTypeAst::Prim(Num::Num),
            ValueTypeAst::Any,
            ValueTypeAst::Prim(Num::Bool),
            ValueTypeAst::Any,
        ]
    );
}

#[test]
fn simple_slice_with_length() {
    let input = InputSpan::new("[Num; N]");
    let (rest, slice) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(*slice.element, ValueTypeAst::Prim(Num::Num));
    assert_matches!(slice.length, TupleLengthAst::Ident(ident) if *ident.fragment() == "N");
}

#[test]
fn simple_slice_without_length() {
    let input = InputSpan::new("[T]");
    let (rest, slice) = slice_definition::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(*slice.element, ValueTypeAst::Ident(ident) if *ident.fragment() == "T");
    assert_eq!(slice.length, TupleLengthAst::Dynamic);
}

#[test]
fn complex_slice_type() {
    let input = InputSpan::new("[(Num, Bool); M]");
    let (rest, slice) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(
        *slice.element,
        ValueTypeAst::Tuple(TupleAst {
            start: vec![ValueTypeAst::Prim(Num::Num), ValueTypeAst::Prim(Num::Bool)],
            middle: None,
            end: vec![],
        })
    );
    assert_matches!(slice.length, TupleLengthAst::Ident(ident) if *ident.fragment() == "M");
}

#[test]
fn embedded_slice_type() {
    let input = InputSpan::new("[(Num, [Bool]); M]");
    let (rest, slice) = slice_definition::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(slice.length, TupleLengthAst::Ident(ident) if *ident.fragment() == "M");
    let first_element = match slice.element.as_ref() {
        ValueTypeAst::Tuple(tuple) => &tuple.start[1],
        _ => panic!("Unexpected slice element: {:?}", slice.element),
    };
    assert_matches!(
        first_element,
        ValueTypeAst::Slice(SliceAst { element, length: TupleLengthAst::Dynamic })
            if **element == ValueTypeAst::Prim(Num::Bool)
    );
}

#[test]
fn simple_fn_type() {
    let input = InputSpan::new("fn() -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(fn_type.len_params.is_empty());
    assert!(fn_type.type_params.is_empty());
    assert!(fn_type.args.start.is_empty());
    assert_eq!(fn_type.return_type, ValueTypeAst::Prim(Num::Num));
}

#[test]
fn simple_fn_type_with_args() {
    let input = InputSpan::new("fn((Num, Num), Bool) -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(fn_type.len_params.is_empty());
    assert!(fn_type.type_params.is_empty());
    assert_eq!(fn_type.args.start.len(), 2);
    assert_eq!(
        fn_type.args.start[0],
        ValueTypeAst::Tuple(TupleAst {
            start: vec![ValueTypeAst::Prim(Num::Num); 2],
            middle: None,
            end: vec![],
        })
    );
    assert_eq!(fn_type.args.start[1], ValueTypeAst::Prim(Num::Bool));
    assert_eq!(fn_type.return_type, ValueTypeAst::Prim(Num::Num));
}

#[test]
fn fn_type_with_type_params() {
    let input = InputSpan::new("fn<T>(Bool, T, T) -> T");
    let (rest, fn_type) = fn_definition::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(fn_type.len_params.is_empty());
    assert_eq!(fn_type.type_params.len(), 1);
    assert_eq!(*fn_type.type_params[0].0.fragment(), "T");
    assert_eq!(fn_type.args.start.len(), 3);
    assert_matches!(fn_type.args.start[1], ValueTypeAst::Ident(ident) if *ident.fragment() == "T");
    assert_matches!(fn_type.return_type, ValueTypeAst::Ident(ident) if *ident.fragment() == "T");
}

#[test]
fn fn_type_accepting_fn_arg() {
    let input = InputSpan::new("fn<len N; T>([T; N], fn(T) -> Bool) -> Bool");
    let (rest, fn_type) = fn_definition::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(fn_type.len_params.len(), 1);
    assert_eq!(fn_type.type_params.len(), 1);
    assert_eq!(fn_type.args.start.len(), 2);
    assert_eq!(fn_type.return_type, ValueTypeAst::Prim(Num::Bool));

    let inner_fn = match &fn_type.args.start[1] {
        ValueTypeAst::Function(inner_fn) => inner_fn.as_ref(),
        ty => panic!("Unexpected arg type: {:?}", ty),
    };
    assert!(inner_fn.len_params.is_empty());
    assert!(inner_fn.type_params.is_empty());
    assert_matches!(
        inner_fn.args.start.as_slice(),
        [ValueTypeAst::Ident(ident)] if *ident.fragment() == "T"
    );
    assert_eq!(inner_fn.return_type, ValueTypeAst::Prim(Num::Bool));
}

#[test]
fn fn_type_returning_fn_arg() {
    let input = InputSpan::new("fn(Num) -> fn(Bool) -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(fn_type.len_params.is_empty());
    assert!(fn_type.type_params.is_empty());
    assert_eq!(fn_type.args.start, vec![ValueTypeAst::Prim(Num::Num)]);

    let returned_fn = match fn_type.return_type {
        ValueTypeAst::Function(returned_fn) => *returned_fn,
        ty => panic!("Unexpected return type: {:?}", ty),
    };
    assert!(returned_fn.len_params.is_empty());
    assert!(returned_fn.type_params.is_empty());
    assert_eq!(returned_fn.args.start, vec![ValueTypeAst::Prim(Num::Bool)]);
    assert_eq!(returned_fn.return_type, ValueTypeAst::Prim(Num::Num));
}
