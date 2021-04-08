use assert_matches::assert_matches;

use super::*;
use crate::{arith::LinConstraints, Num};

#[test]
fn fn_const_params() {
    let input = InputSpan::new("for<len N*>");
    let (rest, constraints) = constraints::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(constraints.type_params.is_empty());
    assert_eq!(constraints.dyn_lengths.len(), 1);
    assert_eq!(*constraints.dyn_lengths[0].fragment(), "N");
}

#[test]
fn multiple_dyn_lengths() {
    let input = InputSpan::new("for<len N*, M*>");
    let (rest, constraints) = constraints::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(constraints.type_params.is_empty());
    assert_eq!(constraints.dyn_lengths.len(), 2);
    assert_eq!(*constraints.dyn_lengths[1].fragment(), "M");
}

#[test]
fn type_param_constraints() {
    let input = InputSpan::new("for<'T: Lin, 'U: Lin>");
    let (rest, constraints) = constraints::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(constraints.dyn_lengths.is_empty());
    assert_eq!(
        constraints
            .type_params
            .iter()
            .map(|(param, _)| *param.fragment())
            .collect::<Vec<_>>(),
        vec!["T", "U"]
    );
}

#[test]
fn mixed_constraints() {
    let input = InputSpan::new("for<len N*; 'T: Lin>");
    let (rest, constraints) = constraints::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(constraints.dyn_lengths.len(), 1);
    assert_eq!(*constraints.dyn_lengths[0].fragment(), "N");
    assert_eq!(
        constraints
            .type_params
            .iter()
            .map(|(param, ty_constraints)| (*param.fragment(), *ty_constraints.terms[0].fragment()))
            .collect::<Vec<_>>(),
        vec![("T", "Lin")]
    );
}

#[test]
fn simple_tuple() {
    let input = InputSpan::new("(Num, _, Bool, _)");
    let (rest, tuple) = tuple_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(
        tuple.start,
        vec![
            ValueTypeAst::Prim(Num::Num),
            ValueTypeAst::Some,
            ValueTypeAst::Prim(Num::Bool),
            ValueTypeAst::Some,
        ]
    );
}

#[derive(Debug)]
struct TupleSample {
    input: &'static str,
    start: &'static [ValueTypeAst<'static>],
    middle_element: ValueTypeAst<'static>,
    end: &'static [ValueTypeAst<'static>],
}

#[test]
fn complex_tuples() {
    const SAMPLES: &[TupleSample] = &[
        TupleSample {
            input: "(Num, _, ...[Bool; _])",
            start: &[ValueTypeAst::Prim(Num::Num), ValueTypeAst::Some],
            middle_element: ValueTypeAst::Prim(Num::Bool),
            end: &[],
        },
        TupleSample {
            input: "(...[Bool; _], Num)",
            start: &[],
            middle_element: ValueTypeAst::Prim(Num::Bool),
            end: &[ValueTypeAst::Prim(Num::Num)],
        },
        TupleSample {
            input: "(_, ...[Bool; _], Num)",
            start: &[ValueTypeAst::Some],
            middle_element: ValueTypeAst::Prim(Num::Bool),
            end: &[ValueTypeAst::Prim(Num::Num)],
        },
    ];

    for sample in SAMPLES {
        let input = InputSpan::new(sample.input);
        let (rest, tuple) = tuple_definition(input).unwrap();

        assert!(rest.fragment().is_empty());
        assert_eq!(tuple.start, sample.start);
        assert_eq!(*tuple.middle.unwrap().element, sample.middle_element);
        assert_eq!(tuple.end, sample.end);
    }
}

#[test]
fn embedded_complex_tuple() {
    let input = InputSpan::new("(Num, ...[Num; _], (...['T; N], () -> 'T))");
    let (rest, tuple) = tuple_definition::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(tuple.start, &[ValueTypeAst::Prim(Num::Num)]);
    let middle = tuple.middle.unwrap();
    assert_eq!(*middle.element, ValueTypeAst::Prim(Num::Num));
    assert_eq!(tuple.end.len(), 1);

    let embedded_tuple = match &tuple.end[0] {
        ValueTypeAst::Tuple(tuple) => tuple,
        other => panic!("Unexpected tuple end: {:?}", other),
    };
    assert!(embedded_tuple.start.is_empty());
    assert_matches!(
        *embedded_tuple.middle.as_ref().unwrap().element,
        ValueTypeAst::Param(id) if *id.fragment() == "T"
    );
    assert_matches!(
        embedded_tuple.end.as_slice(),
        [ValueTypeAst::Function { .. }]
    );
}

#[test]
fn simple_slice_with_length() {
    let input = InputSpan::new("[Num; N]");
    let (rest, slice) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(*slice.element, ValueTypeAst::Prim(Num::Num));
    assert_matches!(slice.length, TupleLenAst::Ident(ident) if *ident.fragment() == "N");
}

#[test]
fn simple_slice_without_length() {
    let input = InputSpan::new("['T]");
    let (rest, slice) = slice_definition::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(*slice.element, ValueTypeAst::Param(ident) if *ident.fragment() == "T");
    assert_eq!(slice.length, TupleLenAst::Dynamic);
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
    assert_matches!(slice.length, TupleLenAst::Ident(ident) if *ident.fragment() == "M");
}

#[test]
fn embedded_slice_type() {
    let input = InputSpan::new("[(Num, [Bool]); M]");
    let (rest, slice) = slice_definition::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(slice.length, TupleLenAst::Ident(ident) if *ident.fragment() == "M");
    let first_element = match slice.element.as_ref() {
        ValueTypeAst::Tuple(tuple) => &tuple.start[1],
        _ => panic!("Unexpected slice element: {:?}", slice.element),
    };
    assert_matches!(
        first_element,
        ValueTypeAst::Slice(SliceAst { element, length: TupleLenAst::Dynamic })
            if **element == ValueTypeAst::Prim(Num::Bool)
    );
}

#[test]
fn simple_fn_type() {
    let input = InputSpan::new("() -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(fn_type.args.start.is_empty());
    assert_eq!(fn_type.return_type, ValueTypeAst::Prim(Num::Num));
}

#[test]
fn simple_fn_type_with_args() {
    let input = InputSpan::new("((Num, Num), Bool) -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
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
    let input = InputSpan::new("(Bool, 'T, 'T) -> 'T");
    let (rest, fn_type) = fn_definition::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(fn_type.args.start.len(), 3);
    assert_matches!(fn_type.args.start[1], ValueTypeAst::Param(ident) if *ident.fragment() == "T");
    assert_matches!(fn_type.return_type, ValueTypeAst::Param(ident) if *ident.fragment() == "T");
}

#[test]
fn fn_type_accepting_fn_arg() {
    let input = InputSpan::new("(['T; N], ('T) -> Bool) -> Bool");
    let (rest, fn_type) = fn_definition::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(fn_type.args.start.len(), 2);
    assert_eq!(fn_type.return_type, ValueTypeAst::Prim(Num::Bool));

    let inner_fn = match &fn_type.args.start[1] {
        ValueTypeAst::Function { function, .. } => function.as_ref(),
        ty => panic!("Unexpected arg type: {:?}", ty),
    };
    assert_matches!(
        inner_fn.args.start.as_slice(),
        [ValueTypeAst::Param(ident)] if *ident.fragment() == "T"
    );
    assert_eq!(inner_fn.return_type, ValueTypeAst::Prim(Num::Bool));
}

#[test]
fn fn_type_returning_fn_arg() {
    let input = InputSpan::new("(Num) -> (Bool) -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(fn_type.args.start, vec![ValueTypeAst::Prim(Num::Num)]);

    let returned_fn = match fn_type.return_type {
        ValueTypeAst::Function { function, .. } => *function,
        ty => panic!("Unexpected return type: {:?}", ty),
    };
    assert_eq!(returned_fn.args.start, vec![ValueTypeAst::Prim(Num::Bool)]);
    assert_eq!(returned_fn.return_type, ValueTypeAst::Prim(Num::Num));
}

#[test]
fn fn_type_with_rest_params() {
    let input = InputSpan::new("(Bool, ...[Num; N]) -> Num");
    let (rest, fn_type) = fn_definition::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(fn_type.args.start.len(), 1);
    assert_eq!(fn_type.args.start[0], ValueTypeAst::Prim(Num::Bool));
    let middle = fn_type.args.middle.unwrap();
    assert_eq!(*middle.element, ValueTypeAst::Prim(Num::Num));
    assert_matches!(middle.length, TupleLenAst::Ident(id) if *id.fragment() == "N");
}

#[test]
fn fn_type_with_constraints() {
    let input = InputSpan::new("for<'T: Lin> ('T) -> 'T");
    let (rest, (constraints, fn_type)) = fn_definition_with_constraints::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(constraints.dyn_lengths.is_empty());
    assert_eq!(constraints.type_params.len(), 1);
    assert_eq!(constraints.type_params[0].1.computed, LinConstraints::LIN);
    assert_eq!(fn_type.args.start.len(), 1);
    assert_matches!(fn_type.args.start[0], ValueTypeAst::Param(id) if *id.fragment() == "T");
    assert_matches!(fn_type.return_type, ValueTypeAst::Param(id) if *id.fragment() == "T");
}

#[test]
fn multiple_fns_with_constraints() {
    let input = InputSpan::new("(for<'T: Lin> ('T) -> 'T, for<'T: Lin> (...['T; _]) -> ())");
    let (rest, ty) = type_definition::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    let (second_fn, first_fn) = match ty {
        ValueTypeAst::Tuple(TupleAst { mut start, .. }) if start.len() == 2 => {
            (start.pop().unwrap(), start.pop().unwrap())
        }
        _ => panic!("Unexpected type: {:?}", ty),
    };

    let first_fn = match first_fn {
        ValueTypeAst::Function {
            constraints: Some(_),
            function,
        } => *function,
        _ => panic!("Unexpected 1st function: {:?}", first_fn),
    };
    assert_matches!(first_fn.return_type, ValueTypeAst::Param(id) if *id.fragment() == "T");

    let second_fn = match second_fn {
        ValueTypeAst::Function {
            constraints: Some(_),
            function,
        } => *function,
        _ => panic!("Unexpected 2nd function: {:?}", second_fn),
    };
    assert!(second_fn.args.start.is_empty());
    assert_matches!(
        *second_fn.args.middle.unwrap().element,
        ValueTypeAst::Param(id) if *id.fragment() == "T"
    );
    assert_matches!(second_fn.return_type, ValueTypeAst::Tuple(t) if t.start.is_empty());
}

#[test]
fn any_type() {
    let input = InputSpan::new("any");
    let (rest, ty) = type_definition::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(ty, ValueTypeAst::Any(constraints) if constraints.terms.is_empty());
}

#[test]
fn any_type_with_bound() {
    let input = InputSpan::new("any Lin");
    let (rest, ty) = type_definition::<Num>(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(ty, ValueTypeAst::Any(constraints) if constraints.terms.len() == 1);

    let bogus_input = InputSpan::new("anyLin");
    let err = type_definition::<Num>(bogus_input).unwrap_err();
    let err = match err {
        NomErr::Failure(err) => err,
        _ => panic!("Unexpected error type: {:?}", err),
    };

    assert_eq!(*err.span().fragment(), "anyLin");
    assert_matches!(
        err.kind(),
        ParseErrorKind::Type(e) if e.to_string() == "Unknown type name: anyLin"
    );
}
