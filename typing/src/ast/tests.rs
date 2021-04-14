use assert_matches::assert_matches;

use std::ops::Range;

use super::*;

fn sp<T>(input: InputSpan<'_>, range: Range<usize>, extra: T) -> Spanned<'_, T> {
    Spanned::from_str(input.fragment(), range).copy_with_extra(extra)
}

#[test]
fn fn_const_params() {
    let input = InputSpan::new("for<len! N>");
    let (rest, constraints) = constraints(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(constraints.type_params.is_empty());
    assert_eq!(constraints.static_lengths.len(), 1);
    assert_eq!(*constraints.static_lengths[0].fragment(), "N");
}

#[test]
fn multiple_static_lengths() {
    let input = InputSpan::new("for<len! N, M>");
    let (rest, constraints) = constraints(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(constraints.type_params.is_empty());
    assert_eq!(constraints.static_lengths.len(), 2);
    assert_eq!(*constraints.static_lengths[1].fragment(), "M");
}

#[test]
fn type_param_constraints() {
    let input = InputSpan::new("for<'T: Lin, 'U: Lin>");
    let (rest, constraints) = constraints(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(constraints.static_lengths.is_empty());
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
    let input = InputSpan::new("for<len! N; 'T: Lin>");
    let (rest, constraints) = constraints(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(constraints.static_lengths.len(), 1);
    assert_eq!(*constraints.static_lengths[0].fragment(), "N");
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
            sp(input, 1..4, ValueTypeAst::Ident),
            sp(input, 6..7, ValueTypeAst::Some),
            sp(input, 9..13, ValueTypeAst::Ident),
            sp(input, 15..16, ValueTypeAst::Some),
        ]
    );
}

#[derive(Debug)]
struct TupleSample {
    input: InputSpan<'static>,
    start: Vec<SpannedValueTypeAst<'static>>,
    middle_element: SpannedValueTypeAst<'static>,
    end: Vec<SpannedValueTypeAst<'static>>,
}

#[test]
fn complex_tuples() {
    let first_input = InputSpan::new("(Num, _, ...[Bool; _])");
    let second_input = InputSpan::new("(...[Bool; _], Num)");
    let third_input = InputSpan::new("(_, ...[Bool; _], Num)");

    let samples = &[
        TupleSample {
            input: first_input,
            start: vec![
                sp(first_input, 1..4, ValueTypeAst::Ident),
                sp(first_input, 6..7, ValueTypeAst::Some),
            ],
            middle_element: sp(first_input, 13..17, ValueTypeAst::Ident),
            end: vec![],
        },
        TupleSample {
            input: second_input,
            start: vec![],
            middle_element: sp(second_input, 5..9, ValueTypeAst::Ident),
            end: vec![sp(second_input, 15..18, ValueTypeAst::Ident)],
        },
        TupleSample {
            input: third_input,
            start: vec![sp(third_input, 1..2, ValueTypeAst::Some)],
            middle_element: sp(third_input, 8..12, ValueTypeAst::Ident),
            end: vec![sp(third_input, 18..21, ValueTypeAst::Ident)],
        },
    ];

    for sample in samples {
        let input = sample.input;
        let (rest, tuple) = tuple_definition(input).unwrap();

        assert!(rest.fragment().is_empty());
        assert_eq!(tuple.start, sample.start);
        assert_eq!(*tuple.middle.unwrap().extra.element, sample.middle_element);
        assert_eq!(tuple.end, sample.end);
    }
}

#[test]
fn embedded_complex_tuple() {
    let input = InputSpan::new("(Num, ...[Num; _], (...['T; N], () -> 'T))");
    let (rest, tuple) = tuple_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(
        tuple.start.as_slice(),
        [sp(input, 1..4, ValueTypeAst::Ident)]
    );
    let middle = tuple.middle.unwrap();
    assert_eq!(
        *middle.extra.element,
        sp(input, 10..13, ValueTypeAst::Ident)
    );
    assert_eq!(tuple.end.len(), 1);

    let embedded_tuple = match &tuple.end[0].extra {
        ValueTypeAst::Tuple(tuple) => tuple,
        other => panic!("Unexpected tuple end: {:?}", other),
    };
    assert!(embedded_tuple.start.is_empty());
    assert_eq!(
        *embedded_tuple.middle.as_ref().unwrap().extra.element,
        sp(input, 24..26, ValueTypeAst::Param)
    );
    assert_matches!(embedded_tuple.end[0].extra, ValueTypeAst::Function { .. });
}

#[test]
fn simple_slice_with_length() {
    let input = InputSpan::new("[Num; N]");
    let (rest, slice) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(slice.element.extra, ValueTypeAst::Ident);
    assert_matches!(slice.length, TupleLenAst::Ident(id) if *id.fragment() == "N");
}

#[test]
fn simple_slice_without_length() {
    let input = InputSpan::new("['T]");
    let (rest, slice) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(slice.element.extra, ValueTypeAst::Param);
    assert_eq!(slice.length, TupleLenAst::Dynamic);
}

#[test]
fn complex_slice_type() {
    let input = InputSpan::new("[(Num, Bool); M]");
    let (rest, slice) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(
        slice.element.extra,
        ValueTypeAst::Tuple(TupleAst {
            start: vec![
                sp(input, 2..5, ValueTypeAst::Ident),
                sp(input, 7..11, ValueTypeAst::Ident),
            ],
            middle: None,
            end: vec![],
        })
    );
    assert_matches!(slice.length, TupleLenAst::Ident(ident) if *ident.fragment() == "M");
}

#[test]
fn embedded_slice_type() {
    let input = InputSpan::new("[(Num, [Bool]); M]");
    let (rest, slice) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(slice.length, TupleLenAst::Ident(ident) if *ident.fragment() == "M");
    let first_element = match &slice.element.extra {
        ValueTypeAst::Tuple(tuple) => &tuple.start[1].extra,
        _ => panic!("Unexpected slice element: {:?}", slice.element),
    };
    assert_matches!(
        first_element,
        ValueTypeAst::Slice(SliceAst { element, length: TupleLenAst::Dynamic })
            if element.extra == ValueTypeAst::Ident
    );
}

#[test]
fn simple_fn_type() {
    let input = InputSpan::new("() -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(fn_type.args.extra.start.is_empty());
    assert_eq!(fn_type.return_type, sp(input, 6..9, ValueTypeAst::Ident));
}

#[test]
fn simple_fn_type_with_args() {
    let input = InputSpan::new("((Num, Num), Bool) -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(fn_type.args.extra.start.len(), 2);
    assert_eq!(
        fn_type.args.extra.start[0].extra,
        ValueTypeAst::Tuple(TupleAst {
            start: vec![
                sp(input, 2..5, ValueTypeAst::Ident),
                sp(input, 7..10, ValueTypeAst::Ident),
            ],
            middle: None,
            end: vec![],
        })
    );
    assert_eq!(
        fn_type.args.extra.start[1],
        sp(input, 13..17, ValueTypeAst::Ident)
    );
    assert_eq!(fn_type.return_type, sp(input, 22..25, ValueTypeAst::Ident));
}

#[test]
fn fn_type_with_type_params() {
    let input = InputSpan::new("(Bool, 'T, 'T) -> 'T");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(fn_type.args.extra.start.len(), 3);
    assert_eq!(
        fn_type.args.extra.start[1],
        sp(input, 7..9, ValueTypeAst::Param)
    );
    assert_eq!(fn_type.return_type, sp(input, 18..20, ValueTypeAst::Param));
}

#[test]
fn fn_type_accepting_fn_arg() {
    let input = InputSpan::new("(['T; N], ('T) -> Bool) -> Bool");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(fn_type.args.extra.start.len(), 2);
    assert_eq!(fn_type.return_type, sp(input, 27..31, ValueTypeAst::Ident));

    let inner_fn = match &fn_type.args.extra.start[1].extra {
        ValueTypeAst::Function(function) => function.as_ref(),
        ty => panic!("Unexpected arg type: {:?}", ty),
    };
    assert_eq!(
        inner_fn.args.extra.start,
        [sp(input, 11..13, ValueTypeAst::Param)]
    );
    assert_eq!(inner_fn.return_type, sp(input, 18..22, ValueTypeAst::Ident));
}

#[test]
fn fn_type_returning_fn_arg() {
    let input = InputSpan::new("(Num) -> (Bool) -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(
        fn_type.args.extra.start,
        vec![sp(input, 1..4, ValueTypeAst::Ident)]
    );

    let returned_fn = match fn_type.return_type.extra {
        ValueTypeAst::Function(function) => *function,
        ty => panic!("Unexpected return type: {:?}", ty),
    };
    assert_eq!(
        returned_fn.args.extra.start,
        vec![sp(input, 10..14, ValueTypeAst::Ident)]
    );
    assert_eq!(
        returned_fn.return_type,
        sp(input, 19..22, ValueTypeAst::Ident)
    );
}

#[test]
fn fn_type_with_rest_params() {
    let input = InputSpan::new("(Bool, ...[Num; N]) -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(fn_type.args.extra.start.len(), 1);
    assert_eq!(
        fn_type.args.extra.start[0],
        sp(input, 1..5, ValueTypeAst::Ident)
    );
    let middle = fn_type.args.extra.middle.unwrap().extra;
    assert_eq!(*middle.element, sp(input, 11..14, ValueTypeAst::Ident));
    assert_matches!(middle.length, TupleLenAst::Ident(id) if *id.fragment() == "N");
}

#[test]
fn fn_type_with_constraints() {
    let input = InputSpan::new("for<'T: Lin> ('T) -> 'T");
    let (rest, ty) = fn_definition_with_constraints(input).unwrap();
    let (constraints, fn_type) = match ty {
        ValueTypeAst::FunctionWithConstraints {
            constraints,
            function,
        } => (constraints.extra, function.extra),
        _ => panic!("Unexpected type: {:?}", ty),
    };

    assert!(rest.fragment().is_empty());
    assert!(constraints.static_lengths.is_empty());
    assert_eq!(constraints.type_params.len(), 1);
    assert_eq!(fn_type.args.extra.start.len(), 1);
    assert_eq!(
        fn_type.args.extra.start[0],
        sp(input, 14..16, ValueTypeAst::Param)
    );
    assert_eq!(fn_type.return_type, sp(input, 21..23, ValueTypeAst::Param));
}

#[test]
fn multiple_fns_with_constraints() {
    let input = InputSpan::new("(for<'T: Lin> ('T) -> 'T, for<'T: Lin> (...['T; _]) -> ())");
    let (rest, ty) = type_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    let (second_fn, first_fn) = match ty {
        ValueTypeAst::Tuple(TupleAst { mut start, .. }) if start.len() == 2 => {
            (start.pop().unwrap().extra, start.pop().unwrap().extra)
        }
        _ => panic!("Unexpected type: {:?}", ty),
    };

    let first_fn = match first_fn {
        ValueTypeAst::FunctionWithConstraints { function, .. } => function.extra,
        _ => panic!("Unexpected 1st function: {:?}", first_fn),
    };
    assert_eq!(first_fn.return_type, sp(input, 22..24, ValueTypeAst::Param));

    let second_fn = match second_fn {
        ValueTypeAst::FunctionWithConstraints { function, .. } => function.extra,
        _ => panic!("Unexpected 2nd function: {:?}", second_fn),
    };
    assert!(second_fn.args.extra.start.is_empty());
    assert_eq!(
        *second_fn.args.extra.middle.unwrap().extra.element,
        sp(input, 44..46, ValueTypeAst::Param)
    );
    assert_matches!(second_fn.return_type.extra, ValueTypeAst::Tuple(t) if t.start.is_empty());
}

#[test]
fn any_type() {
    let input = InputSpan::new("any");
    let (rest, ty) = type_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(ty, ValueTypeAst::Any(constraints) if constraints.terms.is_empty());
}

#[test]
fn any_type_with_bound() {
    let input = InputSpan::new("any Lin");
    let (rest, ty) = type_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(ty, ValueTypeAst::Any(constraints) if constraints.terms.len() == 1);

    let weird_input = InputSpan::new("anyLin");
    let (rest, ty) = type_definition(weird_input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(ty, ValueTypeAst::Ident);
}
