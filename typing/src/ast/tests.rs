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
            sp(input, 1..4, TypeAst::Ident),
            sp(input, 6..7, TypeAst::Some),
            sp(input, 9..13, TypeAst::Ident),
            sp(input, 15..16, TypeAst::Some),
        ]
    );
}

#[derive(Debug)]
struct TupleSample {
    input: InputSpan<'static>,
    start: Vec<SpannedTypeAst<'static>>,
    middle_element: SpannedTypeAst<'static>,
    end: Vec<SpannedTypeAst<'static>>,
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
                sp(first_input, 1..4, TypeAst::Ident),
                sp(first_input, 6..7, TypeAst::Some),
            ],
            middle_element: sp(first_input, 13..17, TypeAst::Ident),
            end: vec![],
        },
        TupleSample {
            input: second_input,
            start: vec![],
            middle_element: sp(second_input, 5..9, TypeAst::Ident),
            end: vec![sp(second_input, 15..18, TypeAst::Ident)],
        },
        TupleSample {
            input: third_input,
            start: vec![sp(third_input, 1..2, TypeAst::Some)],
            middle_element: sp(third_input, 8..12, TypeAst::Ident),
            end: vec![sp(third_input, 18..21, TypeAst::Ident)],
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
    assert_eq!(tuple.start.as_slice(), [sp(input, 1..4, TypeAst::Ident)]);
    let middle = tuple.middle.unwrap();
    assert_eq!(*middle.extra.element, sp(input, 10..13, TypeAst::Ident));
    assert_eq!(tuple.end.len(), 1);
    let end = &tuple.end[0];

    let TypeAst::Tuple(embedded_tuple) = &end.extra else {
        panic!("Unexpected tuple end: {end:?}");
    };
    assert!(embedded_tuple.start.is_empty());
    assert_eq!(
        *embedded_tuple.middle.as_ref().unwrap().extra.element,
        sp(input, 24..26, TypeAst::Param)
    );
    assert_matches!(embedded_tuple.end[0].extra, TypeAst::Function { .. });
}

#[test]
fn simple_slice_with_length() {
    let input = InputSpan::new("[Num; N]");
    let (rest, slice) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(slice.element.extra, TypeAst::Ident);
    assert_eq!(slice.length, sp(input, 6..7, TupleLenAst::Ident));
}

#[test]
fn simple_slice_without_length() {
    let input = InputSpan::new("['T]");
    let (rest, slice) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(slice.element.extra, TypeAst::Param);
    assert_eq!(slice.length, sp(input, 3..3, TupleLenAst::Dynamic));
}

#[test]
fn complex_slice_type() {
    let input = InputSpan::new("[(Num, Bool); M]");
    let (rest, slice) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(
        slice.element.extra,
        TypeAst::Tuple(TupleAst {
            start: vec![
                sp(input, 2..5, TypeAst::Ident),
                sp(input, 7..11, TypeAst::Ident),
            ],
            middle: None,
            end: vec![],
        })
    );
    assert_eq!(slice.length, sp(input, 14..15, TupleLenAst::Ident));
}

#[test]
fn embedded_slice_type() {
    let input = InputSpan::new("[(Num, [Bool]); M]");
    let (rest, slice) = slice_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(slice.length, sp(input, 16..17, TupleLenAst::Ident));
    let first_element = match &slice.element.extra {
        TypeAst::Tuple(tuple) => &tuple.start[1].extra,
        _ => panic!("Unexpected slice element: {:?}", slice.element),
    };
    assert_matches!(
        first_element,
        TypeAst::Slice(SliceAst { element, length })
            if element.extra == TypeAst::Ident && length.extra == TupleLenAst::Dynamic
    );
}

#[test]
fn simple_fn_type() {
    let input = InputSpan::new("() -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert!(fn_type.args.extra.start.is_empty());
    assert_eq!(fn_type.return_type, sp(input, 6..9, TypeAst::Ident));
}

#[test]
fn simple_fn_type_with_args() {
    let input = InputSpan::new("((Num, Num), Bool) -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(fn_type.args.extra.start.len(), 2);
    assert_eq!(
        fn_type.args.extra.start[0].extra,
        TypeAst::Tuple(TupleAst {
            start: vec![
                sp(input, 2..5, TypeAst::Ident),
                sp(input, 7..10, TypeAst::Ident),
            ],
            middle: None,
            end: vec![],
        })
    );
    assert_eq!(
        fn_type.args.extra.start[1],
        sp(input, 13..17, TypeAst::Ident)
    );
    assert_eq!(fn_type.return_type, sp(input, 22..25, TypeAst::Ident));
}

#[test]
fn fn_type_with_type_params() {
    let input = InputSpan::new("(Bool, 'T, 'T) -> 'T");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(fn_type.args.extra.start.len(), 3);
    assert_eq!(fn_type.args.extra.start[1], sp(input, 7..9, TypeAst::Param));
    assert_eq!(fn_type.return_type, sp(input, 18..20, TypeAst::Param));
}

#[test]
fn fn_type_accepting_fn_arg() {
    let input = InputSpan::new("(['T; N], ('T) -> Bool) -> Bool");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(fn_type.args.extra.start.len(), 2);
    assert_eq!(fn_type.return_type, sp(input, 27..31, TypeAst::Ident));

    let TypeAst::Function(inner_fn) = &fn_type.args.extra.start[1].extra else {
        panic!("Unexpected arg type: {fn_type:?}");
    };
    assert_eq!(
        inner_fn.args.extra.start,
        [sp(input, 11..13, TypeAst::Param)]
    );
    assert_eq!(inner_fn.return_type, sp(input, 18..22, TypeAst::Ident));
}

#[test]
fn fn_type_returning_fn_arg() {
    let input = InputSpan::new("(Num) -> (Bool) -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(
        fn_type.args.extra.start,
        vec![sp(input, 1..4, TypeAst::Ident)]
    );

    let TypeAst::Function(returned_fn) = fn_type.return_type.extra else {
        panic!("Unexpected return type: {fn_type:?}");
    };
    assert_eq!(
        returned_fn.args.extra.start,
        vec![sp(input, 10..14, TypeAst::Ident)]
    );
    assert_eq!(returned_fn.return_type, sp(input, 19..22, TypeAst::Ident));
}

#[test]
fn fn_type_with_rest_params() {
    let input = InputSpan::new("(Bool, ...[Num; N]) -> Num");
    let (rest, fn_type) = fn_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(fn_type.args.extra.start.len(), 1);
    assert_eq!(fn_type.args.extra.start[0], sp(input, 1..5, TypeAst::Ident));
    let middle = fn_type.args.extra.middle.unwrap().extra;
    assert_eq!(*middle.element, sp(input, 11..14, TypeAst::Ident));
    assert_eq!(middle.length, sp(input, 16..17, TupleLenAst::Ident));
}

#[test]
fn fn_type_with_constraints() {
    let input = InputSpan::new("for<'T: Lin> ('T) -> 'T");
    let (rest, ty) = fn_definition_with_constraints(input).unwrap();
    let TypeAst::FunctionWithConstraints {
        constraints,
        function,
    } = ty
    else {
        panic!("Unexpected type: {ty:?}");
    };
    let constraints = constraints.extra;
    let fn_type = function.extra;

    assert!(rest.fragment().is_empty());
    assert!(constraints.static_lengths.is_empty());
    assert_eq!(constraints.type_params.len(), 1);
    assert_eq!(fn_type.args.extra.start.len(), 1);
    assert_eq!(
        fn_type.args.extra.start[0],
        sp(input, 14..16, TypeAst::Param)
    );
    assert_eq!(fn_type.return_type, sp(input, 21..23, TypeAst::Param));
}

#[test]
fn multiple_fns_with_constraints() {
    let input = InputSpan::new("(for<'T: Lin> ('T) -> 'T, for<'T: Lin> (...['T; _]) -> ())");
    let (rest, ty) = type_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    let (second_fn, first_fn) = match ty {
        TypeAst::Tuple(TupleAst { mut start, .. }) if start.len() == 2 => {
            (start.pop().unwrap().extra, start.pop().unwrap().extra)
        }
        _ => panic!("Unexpected type: {ty:?}"),
    };

    let TypeAst::FunctionWithConstraints {
        function: first_fn, ..
    } = first_fn
    else {
        panic!("Unexpected 1st function: {first_fn:?}");
    };
    assert_eq!(
        first_fn.extra.return_type,
        sp(input, 22..24, TypeAst::Param)
    );

    let TypeAst::FunctionWithConstraints {
        function: second_fn,
        ..
    } = second_fn
    else {
        panic!("Unexpected 2nd function: {second_fn:?}");
    };
    let FunctionAst { args, return_type } = second_fn.extra;
    assert!(args.extra.start.is_empty());
    assert_eq!(
        *args.extra.middle.unwrap().extra.element,
        sp(input, 44..46, TypeAst::Param)
    );
    assert_matches!(return_type.extra, TypeAst::Tuple(t) if t.start.is_empty());
}

#[test]
fn any_type() {
    let input = InputSpan::new("any");
    let (rest, ty) = type_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(ty, TypeAst::Any);
}

#[test]
fn dyn_type_with_bound() {
    let input = InputSpan::new("dyn Lin");
    let (rest, ty) = type_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_matches!(ty, TypeAst::Dyn(constraints) if constraints.terms.len() == 1);

    let weird_input = InputSpan::new("dynLin");
    let (rest, ty) = type_definition(weird_input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(ty, TypeAst::Ident);
}

#[test]
fn dyn_type_with_object_bound() {
    let input = InputSpan::new("dyn { x: Num } + Lin");
    let (rest, ty) = type_definition(input).unwrap();

    assert!(rest.fragment().is_empty());
    let TypeAst::Dyn(constraints) = ty else {
        panic!("Unexpected type: {ty:?}");
    };
    assert_eq!(constraints.terms.len(), 1);
    let object = constraints.object.unwrap();
    assert_eq!(object.fields.len(), 1);
    let (field_name, field_ty) = &object.fields[0];
    assert_eq!(*field_name.fragment(), "x");
    assert_matches!(field_ty.extra, TypeAst::Ident);
}

#[test]
fn any_type_in_cast_chain() {
    let input = InputSpan::new("any as Num");
    let (rest, ty) = type_definition(input).unwrap();

    assert_eq!(*rest.fragment(), " as Num");
    assert_matches!(ty, TypeAst::Any);
}

#[test]
fn object_types() {
    let input = InputSpan::new("{ x: Num, y: [(Num, 'T)] }");
    let (rest, ty) = object(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(ty.fields.len(), 2);
    let (first_name, first_ty) = &ty.fields[0];
    assert_eq!(*first_name.fragment(), "x");
    assert_eq!(first_ty.extra, TypeAst::Ident);
    let (second_name, second_ty) = &ty.fields[1];
    assert_eq!(*second_name.fragment(), "y");
    assert_matches!(second_ty.extra, TypeAst::Slice(_));
}

#[test]
fn object_constraints() {
    let obj_input = InputSpan::new("{ len: Num }");
    let (obj_rest, obj_constraints) = type_bounds(obj_input).unwrap();
    assert!(obj_rest.fragment().is_empty());
    assert_eq!(obj_constraints.object.unwrap().fields.len(), 1);
    assert!(obj_constraints.terms.is_empty());

    let mixed_input = InputSpan::new("{ len: Num } + Ops");
    let (mixed_rest, mixed_constraints) = type_bounds(mixed_input).unwrap();
    assert!(mixed_rest.fragment().is_empty());
    assert_eq!(mixed_constraints.object.unwrap().fields.len(), 1);
    assert_eq!(mixed_constraints.terms.len(), 1);
}

#[test]
fn object_in_type_param_constraints() {
    let input = InputSpan::new("for<'T: Lin, 'U: { x: 'T } + Lin>");
    let (rest, constraints) = constraints(input).unwrap();

    assert!(rest.fragment().is_empty());
    assert_eq!(constraints.type_params.len(), 2);
    let (type_var, constraints) = &constraints.type_params[1];
    assert_eq!(*type_var.fragment(), "U");
    assert_eq!(constraints.object.as_ref().unwrap().fields.len(), 1);
    assert_eq!(constraints.terms.len(), 1);
    assert_eq!(*constraints.terms[0].fragment(), "Lin");
}
