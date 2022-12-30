//! Unit tests for `Substitutions`.

use assert_matches::assert_matches;

use super::*;
use crate::{
    arith::{ConstraintSet, Linearity, Num, Ops},
    error::TupleContext,
    DynConstraints,
};

fn extract_errors<Prim: PrimitiveType>(
    action: impl FnOnce(OpErrors<'_, Prim>),
) -> Result<(), ErrorKind<Prim>> {
    let mut errors = OpErrors::new();
    action(errors.by_ref());
    let mut errors = errors.into_vec();

    match errors.len() {
        0 => Ok(()),
        1 => Err(errors.pop().unwrap()),
        _ => panic!("Unexpected multiple errors: {errors:?}"),
    }
}

fn unify_lengths<Prim: PrimitiveType>(
    substitutions: &mut Substitutions<Prim>,
    lhs: TupleLen,
    rhs: TupleLen,
) -> Result<(), ErrorKind<Prim>> {
    substitutions
        .unify_lengths(lhs, rhs, TupleContext::Generic)
        .map(drop)
}

fn unify_tuples<Prim: PrimitiveType>(
    substitutions: &mut Substitutions<Prim>,
    lhs: &Tuple<Prim>,
    rhs: &Tuple<Prim>,
) -> Result<(), ErrorKind<Prim>> {
    extract_errors(|errors| {
        substitutions.unify_tuples(lhs, rhs, TupleContext::Generic, errors);
    })
}

fn unify_objects<Prim: PrimitiveType>(
    substitutions: &mut Substitutions<Prim>,
    lhs: &Object<Prim>,
    rhs: &Object<Prim>,
) -> Result<(), ErrorKind<Prim>> {
    extract_errors(|errors| {
        substitutions.unify_objects(lhs, rhs, errors);
    })
}

fn unify<Prim: PrimitiveType>(
    substitutions: &mut Substitutions<Prim>,
    lhs: &Type<Prim>,
    rhs: &Type<Prim>,
) -> Result<(), ErrorKind<Prim>> {
    extract_errors(|errors| substitutions.unify(lhs, rhs, errors))
}

#[test]
fn unifying_lengths_success_without_side_effects() {
    let samples = &[
        (TupleLen::from(2), TupleLen::from(2)),
        (
            UnknownLen::free_var(1).into(),
            UnknownLen::free_var(1).into(),
        ),
        // `Dynamic` should always be accepted on LHS.
        (UnknownLen::Dynamic.into(), TupleLen::from(2)),
        (UnknownLen::Dynamic.into(), UnknownLen::free_var(3).into()),
        (UnknownLen::Dynamic.into(), UnknownLen::Dynamic.into()),
        (UnknownLen::Dynamic.into(), UnknownLen::Dynamic + 1),
    ];

    let mut substitutions = Substitutions::<Num>::default();
    for &(lhs, rhs) in samples {
        unify_lengths(&mut substitutions, lhs, rhs).unwrap();
        assert!(substitutions.length_eqs.is_empty());
    }
}

#[test]
fn unifying_lengths_error() {
    let samples = &[
        (TupleLen::from(1), TupleLen::from(2)),
        (TupleLen::from(2), UnknownLen::Dynamic.into()),
        (UnknownLen::Dynamic + 1, UnknownLen::Dynamic.into()),
        (UnknownLen::free_var(1).into(), UnknownLen::free_var(1) + 1),
        (UnknownLen::free_var(1) + 2, UnknownLen::free_var(1).into()),
    ];

    let mut substitutions = Substitutions::<Num>::default();
    for &(lhs, rhs) in samples {
        let err = unify_lengths(&mut substitutions, lhs, rhs).unwrap_err();

        assert_matches!(err, ErrorKind::TupleLenMismatch { .. });
        assert!(substitutions.length_eqs.is_empty());
    }
}

#[test]
fn unifying_lengths_success_with_new_equation() {
    let len_samples = &[
        TupleLen::from(3),
        UnknownLen::free_var(5).into(),
        UnknownLen::Dynamic.into(),
    ];

    for &rhs in len_samples {
        let mut substitutions = Substitutions::<Num>::default();
        unify_lengths(&mut substitutions, UnknownLen::free_var(2).into(), rhs).unwrap();

        assert_eq!(substitutions.length_eqs.len(), 1);
        assert_eq!(substitutions.length_eqs[&2], rhs);
    }

    let mut substitutions = Substitutions::<Num>::default();
    unify_lengths(
        &mut substitutions,
        TupleLen::from(3),
        UnknownLen::free_var(2).into(),
    )
    .unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&2], TupleLen::from(3));
}

#[test]
fn unifying_compound_length_success() {
    let compound_len = UnknownLen::free_var(0) + 2;
    let mut substitutions = Substitutions::<Num>::default();
    unify_lengths(&mut substitutions, compound_len, TupleLen::from(5)).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], TupleLen::from(3));

    let mut substitutions = Substitutions::<Num>::default();
    unify_lengths(&mut substitutions, TupleLen::from(5), compound_len).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], TupleLen::from(3));

    let other_compound_len = UnknownLen::free_var(1) + 2;
    let mut substitutions = Substitutions::<Num>::default();
    unify_lengths(&mut substitutions, compound_len, other_compound_len).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], UnknownLen::free_var(1).into());
}

#[test]
fn unifying_compound_length_with_dyn_length() {
    let compound_len = UnknownLen::Dynamic + 2;
    let mut substitutions = Substitutions::<Num>::default();

    unify_lengths(&mut substitutions, compound_len, TupleLen::from(5)).unwrap();
    assert!(substitutions.length_eqs.is_empty());

    unify_lengths(&mut substitutions, compound_len, compound_len).unwrap();
    assert!(substitutions.length_eqs.is_empty());

    let err = unify_lengths(&mut substitutions, TupleLen::from(5), compound_len).unwrap_err();
    assert_matches!(err, ErrorKind::TupleLenMismatch { .. });

    let other_compound_len = UnknownLen::free_var(1) + 2;
    unify_lengths(&mut substitutions, compound_len, other_compound_len).unwrap();
    assert!(substitutions.length_eqs.is_empty());

    let other_dyn_len = UnknownLen::Dynamic + 3;
    unify_lengths(&mut substitutions, compound_len, other_dyn_len).unwrap();
    assert!(substitutions.length_eqs.is_empty());

    let err = unify_lengths(&mut substitutions, other_dyn_len, compound_len).unwrap_err();
    assert_matches!(err, ErrorKind::TupleLenMismatch { .. });

    unify_lengths(&mut substitutions, other_compound_len, compound_len).unwrap();
    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&1], UnknownLen::Dynamic.into());
}

#[test]
fn unifying_compound_length_errors() {
    let compound_len = UnknownLen::free_var(0) + 2;
    let mut substitutions = Substitutions::<Num>::default();
    let err = unify_lengths(&mut substitutions, compound_len, TupleLen::from(1)).unwrap_err();

    assert_matches!(err, ErrorKind::TupleLenMismatch { .. });
}

#[test]
fn unresolved_param_error() {
    let mut substitutions = Substitutions::<Num>::default();
    let err = unify_lengths(
        &mut substitutions,
        UnknownLen::param(2).into(),
        TupleLen::from(1),
    )
    .unwrap_err();
    assert_matches!(err, ErrorKind::UnresolvedParam);

    let err = unify(&mut substitutions, &Type::param(2), &Type::NUM).unwrap_err();
    assert_matches!(err, ErrorKind::UnresolvedParam);
}

#[test]
fn unifying_complex_tuples() {
    let xs = Tuple::new(
        vec![Type::NUM, Type::NUM],
        Type::NUM.repeat(UnknownLen::free_var(0)),
        vec![],
    );
    let ys = Tuple::from(Type::NUM.repeat(UnknownLen::free_var(1) + 2));

    let mut substitutions = Substitutions::<Num>::default();
    unify_tuples(&mut substitutions, &xs, &ys).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], UnknownLen::free_var(1).into());

    let zs = Tuple::from(Type::free_var(0).repeat(UnknownLen::free_var(1)));
    let mut substitutions = Substitutions::<Num>::default();
    unify_tuples(&mut substitutions, &xs, &zs).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&1], UnknownLen::free_var(0) + 2);
    assert_eq!(substitutions.eqs.len(), 1);
    assert_eq!(substitutions.eqs[&0], Type::NUM);

    let us = Tuple::new(
        vec![],
        Type::NUM.repeat(UnknownLen::free_var(1)),
        vec![Type::free_var(0), Type::free_var(1), Type::free_var(2)],
    );
    let mut substitutions = Substitutions::<Num>::default();
    unify_tuples(&mut substitutions, &xs, &us).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], UnknownLen::free_var(1) + 1);
    assert_eq!(substitutions.eqs.len(), 3);
    for i in 0..3 {
        assert_eq!(substitutions.eqs[&i], Type::NUM);
    }
}

#[test]
fn any_can_be_unified_with_anything() {
    let mut substitutions = Substitutions::<Num>::default();
    substitutions.eqs.insert(0, Type::Any);

    let rhs_samples = &[
        Type::NUM,
        Type::BOOL,
        (Type::BOOL, Type::NUM).into(),
        Type::NUM.repeat(3).into(),
        Type::NUM.repeat(UnknownLen::free_var(0)).into(),
        Function::new(vec![Type::BOOL, Type::NUM].into(), Type::void()).into(),
        Type::Any,
    ];
    for rhs in rhs_samples {
        unify(&mut substitutions, &Type::free_var(0), rhs).unwrap();
    }
}

#[test]
fn unifying_dyn_type_as_rhs() {
    let mut substitutions = Substitutions::<Num>::default();
    let rhs = Type::Dyn(DynConstraints::just(Linearity));

    unify(&mut substitutions, &Type::Any, &rhs).unwrap();
    assert!(substitutions.eqs.is_empty());
    assert!(substitutions.constraints.is_empty());

    unify(&mut substitutions, &Type::free_var(0), &rhs).unwrap();
    assert_eq!(substitutions.eqs.len(), 1);
    assert_eq!(substitutions.eqs[&0], rhs);

    let invalid_lhs = &[
        Type::NUM,
        Type::BOOL,
        (Type::BOOL, Type::NUM).into(),
        Type::NUM.repeat(3).into(),
        Type::NUM.repeat(UnknownLen::free_var(0)).into(),
        Function::new(vec![Type::BOOL, Type::NUM].into(), Type::void()).into(),
    ];

    for lhs in invalid_lhs {
        let err = unify(&mut substitutions, lhs, &rhs).unwrap_err();
        assert_matches!(err, ErrorKind::TypeMismatch(_, rhs_ty) if rhs_ty == rhs);
    }
}

#[test]
fn unifying_dyn_type_as_lhs() {
    let constraints = DynConstraints::just(Linearity);
    let lhs = Type::Dyn(constraints.clone());
    let valid_rhs = &[Type::Any, Type::NUM, Type::NUM.repeat(3).into()];

    for rhs in valid_rhs {
        let mut substitutions = Substitutions::<Num>::default();
        unify(&mut substitutions, &lhs, rhs).unwrap();
        assert!(substitutions.eqs.is_empty());
        assert!(substitutions.constraints.is_empty());
    }

    // RHS with type vars.
    {
        let rhs = Type::free_var(0);
        let mut substitutions = Substitutions::<Num>::default();
        unify(&mut substitutions, &lhs, &rhs).unwrap();
        assert!(substitutions.eqs.is_empty());
        assert_eq!(substitutions.constraints.len(), 1);
        assert_eq!(substitutions.constraints[&0], constraints.inner);
    }
    {
        let rhs = (Type::free_var(0), Type::free_var(1)).into();
        let mut substitutions = Substitutions::<Num>::default();
        unify(&mut substitutions, &lhs, &rhs).unwrap();
        assert!(substitutions.eqs.is_empty());
        assert_eq!(substitutions.constraints.len(), 2);
        assert_eq!(substitutions.constraints[&0], constraints.inner);
        assert_eq!(substitutions.constraints[&1], constraints.inner);
    }

    // `dyn` RHS.
    {
        let mut substitutions = Substitutions::<Num>::default();
        unify(&mut substitutions, &lhs, &lhs).unwrap();
        assert!(substitutions.eqs.is_empty());
        assert!(substitutions.constraints.is_empty());
    }
    {
        // We cheat here a little bit: `Ops` is not object-safe.
        let mut extended_constraints = ConstraintSet::new();
        extended_constraints.insert(Linearity);
        extended_constraints.insert(Ops);
        let rhs = Type::Dyn(DynConstraints {
            inner: extended_constraints.into(),
        });

        let mut substitutions = Substitutions::<Num>::default();
        unify(&mut substitutions, &lhs, &rhs).unwrap();
        assert!(substitutions.eqs.is_empty());
        assert!(substitutions.constraints.is_empty());
    }
    {
        let ops_constraint = ConstraintSet::just(Ops);
        let rhs = Type::Dyn(DynConstraints {
            inner: ops_constraint.into(),
        });
        let mut substitutions = Substitutions::<Num>::default();
        let err = unify(&mut substitutions, &lhs, &rhs).unwrap_err();
        assert_matches!(err, ErrorKind::FailedConstraint { .. });
    }

    let invalid_rhs = &[
        Type::BOOL,
        (Type::NUM, Type::BOOL).into(),
        Function::new(vec![Type::BOOL, Type::NUM].into(), Type::void()).into(),
    ];

    for rhs in invalid_rhs {
        let mut substitutions = Substitutions::<Num>::default();
        let err = unify(&mut substitutions, &lhs, rhs).unwrap_err();
        assert_matches!(err, ErrorKind::FailedConstraint { .. });
    }
}

#[test]
fn unifying_dyn_object_as_lhs() {
    let constraints = DynConstraints::from(Object::just("x", Type::NUM));
    let lhs = Type::Dyn(constraints.clone());

    {
        let mut substitutions = Substitutions::<Num>::default();
        unify(&mut substitutions, &lhs, &Type::Any).unwrap();
        assert!(substitutions.eqs.is_empty());
        assert!(substitutions.constraints.is_empty());
    }
    {
        let mut substitutions = Substitutions::<Num>::default();
        unify(&mut substitutions, &lhs, &Type::free_var(0)).unwrap();
        assert!(substitutions.eqs.is_empty());
        assert_eq!(substitutions.constraints.len(), 1);
        assert_eq!(substitutions.constraints[&0], constraints.inner);
    }

    // Object RHS.
    {
        let rhs = Object::just("x", Type::BOOL);
        let mut substitutions = Substitutions::<Num>::default();
        let err = unify(&mut substitutions, &lhs, &rhs.into()).unwrap_err();
        assert_matches!(
            err,
            ErrorKind::TypeMismatch(lhs, rhs) if lhs == Type::NUM && rhs == Type::BOOL
        );
    }
    {
        let rhs = Object::just("y", Type::NUM);
        let mut substitutions = Substitutions::<Num>::default();
        let err = unify(&mut substitutions, &lhs, &rhs.into()).unwrap_err();
        assert_matches!(
            err,
            ErrorKind::MissingFields { fields, .. } if fields.contains("x")
        );
    }
    {
        let rhs = Object::just("x", Type::free_var(0));
        let mut substitutions = Substitutions::<Num>::default();
        unify(&mut substitutions, &lhs, &rhs.into()).unwrap();
        assert_eq!(substitutions.eqs[&0], Type::NUM);
    }

    // Dyn constraint RHS.
    {
        let mut substitutions = Substitutions::<Num>::default();
        unify(&mut substitutions, &lhs, &lhs).unwrap();
        assert!(substitutions.eqs.is_empty());
        assert!(substitutions.constraints.is_empty());
    }
    {
        let rhs = Object::just("y", Type::NUM).into_dyn();
        let mut substitutions = Substitutions::<Num>::default();
        let err = unify(&mut substitutions, &lhs, &rhs).unwrap_err();
        assert_matches!(err, ErrorKind::MissingFields { fields, .. } if fields.contains("x"));
    }
    {
        let rhs = Object::just("x", Type::BOOL).into_dyn();
        let mut substitutions = Substitutions::<Num>::default();
        let err = unify(&mut substitutions, &lhs, &rhs).unwrap_err();
        assert_matches!(
            err,
            ErrorKind::TypeMismatch(lhs_ty, rhs_ty)
                if lhs_ty == Type::NUM && rhs_ty == Type::BOOL
        );
    }
    {
        let rhs = Object::just("x", Type::free_var(0)).into_dyn();
        let mut substitutions = Substitutions::<Num>::default();
        unify(&mut substitutions, &lhs, &rhs).unwrap();
        assert_eq!(substitutions.eqs[&0], Type::NUM);
    }
    {
        let rhs = Type::Dyn(DynConstraints::just(Linearity));
        let mut substitutions = Substitutions::<Num>::default();
        let err = unify(&mut substitutions, &lhs, &rhs).unwrap_err();
        assert_matches!(err, ErrorKind::CannotAccessFields);
    }

    let invalid_rhs = &[
        Type::NUM,
        Type::BOOL,
        (Type::BOOL, Type::NUM).into(),
        Type::NUM.repeat(3).into(),
        Type::NUM.repeat(UnknownLen::free_var(0)).into(),
        Function::new(vec![Type::BOOL, Type::NUM].into(), Type::void()).into(),
    ];
    for rhs in invalid_rhs {
        let mut substitutions = Substitutions::<Num>::default();
        let err = unify(&mut substitutions, &lhs, rhs).unwrap_err();
        assert_matches!(err, ErrorKind::CannotAccessFields);
    }
}

#[test]
fn static_length_restrictions() {
    let mut substitutions = Substitutions::<Num>::default();

    let positive_samples = &[
        TupleLen::from(0),
        TupleLen::from(3),
        UnknownLen::free_var(0).into(),
        UnknownLen::free_var(1) + 2,
    ];
    for &sample in positive_samples {
        substitutions.apply_static_len(sample).unwrap();
    }
    assert_eq!(
        substitutions.static_lengths,
        IntoIterator::into_iter([0, 1]).collect::<HashSet<_>>()
    );

    let negative_samples = &[UnknownLen::Dynamic.into(), UnknownLen::Dynamic + 2];
    for &sample in negative_samples {
        let err = substitutions.apply_static_len(sample).unwrap_err();
        assert_matches!(err, ErrorKind::DynamicLen(_));
    }
}

#[test]
fn marking_length_as_static_and_then_failing_unification() {
    let mut substitutions = Substitutions::<Num>::default();
    substitutions.static_lengths.insert(0);

    let positive_samples: &[(TupleLen, _)] = &[
        (UnknownLen::Dynamic.into(), UnknownLen::free_var(0).into()),
        (UnknownLen::Dynamic.into(), UnknownLen::free_var(0) + 1),
    ];
    for &(lhs, rhs) in positive_samples {
        unify_lengths(&mut substitutions, lhs, rhs).unwrap();
        assert!(substitutions.length_eqs.is_empty());
    }

    let failing_samples: &[(TupleLen, _)] = &[
        (UnknownLen::free_var(0).into(), UnknownLen::Dynamic + 2),
        (UnknownLen::free_var(0).into(), UnknownLen::Dynamic.into()),
        (UnknownLen::free_var(0) + 1, UnknownLen::Dynamic + 2),
    ];
    for &(lhs, rhs) in failing_samples {
        let err = unify_lengths(&mut substitutions, lhs, rhs).unwrap_err();
        assert_matches!(err, ErrorKind::DynamicLen(_));
    }
}

#[test]
fn marking_length_as_static_and_then_propagating() {
    let mut substitutions = Substitutions::<Num>::default();
    substitutions.static_lengths.insert(0);
    unify_lengths(
        &mut substitutions,
        UnknownLen::free_var(1).into(),
        UnknownLen::free_var(0).into(),
    )
    .unwrap();

    assert_eq!(substitutions.length_eqs[&1], UnknownLen::free_var(0).into());
    assert_eq!(substitutions.static_lengths.len(), 1);

    let mut substitutions = Substitutions::<Num>::default();
    substitutions.static_lengths.insert(0);
    unify_lengths(
        &mut substitutions,
        UnknownLen::free_var(0).into(),
        UnknownLen::free_var(1).into(),
    )
    .unwrap();

    assert_eq!(substitutions.length_eqs[&0], UnknownLen::free_var(1).into());
    assert!(substitutions.static_lengths.contains(&1));
}

#[test]
fn unifying_objects() {
    let x: Object<Num> = IntoIterator::into_iter([("x", Type::NUM), ("y", Type::NUM)]).collect();
    let y: Object<Num> =
        IntoIterator::into_iter([("x", Type::NUM), ("y", Type::free_var(0))]).collect();

    let mut substitutions = Substitutions::<Num>::default();
    unify_objects(&mut substitutions, &x, &y).unwrap();
    assert_eq!(substitutions.eqs.len(), 1);
    assert_eq!(substitutions.eqs[&0], Type::NUM);

    let truncated_y = IntoIterator::into_iter([("x", Type::NUM)]).collect();
    let mut substitutions = Substitutions::<Num>::default();
    let err = unify_objects(&mut substitutions, &x, &truncated_y).unwrap_err();
    assert_matches!(err, ErrorKind::FieldsMismatch { .. });

    let extended_y =
        IntoIterator::into_iter([("x", Type::NUM), ("y", Type::NUM), ("z", Type::BOOL)]).collect();
    let mut substitutions = Substitutions::<Num>::default();
    let err = unify_objects(&mut substitutions, &x, &extended_y).unwrap_err();
    assert_matches!(err, ErrorKind::FieldsMismatch { .. });
}
