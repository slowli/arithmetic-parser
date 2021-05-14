//! Hub for integration tests.

use std::fmt;

use arithmetic_parser::grammars::{NumGrammar, Typed};
use arithmetic_typing::{
    arith::{Constraint, ConstraintSet},
    error::{Error, ErrorKind, Errors, OpErrors},
    Annotated, FnType, Num, PrimitiveType, Substitutions, Type, UnknownLen,
};

mod annotations;
mod basics;
mod errors;
mod examples;
mod length_eqs;
mod object;

type F32Grammar = Typed<Annotated<NumGrammar<f32>>>;

trait ErrorsExt<'a, Prim: PrimitiveType> {
    fn single(self) -> Error<'a, Prim>;
}

impl<'a, Prim: PrimitiveType> ErrorsExt<'a, Prim> for Errors<'a, Prim> {
    fn single(self) -> Error<'a, Prim> {
        if self.len() == 1 {
            self.into_iter().next().unwrap()
        } else {
            panic!("Expected 1 error, got {:?}", self);
        }
    }
}

/// Constraint for types that can be hashed.
#[derive(Debug, Clone, Copy)]
struct Hashed;

impl fmt::Display for Hashed {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Hash")
    }
}

impl<Prim: PrimitiveType> Constraint<Prim> for Hashed {
    fn apply(
        &self,
        ty: &Type<Prim>,
        substitutions: &mut Substitutions<Prim>,
        errors: OpErrors<'_, Prim>,
    ) {
        use arithmetic_typing::arith::StructConstraint;

        StructConstraint::new(*self, |_| true).apply(ty, substitutions, errors)
    }

    fn clone_boxed(&self) -> Box<dyn Constraint<Prim>> {
        Box::new(*self)
    }
}

fn assert_incompatible_types<Prim: PrimitiveType>(
    err: &ErrorKind<Prim>,
    first: &Type<Prim>,
    second: &Type<Prim>,
) {
    let (x, y) = match err {
        ErrorKind::TypeMismatch(x, y) => (x, y),
        _ => panic!("Unexpected error type: {:?}", err),
    };
    assert!(
        (x == first && y == second) || (x == second && y == first),
        "Unexpected incompatible types: {:?}, expected: {:?}",
        (x, y),
        (first, second)
    );
}

fn hash_fn_type() -> FnType<Num> {
    FnType::builder()
        .with_varargs(ConstraintSet::just(Hashed), UnknownLen::param(0))
        .returning(Type::NUM)
}

#[test]
fn hash_fn_type_display() {
    assert_eq!(hash_fn_type().to_string(), "(...[any Hash; N]) -> Num");
}

/// `zip` function signature.
fn zip_fn_type() -> FnType<Num> {
    FnType::builder()
        .with_arg(Type::param(0).repeat(UnknownLen::param(0)))
        .with_arg(Type::param(1).repeat(UnknownLen::param(0)))
        .returning(Type::slice(
            (Type::param(0), Type::param(1)),
            UnknownLen::param(0),
        ))
        .with_static_lengths(&[0])
        .into()
}

#[test]
fn zip_fn_type_display() {
    let zip_fn_string = zip_fn_type().to_string();
    assert_eq!(
        zip_fn_string,
        "for<len! N> (['T; N], ['U; N]) -> [('T, 'U); N]"
    );
}
