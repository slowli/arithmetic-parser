//! Hub for integration tests.

use std::fmt;

use arithmetic_parser::grammars::NumGrammar;
use arithmetic_typing::{
    arith::{Constraint, Num, ObjectSafeConstraint, Substitutions},
    error::{Error, ErrorKind, Errors, OpErrors},
    visit::Visit,
    Annotated, DynConstraints, Function, PrimitiveType, Type, UnknownLen,
};

mod annotations;
mod basics;
mod errors;
mod examples;
mod length_eqs;
mod object;

type F32Grammar = Annotated<NumGrammar<f32>>;

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
    fn visitor<'r>(
        &self,
        substitutions: &'r mut Substitutions<Prim>,
        errors: OpErrors<'r, Prim>,
    ) -> Box<dyn Visit<Prim> + 'r> {
        use arithmetic_typing::arith::StructConstraint;

        StructConstraint::new(*self, |_| true).visitor(substitutions, errors)
    }

    fn clone_boxed(&self) -> Box<dyn Constraint<Prim>> {
        Box::new(*self)
    }
}

impl<Prim: PrimitiveType> ObjectSafeConstraint<Prim> for Hashed {}

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

fn hash_fn_type() -> Function<Num> {
    Function::builder()
        .with_varargs(DynConstraints::just(Hashed), UnknownLen::param(0))
        .returning(Type::NUM)
}

#[test]
fn hash_fn_type_display() {
    assert_eq!(hash_fn_type().to_string(), "(...[dyn Hash; N]) -> Num");
}

/// `zip` function signature.
fn zip_fn_type() -> Function<Num> {
    Function::builder()
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
