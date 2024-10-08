//! `TypeEnvironment` and related types.

use core::ops;

use arithmetic_parser::{grammars::Grammar, Block};

use self::processor::TypeProcessor;
use crate::{
    alloc::{HashMap, String, ToOwned},
    arith::{
        Constraint, ConstraintSet, MapPrimitiveType, Num, NumArithmetic, ObjectSafeConstraint,
        Substitutions, TypeArithmetic,
    },
    ast::TypeAst,
    error::Errors,
    types::{ParamConstraints, ParamQuantifier},
    visit::VisitMut,
    Function, PrimitiveType, Type,
};

mod processor;

/// Environment containing type information on named variables.
///
/// # Examples
///
/// See [the crate docs](index.html#examples) for examples of usage.
///
/// # Concrete and partially specified types
///
/// The environment retains full info on the types even if the type is not
/// [concrete](Type::is_concrete()). Non-concrete types are tied to an environment.
/// An environment will panic on inserting a non-concrete type via [`Self::insert()`]
/// or other methods.
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse};
/// # use arithmetic_typing::{defs::Prelude, Annotated, TypeEnvironment};
/// # type Parser = Annotated<F32Grammar>;
/// # fn main() -> anyhow::Result<()> {
/// // An easy way to get a non-concrete type is to involve `any`.
/// let code = "(x, ...) = (1, 2, 3) as any;";
/// let code = Parser::parse_statements(code)?;
///
/// let mut env: TypeEnvironment = Prelude::iter().collect();
/// env.process_statements(&code)?;
/// assert!(!env["x"].is_concrete());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct TypeEnvironment<Prim: PrimitiveType = Num> {
    pub(crate) substitutions: Substitutions<Prim>,
    pub(crate) known_constraints: ConstraintSet<Prim>,
    variables: HashMap<String, Type<Prim>>,
}

impl<Prim: PrimitiveType> Default for TypeEnvironment<Prim> {
    fn default() -> Self {
        Self {
            variables: HashMap::new(),
            known_constraints: Prim::well_known_constraints(),
            substitutions: Substitutions::default(),
        }
    }
}

impl<Prim: PrimitiveType> TypeEnvironment<Prim> {
    /// Creates an empty environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Gets type of the specified variable.
    pub fn get(&self, name: &str) -> Option<&Type<Prim>> {
        self.variables.get(name)
    }

    /// Iterates over variables contained in this env.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Type<Prim>)> + '_ {
        self.variables.iter().map(|(name, ty)| (name.as_str(), ty))
    }

    fn prepare_type(ty: impl Into<Type<Prim>>) -> Type<Prim> {
        let mut ty = ty.into();
        assert!(ty.is_concrete(), "Type {ty} is not concrete");
        TypePreparer.visit_type_mut(&mut ty);
        ty
    }

    /// Sets type of a variable.
    ///
    /// # Panics
    ///
    /// - Will panic if `ty` is not [concrete](Type::is_concrete()). Non-concrete
    ///   types are tied to the environment; inserting them into an env is a logical error.
    pub fn insert(&mut self, name: &str, ty: impl Into<Type<Prim>>) -> &mut Self {
        self.variables
            .insert(name.to_owned(), Self::prepare_type(ty));
        self
    }

    /// Inserts a [`Constraint`] into the environment so that it can be used when parsing
    /// type annotations.
    ///
    /// Adding a constraint is not mandatory for it to be usable during type inference;
    /// this method only influences whether the constraint is recognized during type parsing.
    pub fn insert_constraint(&mut self, constraint: impl Constraint<Prim>) -> &mut Self {
        self.known_constraints.insert(constraint);
        self
    }

    /// Inserts an [`ObjectSafeConstraint`] into the environment so that it can be used
    /// when parsing type annotations.
    ///
    /// Other than more strict type requirements, this method is identical to
    /// [`Self::insert_constraint`].
    pub fn insert_object_safe_constraint(
        &mut self,
        constraint: impl ObjectSafeConstraint<Prim>,
    ) -> &mut Self {
        self.known_constraints.insert_object_safe(constraint);
        self
    }

    /// Processes statements with the default type arithmetic. After processing, the environment
    /// will contain type info about newly declared vars.
    ///
    /// This method is a shortcut for calling `process_with_arithmetic` with
    /// [`NumArithmetic::without_comparisons()`].
    pub fn process_statements<'a, T>(
        &mut self,
        block: &Block<'a, T>,
    ) -> Result<Type<Prim>, Errors<Prim>>
    where
        T: Grammar<Type<'a> = TypeAst<'a>>,
        NumArithmetic: MapPrimitiveType<T::Lit, Prim = Prim> + TypeArithmetic<Prim>,
    {
        self.process_with_arithmetic(&NumArithmetic::without_comparisons(), block)
    }

    /// Processes statements with a given `arithmetic`. After processing, the environment
    /// will contain type info about newly declared vars.
    ///
    /// # Errors
    ///
    /// Even if there are any type errors, all statements in the `block` will be executed
    /// to completion and all errors will be reported. However, the environment will **not**
    /// include any vars beyond the first failing statement.
    pub fn process_with_arithmetic<'a, T, A>(
        &mut self,
        arithmetic: &A,
        block: &Block<'a, T>,
    ) -> Result<Type<Prim>, Errors<Prim>>
    where
        T: Grammar<Type<'a> = TypeAst<'a>>,
        A: MapPrimitiveType<T::Lit, Prim = Prim> + TypeArithmetic<Prim>,
    {
        TypeProcessor::new(self, arithmetic).process_statements(block)
    }
}

impl<Prim: PrimitiveType> ops::Index<&str> for TypeEnvironment<Prim> {
    type Output = Type<Prim>;

    fn index(&self, name: &str) -> &Self::Output {
        self.get(name)
            .unwrap_or_else(|| panic!("Variable `{name}` is not defined"))
    }
}

/// Fills in parameters in all encountered top-level functions within a type.
#[derive(Debug)]
struct TypePreparer;

impl<Prim: PrimitiveType> VisitMut<Prim> for TypePreparer {
    fn visit_function_mut(&mut self, function: &mut Function<Prim>) {
        if function.params.is_none() {
            ParamQuantifier::fill_params(function, ParamConstraints::default());
        }
        // We intentionally do not recurse into functions; this is done within `ParamQuantifier`.
    }
}

fn convert_iter<Prim: PrimitiveType, S, Ty, I>(
    iter: I,
) -> impl Iterator<Item = (String, Type<Prim>)>
where
    I: IntoIterator<Item = (S, Ty)>,
    S: Into<String>,
    Ty: Into<Type<Prim>>,
{
    iter.into_iter()
        .map(|(name, ty)| (name.into(), TypeEnvironment::prepare_type(ty)))
}

impl<Prim: PrimitiveType, S, Ty> FromIterator<(S, Ty)> for TypeEnvironment<Prim>
where
    S: Into<String>,
    Ty: Into<Type<Prim>>,
{
    fn from_iter<I: IntoIterator<Item = (S, Ty)>>(iter: I) -> Self {
        Self {
            variables: convert_iter(iter).collect(),
            known_constraints: Prim::well_known_constraints(),
            substitutions: Substitutions::default(),
        }
    }
}

impl<Prim: PrimitiveType, S, Ty> Extend<(S, Ty)> for TypeEnvironment<Prim>
where
    S: Into<String>,
    Ty: Into<Type<Prim>>,
{
    fn extend<I: IntoIterator<Item = (S, Ty)>>(&mut self, iter: I) {
        self.variables.extend(convert_iter(iter));
    }
}

// Helper trait to wrap type mapper and arithmetic.
trait FullArithmetic<Val, Prim: PrimitiveType>:
    MapPrimitiveType<Val, Prim = Prim> + TypeArithmetic<Prim>
{
}

impl<Val, Prim: PrimitiveType, T> FullArithmetic<Val, Prim> for T where
    T: MapPrimitiveType<Val, Prim = Prim> + TypeArithmetic<Prim>
{
}
