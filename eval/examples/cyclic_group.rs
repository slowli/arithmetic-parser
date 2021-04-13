//! Showcases the use of custom literals and arithmetics by implementing
//! Schnorr and DSA signatures on prime-order cyclic groups.
//!
//! The cyclic group used is the prime-order multiplicative subgroup (~Z/qZ) of integers modulo
//! a safe prime `p = 2q + 1`, i.e., the ElGamal construction.
//!
//! ⚠ This implementation is NOT SECURE (e.g., in terms of side-channel attacks)
//! and should be viewed only as a showcase of the crate abilities.

use glass_pumpkin::safe_prime;
use num_bigint::{BigUint, RandBigInt};
use rand::thread_rng;
use sha2::{digest::Digest, Sha256};

use std::{cell::RefCell, fmt};

use arithmetic_eval::{
    arith::{Arithmetic, ArithmeticExt, ModularArithmetic},
    error::{ArithmeticError, AuxErrorInfo},
    fns, Assertions, CallContext, ErrorKind, EvalResult, ExecutableModule, NativeFn, Number,
    Prelude, SpannedValue, Value,
};
use arithmetic_parser::{
    grammars::{Features, NumGrammar, NumLiteral, Parse, Untyped},
    InputSpan, NomResult,
};

/// Literals for our cyclic groups. We type them into scalars and group elements despite
/// both being represented by `BigUint`, since allowed arithmetic ops on scalars and group elements
/// are different:
///
/// - Scalars have the full set of arithmetic ops in Z/qZ
/// - Group elements have only multiplication / division and exponentiation by a scalar
///   in Z/(2q + 1)Z
#[derive(Debug, Clone)]
enum GroupLiteral {
    Scalar(BigUint),
    GroupElement(BigUint),
}

/// `Display` is necessary to output literals using the `fns::Dbg` native function.
impl fmt::Display for GroupLiteral {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(sc) => fmt::Display::fmt(sc, formatter),
            Self::GroupElement(ge) => write!(formatter, "Ge({})", ge),
        }
    }
}

/// Implement parsing of our literals. We only want to parse scalars.
impl NumLiteral for GroupLiteral {
    fn parse(input: InputSpan<'_>) -> NomResult<'_, Self> {
        <BigUint as NumLiteral>::parse(input)
            .map(|(rest, output)| (rest, GroupLiteral::Scalar(output)))
    }
}

/// Mark `GroupLiteral` as a number. This allows to use it as an argument in wrapped functions
/// (see the `arithmetic_eval::fns` module), which we utilize in several cases below.
impl Number for GroupLiteral {}

/// Arithmetic for our cyclic group.
#[derive(Debug)]
struct CyclicGroupArithmetic {
    /// Z/(2q + 1)Z modular arithmetic.
    for_group: ModularArithmetic<BigUint>,
    /// Generator for our cyclic group.
    generator: BigUint,
    /// Z/qZ modular arithmetic.
    for_scalars: ModularArithmetic<BigUint>,
}

impl CyclicGroupArithmetic {
    fn new(bits: usize) -> Self {
        // `BigUint::new` is required because `glass_pumpkin` produces `BigUint`s
        // with an incompatible package version (0.3 vs 0.4).
        let safe_prime = safe_prime::new(bits).unwrap();
        let prime_subgroup_order = &safe_prime >> 1;
        let two = BigUint::from(2_u32);

        // Generator search uses the DSA approach: generate a random element in Z/(2q + 1)Z,
        // and then square it so it falls into the prime-order subgroup.
        let generator = thread_rng()
            .gen_biguint_range(&two, &safe_prime)
            .modpow(&two, &safe_prime);

        Self {
            for_group: ModularArithmetic::new(safe_prime),
            generator,
            for_scalars: ModularArithmetic::new(prime_subgroup_order),
        }
    }

    /// Returns a closure generating random scalars.
    fn rand_scalar(&self) -> impl Fn() -> GroupLiteral {
        let rng = RefCell::new(thread_rng());
        let two = BigUint::from(2_u32);
        let prime_subgroup_order = self.for_scalars.modulus().to_owned();

        move || {
            GroupLiteral::Scalar(
                rng.borrow_mut()
                    .gen_biguint_range(&two, &prime_subgroup_order),
            )
        }
    }

    /// Returns a native function hashing data to a scalar.
    fn hash_to_scalar(&self) -> HashToScalar {
        let max_bit_len = self.for_group.modulus().bits();
        let max_byte_len = (max_bit_len / 8) as usize + (max_bit_len % 8 != 0) as usize;

        HashToScalar {
            modulus: self.for_scalars.modulus().to_owned(),
            max_byte_len,
        }
    }

    /// Converts a group element to a scalar.
    fn to_scalar(&self) -> impl Fn(GroupLiteral) -> GroupLiteral {
        let prime_subgroup_order = self.for_scalars.modulus().to_owned();
        move |value| match value {
            GroupLiteral::Scalar(sc) => GroupLiteral::Scalar(sc),
            GroupLiteral::GroupElement(ge) => GroupLiteral::Scalar(ge % &prime_subgroup_order),
        }
    }

    /// Sets generic imports for the provided `module`.
    fn set_imports(&self, module: &mut ExecutableModule<'_, GroupLiteral>) {
        let generator = GroupLiteral::GroupElement(self.generator.clone());
        let prime_subgroup_order = GroupLiteral::Scalar(self.for_group.modulus().to_owned());
        module
            .set_import("GEN", Value::Number(generator))
            .set_import("ORDER", Value::Number(prime_subgroup_order))
            .set_import("rand_scalar", Value::wrapped_fn(self.rand_scalar()))
            .set_import("hash_to_scalar", Value::native_fn(self.hash_to_scalar()));
    }
}

/// Function that hashes data to a scalar.
#[derive(Debug)]
struct HashToScalar {
    modulus: BigUint,
    /// Upper bound on the byte length of hashed `BigUint`s.
    max_byte_len: usize,
}

impl HashToScalar {
    fn hash_scalar(&self, hasher: &mut Sha256, sc: &BigUint) {
        hasher.update(&[0]); // "scalar" marker

        let mut sc_bytes = sc.to_bytes_le();
        assert!(sc_bytes.len() <= self.max_byte_len);
        sc_bytes.resize(self.max_byte_len, 0);
        hasher.update(&sc_bytes); // little-endian, 0-padded serialization of the value
    }

    fn hash_group_element(&self, hasher: &mut Sha256, ge: &BigUint) {
        hasher.update(&[1]); // "group element" marker

        let mut ge_bytes = ge.to_bytes_le();
        assert!(ge_bytes.len() <= self.max_byte_len);
        ge_bytes.resize(self.max_byte_len, 0);
        hasher.update(&ge_bytes); // little-endian, 0-padded serialization of the value
    }
}

impl NativeFn<GroupLiteral> for HashToScalar {
    fn evaluate<'a>(
        &self,
        args: Vec<SpannedValue<'a, GroupLiteral>>,
        context: &mut CallContext<'_, 'a, GroupLiteral>,
    ) -> EvalResult<'a, GroupLiteral> {
        // It is relatively easy to implement hashing for all types, but we're fine
        // with implementing it only for literals (scalars and group elements).

        let mut hasher = Sha256::default();
        for arg in &args {
            match &arg.extra {
                Value::Number(GroupLiteral::Scalar(sc)) => self.hash_scalar(&mut hasher, sc),
                Value::Number(GroupLiteral::GroupElement(ge)) => {
                    self.hash_group_element(&mut hasher, ge);
                }
                _ => {
                    let err = ErrorKind::native("Cannot hash value");
                    return Err(context
                        .call_site_error(err)
                        .with_span(arg, AuxErrorInfo::InvalidArg));
                }
            }
        }

        let mut hash_scalar = BigUint::from_bytes_le(hasher.finalize().as_slice());
        // Reduce the scalar by the modulus.
        hash_scalar %= &self.modulus;

        Ok(Value::Number(GroupLiteral::Scalar(hash_scalar)))
    }
}

impl Arithmetic<GroupLiteral> for CyclicGroupArithmetic {
    fn add(&self, x: GroupLiteral, y: GroupLiteral) -> Result<GroupLiteral, ArithmeticError> {
        match (x, y) {
            (GroupLiteral::Scalar(x), GroupLiteral::Scalar(y)) => {
                self.for_scalars.add(x, y).map(GroupLiteral::Scalar)
            }
            _ => Err(ArithmeticError::invalid_op("only scalars may be added")),
        }
    }

    fn sub(&self, x: GroupLiteral, y: GroupLiteral) -> Result<GroupLiteral, ArithmeticError> {
        match (x, y) {
            (GroupLiteral::Scalar(x), GroupLiteral::Scalar(y)) => {
                self.for_scalars.sub(x, y).map(GroupLiteral::Scalar)
            }
            _ => Err(ArithmeticError::invalid_op(
                "only scalars may be subtracted",
            )),
        }
    }

    fn mul(&self, x: GroupLiteral, y: GroupLiteral) -> Result<GroupLiteral, ArithmeticError> {
        match (x, y) {
            (GroupLiteral::Scalar(x), GroupLiteral::Scalar(y)) => {
                self.for_scalars.mul(x, y).map(GroupLiteral::Scalar)
            }
            (GroupLiteral::GroupElement(x), GroupLiteral::GroupElement(y)) => {
                self.for_group.mul(x, y).map(GroupLiteral::GroupElement)
            }
            _ => Err(ArithmeticError::invalid_op(
                "multiplication operands must have same type",
            )),
        }
    }

    fn div(&self, x: GroupLiteral, y: GroupLiteral) -> Result<GroupLiteral, ArithmeticError> {
        match (x, y) {
            (GroupLiteral::Scalar(x), GroupLiteral::Scalar(y)) => {
                self.for_scalars.div(x, y).map(GroupLiteral::Scalar)
            }
            (GroupLiteral::GroupElement(x), GroupLiteral::GroupElement(y)) => {
                self.for_group.div(x, y).map(GroupLiteral::GroupElement)
            }
            _ => Err(ArithmeticError::invalid_op(
                "division operands must have same type",
            )),
        }
    }

    fn pow(&self, x: GroupLiteral, y: GroupLiteral) -> Result<GroupLiteral, ArithmeticError> {
        match (x, y) {
            (GroupLiteral::Scalar(x), GroupLiteral::Scalar(y)) => {
                self.for_scalars.pow(x, y).map(GroupLiteral::Scalar)
            }
            (GroupLiteral::GroupElement(x), GroupLiteral::Scalar(y)) => {
                self.for_group.pow(x, y).map(GroupLiteral::GroupElement)
            }
            _ => Err(ArithmeticError::invalid_op("exponent must be a scalar")),
        }
    }

    fn neg(&self, x: GroupLiteral) -> Result<GroupLiteral, ArithmeticError> {
        if let GroupLiteral::Scalar(x) = x {
            self.for_scalars.neg(x).map(GroupLiteral::Scalar)
        } else {
            Err(ArithmeticError::invalid_op("only scalars can be negated"))
        }
    }

    fn eq(&self, x: &GroupLiteral, y: &GroupLiteral) -> bool {
        match (x, y) {
            (GroupLiteral::Scalar(x), GroupLiteral::Scalar(y)) => self.for_scalars.eq(x, y),
            (GroupLiteral::GroupElement(x), GroupLiteral::GroupElement(y)) => {
                self.for_group.eq(x, y)
            }
            _ => false,
        }
    }
}

const SCHNORR_SIGNATURES: &str = r#"
    dbg(GEN, ORDER);

    gen = || {
        sk = rand_scalar();
        (sk, GEN ^ sk)
    };

    sign = |message, sk| {
        r = rand_scalar();
        R = GEN ^ r;
        e = hash_to_scalar(R, message);
        (e, r - sk * e)
    };

    verify = |(e, s), message, pk| {
        R = GEN ^ s * pk ^ e;
        e == hash_to_scalar(R, message)
    };

    // Test!
    (sk, pk) = gen();
    (_, other_pk) = gen();

    5.while(|i| i != 0, |i| {
        message = rand_scalar();
        dbg(message);
        signature = message.sign(sk);
        dbg(signature);

        assert(signature.verify(message, pk));
        assert(!signature.verify(message, other_pk));
        assert(!signature.verify(rand_scalar(), pk));

        i - 1
    })
"#;

const DSA_SIGNATURES: &str = r#"
    dbg(GEN, ORDER);

    gen = || {
        sk = rand_scalar();
        (sk, GEN ^ sk)
    };

    sign = |message, sk| {
        k = rand_scalar();
        r = (GEN ^ k).to_scalar();
        s = (hash_to_scalar(message) + sk * r) / k;
        (r, s)
    };

    verify = |(r, s), message, pk| {
        (u1, u2) = (hash_to_scalar(message), r) / s;
        (GEN ^ u1 * pk ^ u2).to_scalar() == r
    };

    // Test!
    (sk, pk) = gen();
    (_, other_pk) = gen();

    5.while(|i| i != 0, |i| {
        message = rand_scalar();
        dbg(message);
        signature = message.sign(sk);
        dbg(signature);

        assert(signature.verify(message, pk));
        assert(!signature.verify(message, other_pk));
        assert(!signature.verify(rand_scalar(), pk));

        i - 1
    })
"#;

/// Type for a custom grammar definition.
#[derive(Debug, Clone, Copy)]
struct GroupGrammar;

impl Parse<'_> for GroupGrammar {
    type Base = Untyped<NumGrammar<GroupLiteral>>;

    // Disable comparisons in the parser.
    const FEATURES: Features = Features::all()
        .without(Features::TYPE_ANNOTATIONS)
        .without(Features::ORDER_COMPARISONS);
}

fn main() -> anyhow::Result<()> {
    /// Bit length of `p = 2q + 1`. This value is not cryptographically secure!
    const BIT_LENGTH: usize = 256;

    let schnorr_signatures = GroupGrammar::parse_statements(SCHNORR_SIGNATURES)?;
    let mut schnorr_signatures = ExecutableModule::builder("schnorr", &schnorr_signatures)?
        .with_imports_from(&Prelude)
        .with_imports_from(&Assertions)
        .with_import("dbg", Value::native_fn(fns::Dbg))
        .set_imports(|_| Value::void());

    let dsa_signatures = GroupGrammar::parse_statements(DSA_SIGNATURES)?;
    let mut dsa_signatures = ExecutableModule::builder("dsa", &dsa_signatures)?
        .with_imports_from(&Prelude)
        .with_imports_from(&Assertions)
        .with_import("dbg", Value::native_fn(fns::Dbg))
        .set_imports(|_| Value::void());

    for i in 0..5 {
        println!("\nRunning sample #{}", i);

        let arithmetic = CyclicGroupArithmetic::new(BIT_LENGTH);
        arithmetic.set_imports(&mut schnorr_signatures);
        arithmetic.set_imports(&mut dsa_signatures);
        dsa_signatures.set_import("to_scalar", Value::wrapped_fn(arithmetic.to_scalar()));

        let arithmetic = arithmetic.without_comparisons();
        schnorr_signatures.with_arithmetic(&arithmetic).run()?;
        dsa_signatures.with_arithmetic(&arithmetic).run()?;
    }

    Ok(())
}
