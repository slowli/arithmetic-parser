//! Showcases modular arithmetic by implementing a toy version of ElGamal encryption.
//!
//! See the `cyclic_group` example for a more complex usage of the crate.
//!
//! ⚠ This implementation is NOT SECURE (e.g., in terms of side-channel attacks)
//! and should be viewed only as a showcase of the crate abilities.

use std::cell::RefCell;

use arithmetic_eval::{
    arith::{ArithmeticExt, ModularArithmetic},
    env::{Assertions, Prelude},
    fns, Environment, ExecutableModule, Value,
};
use arithmetic_parser::grammars::{NumGrammar, Parse, Untyped};
use glass_pumpkin::safe_prime;
use num_bigint::{BigUint, RandBigInt};
use rand::thread_rng;

// NB: this is nowhere near a secure value (~2,048 bits).
const BIT_LENGTH: usize = 256;

/// Finds a generator of a prime-order multiplicative subgroup in integers modulo `modulus`
/// (which is guaranteed to be a safe prime). Per Fermat's little theorem, a square of any
/// number is guaranteed to be in the group, thus it will be a generator.
fn find_generator(modulus: &BigUint) -> BigUint {
    let two = BigUint::from(2_u32);
    let random_value = thread_rng().gen_biguint_range(&two, modulus);
    random_value.modpow(&two, modulus)
}

const EL_GAMAL_ENCRYPTION: &str = include_str!("elgamal.script");

fn main() -> anyhow::Result<()> {
    let el_gamal_encryption =
        Untyped::<NumGrammar<BigUint>>::parse_statements(EL_GAMAL_ENCRYPTION)?;
    let el_gamal_encryption = ExecutableModule::new("el_gamal", &el_gamal_encryption)?;

    // Run the compiled module with different groups.
    for i in 0..5 {
        println!("\nRunning sample #{i}");

        let modulus = safe_prime::new(BIT_LENGTH)?;
        println!("Generated safe prime: {modulus}");

        let prime_subgroup_order: BigUint = &modulus >> 1;
        let order_value = Value::Prim(prime_subgroup_order.clone());
        let generator = find_generator(&modulus);
        let arithmetic = ModularArithmetic::new(modulus).without_comparisons();

        let rng = RefCell::new(thread_rng());
        let two = BigUint::from(2_u32);
        let rand_scalar = Value::wrapped_fn(move || {
            rng.borrow_mut()
                .gen_biguint_range(&two, &prime_subgroup_order)
        });

        let mut env = Environment::with_arithmetic(arithmetic);
        env.extend(Prelude::iter().chain(Assertions::iter()));
        env.insert_native_fn("dbg", fns::Dbg)
            .insert("GEN", Value::Prim(generator))
            .insert("ORDER", order_value)
            .insert("rand_scalar", rand_scalar);

        el_gamal_encryption.with_env(&env)?.run()?;
    }
    Ok(())
}
