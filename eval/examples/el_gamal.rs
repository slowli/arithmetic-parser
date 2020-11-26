use glass_pumpkin::safe_prime;
use num_bigint::{BigUint, RandBigInt};
use num_traits::{One, Zero};
use rand::thread_rng;

use std::cell::RefCell;

use arithmetic_eval::{
    arith::{ArithmeticExt, ModularArithmetic},
    fns, ExecutableModule, Prelude, Value,
};
use arithmetic_parser::grammars::{NumGrammar, Parse, Untyped};

// NB: this is nowhere near a secure value (~2,048 bits).
const BIT_LENGTH: usize = 256;

fn find_generator(modulus: &BigUint) -> BigUint {
    let mut candidate = BigUint::from(2_u32);
    let order = modulus >> 1;
    loop {
        if candidate.modpow(&order, modulus) == BigUint::one() {
            return candidate;
        }
        candidate += BigUint::one();
    }
}

fn main() -> anyhow::Result<()> {
    let el_gamal_encryption = r#"
        dbg(GEN, ORDER);

        gen = || {
            sk = rand_scalar();
            (sk, GEN ^ sk)
        };

        encrypt = |message, pk| {
            r = rand_scalar();
            shared_secret = pk ^ r;
            (GEN ^ r, message * shared_secret)
        };

        decrypt = |(R, blinded_message), sk| {
            shared_secret = R ^ sk;
            blinded_message / shared_secret
        };

        // Test!
        (sk, pk) = gen();

        5.while(|i| i != 0, |i| {
            message = rand_scalar();
            dbg(message);
            encrypted = message.encrypt(pk);
            dbg(encrypted);
            assert(dbg(encrypted.decrypt(sk)) == message);

            // ElGamal encryption is partially homomorphic: a product of encryptions
            // is an encryption of the product of plaintexts.
            other_message = rand_scalar();
            dbg(other_message);
            encrypted_total = dbg(other_message.encrypt(pk)) * encrypted;
            dbg(encrypted_total);
            assert(dbg(encrypted_total.decrypt(sk)) == dbg(message * other_message));

            i - 1
        })
    "#;

    let el_gamal_encryption =
        Untyped::<NumGrammar<BigUint>>::parse_statements(el_gamal_encryption)?;
    let mut el_gamal_encryption = ExecutableModule::builder("el_gamal", &el_gamal_encryption)?
        .with_imports_from(&Prelude)
        .with_import("dbg", Value::native_fn(fns::Dbg))
        .set_imports(|_| Value::void());

    // Run with different groups.

    for i in 0..5 {
        println!("\nRunning sample #{}", i);
        let rng = RefCell::new(thread_rng());
        let modulus = safe_prime::new(BIT_LENGTH)?;
        println!("Generated safe prime: {}", modulus);
        let prime_subgroup_order: BigUint = &modulus >> 1;
        let generator = find_generator(&modulus);
        let arithmetic = ModularArithmetic::new(modulus).without_comparisons();

        el_gamal_encryption.set_import("GEN", Value::Number(generator));
        el_gamal_encryption.set_import("ORDER", Value::Number(prime_subgroup_order.clone()));
        el_gamal_encryption.set_import(
            "rand_scalar",
            Value::wrapped_fn(move || {
                rng.borrow_mut()
                    .gen_biguint_range(&BigUint::zero(), &prime_subgroup_order)
            }),
        );

        el_gamal_encryption.with_arithmetic(&arithmetic).run()?;
    }

    Ok(())
}
