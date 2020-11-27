use num_bigint::{BigInt, BigUint};
use num_traits::{One, Signed, Zero};

use core::{convert::TryFrom, mem};

use super::{Arithmetic, ArithmeticError, ModularArithmetic};

impl ModularArithmetic<BigUint> {
    fn invert_big(&self, value: BigUint) -> Option<BigUint> {
        let value = value % &self.modulus; // Reduce value since this influences speed.
        let mut t = BigInt::zero();
        let mut new_t = BigInt::one();

        let modulus = BigInt::from(self.modulus.clone());
        let mut r = modulus.clone();
        let mut new_r = BigInt::from(value);

        while !new_r.is_zero() {
            let quotient = &r / &new_r;
            t -= &quotient * &new_t;
            mem::swap(&mut new_t, &mut t);
            r -= quotient * &new_r;
            mem::swap(&mut new_r, &mut r);
        }

        if r > BigInt::one() {
            None // r = gcd(self.modulus, value) > 1
        } else {
            if t.is_negative() {
                t += modulus;
            }
            Some(BigUint::try_from(t).unwrap())
            // ^-- `unwrap` is safe by construction
        }
    }
}

impl Arithmetic<BigUint> for ModularArithmetic<BigUint> {
    fn add(&self, x: BigUint, y: BigUint) -> Result<BigUint, ArithmeticError> {
        Ok((x + y) % &self.modulus)
    }

    fn sub(&self, x: BigUint, y: BigUint) -> Result<BigUint, ArithmeticError> {
        let y_neg = &self.modulus - (y % &self.modulus);
        self.add(x, y_neg)
    }

    fn mul(&self, x: BigUint, y: BigUint) -> Result<BigUint, ArithmeticError> {
        Ok((x * y) % &self.modulus)
    }

    fn div(&self, x: BigUint, y: BigUint) -> Result<BigUint, ArithmeticError> {
        if y.is_zero() {
            Err(ArithmeticError::DivisionByZero)
        } else {
            let y_inv = self.invert_big(y).ok_or(ArithmeticError::NoInverse)?;
            self.mul(x, y_inv)
        }
    }

    fn pow(&self, x: BigUint, y: BigUint) -> Result<BigUint, ArithmeticError> {
        Ok(x.modpow(&y, &self.modulus))
    }

    fn neg(&self, x: BigUint) -> Result<BigUint, ArithmeticError> {
        let x = x % &self.modulus;
        Ok(&self.modulus - x)
    }

    fn eq(&self, x: &BigUint, y: &BigUint) -> bool {
        x % &self.modulus == y % &self.modulus
    }
}

#[cfg(test)]
mod bigint_tests {
    use super::*;
    use crate::arith::{CheckedArithmetic, NegateOnlyZero, OrdArithmetic, Unchecked};

    use num_bigint::{BigInt, BigUint};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use static_assertions::assert_impl_all;

    assert_impl_all!(CheckedArithmetic<NegateOnlyZero>: OrdArithmetic<BigUint>);
    assert_impl_all!(CheckedArithmetic<Unchecked>: OrdArithmetic<BigInt>);
    assert_impl_all!(ModularArithmetic<BigUint>: Arithmetic<BigUint>);

    fn gen_biguint<R: Rng>(rng: &mut R, bits: u64) -> BigUint {
        let bits = usize::try_from(bits).expect("Capacity overflow");
        let (div, rem) = (bits / 8, bits % 8);

        let mut buffer = vec![0_u8; div + (rem != 0) as usize];
        rng.fill_bytes(&mut buffer);
        if rem > 0 {
            // Zero out most significant bits in the first byte.
            let mask = u8::try_from((1_u16 << rem) - 1).unwrap();
            buffer[0] &= mask;
        }

        BigUint::from_bytes_be(&buffer)
    }

    fn mini_fuzz_for_big_prime_modulus(modulus: &BigUint, sample_count: usize) {
        let arithmetic = ModularArithmetic::new(modulus.clone());
        let mut rng = StdRng::seed_from_u64(modulus.bits());
        let signed_modulus = BigInt::from(modulus.clone());

        for _ in 0..sample_count {
            let x = gen_biguint(&mut rng, modulus.bits() - 1);
            let y = gen_biguint(&mut rng, modulus.bits() - 1);
            let expected = (&x + &y) % modulus;
            assert_eq!(arithmetic.add(x.clone(), y.clone()).unwrap(), expected);

            let mut expected =
                (BigInt::from(x.clone()) - BigInt::from(y.clone())) % &signed_modulus;
            if expected < BigInt::zero() {
                expected += &signed_modulus;
            }
            let expected = BigUint::try_from(expected).unwrap();
            assert_eq!(arithmetic.sub(x.clone(), y.clone()).unwrap(), expected);

            let expected = (&x * &y) % modulus;
            assert_eq!(arithmetic.mul(x, y).unwrap(), expected);
        }

        for _ in 0..sample_count {
            let x = gen_biguint(&mut rng, modulus.bits());
            let inv = arithmetic.div(BigUint::one(), x.clone());
            if (&x % modulus).is_zero() {
                // Quite unlikely, but better be safe than sorry.
                assert!(inv.is_err());
            } else {
                let inv = inv.unwrap();
                assert_eq!((inv * &x) % modulus, BigUint::one());
            }
        }

        for _ in 0..(sample_count / 10) {
            let x = gen_biguint(&mut rng, modulus.bits());

            // Check a random small exponent.
            let exp = rng.gen_range(1_u64, 1_000);
            let expected_pow = (0..exp).fold(BigUint::one(), |acc, _| (acc * &x) % modulus);
            assert_eq!(
                arithmetic.pow(x.clone(), BigUint::from(exp)).unwrap(),
                expected_pow
            );

            if !(&x % modulus).is_zero() {
                // Check Fermat's little theorem.
                let pow = arithmetic.pow(x, modulus - 1_u32).unwrap();
                assert_eq!(pow, BigUint::one());
            }
        }
    }

    // Primes taken from https://bigprimes.org/

    #[test]
    fn mini_fuzz_for_128_bit_prime_modulus() {
        let modulus = "904717851509176637007209984924163038177";
        mini_fuzz_for_big_prime_modulus(&modulus.parse().unwrap(), 10_000);
    }

    #[test]
    fn mini_fuzz_for_256_bit_prime_modulus() {
        let modulus =
            "35383204059922826862591333932184957269284020569026927321130404396066349029943";
        mini_fuzz_for_big_prime_modulus(&modulus.parse().unwrap(), 5_000);
    }

    #[test]
    fn mini_fuzz_for_384_bit_prime_modulus() {
        let modulus =
            "680077592003957715873956706738577254635634257392753873876268782486415186187701100959\
             54501183649227109037342431341197";
        mini_fuzz_for_big_prime_modulus(&modulus.parse().unwrap(), 2_000);
    }

    #[test]
    fn mini_fuzz_for_512_bit_prime_modulus() {
        let modulus =
            "134956060831834915306923365068985449378393338769474235719041178417311022526812045709\
             1169866466743447386864273902296614844109589811099153700965207136981133";
        mini_fuzz_for_big_prime_modulus(&modulus.parse().unwrap(), 2_000);
    }
}
