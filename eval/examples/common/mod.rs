use num_bigint::BigUint;
use num_traits::Zero;
use rand::RngCore;

fn gen_uint(rng: &mut impl RngCore, bit_size: u64) -> BigUint {
    let byte_size = (bit_size + 7) / 8;
    let mut bytes = vec![0u8; byte_size as usize];
    rng.fill_bytes(&mut bytes);

    // Mask the upper byte if necessary.
    let mask_bits = (byte_size * 8 - bit_size) as u8;
    let mask = u8::MAX >> mask_bits;
    *bytes.last_mut().unwrap() &= mask;

    let output = BigUint::from_bytes_le(&bytes);
    assert!(output.bits() <= bit_size);
    output
}

fn gen_uint_below(rng: &mut impl RngCore, bound: &BigUint) -> BigUint {
    assert!(!bound.is_zero());

    let bits = bound.bits();
    loop {
        let output = gen_uint(rng, bits);
        if output < *bound {
            return output;
        }
    }
}

pub(crate) fn gen_uint_range(rng: &mut impl RngCore, low: &BigUint, hi: &BigUint) -> BigUint {
    assert!(*low < *hi);
    low + gen_uint_below(rng, &(hi - low))
}

#[test]
fn generating_small_ints() {
    let mut rng = rand::rng();
    let mut stats = [0_usize; 10];
    for _ in 0..900_000 {
        let x = gen_uint_range(&mut rng, &BigUint::from(1_u32), &BigUint::from(10_u32));
        let x = usize::try_from(x).unwrap();
        assert!((1..10).contains(&x));
        stats[x] += 1;
    }

    assert_eq!(stats[0], 0);
    for count in &stats[1..] {
        assert!((99_000..110_000).contains(count), "{stats:?}");
    }
}
