//! Benches for the interpreter.
//!
//! Implemented benches:
//!
//! - Multiplication of `ELEMENTS` randomly selected numbers
//! - List reversal (worst-case by the number of reallocations)

use criterion::{criterion_group, criterion_main, BatchSize, Bencher, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};
use typed_arena::Arena;

use arithmetic_parser::{
    grammars::F32Grammar,
    interpreter::{fns, Interpreter, Value},
    GrammarExt, Span,
};

const SEED: u64 = 123;
const ELEMENTS: u64 = 100;

fn bench_mul_native(bencher: &mut Bencher<'_>) {
    let mut rng = StdRng::seed_from_u64(SEED);

    bencher.iter_batched(
        || {
            (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32, 1.5))
                .collect::<Vec<_>>()
        },
        |values| values.into_iter().product::<f32>(),
        BatchSize::SmallInput,
    );
}

fn bench_mul(bencher: &mut Bencher<'_>) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let arena = Arena::new();

    bencher.iter_batched(
        || {
            let values: Vec<_> = (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32, 1.5).to_string())
                .collect();
            let program = arena.alloc(values.join(" * "));
            F32Grammar::parse_statements(Span::new(program)).unwrap()
        },
        |block| Interpreter::new().evaluate(&block).unwrap(),
        BatchSize::SmallInput,
    );
}

fn bench_mul_fold(bencher: &mut Bencher<'_>) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let program = Span::new("xs.fold(1, |acc, x| acc * x)");
    let program = F32Grammar::parse_statements(program).unwrap();

    bencher.iter_batched(
        || {
            let values: Vec<_> = (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32, 1.5))
                .map(Value::Number)
                .collect();
            let mut interpreter = Interpreter::new();
            interpreter
                .innermost_scope()
                .insert_native_fn("fold", fns::Fold)
                .insert_var("xs", Value::Tuple(values));
            interpreter
        },
        |mut interpreter| interpreter.evaluate(&program).unwrap(),
        BatchSize::SmallInput,
    );
}

fn bench_reverse_native(bencher: &mut Bencher<'_>) {
    let mut rng = StdRng::seed_from_u64(SEED);

    bencher.iter_batched(
        || {
            (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32, 1.5))
                .collect::<Vec<_>>()
        },
        |values| {
            // This is obviously suboptimal, but mirrors the way reversing is done
            // in the interpreter.
            values.into_iter().fold(vec![], |mut acc, x| {
                acc.insert(0, x);
                acc
            })
        },
        BatchSize::SmallInput,
    );
}

fn bench_reverse(bencher: &mut Bencher<'_>) {
    let mut rng = StdRng::seed_from_u64(SEED);

    let rev_fn = "reverse = |xs| xs.fold((), |acc, x| (x,).merge(acc));";
    let rev_fn = F32Grammar::parse_statements(Span::new(rev_fn)).unwrap();
    let program = "xs.reverse()";
    let program = F32Grammar::parse_statements(Span::new(program)).unwrap();

    let mut interpreter = Interpreter::new();
    interpreter
        .innermost_scope()
        .insert_native_fn("fold", fns::Fold)
        .insert_native_fn("merge", fns::Merge);
    interpreter.evaluate(&rev_fn).unwrap();

    bencher.iter_batched(
        || {
            (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32, 1.5))
                .map(Value::Number)
                .collect::<Vec<_>>()
        },
        |values| {
            interpreter
                .innermost_scope()
                .insert_var("xs", Value::Tuple(values));
            interpreter.evaluate(&program).unwrap()
        },
        BatchSize::SmallInput,
    );
}

fn bench_interpreter(criterion: &mut Criterion) {
    criterion
        .benchmark_group("mul")
        .bench_function("native", bench_mul_native)
        .bench_function("int", bench_mul)
        .bench_function("int_fold", bench_mul_fold)
        .throughput(Throughput::Elements(ELEMENTS));

    criterion
        .benchmark_group("reverse")
        .bench_function("native", bench_reverse_native)
        .bench_function("int", bench_reverse);
}

criterion_group!(benches, bench_interpreter);
criterion_main!(benches);
