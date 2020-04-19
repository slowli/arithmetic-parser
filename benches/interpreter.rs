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
    eval::{fns, CallContext, Interpreter, NativeFn, Value},
    grammars::F32Grammar,
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

fn bench_mul_native_with_value(bencher: &mut Bencher<'_>) {
    let mut rng = StdRng::seed_from_u64(SEED);

    bencher.iter_batched(
        || {
            (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32, 1.5))
                .map(Value::<F32Grammar>::Number)
                .collect::<Vec<_>>()
        },
        |values| {
            values
                .into_iter()
                .fold(Value::<F32Grammar>::Number(1.0), |acc, x| match (acc, x) {
                    (Value::Number(acc), Value::Number(x)) => Value::Number(acc * x),
                    _ => unimplemented!("oops"),
                })
        },
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
            let program = F32Grammar::parse_statements(Span::new(program)).unwrap();
            Interpreter::new().compile(&program).unwrap()
        },
        |block| block.run().unwrap(),
        BatchSize::SmallInput,
    );
}

fn bench_mul_fold(bencher: &mut Bencher<'_>) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let program = Span::new("xs.fold(1, |acc, x| acc * x)");
    let program = F32Grammar::parse_statements(program).unwrap();

    bencher.iter_batched(
        || {
            let mut interpreter = Interpreter::new();
            let mut module = interpreter
                .insert_native_fn("fold", fns::Fold)
                .insert_var("xs", Value::Tuple(vec![]))
                .compile(&program)
                .unwrap();

            let values: Vec<_> = (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32, 1.5))
                .map(Value::Number)
                .collect();
            module.set_import("xs", Value::Tuple(values));
            module
        },
        |module| module.run(),
        BatchSize::SmallInput,
    );
}

fn bench_fold_fn(bencher: &mut Bencher<'_>) {
    let mut ctx = CallContext::mock();
    let acc = ctx.apply_call_span(Value::Number(1.0));
    let fold_fn = fns::Binary::new(|x, y| x * y);
    let fold_fn = ctx.apply_call_span(Value::native_fn(fold_fn));

    let mut rng = StdRng::seed_from_u64(SEED);

    bencher.iter_batched(
        || {
            (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32, 1.5))
                .map(Value::<F32Grammar>::Number)
                .collect::<Vec<_>>()
        },
        |array| {
            let array = ctx.apply_call_span(Value::Tuple(array));
            fns::Fold
                .evaluate(vec![array, acc.clone(), fold_fn.clone()], &mut ctx)
                .unwrap()
        },
        BatchSize::SmallInput,
    );
}

fn bench_interpreted_fn(bencher: &mut Bencher<'_>) {
    let mut ctx = CallContext::mock();
    let interpreted_fn = F32Grammar::parse_statements(Span::new("|x, y| x * y")).unwrap();
    let interpreted_fn = Interpreter::new().evaluate(&interpreted_fn).unwrap();
    let interpreted_fn = match interpreted_fn {
        Value::Function(function) => function,
        _ => unreachable!("Wrong function type"),
    };

    let mut rng = StdRng::seed_from_u64(SEED);

    bencher.iter_batched(
        || {
            (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32, 1.5))
                .map(Value::<F32Grammar>::Number)
                .collect::<Vec<_>>()
        },
        |array| {
            let results = array.chunks(2).map(|chunk| {
                let args = chunk
                    .iter()
                    .map(|val| ctx.apply_call_span(val.to_owned()))
                    .collect();
                interpreted_fn.evaluate(args, &mut ctx).unwrap()
            });
            results.collect::<Vec<_>>()
        },
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
            interpreter.insert_var("xs", Value::Tuple(values));
            interpreter.evaluate(&program).unwrap()
        },
        BatchSize::SmallInput,
    );
}

fn bench_interpreter(criterion: &mut Criterion) {
    criterion
        .benchmark_group("mul")
        .bench_function("native", bench_mul_native)
        .bench_function("native_value", bench_mul_native_with_value)
        .bench_function("int", bench_mul)
        .bench_function("int_fold", bench_mul_fold)
        .throughput(Throughput::Elements(ELEMENTS));

    criterion.bench_function("fold_fn", bench_fold_fn);
    criterion.bench_function("int_fn", bench_interpreted_fn);

    criterion
        .benchmark_group("reverse")
        .bench_function("native", bench_reverse_native)
        .bench_function("int", bench_reverse);
}

criterion_group!(benches, bench_interpreter);
criterion_main!(benches);
