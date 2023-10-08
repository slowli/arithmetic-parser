//! Benches for the interpreter.
//!
//! Implemented benches:
//!
//! - Multiplication of `ELEMENTS` randomly selected numbers
//! - List reversal (worst-case by the number of re-allocations)
//! - Quicksort implementation from the README
//!
//! Generally, the interpreter seems to be ~50x slower than the comparable native code.

use criterion::{criterion_group, criterion_main, BatchSize, Bencher, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};
use typed_arena::Arena;

use std::cmp::Ordering;

use arithmetic_eval::{
    env::Prelude,
    exec::{ExecutableModule, WildcardId},
    fns, CallContext, Environment, NativeFn, Value,
};
use arithmetic_parser::{
    grammars::{F32Grammar, Parse, Untyped},
    Location,
};

const SEED: u64 = 123;
const ELEMENTS: u64 = 50;
const SORT_ELEMENTS: u64 = 1_000;

fn bench_mul_native(bencher: &mut Bencher<'_>) {
    let mut rng = StdRng::seed_from_u64(SEED);

    bencher.iter_batched(
        || {
            (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32..1.5))
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
                .map(|_| rng.gen_range(0.5_f32..1.5))
                .map(Value::Prim)
                .collect::<Vec<_>>()
        },
        |values| {
            values
                .into_iter()
                .fold(Value::Prim(1.0), |acc, x| match (acc, x) {
                    (Value::Prim(acc), Value::Prim(x)) => Value::Prim(acc * x),
                    _ => unreachable!(),
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
                .map(|_| rng.gen_range(0.5_f32..1.5).to_string())
                .collect();
            let program = arena.alloc(values.join(" * "));
            let program = Untyped::<F32Grammar>::parse_statements(program.as_str()).unwrap();
            ExecutableModule::new(WildcardId, &program).unwrap()
        },
        |module| module.with_env(&Environment::new()).unwrap().run().unwrap(),
        BatchSize::SmallInput,
    );
}

fn bench_mul_fold(bencher: &mut Bencher<'_>) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let program = "xs.fold(1, |acc, x| acc * x)";
    let program = Untyped::<F32Grammar>::parse_statements(program).unwrap();

    bencher.iter_batched(
        || {
            let module = ExecutableModule::new(WildcardId, &program).unwrap();
            let values: Vec<_> = (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32..1.5))
                .map(Value::Prim)
                .collect();

            let mut env = Environment::new();
            env.insert_native_fn("fold", fns::Fold);
            env.insert("xs", Value::from(values));
            (module, env)
        },
        |(module, env)| module.with_env(&env).unwrap().run(),
        BatchSize::SmallInput,
    );
}

fn bench_fold_fn(bencher: &mut Bencher<'_>) {
    let env = Environment::new();
    let mut ctx = CallContext::mock(WildcardId, Location::from_str("", ..), &env);
    let acc = ctx.apply_call_location(Value::Prim(1.0));
    let fold_fn = fns::Binary::new(|x: f32, y| x * y);
    let fold_fn = ctx.apply_call_location(Value::native_fn(fold_fn));

    let mut rng = StdRng::seed_from_u64(SEED);

    bencher.iter_batched(
        || {
            (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32..1.5))
                .map(Value::Prim)
                .collect::<Vec<_>>()
        },
        |array| {
            let array = ctx.apply_call_location(Value::from(array));
            fns::Fold
                .evaluate(vec![array, acc.clone(), fold_fn.clone()], &mut ctx)
                .unwrap()
        },
        BatchSize::SmallInput,
    );
}

fn bench_interpreted_fn(bencher: &mut Bencher<'_>) {
    let interpreted_fn = Untyped::<F32Grammar>::parse_statements("|x, y| x * y").unwrap();
    let interpreted_fn = ExecutableModule::new(WildcardId, &interpreted_fn).unwrap();
    let interpreted_fn = interpreted_fn.with_env(&Environment::new()).unwrap().run();
    let interpreted_fn = match interpreted_fn {
        Ok(Value::Function(function)) => function,
        _ => unreachable!("Unexpected function type"),
    };

    let mut rng = StdRng::seed_from_u64(SEED);
    let env = Environment::new();
    let mut ctx = CallContext::mock(WildcardId, Location::from_str("", ..), &env);

    bencher.iter_batched(
        || {
            (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32..1.5))
                .map(Value::Prim)
                .collect::<Vec<_>>()
        },
        |array| {
            let results = array.chunks(2).map(|chunk| {
                let args = chunk
                    .iter()
                    .map(|val| ctx.apply_call_location(val.to_owned()))
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
                .map(|_| rng.gen_range(0.5_f32..1.5))
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

    let rev_fn = "|xs| fold(xs, (), |acc, x| merge((x,), acc))";
    let rev_fn = Untyped::<F32Grammar>::parse_statements(rev_fn).unwrap();
    let rev_fn = ExecutableModule::new("rev_fn", &rev_fn).unwrap();
    let rev_fn = {
        let mut env = Environment::new();
        env.insert_native_fn("fold", fns::Fold)
            .insert_native_fn("merge", fns::Merge);
        rev_fn.with_env(&env).unwrap().run().unwrap()
    };
    assert!(rev_fn.is_function());

    let program = "reverse(xs)";
    let program = Untyped::<F32Grammar>::parse_statements(program).unwrap();
    let program = ExecutableModule::new("rev_fn", &program).unwrap();
    let mut env = Environment::new();
    env.insert("reverse", rev_fn);

    bencher.iter_batched(
        || {
            (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32..1.5))
                .map(Value::Prim)
                .collect::<Vec<_>>()
        },
        move |values| {
            env.insert("xs", Value::from(values));
            program.with_env(&env).unwrap().run().unwrap()
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

fn bench_quick_sort_native(bencher: &mut Bencher<'_>) {
    let mut rng = StdRng::seed_from_u64(SEED);
    bencher.iter_batched(
        || {
            (0..SORT_ELEMENTS)
                .map(|_| rng.gen_range(0.0_f32..100.0))
                .collect::<Vec<_>>()
        },
        |mut items| {
            items.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal));
            items
        },
        BatchSize::SmallInput,
    );
}

fn bench_quick_sort_native_slow(bencher: &mut Bencher<'_>) {
    // Subpar implementation of quick sort that is more or less equivalent to
    // the interpreted one.
    fn quick_sort(items: &[f32]) -> Vec<f32> {
        if let Some(pivot) = items.first().copied() {
            let lesser: Vec<_> = items[1..].iter().copied().filter(|&x| x < pivot).collect();
            let lesser = quick_sort(&lesser);
            let greater: Vec<_> = items[1..].iter().copied().filter(|&x| x >= pivot).collect();
            let greater = quick_sort(&greater);

            let mut all = lesser;
            all.push(pivot);
            all.extend_from_slice(&greater);
            all
        } else {
            vec![]
        }
    }

    let mut rng = StdRng::seed_from_u64(SEED);
    bencher.iter_batched(
        || {
            (0..SORT_ELEMENTS)
                .map(|_| rng.gen_range(0.0_f32..100.0))
                .collect::<Vec<_>>()
        },
        |items| quick_sort(&items),
        BatchSize::SmallInput,
    );
}

fn bench_quick_sort_interpreted(bencher: &mut Bencher<'_>) {
    let program = r#"
        quick_sort = |xs, quick_sort| {
            if(xs == (), || (), || {
                (pivot, ...rest) = xs;
                lesser_part = quick_sort(rest.filter(|x| x < pivot), quick_sort);
                greater_part = quick_sort(rest.filter(|x| x >= pivot), quick_sort);
                lesser_part.push(pivot).merge(greater_part)
            })()
        };
        |xs| quick_sort(xs, quick_sort)
    "#;
    let program = Untyped::<F32Grammar>::parse_statements(program).unwrap();
    let sort_module = ExecutableModule::new("sort", &program).unwrap();

    let mut env = Environment::new();
    env.extend(Prelude::iter());
    let sort_fn = sort_module.with_env(&env).unwrap().run().unwrap();
    let sort_fn = match sort_fn {
        Value::Function(function) => function,
        other => panic!("Unexpected module export: {other:?}"),
    };

    let mut rng = StdRng::seed_from_u64(SEED);
    bencher.iter_batched(
        || {
            Value::Tuple(
                (0..SORT_ELEMENTS)
                    .map(|_| Value::Prim(rng.gen_range(0.0_f32..100.0)))
                    .collect(),
            )
        },
        |items| {
            let mut ctx = CallContext::mock("test", Location::from_str("", ..), &env);
            let items = Location::from_str("", ..).copy_with_extra(items);
            sort_fn.evaluate(vec![items], &mut ctx).unwrap()
        },
        BatchSize::SmallInput,
    );
}

fn bench_quick_sort(criterion: &mut Criterion) {
    criterion
        .benchmark_group("quick_sort")
        .bench_function("native", bench_quick_sort_native)
        .bench_function("native_slow", bench_quick_sort_native_slow)
        .bench_function("int", bench_quick_sort_interpreted)
        .throughput(Throughput::Elements(SORT_ELEMENTS));
}

criterion_group!(benches, bench_interpreter, bench_quick_sort);
criterion_main!(benches);
