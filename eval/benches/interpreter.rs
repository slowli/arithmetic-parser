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
    arith::StdArithmetic,
    env::{Filler, Prelude},
    exec::{ExecutableModule, WildcardId},
    fns, CallContext, NativeFn, Value,
};
use arithmetic_parser::{
    grammars::{F32Grammar, Parse, Untyped},
    MaybeSpanned,
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

            ExecutableModule::builder(WildcardId, &program)
                .unwrap()
                .build()
        },
        |module| module.run().unwrap(),
        BatchSize::SmallInput,
    );
}

fn bench_mul_fold(bencher: &mut Bencher<'_>) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let program = "xs.fold(1, |acc, x| acc * x)";
    let program = Untyped::<F32Grammar>::parse_statements(program).unwrap();

    bencher.iter_batched(
        || {
            let mut module = ExecutableModule::builder(WildcardId, &program)
                .unwrap()
                .with_import("fold", Value::native_fn(fns::Fold))
                .with_import("xs", Value::void())
                .build();

            let values: Vec<_> = (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32..1.5))
                .map(Value::Prim)
                .collect();
            module.set_import("xs", Value::from(values));
            module
        },
        |module| module.run(),
        BatchSize::SmallInput,
    );
}

fn bench_fold_fn(bencher: &mut Bencher<'_>) {
    let mut ctx = CallContext::mock(&WildcardId, MaybeSpanned::from_str("", ..), &StdArithmetic);
    let acc = ctx.apply_call_span(Value::Prim(1.0));
    let fold_fn = fns::Binary::new(|x: f32, y| x * y);
    let fold_fn = ctx.apply_call_span(Value::native_fn(fold_fn));

    let mut rng = StdRng::seed_from_u64(SEED);

    bencher.iter_batched(
        || {
            (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32..1.5))
                .map(Value::Prim)
                .collect::<Vec<_>>()
        },
        |array| {
            let array = ctx.apply_call_span(Value::from(array));
            fns::Fold
                .evaluate(vec![array, acc.clone(), fold_fn.clone()], &mut ctx)
                .unwrap()
        },
        BatchSize::SmallInput,
    );
}

fn bench_interpreted_fn(bencher: &mut Bencher<'_>) {
    let mut ctx = CallContext::mock(&WildcardId, MaybeSpanned::from_str("", ..), &StdArithmetic);
    let interpreted_fn = Untyped::<F32Grammar>::parse_statements("|x, y| x * y").unwrap();
    let interpreted_fn = ExecutableModule::builder(WildcardId, &interpreted_fn)
        .unwrap()
        .build()
        .run()
        .unwrap();
    let interpreted_fn = match interpreted_fn {
        Value::Function(function) => function,
        _ => unreachable!("Wrong function type"),
    };

    let mut rng = StdRng::seed_from_u64(SEED);

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

    let rev_fn = "|xs| xs.fold((), |acc, x| (x,).merge(acc))";
    let rev_fn = Untyped::<F32Grammar>::parse_statements(rev_fn).unwrap();
    let rev_fn = ExecutableModule::builder("rev_fn", &rev_fn)
        .unwrap()
        .with_import("fold", Value::native_fn(fns::Fold))
        .with_import("merge", Value::native_fn(fns::Merge))
        .build();
    let rev_fn = rev_fn.run().unwrap();
    assert!(rev_fn.is_function());

    let program = "xs.reverse()";
    let program = Untyped::<F32Grammar>::parse_statements(program).unwrap();
    let mut program = ExecutableModule::builder("rev_fn", &program)
        .unwrap()
        .with_import("reverse", rev_fn)
        .with_imports_from(&Filler::void(&["xs"]))
        .build();

    bencher.iter_batched(
        || {
            (0..ELEMENTS)
                .map(|_| rng.gen_range(0.5_f32..1.5))
                .map(Value::Prim)
                .collect::<Vec<_>>()
        },
        |values| {
            program.set_import("xs", Value::from(values));
            program.run().unwrap()
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
                lesser_part = rest.filter(|x| x < pivot).quick_sort(quick_sort);
                greater_part = rest.filter(|x| x >= pivot).quick_sort(quick_sort);
                lesser_part.push(pivot).merge(greater_part)
            })()
        };
        |xs| xs.quick_sort(quick_sort)
    "#;
    let program = Untyped::<F32Grammar>::parse_statements(program).unwrap();
    let sort_module = ExecutableModule::builder("sort", &program)
        .unwrap()
        .with_imports_from(&Prelude)
        .build();
    let sort = match sort_module.run().unwrap() {
        Value::Function(function) => function,
        other => panic!("Unexpected module export: {:?}", other),
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
            let mut ctx =
                CallContext::mock(&"test", MaybeSpanned::from_str("", ..), &StdArithmetic);
            let items = MaybeSpanned::from_str("", ..).copy_with_extra(items);
            sort.evaluate(vec![items], &mut ctx).unwrap()
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
