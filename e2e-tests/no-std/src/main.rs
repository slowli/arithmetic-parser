//! Test no-std application for arithmetic parser / interpreter.

#![no_std]
#![no_main]
#![feature(default_alloc_error_handler)]

extern crate alloc;

use alloc_cortex_m::CortexMHeap;
use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln, syscall};
use panic_halt as _;
use rand_chacha::{
    rand_core::{RngCore, SeedableRng},
    ChaChaRng,
};

use alloc::vec::Vec;
use core::cell::RefCell;

use arithmetic_eval::{
    arith::CheckedArithmetic, fns, Assertions, CallContext, EvalResult, ExecutableModule, NativeFn,
    Prelude, SpannedValue, Value,
};
use arithmetic_parser::{
    grammars::{NumGrammar, Parse, Untyped},
    CodeFragment,
};

#[global_allocator]
static ALLOCATOR: CortexMHeap = CortexMHeap::empty();

const HEAP_SIZE: usize = 49_152;

const MINMAX_SCRIPT: &str = r#"
    minmax = |xs| xs.fold(#{ min: MAX_VALUE, max: MIN_VALUE }, |acc, x| #{
         min: if(x < acc.min, x, acc.min),
         max: if(x > acc.max, x, acc.max),
    });
    xs = dbg(array(10, |_| rand_num()));
    { min, max } = dbg(xs.minmax());
    assert(xs.fold(true, |acc, x| acc && x >= min && x <= max));
"#;

/// Analogue of `arithmetic_eval::fns::Dbg` that writes to the semihosting interface.
struct Dbg;

impl NativeFn<i32> for Dbg {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, i32>>,
        ctx: &mut CallContext<'_, 'a, i32>,
    ) -> EvalResult<'a, i32> {
        ctx.check_args_count(&args, 1)?;
        let arg = args.pop().unwrap();

        match arg.fragment() {
            CodeFragment::Str(code) => hprintln!(
                "[{line}:{col}] {code} = {val}",
                line = arg.location_line(),
                col = arg.get_column(),
                code = code,
                val = arg.extra
            ),
            CodeFragment::Stripped(_) => hprintln!(
                "[{line}:{col}] {val}",
                line = arg.location_line(),
                col = arg.get_column(),
                val = arg.extra
            ),
        }
        .unwrap();
        Ok(arg.extra)
    }
}

fn main_inner() {
    let minmax = Untyped::<NumGrammar<i32>>::parse_statements(MINMAX_SCRIPT).unwrap();

    let epoch_seconds = unsafe { syscall!(TIME) };
    // Using a timestamp as an RNG seed is unsecure and done for simplicity only.
    // Modern bare metal envs come with a hardware RNG peripheral that should be used instead.
    let rng = ChaChaRng::seed_from_u64(epoch_seconds as u64);
    let rng = RefCell::new(rng);
    let rand_num = Value::wrapped_fn(move || rng.borrow_mut().next_u32() as i32);

    let minmax = ExecutableModule::builder("minmax", &minmax)
        .unwrap()
        .with_imports_from(&Prelude)
        .with_imports_from(&Assertions)
        .with_import("dbg", Value::native_fn(Dbg))
        .with_import("array", Value::native_fn(fns::Array))
        .with_import("rand_num", rand_num)
        .with_import("MIN_VALUE", Value::Prim(i32::MIN))
        .with_import("MAX_VALUE", Value::Prim(i32::MAX))
        .set_imports(|_| Value::void());

    minmax
        .with_arithmetic(&<CheckedArithmetic>::new())
        .run()
        .unwrap();
}

#[entry]
fn main() -> ! {
    let start = cortex_m_rt::heap_start() as usize;
    unsafe {
        ALLOCATOR.init(start, HEAP_SIZE);
    }

    main_inner();

    debug::exit(debug::EXIT_SUCCESS);
    unreachable!("Program must exit by this point");
}
