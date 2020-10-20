#![no_std]

extern crate alloc;

use alloc::string::ToString;
use core::f64;

use arithmetic_eval::{fns, Interpreter, Value};
use arithmetic_parser::{grammars::F64Grammar, GrammarExt};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    pub type Error;

    #[wasm_bindgen(constructor)]
    fn new(message: &str) -> Error;
}

#[allow(clippy::type_complexity)]
fn initialize_interpreter(interpreter: &mut Interpreter<'_, F64Grammar>) {
    const CONSTANTS: &[(&str, f64)] = &[
        ("E", f64::consts::E),
        ("PI", f64::consts::PI),
        ("Inf", f64::INFINITY),
    ];

    const UNARY_FNS: &[(&str, fn(f64) -> f64)] = &[
        // Rounding functions.
        ("floor", f64::floor),
        ("ceil", f64::ceil),
        ("round", f64::round),
        ("frac", f64::fract),
        // Exponential functions.
        ("exp", f64::exp),
        ("ln", f64::ln),
        ("sinh", f64::sinh),
        ("cosh", f64::cosh),
        ("tanh", f64::tanh),
        ("asinh", f64::asinh),
        ("acosh", f64::acosh),
        ("atanh", f64::atanh),
        // Trigonometric functions.
        ("sin", f64::sin),
        ("cos", f64::cos),
        ("tan", f64::tan),
        ("asin", f64::asin),
        ("acos", f64::acos),
        ("atan", f64::atan),
    ];

    const BINARY_FNS: &[(&str, fn(f64, f64) -> f64)] = &[
        ("min", |x, y| if x < y { x } else { y }),
        ("max", |x, y| if x > y { x } else { y }),
    ];

    interpreter.insert_native_fn("cmp", fns::Compare);

    for (name, c) in CONSTANTS {
        interpreter.insert_var(name, Value::Number(*c));
    }
    for (name, unary_fn) in UNARY_FNS {
        interpreter.insert_native_fn(name, fns::Unary::new(*unary_fn));
    }
    for (name, binary_fn) in BINARY_FNS {
        interpreter.insert_native_fn(name, fns::Binary::new(*binary_fn));
    }
}

#[wasm_bindgen]
pub fn evaluate(program: &str) -> Result<JsValue, JsValue> {
    let statements =
        F64Grammar::parse_statements(program).map_err(|err| Error::new(&err.to_string()))?;

    let mut interpreter = Interpreter::with_prelude();
    initialize_interpreter(&mut interpreter);

    let value = interpreter
        .evaluate(&statements)
        .map_err(|err| Error::new(&err.to_string()))?;

    match value {
        Value::Number(number) => Ok(JsValue::from(number)),
        Value::Bool(flag) => Ok(JsValue::from(flag)),
        _ => Err(Error::new("returned value is not presentable").into()),
    }
}
