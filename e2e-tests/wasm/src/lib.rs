#![no_std]

extern crate alloc;

use alloc::string::ToString;
use core::f64;

use arithmetic_eval::{
    env::{Assertions, Environment, Prelude, VariableMap},
    exec::WildcardId,
    fns, Value,
};
use arithmetic_parser::grammars::{F64Grammar, Parse, Untyped};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    pub type Error;

    #[wasm_bindgen(constructor)]
    fn new(message: &str) -> Error;
}

#[allow(clippy::type_complexity)]
fn initialize_env(env: &mut Environment<'_, f64>) {
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

    for (name, c) in CONSTANTS {
        env.insert(name, Value::Prim(*c));
    }
    for (name, unary_fn) in UNARY_FNS {
        env.insert_native_fn(name, fns::Unary::new(*unary_fn));
    }
    for (name, binary_fn) in BINARY_FNS {
        env.insert_native_fn(name, fns::Binary::new(*binary_fn));
    }
}

#[wasm_bindgen]
pub fn evaluate(program: &str) -> Result<JsValue, JsValue> {
    let block = Untyped::<F64Grammar>::parse_statements(program)
        .map_err(|err| Error::new(&err.to_string()))?;

    let mut env = Prelude.iter().chain(Assertions.iter()).collect();
    initialize_env(&mut env);

    let value = env
        .compile_module(WildcardId, &block)
        .map_err(|err| Error::new(&err.to_string()))?
        .run()
        .map_err(|err| Error::new(&err.to_string()))?;

    match value {
        Value::Prim(number) => Ok(JsValue::from(number)),
        Value::Bool(flag) => Ok(JsValue::from(flag)),
        _ => Err(Error::new("returned value is not presentable").into()),
    }
}
