# Testing Usability of `arithmetic-*` Crates in WASM

This simple crate tests that `arithmetic-parser` and `arithmetic-eval` build
and can be used in WASM.

## Compiling

1. Install WASM target for Rust via `rustup`: `rustup target add wasm32-unknown-unknown`.
2. Install [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/).
3. Install [Node](https://nodejs.org/).
4. Switch to the directory with this README and run `wasm-pack build --target nodejs`.
5. Run the testing script: `node test.js`. 
