//! Shows how to use owned modules.

use anyhow::anyhow;
use assert_matches::assert_matches;

use arithmetic_eval::{AuxErrorInfo, EvalError, ExecutableModule, Interpreter, Value};
use arithmetic_parser::{grammars::F64Grammar, Code, GrammarExt, InputSpan, StripCode};

fn create_module(
    program: &str,
    var_name: &str,
) -> anyhow::Result<ExecutableModule<'static, F64Grammar>> {
    let block = F64Grammar::parse_statements(InputSpan::new(program))
        .map_err(|e| anyhow!("Parse error: {}", e.extra))?;

    let mut interpreter = Interpreter::with_prelude();
    interpreter.insert_var(var_name, Value::void());
    let module = interpreter.compile(&block).map_err(|e| e.strip_code())?;

    // By default, the module is tied by its lifetime to the `program`. However,
    // we can break this tie using the `StripCode` trait.
    Ok(module.strip_code())
}

fn main() -> anyhow::Result<()> {
    let var_name = "xs";
    let dynamic_program = format!("{}.fold(0, |acc, x| acc + x)", var_name);
    let module = create_module(&dynamic_program, var_name)?;

    // The code is dropped here, but the module is still usable.
    let mut imports = module.imports().to_owned();
    imports[var_name] = Value::Tuple(vec![Value::Number(3.0), Value::Number(-5.0)]);
    let sum = module.run_with_imports(imports.clone())?;
    assert_eq!(sum, Value::Number(-2.0));

    // Errors are handled as well.
    imports[var_name] = Value::Number(1.0);
    let err = module.run_with_imports(imports).unwrap_err();
    println!("Expected error\n{:#}", err);
    assert_matches!(
        err.source(),
        EvalError::NativeCall(ref msg) if msg == "`fold` requires first arg to be a tuple"
    );
    let arg_span = err.aux_spans()[0];
    assert_matches!(arg_span.extra, AuxErrorInfo::InvalidArg);

    // Naturally, all spans do not retain refs to source code, but rather contain info
    // sufficient to be recoverable.
    assert_eq!(arg_span.code_or_location("var"), "var at 1:1");
    assert_eq!(*arg_span.fragment(), Code::Stripped(2));

    Ok(())
}
