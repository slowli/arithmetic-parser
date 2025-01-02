searchState.loadedDescShard("arithmetic_eval", 0, "Simple interpreter for ASTs produced by <code>arithmetic-parser</code>.\nArray (a tuple of arbitrary size).\nBoolean value.\nBoolean value.\nContext for native function calls.\nFunction definition. Functions can be either native …\nFunction value.\nFunction.\nInterpreted function.\nFunction defined within the interpreter.\nNative function.\nFunction on zero or more <code>Value</code>s.\nMarker trait for possible literals.\nObject with zero or more named fields.\nObject.\nObject with zero or more named fields.\nOpaque reference to a native value.\nPrimitive type other than <code>Bool</code>ean.\nPrimitive value, such as a number. This does not include …\nOpaque reference to a value.\nOpaque reference to a native value.\nValue together with a span that has produced it.\nTuple of zero or more values.\nTuple of a specific size.\nTuple of zero or more values.\nValues produced by expressions during their interpretation.\nPossible high-level types of <code>Value</code>s.\nApplies the call span to the specified <code>value</code>.\nReturns the number of arguments for this function.\n<code>Arithmetic</code> trait and its implementations.\nReturns the call location of the currently executing …\nCreates an error spanning the call site.\nReturns values captured by this function.\nChecks argument count and returns an error if it doesn’t …\nChecks whether this object has a field with the specified …\nTries to downcast this reference to a specific type.\n<code>Environment</code> and other types related to <code>Value</code> collections.\nEvaluation errors.\nExecutes the function on the specified arguments.\nEvaluates this function with the provided arguments and …\nEvaluates the function on the specified arguments.\n<code>ExecutableModule</code> and related types.\nExtra information that can be embedded by the user.\nIterates over field names.\nStandard functions for the interpreter, and the tools to …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the value of a field with the specified name, or …\nInserts a field into this object.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nChecks if this object is empty (has no fields).\nChecks if this tuple is empty (has no elements).\nChecks if this value is a function.\nChecks if the provided function is the same as this one.\nChecks if this value is void (an empty tuple).\nIterates over name-value pairs for all fields defined in …\nIterates over the elements in this tuple.\nCreates an object with a single field.\nReturns the number of fields in this object.\nReturns the number of elements in this tuple.\nCreates a mock call context with the specified module ID …\nReturns ID of the module defining this function.\nCreates a native function.\nCreates a value for a native function.\nCreates a reference to <code>value</code> that implements equality …\nCreates a reference to a native variable.\nPushes a value to the end of this tuple.\nRemoves and returns the specified field from this object.\nReturns the type of this value.\nCreates a new empty tuple (aka a void value).\nCreates a void value (an empty tuple).\nCreates a reference to <code>value</code> with the identity comparison: …\nCreates a wrapped function.\nEncapsulates arithmetic operations on a certain primitive …\nExtension trait for <code>Arithmetic</code> allowing to combine the …\nMarker for <code>CheckedArithmetic</code> signalling that negation …\nArithmetic on an integer type (e.g., <code>i32</code>) that checks …\nHelper trait for <code>CheckedArithmetic</code> describing how number …\nEncapsulates extension of an unsigned integer type into …\nWrapper type allowing to extend an <code>Arithmetic</code> to an …\nModular arithmetic on integers.\nMarker for <code>CheckedArithmetic</code> signalling that negation is …\nExtends an <code>Arithmetic</code> with a comparison operation on …\nSigned double-width extension type.\nArithmetic on a number type that implements all necessary …\nMarker for <code>CheckedArithmetic</code> signalling that negation …\nUnsigned double-width extension type.\nArithmetic on an integer type (e.g., <code>i32</code>), in which all …\nAdds two values.\nNegates the provided <code>value</code>, or returns <code>None</code> if the value …\nDivides two values.\nChecks if two values are equal. Note that equality can be …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nReturns the modulus for this arithmetic.\nMultiplies two values.\nNegates a value.\nCreates a new arithmetic instance.\nCreates a new arithmetic with the specified <code>modulus</code>.\nCompares two values. Returns <code>None</code> if the numbers are not …\nRaises <code>x</code> to the power of <code>y</code>.\nSubtracts two values.\nCombines this arithmetic with the specified comparison …\nCombines this arithmetic with a comparison function …\nCombines this arithmetic with a comparison function that …\nContainer for assertion functions: <code>assert</code>, <code>assert_eq</code> and …\nContainer with the comparison functions: <code>cmp</code>, <code>min</code> and <code>max</code>.\nEnvironment containing named <code>Value</code>s.\nResult of converting <code>Environment</code> into an iterator.\nIterator over references of the <code>Environment</code> entries.\nCommonly used constants and functions from the <code>fns</code> module.\nChecks if this environment contains a variable with the …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nGets a variable by name.\nInserts a variable with the specified name.\nInserts a native function with the specified name.\nInserts a wrapped function with the specified name.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCreates an iterator over contained values and the …\nCreates an iterator over contained values and the …\nCreates an iterator over contained values and the …\nIterates over variables.\nCreates a new environment.\nCreates an environment with the specified arithmetic.\nString representation of an argument value (e.g., for a …\nMismatch between the number of arguments in the function …\n<code>Arithmetic</code> error, such as division by zero.\nArithmetic errors raised by <code>Arithmetic</code> operations on …\nAn error has occurred during assignment.\nA duplicated variable is in an lvalue of an assignment.\nAuxiliary information about an evaluation error.\nElement of a backtrace, i.e., a function / method call.\nBinary operation.\nAn error has occurred when evaluating a binary operation.\nA field cannot be accessed for the value (i.e., it is not …\nValue is not callable (i.e., it is not a function).\nValue cannot be compared to other values. Only primitive …\nCannot destructure a non-tuple variable.\nValue cannot be indexed (i.e., it is not a tuple).\nDivision by zero.\nContains the error value\nEvaluation error together with one or more relevant code …\nKinds of errors that can occur when compiling or …\nError with the associated backtrace.\nResult of an expression evaluation.\nExpression.\nField set differs between LHS and RHS, which are both …\nA duplicated variable is in function args.\nFunction arguments declaration for …\nIndex is out of bounds for the indexed tuple.\nInteger overflow or underflow.\nInvalid argument.\nExponent of <code>Arithmetic::pow()</code> cannot be converted to <code>usize</code>…\nField name is invalid.\nInvalid operation with a custom error message.\nCode fragment together with information about the module …\nLvalue.\nGeneric error during execution of a native function.\nObject does not have a required field.\nInteger used as a denominator in <code>Arithmetic::div()</code> has no …\nContains the success value\nPrevious variable assignment for …\nRepeated assignment to the same variable in function args …\nContext for <code>ErrorKind::RepeatedAssignment</code>.\nRepeated field in object initialization (e.g., …\nRvalue containing an invalid assignment for …\nStatement.\nMismatch between length of tuples in a binary operation or …\nContext for <code>ErrorKind::TupleLenMismatch</code>.\nUnary operation.\nRHS of a binary operation on differently shaped objects.\nRHS of a binary operation on differently sized tuples.\nVariable with the enclosed name is not defined.\nUnexpected operand type for the specified operation.\nVariable is not initialized.\nConstruct not supported by the interpreter.\nDescription of a construct not supported by a certain …\nError while converting arguments for <code>FnWrapper</code>.\nReturns auxiliary spans for this error.\nIterates over the error backtrace, starting from the most …\nCode span of the function call.\nCode span of the function definition. <code>None</code> for native …\nFunction name.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns information helping fix the error.\nReturns the code fragment within the module. The fragment …\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCreates a new invalid operation error with the specified …\nReturns the source of the error.\nReturns the main span for this error.\nReturns a short description of the spanned information.\nReturns the ID of the module containing this fragment.\nCreates a native error.\nReturns the source of the error.\nReturned shortened error cause.\nCreates an error for an lvalue type not supported by the …\nAdds an auxiliary span to this error. The <code>span</code> must be in …\nAvailable fields in the object in no particular order.\nNumber of args at the function call.\nContext in which the error has occurred.\nContext in which the error has occurred.\nNumber of args at the function definition.\nMissing field.\nIndex.\nActual tuple length.\nLength of a tuple on the left-hand side.\nFields in LHS.\nOperation being performed.\nOperation which failed.\nLength of a tuple on the right-hand side.\nFields in RHS.\nCompiler extensions defined for some AST nodes, most …\nExecutable module together with its imports.\nIndexed module ID containing a prefix part (e.g., <code>snippet</code>).\nIdentifier of an <code>ExecutableModule</code>. This is usually a “…\nModule identifier that has a single possible value, which …\nContainer for an <code>ExecutableModule</code> together with an …\nReturns a reference to the boxed value if it is of type <code>T</code>, …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nGets the identifier of this module.\nReturns a shared reference to imports of this module.\n0-based index of the module.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nReturns <code>true</code> if the boxed type is the same as <code>T</code>.\nChecks if the specified variable is an import.\nCreates a new ID instance.\nCreates a new module.\nPrefix that can identify the nature of the module, such as …\nRuns the module in the previously provided <code>Environment</code>.\nReturns variables not defined within the AST node, …\nCombines this module with the specified <code>Environment</code>. The …\nAnalogue of <code>Self::with_env()</code> that modifies the provided …\nFunction that checks whether all of array items satisfy …\nFunction that checks whether any of array items satisfy …\nFunction generating an array by mapping its indexes.\nLocation within an array.\nAssertion function.\nAssertion that two values are close to each other.\nEquality assertion function.\nAssertion that the provided function raises an error. …\nBinary function wrapper.\nComparator functions on two primitive arguments. All …\nActs similarly to the <code>dbg!</code> macro, outputting the …\nAllows to define a value recursively, by referencing a …\nGeneric error output encompassing all error types …\nFilter function that evaluates the provided function on …\nWrapper of a function containing information about its …\nReduce (aka fold) function that reduces the provided tuple …\nError raised when a value cannot be converted to the …\nError kinds for <code>FromValueError</code>.\nElement of the <code>FromValueError</code> location.\n<code>if</code> function that eagerly evaluates “if” / “else” …\nConverts type into <code>Value</code> or an error. This is used to …\nMismatch between expected and actual value type.\nFunction returning array / object length.\nMap function that evaluates the provided function on each …\nReturns the maximum of the two values. If the values are …\nFunction that merges two tuples.\nError message. The error span will be defined as the call …\nReturns the minimum of the two values. If the values are …\nFunction that appends a value onto a tuple.\nQuaternary function wrapper.\nReturns an <code>Ordering</code> wrapped into an <code>OpaqueRef</code>, or …\nError together with the defined span(s).\nTernary function wrapper.\nFallible conversion from <code>Value</code> to a function argument.\nLocation within a tuple.\nUnary function wrapper.\nLoop function that evaluates the provided closure while a …\nReturns the zero-based index of the argument where the …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nPerforms the conversion.\nReturns the error kind.\nReturns the error location, starting from the outermost …\nCreates a new wrapper.\nCreates a function with the specified tolerance threshold. …\nCreates an assertion function with a custom error matcher. …\nAttempts to convert <code>value</code> to a type supported by the …\nWraps a function enriching it with the information about …\nActual value type.\nExpected value type.\nZero-based index of the erroneous element.\nZero-based index of the erroneous element.\nTuple size.\nFactual array size.")