searchState.loadedDescShard("arithmetic_parser", 0, "Parser for arithmetic expressions with flexible definition …\nAddition (<code>+</code>).\nAddition or subtraction: <code>+</code> or <code>-</code>.\nBoolean AND (<code>&amp;&amp;</code>).\nBoolean AND (<code>&amp;&amp;</code>).\nAssigment, e.g., <code>(x, y) = (5, 8)</code>.\nAssigment, e.g., <code>(x, y) = (5, 8)</code>.\nMinimum length.\nBinary operation, e.g., <code>x + 1</code>.\nBinary operation, e.g., <code>x + 1</code>.\nBinary operation.\nBinary arithmetic operation.\nBinary operation.\nBlock of statements.\nBlock expression, e.g., <code>{ x = 3; x + y }</code>.\nBlock expression, e.g., <code>{ x = 3; x + y }</code>.\nFunction or method call.\nCast, e.g., <code>x as Bool</code>.\nChained comparison, such as <code>1 &lt; 2 &lt; 3</code>.\nComment.\nEquality and order comparisons: <code>==</code>, <code>!=</code>, <code>&gt;</code>, <code>&lt;</code>, <code>&gt;=</code>, <code>&lt;=</code>.\nParsing context.\nTuple destructuring, such as <code>(a, b, ..., c)</code>.\nRest syntax, such as <code>...rest</code> in <code>(a, ...rest, b)</code>.\nDivision (<code>/</code>).\nEquality (<code>==</code>).\nContains the error value\nParsing error with a generic code span.\nParsing error kind.\nExact length.\nArithmetic expression with an abstract types for type …\nExpression, e.g., <code>x + (1, 2)</code>.\nExpression, e.g., <code>x + (1, 2)</code>.\nArithmetic expression.\nExpression.\nType of an <code>Expr</code>.\nField access, e.g., <code>foo.bar</code>.\nField access, e.g., <code>foo.bar</code>.\nFunction definition, e.g., <code>|x, y| x + y</code>.\nFunction definition, e.g., <code>|x, y| { x + y }</code>.\nFunction definition, e.g., <code>|x, y| { x + y }</code>.\nFunction invocation.\nFunction call, e.g., <code>foo(x, y)</code> or <code>|x| { x + 5 }(3)</code>.\nFunction call, e.g., <code>foo(x, y)</code> or <code>|x| { x + 5 }(3)</code>.\n“Greater or equal” comparison.\n“Greater than” comparison.\nInput is incomplete.\nCode span.\n“Lesser or equal” comparison.\nLeftover symbols after parsing.\nLiteral (semantic depends on <code>T</code>).\nLiteral (semantic depends on the grammar).\nError parsing literal.\nLiteral is used where a name is expected, e.g., as a …\nCode span together with information related to where it is …\nValue with an associated code location. Unlike <code>Spanned</code>, …\n“Lesser than” comparison.\nAssignable value.\nLvalue.\nLength of an assigned lvalue.\nType of an <code>Lvalue</code>.\nMethod call, e.g., <code>foo.bar(x, 5)</code>.\nMethod call, e.g., <code>foo.bar(x, 5)</code>.\nMultiplication (<code>*</code>).\nMultiplication or division: <code>*</code> or <code>/</code>.\nNamed rest syntax, e.g., <code>...rest</code>.\nNegation (<code>-</code>).\nNumeric or Boolean negation: <code>!</code> or unary <code>-</code>.\nParsing outcome generalized by the type returned on …\nInput is not in ASCII.\nBoolean negation (<code>!</code>).\nNon-equality (<code>!=</code>).\nObject expression, e.g., <code>#{ x, y: x + 2 }</code>.\nObject expression, e.g., <code>#{ x = 1; y = x + 2; }</code>.\nObject destructuring, e.g., <code>{ x, y }</code>.\nObject destructuring, e.g., <code>{ x, y }</code>.\nObject destructuring, such as <code>{ x, y: new_y }</code>.\nSingle field in <code>ObjectDestructure</code>, such as <code>x</code> and <code>y: new_y</code> …\nObject expression, such as <code>#{ x, y: x + 2 }</code>.\nContains the success value\nGeneric operation, either unary or binary.\nPriority of an operation.\nBoolean OR (<code>||</code>).\nBoolean OR (<code>||</code>).\nOther parsing error.\nPower (<code>^</code>).\nPower (<code>^</code>).\nValue with an associated code span.\n<code>Expr</code> with the associated type and code span.\n<code>Lvalue</code> with the associated code span.\nStatement with the associated code span.\nStatement: an expression or a variable assignment.\nStatement.\nType of a <code>Statement</code>.\nSubtraction (<code>-</code>).\nTuple expression, e.g., <code>(x, y + z)</code>.\nTuple expression, e.g., <code>(x, y + z)</code>.\nTuple destructuring, e.g., <code>(x, y)</code>.\nTuple destructuring, e.g., <code>(x, y)</code>.\nError parsing type annotation.\nType cast, e.g., <code>x as Bool</code>.\nUnary operation, e.g., <code>-x</code>.\nUnary operation, e.g., <code>-x</code>.\nUnary operation.\nUnary operation.\nUnary operation.\nNo rules where expecting this character.\nUnexpected expression end.\nUnfinished comment.\nUnnamed rest syntax, i.e., <code>...</code>.\nUnary or binary operation switched off in the parser …\nDescription of a construct not supported by a certain …\nVariable name.\nVariable use, e.g., <code>x</code>.\nVariable use, e.g., <code>x</code>.\nSimple variable, e.g., <code>x</code>.\nSimple variable, e.g., <code>x</code>.\nFunction arguments, e.g., <code>x, y</code>.\nReturns a copy of this span with borrowed <code>extra</code> field.\nReturns the string representation of this operation.\nReturns LHS of the binary expression. If this is not a …\nReturns RHS of the binary expression. If this is not a …\nBinding for the field, such as <code>(x, ...tail)</code> in …\nFunction body, e.g., <code>x + y</code>.\nReturns optional error context.\nCopies this span with the provided <code>extra</code> field.\nCreates an empty block.\nEnd part of the destructuring, e.g., <code>c</code> in <code>(a, b, ..., c)</code>.\nExtra information that can be embedded by the user.\nExtra information that can be embedded by the user.\nExtra information that can be embedded by the user.\nExtra information that can be embedded by the user. …\nExtra information that can be embedded by the user.\nExtra information that can be embedded by the user.\nExtra information that can be embedded by the user.\nField name, such as <code>xs</code> in <code>xs: (x, ...tail)</code>.\nFields mentioned in the destructuring.\nFields. Each field is the field name and an optional …\nThe fragment that is spanned. The fragment represents a …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCreates a span from a <code>range</code> in the provided <code>code</code>. This is …\nCreates a location from a <code>range</code> in the provided <code>code</code>. This …\nCreates a location from a <code>range</code> in the provided <code>code</code>. This …\nCreates a span from a <code>range</code> in the provided <code>code</code>. This is …\nThe column of the fragment start.\nGrammar functionality and a collection of standard …\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nChecks if this operation is arithmetic.\nChecks if this operation is a comparison.\nChecks if the destructuring is empty.\nReturns <code>true</code> if this is <code>Incomplete</code>.\nChecks if this operation is an order comparison.\nChecks if the provided string is a valid variable name.\nReturns the kind of this error.\nReturns the length of destructured elements.\nCreates a <code>Literal</code> variant with the specified error.\nReturns the span of this error.\nThe line number of the fragment relatively to the input of …\nThe offset represents the position of the fragment …\nMaps the <code>extra</code> field of this span using the provided …\nMaps the fragment field of this span using the provided …\nChecks if this length matches the provided length of the …\nReturns the maximum priority.\nMiddle part of the destructuring, e.g., <code>rest</code> in …\nReturns a relative priority of this operation.\nReturns the priority of this operation.\nThe last statement in the block which is returned from the …\nReturns this location in the provided <code>code</code>. It is caller’…\nReturns this location in the provided <code>code</code>. It is caller’…\nStart part of the destructuring, e.g, <code>a</code> and <code>b</code> in …\nStatements in the block.\nTries to convert this rest declaration into an lvalue. …\nReturns a string representation of this location in the …\nReturns a string representation of this location in the …\nReturns the type of this expression.\nReturns type of this lvalue.\nReturns the type of this statement.\nRemoves <code>extra</code> field from this span.\nWrapper around parsers allowing to capture both their …\nType annotation of the value.\nVariable span, e.g., <code>rest</code>.\nParsing context.\nParsing context.\nParsing context.\n<code>nom</code>-defined error kind.\nFunction arguments.\nArguments; e.g., <code>x, 5</code> in <code>foo.bar(x, 5)</code>.\nInner expression.\nLHS of the operation.\nFunction value. In the simplest case, this is a variable, …\nName of the called method, e.g. <code>bar</code> in <code>foo.bar</code>.\nName of the called method, e.g. <code>bar</code> in <code>foo.bar(x, 5)</code>.\nOperator.\nOperator.\nReceiver of the call, e.g., <code>foo</code> in <code>foo.bar(x, 5)</code>.\nReceiver of the call, e.g., <code>foo</code> in <code>foo.bar(x, 5)</code>.\nRHS of the operation.\nSeparator between the receiver and the called method, …\nType annotation for the case, e.g., <code>Bool</code> in <code>x as Bool</code>.\nValue being cast, e.g., <code>x</code> in <code>x as Bool</code>.\nType annotation of the value.\nLHS of the assignment.\nRHS of the assignment.\nEnables parsing blocks.\nEnables all Boolean operations.\nEnables parsing equality comparisons (<code>==</code>, <code>!=</code>), the <code>!</code> unary …\nBase for the grammar providing the literal and type …\nType alias for a grammar on <code>f32</code> literals.\nType alias for a grammar on <code>f64</code> literals.\nFeatures supported by this grammar.\nEnables parsing function definitions.\nParsing features used to configure <code>Parse</code> implementations.\nExtension of <code>ParseLiteral</code> that parses type annotations.\nHelper trait allowing <code>Parse</code> to accept multiple types as …\nType of the literal used in the grammar.\nEnables parsing methods.\nList of mocked type annotations.\nTrait allowing to mock out type annotation support …\nSingle-type numeric grammar parameterized by the literal …\nNumeric literal used in <code>NumGrammar</code>s.\nEnables parsing objects.\nEnables parsing order comparisons (<code>&gt;</code>, <code>&lt;</code>, <code>&gt;=</code>, <code>&lt;=</code>).\nUnites all necessary parsers to form a complete grammar …\nEncapsulates parsing literals in a grammar.\nEnables parsing tuples.\nEnables parsing type annotations.\nType of the type annotation used in the grammar. This type …\nWrapper for <code>Grammar</code> types that allows to convert the type …\nWrapper for <code>ParseLiteral</code> types that allows to use them as …\nDecorator for a grammar that mocks type parsing.\nGet a flags value with all known bits set.\nThe bitwise and (<code>&amp;</code>) of the bits in two flags values.\nThe bitwise and (<code>&amp;</code>) of the bits in two flags values.\nThe bitwise or (<code>|</code>) of the bits in two flags values.\nThe bitwise or (<code>|</code>) of the bits in two flags values.\nGet the underlying bits value.\nThe bitwise exclusive-or (<code>^</code>) of the bits in two flags …\nThe bitwise exclusive-or (<code>^</code>) of the bits in two flags …\nThe bitwise negation (<code>!</code>) of the bits in a flags value, …\nWhether all set bits in a source flags value are also set …\nThe intersection of a source flags value with the …\nGet a flags value with all bits unset.\nEnsures that the child parser does not consume a part of a …\nThe bitwise or (<code>|</code>) of the bits in each flags value.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nConvert from a bits value.\nConvert from a bits value exactly.\nConvert from a bits value, unsetting any unknown bits.\nThe bitwise or (<code>|</code>) of the bits in each flags value.\nGet a flags value with the bits of a flag with the given …\nThe bitwise or (<code>|</code>) of the bits in two flags values.\nThe bitwise and (<code>&amp;</code>) of the bits in two flags values.\nWhether any set bits in a source flags value are also set …\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nConverts input into a span.\nWhether all known bits in this flags value are set.\nWhether all bits in this flags value are unset.\nYield a set of contained flags values.\nYield a set of contained named flags values.\nThe bitwise negation (<code>!</code>) of the bits in a flags value, …\nTries to parse a literal.\nAttempts to parse a literal.\nParses a list of statements.\nParses a list of statements.\nParses a potentially incomplete list of statements.\nParses a potentially incomplete list of statements.\nAttempts to parse a type annotation.\nThe intersection of a source flags value with the …\nCall <code>insert</code> when <code>value</code> is <code>true</code> or <code>remove</code> when <code>value</code> is …\nThe intersection of a source flags value with the …\nThe intersection of a source flags value with the …\nThe bitwise exclusive-or (<code>^</code>) of the bits in two flags …\nThe bitwise exclusive-or (<code>^</code>) of the bits in two flags …\nThe bitwise or (<code>|</code>) of the bits in two flags values.\nCreates a copy of these <code>Features</code> without any of the flags …")