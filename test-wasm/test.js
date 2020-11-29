#!/usr/bin/env node

const { strict: assert } = require('assert');
const { evaluate } = require('./pkg')

// Normal cases.
const evaluated = evaluate(`
  // The interpreter supports all parser features, including
  // function definitions, tuples and blocks.
  order = |x, y| (min(x, y), max(x, y));
  assert_eq(order(0.5, -1), (-1, 0.5));
  (_, M) = order(3^2, { x = 3; x + 5 });
  M`)
assert.equal(evaluated, 9)

const evaluatedFlag = evaluate(`
  max_value = |...xs| {
      fold(xs, -Inf, |acc, x| if(x > acc, x, acc))
  };
  max_value(1, -2, 7, 2, 5) == 7 && max_value(3, -5, 9) == 9
`)
assert(evaluatedFlag)

// Parse errors.
assert.throws(() => evaluate('1 +'), /1:4: Unfinished arithmetic expression/)

// Evaluation errors.
assert.throws(() => evaluate('2 + test(1, 2)'), /1:5: Variable `test` is not defined/)
