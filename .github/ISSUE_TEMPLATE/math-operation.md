---
name: Admitted mathematical operation
about: Propose one admitted, user-facing operation with a stable postcondition
title: "[Operation]: "
labels: ["request: math-operation"]
assignees: []
---

## Parent gap / RFC
<!-- Link the gap issue or domain RFC this operation satisfies. -->

## Public mathematical postcondition
<!-- One sentence stating exactly what the operation establishes or returns. -->

## Proposed operation ID
<!-- `domain.noun.verb` -->

## Input
<!-- The exact mathematical values required. -->

## Result
<!-- The returned mathematical value, witness, certificate, or closed outcome. -->

## Completeness semantics
<!-- When may the result say EXACT, DOES_NOT_EXIST, UNKNOWN, UNRESOLVED, or reject the request as out of scope? -->

## Bounds and cost class
<!-- Public input, search, output, coefficient, dimension, and time budgets. -->

## Maintained implementation
<!--
Identify the implementation class:

- maintained in-process Python backend;
- thin native binding;
- bounded Jacobian-owned kernel; or
- child-process backend.

Explain why its dependency and operational cost are proportionate to the
admitted mathematical domain. List optional reference implementations
separately; they are evidence, not runtime dependencies.
-->

## Defining invariant
<!--
How can an independent validator tell that a returned object is mathematically
valid? State the reconstruction equation, defining identity, or preservation
law. Do not use "the backend returned it" as the invariant.

Examples: a factorization reconstructs the input from its unit and factors; an
inverse satisfies both matrix products; a decomposition satisfies its local
rules and reconstructs the original object; a Groebner basis generates the
same ideal and its relevant S-polynomials reduce to zero.
-->

## Evidence plan
<!--
Which tests will establish the invariant and catch plausible weaker or
incorrect implementations? Include the applicable evidence:

- a known-answer or source-backed reference fixture when conventions matter;
- boundary, degenerate, and adversarial cases;
- reconstruction, defining-identity, or preservation tests;
- a bounded exhaustive check or independent differential oracle when
  proportionate.

For non-unique results, state the mathematical equivalence to compare instead
of requiring incidental ordering, temporary identifiers, or a particular
witness.
-->

## Why this is not caller-side composition
<!-- Which indispensable mathematical postcondition is not already available? -->

## Deferred operation family
<!-- Related operations explicitly not part of this issue. -->
