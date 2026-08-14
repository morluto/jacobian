# Typed polynomial expression normalization

`polynomial.expression.normalize` converts one bounded typed expression over an
explicit rational polynomial ring to canonical sparse coefficients.
`polynomial.expression_normalization.verify` independently checks the stored
relation. Provider computation and verification are separate trust boundaries.

## Typed input

The source artifact declares `QQ`, one ordered tuple of 1 to 4 distinct
variables, and a version-1 expression AST. The only node kinds are:

- `rational`, with a reduced canonical numerator and positive denominator;
- `variable`, whose name must occur in the declared tuple;
- `add` and `multiply`, each with 2 to 16 operands;
- `negate`; and
- `power`, with an integer exponent from 0 to 32.

Power zero denotes the polynomial-ring identity, including for a syntactic zero
base. The contract excludes formula strings, expression division, functions,
assumptions, and branch-sensitive operations. Jacobian constructs SymPy
objects directly from validated nodes and never passes user text to `sympify`
or `parse_expr`. This boundary matters because the
[SymPy parsing documentation][sympy-parsing] warns that `parse_expr` uses
`eval` and must not receive unsanitized input.

The complete tree is limited to 128 nodes and depth 16. Validation also rejects
an expression when conservative bounds exceed:

- the 1,024-term expanded-term budget;
- exponent 127 for any declared variable;
- 256 decimal digits in one rational numerator or denominator; or
- a 4,096-digit conservative coefficient-arithmetic budget.

An expansion-bound failure uses
`EXPANSION_TERM_BUDGET_EXCEEDED` at the `bounded_normalization` stage. Its
structured details report the 1,024-term limit, the conservative expanded-term
upper bound and its bound kind, the largest requested power when present,
alternatives, and
`retryable_with_same_input: false`. It also reports null normalization evidence
and unavailable checker input. This is a resource-bound non-conclusion, not an
invalid-variable diagnostic, and validation occurs before artifact writes.

The request includes a wall-clock budget from 1 to 60 seconds; the default is
10 seconds.

## Computed normalization

The base environment pins SymPy 1.14.0 for this profile. An isolated process
constructs rational literals and symbols from the AST, then calls
`Poly(expression, *variables, domain=QQ)`. SymPy's
[polynomial domain documentation][sympy-domains] defines `QQ` as the rational
field used for exact polynomial coefficients.

The result omits zero coefficients and orders terms by descending
lexicographic exponent tuple. Jacobian records the source identity, full
canonical polynomial, SymPy distribution digest, exact operation profile, and
resource budget. Successful provider output has `COMPUTED` assurance and
`conclusion: UNKNOWN`; it does not verify its own equivalence claim.

Timeout, process failure, excessive output, a malformed worker response, or a
runtime identity change produces no normalization artifact and no mathematical
conclusion.

The operation normalizes one concrete bounded expression. Any finite
collection of such results remains finite evidence and does not verify an
identity parameterized over all exponents or orders. If the conservative
expanded-term bound exceeds the hard limit, the diagnostic reports
`EXPANSION_TERM_BUDGET_EXCEEDED`, marks the same input non-retryable, exposes no
checker payload, and states that increasing the exponent or size under the same
full-expansion approach is not a universal-proof path.

## Independent verification

With bundled references enabled,
`polynomial.expression_normalization.verify` runs an operator-authorized
standard-library checker in a clean process. It imports neither SymPy nor the
producer. The checker:

1. validates artifact schemas, semantics, payload digests, exact source
   bindings, provider profile, resource budget, and parent lineage;
2. independently validates every AST node and all structural and arithmetic
   bounds;
3. recursively expands the complete AST with `fractions.Fraction`;
4. independently validates canonical sparse output, including reduced
   rationals, exponent dimensions, unique monomials, and term order; and
5. compares every exact coefficient by exponent tuple.

Only full equality creates a verification record and returns `VERIFIED`.
Mismatch, malformed evidence, timeout, cancellation, or checker failure returns
`UNKNOWN`; rejection does not establish an opposite mathematical claim.

## Artifact binding

The source expression and normalization candidate are immutable artifacts
under one versioned semantics descriptor. The candidate binds:

- the source URI, object digest, and payload digest;
- the exact ordered variable tuple;
- independently reproducible node, depth, term, and coefficient bounds;
- the complete canonical sparse polynomial;
- the pinned SymPy runtime identity and operation profile; and
- the enforced wall-clock budget.

The verification witness separately binds the exact source, candidate,
semantics, and checker identity. Replacing any node, coefficient, exponent,
digest, binding, runtime profile, or lineage edge invalidates replay.

[sympy-domains]: https://docs.sympy.org/1.14.0/modules/polys/domainsref.html
[sympy-parsing]: https://docs.sympy.org/1.14.0/modules/parsing.html
