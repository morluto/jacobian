# Recurrences and rational generating series

[Documentation home](../index.md)

- Status: Experimental exact combinatorics contract
- Producer: pinned SymPy 1.14.0
- Independent replay: standard-library `Fraction` clean-process checker
- Producer assurance: at most `COMPUTED`

The explicit `combinatorics` bundle includes two generic, bounded operations:

- `combinatorics.recurrence.linear.evaluate`; and
- `combinatorics.generating_function.coefficients.compute`.

They complement rather than replace the named Fibonacci, consecutive-Fibonacci,
and Lucas capabilities. `combinatorics.result.verify` independently replays a
stored result from either generic producer when bundled checker authorization
is enabled.

## Constant-coefficient recurrence evaluation

`combinatorics.recurrence.linear.evaluate` uses the explicit convention

\[
a_n=\sum_{j=1}^{d} c_j a_{n-j},
\]

where the supplied coefficient vector is \((c_1,\ldots,c_d)\) and the initial
vector is exactly \((a_0,\ldots,a_{d-1})\). Requests select either a consecutive
prefix or a strictly increasing list of indices.

The result returns the requested indexed values and the complete recurrence
prefix from zero through the greatest requested index. The latter is a
first-class replay artifact inside the result, not a claim that a sparse
projection alone certifies the recurrence. The independent checker reconstructs
every rational with the Python standard library and replays every initial value
and recurrence step.

## Rational generating-function coefficients

`combinatorics.generating_function.coefficients.compute` accepts canonical
ascending coefficient vectors for \(N(x)\) and \(D(x)\), requires \(D(0)\ne0\),
and computes exactly one finite coefficient prefix of

\[
A(x)=\frac{N(x)}{D(x)}
\]

at expansion point zero.

For a requested order \(k\), the result contains exactly \(a_0,\ldots,a_{k-1}\)
and the coefficient vector of

\[
D(x)\sum_{i=0}^{k-1}a_i x^i-N(x)\pmod{x^k}.
\]

Every reported residual coefficient is exactly zero. This establishes only the
declared finite congruence through \(x^{k-1}\); it does not materialize or claim
an infinite power series. The independent checker recomputes the prefix and
the truncated polynomial product without importing SymPy or producer code.

## Bounds and canonical input

Version 2 of the combinatorics semantics uses these fail-closed limits:

| Quantity | Limit |
| --- | ---: |
| Linear recurrence order | 16 |
| Greatest recurrence index | 512 |
| Sparse requested indices | 256 |
| Numerator or denominator degree | 32 |
| Rational-series truncation order | 512 |
| Input rational numerator/denominator digits | 64 |
| Result rational numerator/denominator digits | 32,768 |

Rationals must be reduced with a positive denominator. Polynomial coefficient
vectors are ascending and omit trailing zeros, except that the zero polynomial
is represented by the one-term vector `[0]`. Cross-field validation completes
before computation or artifact writes.

## Assurance boundary

SymPy computes candidates at exact rational arithmetic and the producer records
its pinned runtime identity, but the result remains `COMPUTED`. Verification
binds the exact input artifact, result artifact, combinatorics semantics,
witness format, candidate digest, checker source identity, and verification
record.

Malformed results, false recurrence terms, false series coefficients, nonzero
or forged residuals, lineage mismatch, checker interruption, and runtime drift
produce no verification record and no opposite mathematical conclusion.

The public overlap regression freezes four discovery intents: named Fibonacci
and Lucas queries must continue to select their named capabilities, while
generic recurrence and rational-series queries select the new operations. This
is deterministic catalog/discovery harness validation, not a model-in-the-loop
performance or statistical claim.
