# Exact rational matrix determinants

`matrix.determinant.compute` computes one exact determinant for a square matrix
over `QQ`. `matrix.determinant.verify` independently recomputes and checks an
authoritative inline input/result pair. Computation and verification are
separate trust boundaries.

## Input and result contracts

The producer accepts one `DeterminantRationalMatrix`: a nonempty square matrix
with order 1 through 64. Every entry is a canonical reduced rational:

```json
{"num": "-3", "den": "7"}
```

The determinant request applies a shape-aware scalar budget before computation.
It is at most 256 decimal digits for small matrices and decreases as the
matrix order grows; order 64 permits 7 digits per numerator or denominator.
This conservative bound accounts for the product of all input denominators and
the Leibniz sum, and keeps every accepted inline determinant within the
32,768-digit `CanonicalRational` result limit. A request that exceeds the
shape-specific budget is rejected before provider execution.

`matrix.determinant.compute` uses SymPy's exact matrix determinant API with the
fraction-free Bareiss method. It returns the bounded canonical determinant
inline with its method and backend version; ordinary computation writes no
artifact.

The compute result creates no verification record. Backend success is an
execution result, not independent verification. SymPy documents both the
[`Matrix.det` API][sympy-det] and its Bareiss method.

## Independent verification

With bundled references enabled, `matrix.determinant.verify` accepts the same
bounded `MatrixDeterminantRequest` input and `MatrixDeterminantResult` candidate
models used by computation. It runs an operator-authorized Python-FLINT checker
in a clean process and binds canonical digests for that exact inline pair. The
verifier accepts only when all of the following hold:

1. input, candidate, witness, semantics, and canonical digest bindings agree;
2. the input is a square matrix and the candidate has the exact determinant
   result shape;
3. all rational values are canonical, reduced, and have positive
   denominators; and
4. exact Python-FLINT replay recomputes the declared determinant.

The checker does not import SymPy or producer code. Accepted replay creates a
bound witness and verification record and may report `VERIFIED`. A wrong value,
malformed binding, timeout, cancellation, or checker error reports `UNKNOWN`
and creates no verification record.

Verification covers only the equality

```text
declared value = det(stored source matrix)
```

It does not separately conclude invertibility, rank, orientation, volume, or
any downstream theorem. An agent can compose those later from the inline value
or from the associated verification record.

[sympy-det]: https://docs.sympy.org/latest/modules/matrices/matrices.html#sympy.matrices.matrixbase.MatrixBase.det
