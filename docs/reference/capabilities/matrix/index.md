# Matrix capability references

[Documentation home](../../../index.md) · [Capability surface](../../tools.md)

## Shared matrix values

`RationalMatrix` and `IntegerMatrix` in
[`jacobian.contracts.matrices`](../../../../src/jacobian/contracts/matrices.py)
are the authoritative shared inline matrix values across the matrix and
lattice domains. Both are bounded Pydantic `ContractModel` types with
rectangular-shape and scalar-digit validators. They are not operation-specific:
the same `RationalMatrix` carries a determinant input, an RREF result, a matrix
product, and a nullspace operand; the same `IntegerMatrix` carries an HNF
source, a Smith normal form, an adjugate, and an LLL reduced basis.

Operation request models in
[`jacobian.contracts.matrix_operations`](../../../../src/jacobian/contracts/matrix_operations.py)
embed `RationalMatrix` or `IntegerMatrix` and apply their own 256-digit
execution budget via `require_matrix_scalar_digits`. The shared matrix models
themselves permit up to 32,768 canonical digits; each operation request
tightens this to 256 decimal digits per scalar component before computation.
Result models reuse the same shared types directly—`RrefResult.reduced_matrix`
is a `RationalMatrix`, `SmithNormalFormResult.normal_form` is an
`IntegerMatrix`, and so on—so producer and consumer share one canonical wire
and persistence form without per-operation matrix schemas.

## Conversion and kernel layer

Contract-to-backend conversions and typed mathematical kernels live in
[`jacobian.domains.matrix_lattice`](../../../../src/jacobian/domains/matrix_lattice):
`conversions.py` translates between `RationalMatrix` / `IntegerMatrix` and
SymPy matrices using the canonical integer API, and `kernels.py` exposes typed
SymPy kernels (`rref`, `inverse`, `trace`, `characteristic_polynomial`,
`smith_normal_form`, `matrix_product`, `rational_linear_solve`, `adjugate`).
These modules sit below both the capability declarations in `capabilities.py`
and the native [`jacobian.math.matrices`](../../../../src/jacobian/math/matrices.py)
API. When a native function corresponds to a capability—`jacobian.math.matrices.rref`,
`.inverse`, `.trace`—both paths call the same kernel. The native API accepts
SymPy `MatrixBase` values directly and does not route through `math.run` or
construct a capability runtime.

## Artifact-backed capabilities

`matrix.normal_form.hermite` is deliberately artifact-backed: it stores the
source matrix, HNF candidate, and left transformation under versioned
`jacobian.matrix-normal-form` semantics because the full transformation needs
durable identity and independent retrieval.

`matrix.determinant.compute` and `matrix.rank.compute` are inline
`ComputedOperation` results in the `matrix_lattice` bundle. They write no
artifact: their exact scalar result and pivot columns remain directly reusable
in a later typed request. Their matching `.verify` capabilities accept those
authoritative inline request and result values directly, then create replay
evidence and a verification record only inside the independent-checker
boundary.

The remaining matrix and lattice operations—RREF, nullspace, matrix product,
inverse, trace, characteristic polynomial, Smith normal form (diagonal only),
adjugate, rational linear solve, and LLL basis reduction—are declared in the
`matrix_lattice` bundle. The computed operations return results inline without
artifacts; LLL basis reduction is a `MaterializedOperation` that stores the
reduced basis and transformation because the bounded worker result needs
durable identity.

## Capability reference pages

- [Integer matrix Hermite normal form](matrix-hermite-normal-form.md)
- [Exact rational matrix determinants](matrix-rational-determinant.md)
