# Exact subsystem-aware Hermitian matrices

[Documentation home](../../../index.md) · [Matrix operations](index.md) ·
[Tool surface](../../tools.md)

`FactorizedHermitianMatrix` binds one square rational symmetric matrix to an
ordered tuple of named subsystem factors. Its basis linearization is always
lexicographic with the final factor varying fastest: for factors `(Y, Z)` the
coordinate order is `(Y0,Z0), (Y0,Z1), (Y1,Z0), (Y1,Z1)`. Equal dimensions do
not make differently labelled or reordered factors interchangeable.

The bounded operations consume and produce this value directly:

- `matrix.subsystem.kronecker_product.compute` forms one matrix product and
  concatenates its factor order;
- `matrix.subsystem.partial_trace.compute` traces named factors and retains
  every remaining factor in source order; and
- `matrix.subsystem.psd_order.decide` decides whether `right - left` is
  positive semidefinite only when both sources have exactly the same factor
  binding. Its result retains both sources, their exact difference, inertia,
  and a rational negative quadratic witness when the order fails.

The admitted product dimension is at most 16, with at most four factors. A
Kronecker operand component has at most 128 decimal digits and the product
preflight proves every resulting component has at most 256. Partial trace and
PSD order accept those 256-digit produced components; their own preflight
accounts respectively for the traced common-denominator sum and for the
`right - left` difference plus a replayable rational witness. The kernels are
exact rational computations; they do not use floating-point matrix predicates
or a tensor registry. The catalog schemas and invocation examples are
authoritative for wire shapes.
