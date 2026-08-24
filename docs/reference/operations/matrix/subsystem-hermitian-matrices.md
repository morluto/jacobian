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

The admitted product dimension is at most 16, with at most four factors.
Kronecker admission couples the two operands rather than fixing per-operand
ceilings: preflight bounds every product component by the larger of
`left_num + right_num` and `left_den + right_den` decimal digits against a
combined 256-digit envelope, so one 200-digit operand paired with a one-digit
operand fits while two moderately wide operands may not. Partial trace accepts
those produced components subject to its traced common-denominator sum bound
of 4,098 digits. PSD order likewise has no fixed component ceiling: it admits
operands whose derived `right - left` component bound stays within 513 digits
and whose dimension-scaled replayable-witness bound stays within the canonical
rational limit, so operands above 256 digits are accepted whenever those
derived bounds fit. A large Kronecker product therefore does not imply that
every downstream decision fits in one bounded call. The kernels are exact
rational computations; they do not use floating-point matrix predicates or a
tensor registry. The catalog schemas and invocation examples are authoritative
for wire shapes.
