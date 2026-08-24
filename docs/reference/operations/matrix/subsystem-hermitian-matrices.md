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
Kronecker and partial-trace admission fix no per-operand ceilings: preflight
derives the operation's exact result coefficients and admits them against
the documented component envelopes — 256 decimal digits for one product
component and 4,098 digits for one partial-trace component — and partial
trace additionally charges its actual contracted intermediates, cancelling
equal-denominator terms before any cross-denominator addition so delayed
cancellation is measured as executed, against a 16,392-digit work envelope.
Because admission measures the emitted values themselves, the envelopes stay closed
under composition: a produced matrix re-enters its own consumer unchanged
whenever its next exact result fits, so a 200-digit operand still composes
with an identity operand at the 256-digit boundary, while rejection means
the requested computation's true coefficients exceed the envelope rather
than that an input estimate did. PSD order likewise has no fixed component
ceiling: it measures the exact reduced `right - left` components, admits
the pair when they stay within 513 digits and the dimension-scaled
replayable-witness bound stays within the canonical rational limit, and
identical or nearly equal operands whose reduced difference is tiny
therefore admit trivially; because the result echoes both operands and
their difference, admission also reserves the serialized transport budget,
so every accepted call returns its typed result. A large Kronecker product
therefore does not imply that every downstream decision fits in one bounded
call. The kernels are exact rational computations; they do not use
floating-point matrix predicates or a tensor registry. The catalog schemas
and invocation examples are authoritative for wire shapes.
