# Exact affine-semigroup factorization counts

`affine_semigroup.factorization_count.compute` returns the exact cardinality
of

```text
F_A(b) = {u in N^n : A u = b}
```

for one target on the retained row axis of a positive affine semigroup. A
strictly positive rational grading gives each generator positive degree, so
every fixed target fiber is finite. The operation returns the count bound to
its source semigroup and target, without materializing the factorization set.
The map `u |-> A u` and its factorization fibers are standard affine-semigroup
notions; see García-Sánchez, Ojeda, and Rosales, [*Affine semigroups having a
unique Betti element*](https://arxiv.org/abs/1203.4138), Section 1.

The current source envelope is at most 8 ambient rows and 10 generators, with
8-digit matrix entries, 64-digit grading numerators and denominators, and
32-digit target coordinates.

For one-row configurations, the operation divides by the gcd of the generator
weights and applies the coefficient recurrence for
`product_i (1 - z^w_i)^(-1)`. Each labelled generator contributes a separate
factor, so duplicate columns count as distinct coordinates. It admits at most
250,000 dynamic-program states and 2,000,000 updates. For higher-row
configurations, it uses the positive grading to bound every coefficient by
`floor(degree(b) / degree(a_i))`, then scans only that finite coefficient box.
The shared fiber admission limits the box to 50,000 candidate states; exact
traversal and matrix-check work is then limited to 2,000,000 units. A
combinatorial upper bound on the count digits is checked before either method
expands. These limits apply to mathematical states and exact scalar growth;
transport owns serialization limits.

The existing `affine_semigroup.factorizations.compute` returns every vector in
an admitted fiber. This count operation is useful when the count is the needed
postcondition and the vector set would be a larger result. It does not provide
a factorization witness or make a nonmembership claim from an interrupted or
rejected search.

For generators `(1,1)`, the target `4` has five factorizations, one for each
choice of the first generator's coefficient from zero through four. The empty
target fiber has count zero; the zero target has count one for a positive
semigroup.
