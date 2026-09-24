# Bounded finite groups from multiplication tables

`finite_group.table.construct.compute` admits an indexed multiplication table
of order at most 24 after checking its proposed two-sided identity, unique
two-sided inverses, and associativity. It returns a `FiniteGroupTable` with a
derived inverse-index tuple and a `FiniteGroupTableElement` for the identity
that retains its parent group. Indices are part of the value: relabelling the
same abstract group gives a different table value.

The bound is based on the cubic associativity check: an order-24 table requires
at most 13,824 triple comparisons, with at most 576 table entries. S3 is
included as a noncommutative fixture. The carrier establishes exact finite
group laws for this indexed table; it does not provide an isomorphism
canonicalizer or arbitrary matrix representation.

The current `lattice_gauge.*` operations accept permutation labels in `S_d`,
not elements of an arbitrary supplied subgroup and not `FiniteGroupTable`
elements. A future gauge integration should bind each edge/frame value to one
group parent and ensure closure in that exact group. The present carrier alone
does not make arbitrary finite-group gauge fields accepted.

The multiplication-table value stores the exact indexed operation and its
derived inverse map. Consumers accepting a caller-supplied value must re-admit
the group laws they rely on; construction is the operation that establishes
those laws for its own result.
