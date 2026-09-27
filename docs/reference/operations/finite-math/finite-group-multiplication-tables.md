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

The `lattice_gauge.finite_group.holonomy.compute` operation composes edge
indices from one `FiniteGroupTable` along an oriented path. Backward traversal
uses the table's inverse, and multiplication follows path order. Its distinct
`FiniteGroupGaugeField` retains the table as the exact group parent. It does
not change the existing permutation-valued `S_d` or rational `SU(2)` field
contracts. The operation computes open-path holonomy and does not claim
plaquette observables or Wilson traces for arbitrary groups.

The multiplication-table value stores the exact indexed operation and its
derived inverse map. Consumers accepting a caller-supplied value must re-admit
the group laws they rely on; construction is the operation that establishes
those laws for its own result.
