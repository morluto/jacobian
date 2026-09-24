# Finite delta-matroids

[Finite mathematics operations](index.md) · [Tool surface](../../tools.md)

`delta_matroid.from_feasible_sets.compute` recognizes one complete bounded
family of feasible subsets under the symmetric-exchange axiom. It returns a
canonical `FiniteDeltaMatroid` when the family is valid, or the first
deterministic obstruction when it is not.

`FiniteDeltaMatroid` retains the labelled ground set and the complete feasible
family. Feasible rows use sorted ground-index tuples and are serialized in
lexicographic order. Omitted rows are infeasible; they are never unknown.

The operation checks every ordered pair of feasible sets and every element of
their symmetric difference. It bounds the complete family before executing its
axiom and explicit verification passes: 16,384 total row memberships, 2,048
UTF-8 bytes of ground labels
(each label must be UTF-8-representable), 250,000 symmetric-exchange candidate
checks per complete axiom replay. There
are deliberately no separate delta-matroid ground-size or row-count caps: the
membership envelope bounds the row count, while label and candidate-work
bounds control the actual work, so a sparse family over hundreds of
short labels — or a dense short-row family such as every subset of size at most
two over 16 elements (137 rows, 220,832 candidate checks) — is admitted when
these bounds hold.

One recognized request performs at most four complete axiom passes across its
operation and explicit bounded verification obligations: the obstruction
decision, canonical value construction, defining-invariant verification, and
result-binding obstruction check. The aggregate worst case is therefore
1,000,000 candidate checks per accepted call, which is part of this operation's
advertised envelope rather than a universal result-construction rule.

The catalog also exposes exact twists, widths, duals, deletion/contraction
minors, and binary principal-minor reconstruction. These are separate
mathematical postconditions rather than fields of the recognition result.
Graph conversions, interlace and transition polynomials, lower/upper matroids,
and graph local-complement profiles remain outside the current contract.

`delta_matroid.twist.compute` returns the canonical twisted
`FiniteDeltaMatroid`. `delta_matroid.width.compute` returns the exact width
`max(|F|)-min(|F|)` over the complete feasible family.

`delta_matroid.relabel.compute` renames and reorders the ground axis through a
bijective `target_to_source` map. Each feasible subset is transported by the
inverse `source_to_target` map, with its indices sorted in the target axis; the
result retains both source and target delta-matroids and both maps. The empty
ground set and the identity permutation are valid. The operation admits at most
2,049 ground positions, 16,384 source feasible-set memberships, 2,048 UTF-8
bytes of target labels, 329,784 units of axis-check and row-transport work, 2,000,000
estimated output bytes, and 2,829,784 total reserved work units, including two
250,000-candidate source-exchange passes for admission and recognition. Source
exchange checks and relabelling work are admitted before target feasible rows
are materialized. Relabelling preserves the symmetric-exchange axiom because
a bijection preserves symmetric difference and membership.
