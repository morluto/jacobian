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

`delta_matroid.twist.compute` returns the canonical twisted `FiniteDeltaMatroid`.
`delta_matroid.distance.compute` returns the exact minimum
`|X symmetric_difference F|` over every feasible set `F`, together with the
lexicographically first nearest feasible set. Its work is linear in the
admitted complete feasible family after source exchange admission.
`delta_matroid.width.compute` returns the exact width
`max(|F|)-min(|F|)` over the complete feasible family. The dual and
deletion/contraction minor operations return complete reusable delta-matroid
values. `delta_matroid.from_binary_matrix.compute` enumerates all principal
submatrices of an admitted symmetric `GF(2)` matrix and selects exactly those
with nonzero determinant.

`delta_matroid.twist_width_profile.compute` returns one width for every subset
of the ground set. The tuple position is the integer subset mask, with bit
`i` selecting ground element `i`; thus position zero is the untwisted width.
Before exchange validation or profile computation, admission bounds the state
count to 4,096 and the product of states and feasible rows to 262,144. The
profile is complete within this admitted scope; an over-limit request is
rejected, never returned as a partial profile.

`delta_matroid.feasible_size_profile.compute` returns the exact histogram
`(c_0, ..., c_|E|)`, where `c_k` counts feasible sets with cardinality `k`.
The result retains the labelled ground axis. Admission bounds the complete
output to 4,096 entries and 32,768 encoded bytes before validating the source
exchange axiom.

`delta_matroid.direct_sum.compute` combines two values whose labelled grounds
are disjoint. Its ground axis concatenates the left and right grounds, with the
corresponding index injections returned explicitly. Each output feasible set
is the union of one left feasible set and one right feasible set. Admission
checks the combined ground size, feasible-pair count, feasible memberships,
symmetric-exchange work ceiling, and a conservative encoded-result size before
constructing any pairwise unions. The operation accepts at most 2,048 ground
labels, 250,000 feasible-set pairs, and 2,000,000 estimated result bytes, while
the result must also fit the carrier's 16,384 memberships and 250,000 exchange
candidate checks. Input ground labels must be disjoint; overlapping names are
rejected rather than silently tagged.

Graph conversions, lower/upper matroids, subset-distance profiles, other
direct-sum variants, relabelling, parity/size distributions, and interlace
polynomials remain
outside the currently published delta-matroid operations.
