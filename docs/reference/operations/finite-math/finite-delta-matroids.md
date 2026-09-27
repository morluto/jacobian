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
minors, binary principal-minor reconstruction, relabelling, and the
subset-distance interlace polynomial. These are separate mathematical
postconditions rather than fields of the recognition result. Graph conversions,
other interlace or transition polynomials, and graph local-complement profiles
remain outside the current contract.

`delta_matroid.twist.compute` returns the canonical twisted `FiniteDeltaMatroid`.
The native `lower_matroid` and `upper_matroid` conversions return canonical
`FiniteBasisMatroid` values directly, whose bases are all minimum-cardinality or
all maximum-cardinality feasible sets, respectively. These deterministic
projections remain available through the Python API without catalog
declarations. The source symmetric-exchange axiom is replayed at each operation
boundary, and the extremal-bases theorem establishes basis exchange for the
returned values. Constructing or deserializing a generic `FiniteBasisMatroid`
checks its canonical structure and cardinalities only; consumers of authored
carriers must call `require_basis_exchange()` before relying on the matroid
claim. The lower/upper matroid theorem is stated in Section 6.1 of Dupont, Fink,
and Moci, [*Universal Tutte characters via combinatorial
coalgebras*](https://doi.org/10.5802/alco.35).

These conversions retain the 16,384 source-membership, 2,048 UTF-8 source-label,
and 250,000 source exchange-work limits. The target `FiniteBasisMatroid` admits
at most 64 ground labels, 4,096 basis rows, 65,536 basis memberships, 1,024
UTF-8 bytes per label, 16,384 aggregate label bytes, and 2,000,000 worst-case
basis-exchange candidate checks. Output bounds are checked before materializing
the selected basis family. A source with more than 64 ground elements remains a
valid delta-matroid input but is refused for these conversions because the
canonical basis carrier cannot represent that output size.

`delta_matroid.width.compute` returns the exact width
`max(|F|)-min(|F|)` over the complete feasible family and has no extra row
ceiling.

`delta_matroid.relabel.compute` renames and reorders the ground axis through a
bijective `target_to_source` map. Each feasible subset is transported by the
inverse `source_to_target` map, with its indices sorted in the target axis; the
result retains both source and target delta-matroids and both maps. The empty
ground set and the identity permutation are valid. The operation admits at most
2,049 ground positions, 16,384 source feasible-set memberships, 2,048 UTF-8
bytes of target labels, 329,784 units of axis-check and row-transport work,
73,740 materialized result cells (retained labels, rows, memberships, and axis
maps), and 903,524 total reserved work units, including two 250,000-candidate
source-exchange passes for admission and recognition. Source exchange checks and
relabelling work are admitted before target feasible rows are materialized.
Relabelling preserves the symmetric-exchange axiom because a bijection preserves
symmetric difference and membership.

`delta_matroid.distance_interlace_polynomial.compute` returns the exact
distance histogram and the polynomial
`Q_D(x) = sum_{X subset E} (x - 1)^{d_D(X)}`, where
`d_D(X) = min_{F feasible} |X symmetric_difference F|`. Coefficients are
integers in descending-degree order, using the shared `IntegerPolynomial`
value. This is the distance specialization in [Brijder and Hoogeboom's
delta-matroid interlace-polynomial treatment](https://arxiv.org/abs/1010.4678),
with the variable shift fixed as `y = x - 1`; it does not imply any other
interlace polynomial convention. Admission validates the complete source,
then bounds `2^|E| * |F|` subset-feasible comparisons, the number of
output terms, and coefficient bit lengths before enumerating subsets.
