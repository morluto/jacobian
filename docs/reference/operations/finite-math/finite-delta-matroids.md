# Finite delta-matroids

[Documentation home](../../../index.md) · [Tool surface](../../tools.md) · [Operation references](../index.md) · [This domain](index.md)

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

`delta_matroid.twist.compute` returns a `DeltaMatroidTwistResult` that retains
the source `FiniteDeltaMatroid`, the sorted ground-index subset, and the exact
twisted target. Its postcondition is that each feasible row `F` in the source
maps bijectively to `F △ A` in the target, on the same ordered ground axis.
Twisting preserves the feasible-row count and ground axis; both source and
target therefore use the same admitted membership and label bounds. The native
`twist` function continues to return the target `FiniteDeltaMatroid` directly.
This adds the source-bound twist result contract only; it does not complete the
broader delta-matroid capabilities in issue #1954.
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
output to 4,096 entries and 32,768 retained allocation units (one per entry
index position, per decimal digit of each count, and per ground label
codepoint) before validating the source exchange axiom.
`delta_matroid.direct_sum.compute` combines two values whose labelled grounds
are disjoint. Its ground axis concatenates the left and right grounds, with the
corresponding index injections returned explicitly. Each output feasible set
is the union of one left feasible set and one right feasible set. Admission
revalidates both native operands as canonical carriers, then checks the
combined ground size, feasible-pair count, and result feasible memberships
before constructing any pairwise unions. The operation accepts at most 2,048
ground labels and 250,000 feasible-set pairs, while each source and the result
must fit the carrier's 16,384-entry membership and 2,048-label-byte envelopes.
The product family satisfies symmetric exchange by the direct-sum theorem, so
admission does not charge a recognition replay this operation never performs.
Input ground labels must be disjoint; overlapping names are rejected rather
than silently tagged.
Graph conversions, other interlace or transition polynomials, and graph
local-complement profiles remain outside the current contract.
minors, binary principal-minor reconstruction, relabelling, and the
subset-distance interlace polynomial. These are separate mathematical
postconditions rather than fields of the recognition result.

`delta_matroid.twist.compute` returns the canonical twisted
`FiniteDeltaMatroid`. `delta_matroid.width.compute` returns the exact width
`max(|F|)-min(|F|)` over the complete feasible family.

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
relabelling work are admitted before target feasible rows are materialized. Relabelling preserves the symmetric-exchange axiom because
a bijection preserves symmetric difference and membership.
`delta_matroid.twist.compute` returns the canonical twisted `FiniteDeltaMatroid`.
Width `max(|F|)-min(|F|)` is a native projection of the feasible family and is
not a catalog operation. It scans every retained feasible-row length of the
canonical value and has no extra row ceiling.

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

`delta_matroid.distance_profile.compute` returns the distance from every
ground subset `X` to the nearest feasible set, where
`d_D(X)=min_{F in feasible(D)} |X symmetric_difference F|`. The profile also
returns the number of nearest feasible sets for each subset and the histogram
of distances. Subsets use integer masks in ascending order; bit `i` denotes
ground index `i`. This is an exact complete profile, not a prefix or a nearest
set witness.

Admission first limits the complete subset domain to 4,096 masks and the
product of subset states and feasible rows to 262,144 distance evaluations.
The source family is then admitted under its existing membership, label-byte,
and symmetric-exchange bounds. These limits are separate: a small feasible
family over too many ground elements exceeds the state bound, and a larger
family on an otherwise admissible ground set can exceed the distance-work
bound. The empty-ground delta-matroid has one mask and distance zero.

`delta_matroid.binary.from_matrix_twist.compute` constructs `D(A)*T` from a
labelled symmetric matrix `A` over GF(2) and a sorted ground-index subset `T`.
The convention is `X` feasible in `D(A)` exactly when the principal matrix
`A[X]` is nonsingular, with the empty principal matrix nonsingular; the twist
then maps each feasible index set `X` to `X symmetric_difference T`. The
returned binary presentation retains `A`, `T`, and the complete resulting
`FiniteDeltaMatroid` on the same labelled ground axis.

The principal-minor kernel admits at most eight ground elements and 250,000
elimination work units before enumerating subsets. The twist transport is
bounded by `2^n (1 + 2n + n^2)` work units, at most 256 rows, and at most
`n 2^(n-1) <= 1,024` feasible-row memberships (`0` memberships for the empty
ground). The retained result has at most 1,376 matrix, row, membership, and
twist cells and at most 4,096 aggregate ground-label bytes across its matrix
and delta-matroid axes. These are mathematical allocation bounds; they are not
a deployment wire-byte ceiling.

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
`delta_matroid.twist_polynomial.compute` returns the exact twist polynomial
\[
  \mathrm{Tw}_D(z)=\sum_{A\subseteq E}z^{\mathrm{width}(D*A)},
\]
with a coefficient for every width from zero through `|E|`. The coefficients
are also returned as Jacobian's canonical descending-degree `IntegerPolynomial`
in the formal variable `z`; the complete histogram retains trailing zero
widths and the labelled ground axis. The coefficient sum is exactly
`2**|E|` because every ground subset supplies one twist. The operation admits
at most 12 ground elements, 4,096 twist subsets, and 262,144
twist-subset/feasible-set evaluations before it replays source exchange and
computes the complete histogram. These bounds also limit the result to 13
histogram entries, with each coefficient at most 4,096 (four decimal digits).
The kernel holds at most one bit mask per admitted feasible row; source
admission limits memberships to 16,384, so at most 16,385 rows (including the
unique empty row) are materialized as masks. Labels never enter the mask sweep,
so the recognition operation's 2,048-byte label envelope does not apply here.
Native source revalidation and UTF-8 checks copy retained labels, however, so
this operation admits at most 1,000,000 aggregate ground-label codepoints
before copying them. Labels must remain unique and UTF-8-representable;
wire-byte limits belong to the delivery boundary.
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
