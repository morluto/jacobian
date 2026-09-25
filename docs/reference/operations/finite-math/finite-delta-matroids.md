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

The recognition result itself deliberately does not carry twists, minors,
binary matrix presentations, graph conversions, or interlace polynomials.
Those are separate mathematical postconditions rather than fields of the
recognition result.

`delta_matroid.twist.compute` returns the canonical twisted `FiniteDeltaMatroid`.
`delta_matroid.width.compute`, `delta_matroid.dual.compute`,
`delta_matroid.minor.compute`, and `delta_matroid.from_binary_matrix.compute`
are separate published operations.

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
ground). The retained result has at most 1,352 matrix, row, membership, and
twist cells and at most 4,096 aggregate ground-label bytes across its matrix
and delta-matroid axes. These are mathematical allocation bounds; they are not
a deployment wire-byte ceiling.
