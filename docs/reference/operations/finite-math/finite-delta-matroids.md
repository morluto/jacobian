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
Width `max(|F|)-min(|F|)` is a native projection of the feasible family and is
not a catalog operation. It scans every retained feasible-row length of the
canonical value and has no extra row ceiling.

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
so the recognition operation's 2,048-byte label envelope does not apply here:
labels must be UTF-8-representable and unique, but their size is not itself a
mathematical work or cardinality limit.
