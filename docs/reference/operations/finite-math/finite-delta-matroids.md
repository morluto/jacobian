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
`delta_matroid.lower_matroid.compute` and
`delta_matroid.upper_matroid.compute` return `FiniteBasisMatroid` values whose
bases are, respectively, all minimum-cardinality or all maximum-cardinality
feasible sets. They preserve the source ground axis and include
`source_feasible_indices`, aligned with the result bases, as an exact map back
to the source's canonical feasible rows. The source symmetric-exchange axiom is
replayed at each operation boundary. The basis-family exchange axiom is checked
by the result carrier.
The lower/upper matroid theorem is stated in Section 6.1 of Dupont, Fink, and
Moci, [*Universal Tutte characters via combinatorial coalgebras*](https://doi.org/10.5802/alco.35).

These conversions retain the 16,384 source-membership, 2,048 UTF-8 source-label,
and 250,000 source exchange-work limits. The target `FiniteBasisMatroid` admits
at most 64 ground labels, 4,096 basis rows, 65,536 basis memberships, 1,024
UTF-8 bytes per label, 16,384 aggregate label bytes, and 2,000,000 worst-case
basis-exchange candidate checks. Output bounds are checked before materializing
the selected basis family. A source with more than 64 ground elements remains a
valid delta-matroid input but is refused for these conversions because the
canonical basis carrier cannot represent that output size.

Width `max(|F|)-min(|F|)` is also available as
`delta_matroid.width.compute`; it scans each retained feasible-row length.
