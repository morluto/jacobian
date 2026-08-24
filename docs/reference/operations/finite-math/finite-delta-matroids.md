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
their symmetric difference. It preflights the complete family before replaying
the axiom: 1,024 total row memberships, 2,048 UTF-8 bytes of ground labels
(each label must be UTF-8-representable), 250,000 symmetric-exchange candidate
checks per complete axiom replay, and a 65,536-byte serialized result. There
are deliberately no separate delta-matroid ground-size or row-count caps: the
membership envelope bounds the row count, and the label-byte, candidate-work,
and result bounds control the actual work, so a sparse family over hundreds of
short labels — or a dense short-row family such as every subset of size at most
two over 16 elements (137 rows, 220,832 candidate checks) — is admitted when
these bounds hold.

One recognized request performs at most four complete axiom replays: the
operation's obstruction decision, the canonical value construction, the
returned value's own defining-invariant validation, and the result-binding
obstruction replay. The aggregate worst case is therefore 1,000,000 candidate
checks per accepted call, which is part of the advertised envelope.

This initial operation deliberately does not construct twists, minors, binary
matrix presentations, graph conversions, or interlace polynomials. Those are
separate mathematical postconditions rather than fields of the recognition
result.
