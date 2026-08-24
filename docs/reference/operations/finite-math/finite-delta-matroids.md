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
the axiom: at most 128 feasible rows, 1,024 total row memberships, 2,048 UTF-8
bytes of ground labels, 250,000 symmetric-exchange candidate checks, and a
65,536-byte serialized result. The shared `FiniteFeasibleSetSystem` carrier
also bounds the labelled ground set to 64 elements. There is deliberately no
smaller delta-matroid ground-size cap: a sparse complete family on all 64
carrier elements is admitted when these work and result bounds hold.

This initial operation deliberately does not construct twists, minors, binary
matrix presentations, graph conversions, or interlace polynomials. Those are
separate mathematical postconditions rather than fields of the recognition
result.
