# Maximum-cardinality weighted matroid intersection

[Combinatorics operations](index.md) · [Operation references](../index.md)

`matroid.intersection.maximum_cardinality_weighted.compute` takes two linear
matroids on the same labelled ground and an exact integer weight function. It
returns a common independent set with maximum cardinality; among common
independent sets of that cardinality, it maximizes the original total weight.
This is a lexicographic objective. Negative weights do not cause the result to
shrink when a larger common independent set exists.

The operation reuses the exact weighted-intersection optimizer with a scalar
cardinality shift. For ground size `n`, let `W=max_e |w_e|` and set
`C=2*n*W+1`. For any feasible `I,J` with `|J|>|I|`,

```text
(w(J) + C|J|) - (w(I) + C|I|)
    >= -(|J|+|I|)W + C
    >= -2nW + (2nW+1)
    = 1.
```

Thus every maximizer of `w(I)+C|I|` has maximum possible cardinality. For equal
cardinality, the shift is constant, so the scalarized objective maximizes the
original weight. The returned nested weighted-intersection result retains the
source matroids and its exact split witness. Subtracting the returned bonus
from each shifted objective entry recovers the original weight function.

The existing weighted-intersection ground, row, work, split-growth, objective
digit, and output limits apply. The shifted objective is also required to fit
the existing fewer-than-12-digit weight envelope; requests outside that
conservative exact regime are rejected before the optimizer expands matrix
ranks. The lexicographic wrapper adds only a linear-size shift and summary.

For the underlying weighted matroid-intersection exchange algorithm and its
proof, see Schrijver, *Combinatorial Optimization*, Chapter 13, §13.7
([book PDF](https://www.mathematik.uni-muenchen.de/~kpanagio/KombOpt/book.pdf)).
The scalarization argument above is independent of a particular optimizer.
Consumers relying on the nested source-bound optimality claim can replay it
with `verify_weighted_intersection_result`, as for the unrestricted weighted
intersection result.
