# Semistandard Young tableau enumeration

`combinatorics.semistandard_young_tableaux.enumerate` returns the complete
family for a straight shape `lambda` and entries in `{1,...,k}`. Rows are
weakly increasing and columns are strictly increasing. Results are ordered
lexicographically by their row tuples.

The operation uses the existing exact `IntegerPartition` and
`SemistandardYoungTableau` values. The complete cardinality is computed first
from the hook-content formula

```text
product over cells (i,j) in lambda of (k + j - i) / hook(i,j),
```

with rows and columns indexed from one. A count beyond the admitted family
bound is refused before tableau construction. Empty shape has one empty
tableau, including for an empty alphabet; a nonempty shape with too few labels
has no semistandard tableaux. See [Stanley's hook-content formula](https://math.mit.edu/~rstan/transparencies/hooks.pdf).

The public envelope admits alphabet size at most 4096, at most 4096 tableaux,
100000 aggregate cells, 25000000 construction steps, and a two-megabyte result
estimate. The result contains the canonical partition and typed tableau values.

[Combinatorics operations](index.md)
