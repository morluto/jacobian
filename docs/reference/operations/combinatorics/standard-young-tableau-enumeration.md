# Standard Young tableau enumeration

`combinatorics.standard_young_tableaux.enumerate` returns every standard
Young tableau of one partition shape, in lexicographic order on the tuple of
rows. Entries are exactly `1, ..., n`, each occurs once, and rows and columns
increase strictly. The empty partition has one empty tableau.

The implementation fills each shape by removing a removable corner containing
the largest entry, recursively enumerating the smaller shape, and restoring
that entry. The exact hook-length count is computed first. The operation admits
the complete family only when it fits all published limits: 4,096 tableaux,
100,000 total tableau cells, 25,000,000 construction-work cells, and an
estimated 2,000,000 output bytes. Construction work bounds the sum of
corner-recursion cell copies plus worst-case tableau sorting comparisons. The
recursion streams results without a cross-request cache. At most one partial
tableau per level is retained, so active branch tableau data is bounded by
`n(n+1)/2 <= 125250` cells for `n <= 500`; the admitted result separately
bounds retained output. The operation returns the full family or a resource
error; it never returns a prefix.

The count uses the Frame–Robinson–Thrall hook-length formula
`f^lambda = n! / product(h(u))`. The convention for standard tableaux and the
formula are stated in Stanley, *Enumerative Combinatorics*, Vol. 2, §8.1:
[author's lecture notes](https://math.mit.edu/~rstan/algcomb/algcomb.pdf).

This is a bounded enumeration of straight-shape standard tableaux. It makes no
claim about shifted, skew, semistandard, or other tableau families.
