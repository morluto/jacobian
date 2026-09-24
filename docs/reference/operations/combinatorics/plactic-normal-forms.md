# Plactic normal forms of words

[Documentation home](../../../index.md) · [Operation references](../index.md) · [Combinatorics](index.md)

`word.plactic_normal_form.compute` applies ordinary row-insertion RSK under
`ROW_INSERTION_RSK_V1`, retaining the source word's explicit ordered alphabet.
It returns the insertion tableau and one canonical representative of its
plactic class. The representative reads every tableau row from left to right,
starting with the bottom row and proceeding upward.

For example, over `1 < 2 < 3`, both `(1, 3, 2)` and `(3, 1, 2)` insert to
the tableau with rows `((1, 2), (3,))`. Its canonical row-reading word is
`(3, 1, 2)`. The operation therefore supports a direct comparison of plactic
classes by comparing the returned insertion tableaux or normal forms.

The word and tableau remain finite exact values. The request inherits the
finite-word limits on alphabet and length; the result is bounded by the same
length and has no search or enumeration of the whole plactic class. The result
model checks that the normal-form word is exactly the declared row reading and
uses the same ordered alphabet as the source. The kernel derives the tableau
from ordinary row insertion, so inserting the returned normal form recovers
that tableau.

This operation returns one canonical representative. It does not enumerate
Knuth-equivalent words or materialize the congruence graph.

`word.plactic_equivalence.compute` compares two words over the same exact
ordered alphabet. It returns both source-bound insertion tableaux and row-
reading representatives, plus `equivalent`, which is true exactly when the
two insertion tableaux (and therefore their plactic classes) agree. Both
insertions are admitted together before either starts. For source lengths
`m` and `n`, the kernel performs at most
`m² ceil(log2(m+1)) + n² ceil(log2(n+1))` row-search comparisons; the combined
result has a preflighted byte bound derived from both alphabet and word
payloads. It does not search the Knuth graph.

This equality criterion is the ordinary row-insertion characterization of
plactic equivalence. Sage's maintained implementation similarly exposes
plactic elements through their insertion tableaux and canonical row-reading
representatives: [Sage plactic monoid reference](https://doc-release--sagemath.netlify.app/html/en/reference/monoids/sage/monoids/plactic_monoid).

## Reading an existing RSK tableau

`tableau.row_reading_word.compute` accepts an existing alphabet-bound ordinary
RSK pair and returns its insertion tableau as a `FiniteWord`. Its explicit
convention is `TABLEAU_ROW_READING_BOTTOM_TO_TOP_LEFT_TO_RIGHT_V1`: start at
the lowermost row, read each row left-to-right, then move upward. This agrees
with Sage's `Tableau.to_word_by_row()` convention. The ordered alphabet comes
from the pair, so the resulting word composes directly with word operations.
The operation admits at most 500 tableau cells and validates the insertion
tableau's semistandard and alphabet-rank conditions before constructing output.

[Sage tableau reference](https://doc.sagemath.org/html/en/reference/combinat/sage/combinat/tableau.html)
