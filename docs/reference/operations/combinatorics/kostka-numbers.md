# Fixed-content semistandard tableau counts

`combinatorics.semistandard_young_tableaux.fixed_content_count` returns the
number of semistandard Young tableaux of a straight shape `lambda` with one
specified sparse content map. Rows are weakly increasing and columns are
strictly increasing. For partition content `mu`, this is the Kostka number
`K_{lambda,mu}`. The operation also accepts a general finite content map whose
multiplicities need not form a partition.

The content carrier is a sparse ordered tuple of `{entry, multiplicity}`
terms. Entry labels are preserved exactly, missing labels have multiplicity
zero, and terms must be strictly increasing by entry. The operation does not
sort or compress the labels. The returned count retains both the source shape
and exact content.

This count differs from
`combinatorics.semistandard_young_tableaux.count`, which counts tableaux over
the entire alphabet `{1,...,m}` by the hook-content formula. Fixed-content
counting admits the complete multiset-word prefix tree before row-major search;
work is bounded by the word count, prefix count, and number of active content
labels. It uses direct exact reductions when the shape is one row, the content
has all distinct labels, the shape/content sizes disagree, or too few distinct
labels exist for the columns.

The mathematical definition of Kostka numbers as fixed-content semistandard
tableau counts is documented in [Sage's semistandard tableau reference](https://doc.sagemath.org/html/en/reference/combinat/sage/combinat/tableau.html); Sage is a reference only, and Jacobian returns its own typed exact values.
