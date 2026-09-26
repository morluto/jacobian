# Type-A highest-weight characters

`root_system.highest_weight_character.compute` returns the complete weight and
multiplicity table for an irreducible representation of the irreducible finite
Cartan type `A_n`, with dominant highest weight in fundamental-weight
coordinates. Terms use the ordered simple-coroot pairing axis, are sorted and
unique, and retain the Cartan matrix and highest weight. Multiplicities are
positive integers; the highest weight occurs once.

The kernel converts the highest weight to a partition for `GL_(n+1)` and
enumerates the exact type-A weight set by the content criterion: a weak
composition contributes precisely when its decreasing rearrangement is
dominated by the highest-weight partition. Freudenthal recursion computes the
multiplicities on this complete set. Before expanding candidates, the
operation bounds scanned weak compositions by work, and retained terms by the
minimum of the composition count and exact Weyl dimension before applying the
4,096-term capacity and conservative result-size estimate. It also bounds
positive-root summations and integer growth. Non-type-A and reducible Cartan data are
currently outside this operation's scope.

The recursion follows Freudenthal's multiplicity formula; see Freudenthal and
de Vries, [*Linear Lie Groups*](https://books.google.com/books?id=yEtux7Pr4DUC).
The complete multiplicity values are checked in fixtures against hand tables,
Weyl symmetry, and the independent Weyl dimension operation.
