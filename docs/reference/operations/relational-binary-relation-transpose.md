# Transpose a finite binary relation

The operation `relational_structure.transpose_binary_relation.compute` returns
a canonical finite relational structure with one selected binary relation
transposed. For a relation table `R`, it replaces every tuple `(a, b)` by
`(b, a)`. The carrier, ranked signature, unary and other relation tables, and
nullary truth values remain bound to the source structure.

The request selects a relation symbol by ID. It must occur exactly once in
the source signature and have arity two. The operation admits the complete
source and output reconstruction before constructing the returned value.
Applying the operation twice to the same symbol recovers the source.

Use this operation for converse relations, inverse directed edges, or reversing
the orientation of a finite relation. The exact request schema and executable
example are published by `math.find` with the operation.
