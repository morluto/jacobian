# Transpose a finite binary relation

The native Python helper
`jacobian.math.logic.relational_structures.transpose_binary_relation`
returns a canonical finite relational structure with one selected binary
relation transposed. For a relation table `R`, it replaces every tuple
`(a, b)` by `(b, a)`. The carrier, ranked signature, unary and other relation
tables, and nullary truth values remain bound to the source structure.

The helper accepts a `FiniteRelationalStructure` and a relation symbol ID. The
symbol must occur exactly once in the source signature and have arity two. The
operation admits the complete source and output reconstruction before
constructing the returned value. Applying the operation twice to the same
symbol recovers the source.

Use this native API for converse relations, inverse directed edges, or
reversing the orientation of a finite relation. It is not a `math.find` or
`math.run` catalog operation.
