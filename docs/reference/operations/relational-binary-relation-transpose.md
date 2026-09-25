# Transpose a binary relation

`relational_structure.transpose_binary_relation.compute` takes a canonical
`FiniteRelationalStructure` and one binary relation symbol ID. It returns a
canonical structure on the same carrier and ranked signature, replacing only
the selected table

```text
R' = {(y, x) : (x, y) in R}.
```

Every other relation table stays unchanged. The selected symbol must exist in
the source signature and have arity two. Its full table is transported; the
canonical structure bounds each table at 4,096 rows and each signature at
eight symbols of arity at most four. The operation admits reconstruction of
the complete output and the selected two-coordinate transport against the
schema-derived 172,040 row and coordinate visit bound before constructing the
result. Empty tables and loops are preserved, and transposition is an
involution.

The output is the ordinary finite relational structure value. It can be
serialized and passed directly to homomorphism, CSP, or structural operations.
