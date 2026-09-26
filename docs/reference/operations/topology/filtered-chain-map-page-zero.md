# Maps on the associated graded

`homological.filtered_chain_map.page_zero.compute` takes the same exact chain
map and source/target filtrations as
`homological.filtered_chain_map.compute`. It requires the supplied matrices to
be a chain map that preserves each filtration level.

For every filtration level and chain degree, the operation returns the induced
map

```text
F_p C_n / F_(p-1) C_n -> F_p D_n / F_(p-1) D_n
```

in the quotient bases selected by the associated-graded operations. These
degreewise matrices are retained with both original complexes and filtrations,
together with the selected source and target quotient representatives. Those
representatives are the explicit row and column axes of every matrix, so a
consumer can interpret or compose the serialized map without replaying the
private basis-selection algorithm. The kernel checks that each matrix commutes
with the associated-graded differentials, so the returned value is an exact map
of `E_0` pages. It does not yet return maps on later spectral-sequence pages.
