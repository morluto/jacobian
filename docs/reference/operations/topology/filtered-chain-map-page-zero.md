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
degreewise matrices are retained with both original complexes and filtrations.
The kernel checks that each matrix commutes with the associated-graded
differentials, so the returned value is an exact map of `E_0` pages. For a
later bounded page, use
[`homological.filtered_chain_map.page.compute`](filtered-chain-map-pages.md).
