# Maps on spectral-sequence pages

`homological.filtered_chain_map.page.compute` takes a retained filtered
chain-map value and a page index `r` in the admitted finite window. The map
must satisfy both the chain-map equation and filtration preservation for every
level.

The result contains source and target `E_r` page values, including their exact
representatives and differentials, plus one matrix for each filtration level
and chain degree. Each matrix is expressed in the page quotient bases. The
kernel applies the chain map to source representatives, reduces their images
modulo the target `E_r` denominator, and checks that the resulting matrices
commute with every in-window `d_r`.

The output preserves empty page groups as zero-row/zero-column matrices. The
operation recomputes the map equations at admission rather than trusting the
status fields on a serialized filtered-map value. Its work and exact result
size are bounded before page representatives are expanded.
