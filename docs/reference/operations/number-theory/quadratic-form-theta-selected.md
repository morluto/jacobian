# Positive-definite representation numbers and fibers

[Documentation home](../../../index.md) · [Tool surface](../../tools.md) · [Operation references](../index.md) · [This domain](index.md)

`quadratic_form.theta_selected_coefficients.compute` returns exact representation numbers `r_Q(n)` at a strictly increasing tuple of requested indices. `quadratic_form.representing_vectors.compute` returns the corresponding complete integer-vector fibers: where counting lists the number, the fiber lists the vectors in lexicographic order on the form's ordered axis.

Both operations require a positive-definite rational quadratic form. Positive definiteness is established exactly by Sylvester's criterion on the integral polar matrix, so the selected coefficients are finite and the representing-vector enumeration is complete. A negative-definite, indefinite, or degenerate form is refused with a domain error rather than returned as a truncated table.

Enumeration work is admitted before searching: the proved coordinate box has volume `prod(2r_i+1)`, and that volume, multiplied by the form support, must fit the work envelope. Result size is also bounded separately. A request that would enumerate too much is rejected even when very few vectors would be retained.

For `x^2 + xy + y^2`, the first representation count is `r(0)=1`, while six vectors represent `3`: `(-1,-1)`, `(-1,2)`, `(2,-1)`, `(-2,1)`, `(1,-2)`, and `(1,1)`. For `x^2`, the fibers are two signed points for every nonzero square: `r(1)=2` and `r(4)=2`, while an empty fiber is returned as zero rather than omitted.

These are distinct from the deliberately unpublished
`quadratic_form.representation_numbers.compute` and
`quadratic_form.theta_series_prefix.compute`. Those historical operations used floating spectral approximations and published finite tables even for indefinite or degenerate forms. The two operations here keep the same mathematical question but repair the contract: exact positive-definiteness checks, proved complete finite enumeration, and work admitted from the box volume before expansion.
