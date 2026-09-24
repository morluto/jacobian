# Exact Dirichlet-character kernel

`dirichlet_character.kernel.compute` returns the complete subgroup
\(\ker(\chi)=\{a\in(\mathbb Z/N\mathbb Z)^*: \chi(a)=1\}\) as increasing
canonical residue representatives. The result retains the source character
and its exact unit-group parent, together with the subgroup index. The operation
evaluates the exact dual-coordinate homomorphism on every admitted unit, so the
returned list is complete; the wire model enforces canonical ordering and the
subgroup index relation without recomputing the kernel.

Work is bounded by the number of units times the number of cyclic coordinates;
the current admitted modulus is at most `2,048`. Sage also exposes the kernel
of an exact Dirichlet character in its [Dirichlet-character reference](https://doc.sagemath.org/html/en/reference/modfrm/sage/modular/dirichlet.html).
