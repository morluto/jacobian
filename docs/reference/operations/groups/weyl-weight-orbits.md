# Integral weight orbits

`weyl_group.weight.orbit.compute` returns the complete orbit of an integral
weight under the finite Weyl group of the supplied Cartan matrix. Coordinates
are in the ordered fundamental-weight basis, equivalently the integer
pairings with the ordered simple coroots. The result retains the Cartan matrix
and sorts the distinct orbit elements lexicographically.

For a simple reflection, these coordinates use

```text
s_i(lambda) = lambda - lambda_i alpha_i
alpha_i = sum_j A[j,i] omega_j.
```

The input must be an integer weight; rational weight-space vectors are outside
this operation's contract. Rank is at most 8 and the exact orbit has at most
4096 values. Before orbit expansion, the operation bounds all coordinates by
the invariant positive-definite weight-space norm and computes the orbit size
from the Weyl-group order divided by the dominant representative's parabolic
stabilizer order. It returns the whole orbit or rejects the request; no partial
orbit is returned.

For `A2`, the orbit of the first fundamental weight `(1,0)` is
`{(-1,1), (0,-1), (1,0)}`. For the Cartan matrix `[[2,-2],[-1,2]]` of `B2`,
the same coordinate vector has orbit
`{(-1,0), (-1,1), (1,-1), (1,0)}`.

[Operation references](../index.md) · [Tool surface](../../tools.md)
