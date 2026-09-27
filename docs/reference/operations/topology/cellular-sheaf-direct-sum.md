# Cellular sheaf direct sum

`cellular_sheaf.direct_sum.compute` forms the pointwise direct sum of two
checked cellular sheaves on the same finite simplicial complex and over the
same exact field (`QQ` or the identical `GF(p)`). At each simplex `sigma`, the
new stalk is `F(sigma) ⊕ G(sigma)`. Each comparable-face restriction is the
block diagonal map `rho_F ⊕ rho_G`; therefore its functoriality follows from
the two input sheaf diagrams.

The result retains both source sheaves, the direct-sum sheaf, and a canonical
injection from each source stalk. Summand basis coordinates are named `L0`,
`L1`, … and `R0`, `R1`, … in their original basis order. The retained source
values bind these positional axes to the original basis labels. The injection
matrices are explicit, so callers can transport sections and cochains without
reconstructing a basis convention.

The operation requires identical complex structure, field, and prime modulus.
It admits each stalk at rank at most 8, total stalk coordinates at most 512,
restriction matrices at at most 65,536 scalar cells, and the output at at most
8,000,000 serialized characters. Each direct-sum stalk and the aggregate
restriction matrices are admitted before block matrices are allocated. Over
`QQ`, matrix entries are `CanonicalRational` values with reduced numerators and
positive denominators. Over `GF(p)`, entries are strict integer residues in
`0..p-1`. Serialized JSON preserves these field types, and numeric scalar
strings are rejected. No floating arithmetic is used.

The pointwise construction is compatible with sections and the sheaf cochain
complex: the corresponding spaces and differentials split by the two summands.
The focused tests check this on a rank-one constant sheaf over `GF(2)` by
exhaustively enumerating every compatible stalk assignment, and compare the
resulting Betti numbers with the independently known interval values. The
operation also preserves rational maps exactly and its result round-trips
through serialization.
