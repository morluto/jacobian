# Finite-group Adams operation

`character.adams_operation.compute` accepts a virtual character expressed in
the canonical irreducible basis of its retained finite-group character table
and a positive integer `k`. It returns the exact table-bound virtual character
defined by

\[
\psi^k(\chi)(g)=\chi(g^k).
\]

The formula is the standard Adams operation on the representation ring of a
finite group; Meir and Szymik state this character formula and its virtual
character-valued codomain in §2.1 of [*Adams operations and symmetries of
representation categories*](https://arxiv.org/abs/1704.03389). The result
remains an element of the same virtual-character ring, including when some
output multiplicities are negative.

The implementation admits the concrete permutation-group closure, repeated
power-map work, exact character-value expansion and inner-product recovery,
canonical table, and serialized output before conjugacy expansion. It accepts
the current table families: the trivial group, cyclic groups through order 60,
and `S3`. There is no fixed exponent ceiling: admission charges for the bit
length of `k`, the number of classes, and the permutation degree, so large
exponents remain admissible when powering fits the work envelope.
