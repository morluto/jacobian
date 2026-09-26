# Exact finite sequence twist by a Dirichlet character

`sequence.dirichlet_character_twist.compute` accepts an exact finite integer,
rational, or cyclotomic sequence and one canonical Dirichlet character. Integer
and rational inputs require the first index `index_origin`; an existing
cyclotomic input keeps its authored origin. At offset \(j\) it returns

\[
b_j=\chi(\text{index\_origin}+j)a_j.
\]

The output is a `FiniteCyclotomicSequence` with the same index origin and one
declared coefficient field `QQ[zeta_l]`, where `l` is the least common multiple
of the source field order (or 1 for rational inputs) and the character's exact
value order. Nonunits contribute zero. Source and character coefficients are
embedded exactly in that common field, including for empty sequences.

The operation composes: twisting by \(\chi\) and then \(\psi\) gives the same
coefficient values as twisting by \(\chi\psi\), after exact transport to a
common coefficient field. This keeps the original index axis and lets the
returned sequence feed directly into another twist.

Admission bounds the source digit total, character value order, cyclotomic
coefficient growth, coefficient cells, work, and serialized output before
constructing twisted coefficients. The finite sequence length follows the
shared sequence carrier bound; `index_origin` is a signed 32-bit integer.
Modular-form space transport remains owned by the modular-form operations.
