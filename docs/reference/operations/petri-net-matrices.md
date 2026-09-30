# Petri-net matrices

[Documentation home](../../index.md) · [Tool surface](../tools.md) · [Operation references](index.md)

`petri_net.matrices.compute` returns the exact precondition matrix `Pre`,
postcondition matrix `Post`, and incidence matrix `C = Post - Pre` for one finite
ordinary weighted place/transition net. Rows follow the net's ordered place
axis and columns follow its ordered transition axis. The result retains the
source net, so the matrices remain bound to those exact axes, including when
place or transition IDs are omitted.

It also returns each transition's input and output place support and each
place's consumer and producer transition support. These are ordered tuples of
indices on the same axes; an arc belongs to a support exactly when its
corresponding Pre or Post weight is positive.

Each matrix is the shared exact `ZZ` matrix value and keeps its shape when an
axis is empty. In particular, a net with zero places and three transitions
produces three `0 × 3` matrices. The entrywise relation `C[p,t] =
Post[p,t] - Pre[p,t]` is the incidence convention used by the Petri-net
marking equation; `C` records the token change caused by firing one transition
when it is enabled. The projection does not determine enabledness or claim that
a formal transition-count vector is fireable.

The operation admits a net with at most 64 places, 64 transitions, and arc
weights at most 1000. Its dense work is bounded by `|P| × |T|`. Before constructing
the derived matrices and maps, native admission bounds the retained result,
including its source net, matrix entries, and support-map indices, by 4,194,304
cells. Label text and encoded byte length do not change this native limit;
serialized payload limits belong to the delivery layer.

For the standard pre/post/incidence and state-equation conventions, see
[Murata, “Petri Nets: Properties, Analysis, and Applications,” §3](https://www.dsc.ufcg.edu.br/~abrantes/CursosAnteriores/MVSRP/murata89.pdf).
