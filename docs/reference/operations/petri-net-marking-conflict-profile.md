# Petri-net marking conflict profile

`petri_net.marking.conflict_profile.compute` partitions every unordered pair
of distinct transitions enabled individually at a supplied marking. A pair is
jointly enabled as a simultaneous step exactly when its aggregate input demand
fits in every place:

```text
M(p) >= Pre(p,t) + Pre(p,u) for every place p.
```

The result retains the exact net and marking, the individually enabled
transition indices, and a complete partition of their distinct pairs into
`jointly_enabled_pairs` and `conflicting_pairs`. Pair indices are ordered
increasingly, and each family is lexicographically ordered.

This is a local step-semantics profile. A pair in the jointly-enabled family
can fire simultaneously; the result says nothing about sequential firing
orders. A pair in the conflict family is blocked as a simultaneous step due
to insufficient aggregate input tokens, even though either transition can
fire alone. The same enabling inequality defines simultaneous steps in
[Murata, “Petri Nets: Properties, Analysis, and Applications,” §3.2](https://www.dsc.ufcg.edu.br/~abrantes/CursosAnteriores/MVSRP/murata89.pdf).

The computation considers at most 2,016 pairs on the bounded 64-transition
net axis and preflights pair-place work before constructing pair families.
