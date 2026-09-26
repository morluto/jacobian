# Simultaneous steps in weighted Petri nets

[Documentation home](../../../index.md) · [Combinatorics operations](index.md)

`petri_net.marking.concurrent_step.compute` applies a transition multiset in
one atomic step. If `x[t]` is the multiplicity of transition `t`, the step is
enabled at marking `M` exactly when

```text
M >= Pre x
```

coordinatewise. The resulting marking is `M - Pre x + Post x`. The operation
returns aggregate required tokens and complete deficits when blocked. A target
outside the declared per-place token envelope is returned as an envelope
escape, not as a valid `Marking`.

This semantics checks all consumption against the *source* marking before
applying any production. It therefore differs from firing a sequence: a
producer followed by a consumer may be a valid sequence even when the same
multiset cannot occur simultaneously. It also does not claim that the
multiset admits any sequential ordering.

Transition multiplicities have a total cap of 1,000. With the existing
64-place and 64-transition axes this bounds aggregate matrix-vector work to
4,096 multiply-adds per place profile. The semantics of a transition multiset
as a simultaneous step is standard for place/transition nets; see
[Semantics of Petri Nets: A Comparison, §3.1](https://www.informs-sim.org/wsc07papers/074.pdf).
