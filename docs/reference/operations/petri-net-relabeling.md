# Petri-net axis relabeling

`petri_net.relabel.compute` applies explicit bijections to the ordered place
and transition axes of a weighted place/transition net. Each map sends a source
index to its target index:

```text
place_source_to_target : {0, ..., |P|-1} -> {0, ..., |P|-1}
transition_source_to_target : {0, ..., |T|-1} -> {0, ..., |T|-1}
```

Both maps must be permutations of their complete axes. If `p` maps to `p'` and
`t` maps to `t'`, the target net satisfies

```text
Pre'(p', t') = Pre(p, t)
Post'(p', t') = Post(p, t).
```

Optional place and transition IDs follow their corresponding elements. The
result retains the source net, target net, and both maps as one typed
isomorphism value. It does not infer matches from arc structure or claim that
two arbitrary nets are isomorphic.

For any source marking `M`, transport it by assigning
`M'(place_source_to_target[p]) = M(p)`. If transition `t` fires from `M` to
`N`, its mapped transition fires from `M'` to the transported marking `N'` in
the target net. This follows directly from preservation of both arc matrices.

The operation uses the bounded dense net carrier (at most 64 places and 64
transitions). It admits matrix work and the serialized result before building
the target net. The contract is the finite bijective, weight-preserving
instance of a place/transition-net isomorphism; see [Petri Net
Transformations](https://doi.org/10.5772/5310) for the broader morphism
framework.
