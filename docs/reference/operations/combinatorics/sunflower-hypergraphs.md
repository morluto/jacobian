# Complete sunflower hypergraphs

`set_system.sunflower_family.construct` receives an indexed finite family

\[
  \mathcal F=(S_0,\ldots,S_{m-1}), \qquad S_i\subseteq\{0,\ldots,n-1\},
\]

and a declared petal count `r`. It returns every increasing `r`-tuple of
distinct source indices for which

\[
  S_i\cap S_j=C\quad\text{for every two distinct selected indices }i,j.
\]

The returned `core` is that exact set `C`; an empty core is valid. The source
member IDs are the canonical hypergraph vertices, and each qualifying tuple is
one edge with the deterministic ordinal ID `sunflower_k` (one-based over the
found rows in enumeration order; ordinals stay within the hypergraph label
limit for every admitted petal count, while `source_indices` retains the
member tuple). The result also carries
the row count, the exact `sunflower_free` status, and the canonical
`FiniteHypergraph` projection. Source order is identity, so permuting source
members permutes vertex labels and rows rather than changing set membership.
For `r=2`, every distinct source pair qualifies with its pairwise intersection;
when `r` exceeds the family size, the complete result is empty and
`sunflower_free` is true.

The operation is complete for every request admitted by its semantic envelope:
the candidate count \(\binom m r\), pairwise-intersection work, source
memberships, canonical hypergraph incidences, and retained result allocation
are preflighted before candidate expansion. The shared hypergraph carrier
admits at most 256 source vertices; a large `r` is accepted when the complete
candidate and result bounds still fit. A resource or deadline failure is an
operation error, never an incomplete row family and never `sunflower_free`.

The request must use canonical `IndexedFiniteSetFamily` members: each member is
sorted, duplicate-free, and distinct by source index. `r` is at least 2 and at
most 256. The operation does not select a maximum sunflower, solve a
sunflower-free extremal problem, or apply a sunflower lemma; those are caller
composition over this exact relation.

Example:

```json
{
  "source": {
    "ground_set_size": 5,
    "members": [[0, 1], [0, 2], [0, 3], [0, 4]]
  },
  "petal_count": 4
}
```

This returns one edge on source vertices `0,1,2,3`, whose exact core is
`[0]`. The canonical `hypergraph` can be passed unchanged to operations such
as `hypergraph.parameters.compute`, `hypergraph.independence_number.compute`,
or `hypergraph.maximum_weight_packing.compute`; those operations analyze the
returned relation, while this operation remains responsible only for complete
sunflower construction.

Native ``construct_sunflower_hypergraph`` is the same family construction at
petal count 3 and is not a second catalog operation.
