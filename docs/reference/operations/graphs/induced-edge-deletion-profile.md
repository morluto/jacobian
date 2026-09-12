# Induced edge-deletion profiles

`graph.coloring.induced_edge_deletion_profile.compute` returns the exact
source-bound distance from every induced subgraph to `r`-colourability. For a
finite simple graph `G` and target `r >= 1`, each row is

\[
D_{G,r}(S)=\min\{|F|:F\subseteq E(G[S]),\ \chi(G[S]-F)\leq r\}
\]

for one canonical vertex subset `S`. The row retains the lexicographically
smallest deleted source-edge tuple attaining that minimum, and the result also
derives the maximum row value and its attaining-row count for each subset size.
Rows are ordered by increasing subset size and then lexicographically by vertex
labels; the profile always contains exactly `2^|V(G)|` rows, including the empty
subset.

## Exact bounded kernel

The complete Boolean-lattice profile is admitted for at most eight vertices,
with aggregate induced-edge materialization, solver-call, conflict, retained
label, and output bounds checked before computation. `r=1` and `r >= |S|` are
closed-form cases. Every `r=2` row uses exhaustive Boolean cut enumeration:
minimum edge deletion to bipartiteness is the complement of a maximum cut. This
is exact for the admitted vertex bound and makes no Z3 calls or conflict-ledger
charges, including for non-bipartite hosts.

For `r >= 3`, the bounded Z3 colouring/deletion kernel is used only after
admission. Each solver transaction receives the remaining request deadline and
is interruptible by request cancellation; an interrupted or exhausted request
establishes no optimum. The pinned `z3-solver` dependency is therefore a private
implementation detail, not part of the mathematical result.

The operation does not select a subset-size layer, construct a host graph, or
draw an asymptotic Erdős-problem conclusion. Callers receive the complete finite
profile and may compose its rows or per-size projection with their own research.
