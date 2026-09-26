# Finite lattice gauge transport

The current exact group carrier is the permutation group `S_d` for bounded
`1 <= d <= 8`. Edges have canonical orientations; backward path traversal uses
the exact inverse permutation. Holonomy multiplies contributions in path order
and retains the source field, lattice, path, and endpoints.

`lattice_gauge.transform.compute` applies the supplied vertex values using the
fixed convention

\[
U'_{u\to v}=g_u U_{u\to v} g_v^{-1}.
\]

Thus an open path from `s` to `t` has transformed holonomy
`g_s Hol_U(path) g_t^-1`, while a based loop is conjugated at its basepoint.
No gauge-orbit search is performed. Quaternion and matrix group carriers remain outside this permutation-valued
slice; the natural permutation-representation trace is provided separately
for closed paths.

## Finite multiplication-table groups

`lattice_gauge.finite_group.holonomy.compute` accepts a separate
`FiniteGroupGaugeField` whose edge values are indices into one exact
`FiniteGroupTable`. The operation re-admits the table's identity, inverse, and
associativity laws, validates full edge coverage and path chaining, and returns
each oriented contribution plus the ordered product in that same table
parent. Backward traversal resolves through the parent's inverse map. A
zero-step path returns the table identity at its named lattice vertex.

The path convention is `h_0 = 1` and `h_i = h_{i-1} * U_i` in traversal
order. This matters for noncommutative groups such as S3. This slice does not
provide vertex gauge transformations, plaquette curvature, or Wilson traces
for arbitrary table groups.

`lattice_gauge.finite_group.complex.construct.compute` supplies the missing
source-bound 2-cell carrier for that path operation. Each face stores an
ordered, closed attaching walk over the exact `GaugeLattice` and retains the
same `FiniteGroupTable` parent used by finite-group edge fields. The walk is
the oriented attaching map, so reversing a face reverses step order and flips
each `forward` bit. A constant attachment is an empty walk with an explicit
basepoint; a backtracking walk remains a separate, nonempty face. Face IDs are
unique and sorted. This is the finite combinatorial form of attaching a
2-cell along a map from its boundary circle into the 1-skeleton ([Hatcher,
Algebraic Topology, Appendix](https://pi.math.cornell.edu/~hatcher/AT/ATapp.pdf)).

Construction admits at most 128 faces, 256 steps per face, 4096 aggregate
steps, and a conservative serialized result estimate below two megabytes before
returning the value. The value records topology only: no curvature product,
flatness conclusion, or Wilson observable is computed by this constructor.

## Identity cases

`lattice_gauge.holonomy.compute` accepts a zero-length path only when the path
names its `basepoint`. The basepoint must belong to the source lattice. Its
holonomy is the identity element of the field's permutation group, its start
and end vertices are both that basepoint, and its edge-contribution tuple is
empty. This represents the identity morphism at one vertex; it does not infer
an endpoint for an unbased empty edge list.

The permutation group `S_1` is admitted as the trivial structure group. Its
only element is the degree-one permutation `[0]`. A field over a nonempty
lattice assigns that identity to each edge, and every accepted path therefore
has identity holonomy. This retains the same typed group and edge-field
contracts used for `S_d` with `2 <= d <= 8`.

An empty identity path is not a plaquette boundary. The plaquette operation
requires a nonempty closed oriented edge walk, so an omitted face cannot be
confused with a zero-curvature face.

The native helper `jacobian.math.gauge.permutation_wilson_trace` accepts a
closed path over the same bounded permutation-group field and returns its
holonomy together with the character of the natural degree-`d` permutation
representation. That exact integer is the number of points fixed by the
holonomy (equivalently, the trace of its permutation matrix), so it is
constant on conjugacy classes. The helper recomputes the holonomy from the
field and path; it does not rely on a caller-asserted holonomy. It uses the
specified natural representation; it does not accept an arbitrary
caller-supplied representation or claim to evaluate every irreducible
character. The request is bounded by the existing degree-eight and
path-length-256 gauge envelope. This cheap deterministic projection of the
published holonomy is native API rather than a catalog operation; the Wilson
loop character is discovered through `lattice_gauge.holonomy.compute`.
