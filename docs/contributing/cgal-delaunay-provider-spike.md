# CGAL exact-Delaunay optional-provider spike

This spike tests CGAL 6.2 as a producer for one future atomic outcome:
the Delaunay triangulation of a bounded set of exact rational 2D sites. It does
not register a capability, authorize CGAL as a checker, or include Voronoi
construction in the same outcome.

## Decision

The exact-kernel reproduction passes; a production capability remains
`REVISE`.

- The producer uses
  `Exact_predicates_exact_constructions_kernel` and
  `Delaunay_triangulation_2`.
- CGAL's 2D Triangulations package is
  `GPL-3.0-or-later OR LicenseRef-Commercial`. Jacobian must not distribute the
  GPL adapter executable with its MIT core. Any production deployment needs a
  recorded legal/licensing decision and should remain an operator-installed
  T2 subprocess.
- The local reproduction compiled successfully with GCC 12.2 and Boost 1.74,
  but CGAL 6.2 documents GNU g++ 13.3 or later as supported. Production images
  must use a supported toolchain and record it.
- Exact output remains `COMPUTED` at most. CGAL cannot independently verify its
  own triangulation.

The [official 6.2 release](https://www.cgal.org/2026/06/11/cgal62/),
[2D Triangulations manual](https://doc.cgal.org/latest/Triangulation_2/index.html),
[kernel manual](https://doc.cgal.org/latest/Kernel_23/index.html), and
[package license inventory](https://doc.cgal.org/latest/Manual/packages.html)
define the upstream boundary. The frozen source, adapter, commands, and outputs
are in
[`benchmarks/cgal_delaunay_pin.json`](../../benchmarks/cgal_delaunay_pin.json).

## Reproduce

Compile the checked-in fixed transport against the official library archive:

```sh
g++ -std=c++17 -O2 \
  -I/path/to/CGAL-6.2/include \
  benchmarks/cgal_delaunay_spike.cpp \
  -lgmp -lmpfr -o /tmp/cgal-delaunay-spike

uv run python benchmarks/cgal_delaunay_spike.py \
  --source-archive /path/to/CGAL-6.2-library.tar.xz \
  --executable /tmp/cgal-delaunay-spike \
  --output /tmp/cgal-delaunay-provider-spike.json
```

The runner hashes the official archive, inspects the exact version and
package-specific SPDX header without extraction, binds the checked-in adapter
source, measures the executable, probes CGAL/compiler/Boost versions, and runs
two bounded cases.

The unique case contains six exact rational sites, including a non-integral
site. It returns five consistently oriented triangles, ten edges, and five
convex-hull boundary edges. The degenerate case uses four cocircular square
vertices and returns `applicable false / COCIRCULAR` under `REQUIRE_UNIQUE`.

Missing files, version/source/adapter mismatch, malformed output, timeout,
cancellation, output overflow, and process crash are non-conclusions.

## Production contract gate

A future `geometry.points.delaunay_triangulation.compute` request must bind:

- canonical exact rational sites and stable site IDs;
- duplicate handling and at least three non-collinear unique points;
- bounded site/output size;
- `REQUIRE_UNIQUE` or a separately specified canonical tie-break;
- exact-kernel provider identity and adapter/build provenance; and
- explicit scope, execution, conclusion, completeness, and assurance.

Its result should expose triangles, edges, adjacency, convex-hull boundary,
orientation convention, empty-circumcircle evidence, and relationships to the
exact point-set artifact.

An independent checker using rational arithmetic can validate:

1. all triangle IDs refer to input sites and every triangle is non-degenerate
   and consistently oriented;
2. edge incidence is one on the hull and two in the interior;
3. triangle interiors do not cross;
4. the boundary is the convex hull and triangles cover it;
5. every other site is outside or on each triangle circumcircle; and
6. `REQUIRE_UNIQUE` rejects any four cocircular sites.

These checks are feasible without importing or invoking CGAL, but are not part
of this provider spike. Until they exist and are operator-authorized, no result
may be `VERIFIED`.

Voronoi construction is a separate outcome. If later implemented, its artifact
should relate to the Delaunay artifact rather than adding dual geometry to this
contract.

## Absence isolation

The spike is under `benchmarks/` and is never imported during portfolio,
kernel, CLI, or MCP startup. The absence test compares the complete catalog
before and after a missing-provider probe. No capability ID is added or
removed, and the existing exact planar geometry foundation remains available.
