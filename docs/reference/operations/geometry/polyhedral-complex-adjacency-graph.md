# Rational polytopal complex adjacency graph

`polyhedral_complex.adjacency_graph.compute` takes a finite family of bounded
rational maximal cells and returns their exact facet-adjacency graph. It first
uses the canonical face-closure operation, so the cells must share one labelled
ambient coordinate space, be full-dimensional there (a pure complex), and meet
face-to-face. A lower-dimensional cell presentation is rejected by closure.

The result contains a reusable `SimpleUndirectedGraph`, exact vertex-to-cell
geometry, and one exact shared-face record per edge. Two cells are adjacent
only when their intersection has dimension one less than the ambient dimension.
A vertex-only or lower-dimensional contact creates no edge.

Canonical cell IDs label graph vertices. Each labelled graph edge names its
shared facet ID, dimension, support cells, and exact vertices. The result is
bounded by the face-closure limit of 16 cells in dimension at most 4, at most
120 edges, 34,816 output coordinate values, and 2.5 million exact coefficient
digits, with at most 1,024 digits in any individual coordinate component. The
closure's face-enumeration and pairwise-intersection limits are applied before
graph construction. The result value also admits exact vertex and supporting
facet validation under a 10-million-bit aggregate coordinate-work bound.

This is top-cell facet adjacency. It does not construct the full incidence graph
between cells of different dimensions.
