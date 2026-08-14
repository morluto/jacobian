# Exact planar geometry

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

Planar geometry operations accept bounded exact rational points, segments,
lines, triangles, and polygons. The catalog includes direct computations for
intersection, projection, midpoint, orientation, centroid, circumcircle,
convex hull, signed area, triangulation, and projective line-arrangement flats,
as well as direct predicates for parallelism, perpendicularity, collinearity,
concyclicity, and polygon simplicity.

Inputs are validated as their owning geometric value before computation. Results
are returned inline and can be passed to another compatible typed operation;
there is no geometry artifact or generic verification service.
