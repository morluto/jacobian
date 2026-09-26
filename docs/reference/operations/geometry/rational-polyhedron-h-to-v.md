# Exact rational H-to-V polyhedron conversion

`polytope.rational.h_to_v.compute` converts a rational H-presentation
`{x : a_i · x <= b_i}` into a serializable V-presentation. A nonempty result
represents

`conv(points) + cone(rays) + span(lineality)`.

Finite points, oriented recession rays, and lineality directions remain separate
typed families. The carrier records the ambient ordered axes, emptiness, and
affine dimension, so an empty system, an affine set, a bounded polytope, and an
unbounded polyhedron retain their distinct states. Rational coordinates and
directions remain exact. The operation does not promise minimal or canonical
extreme generators; consumers must establish any stronger relations they rely
on when accepting a deserialized claim.

The conversion adapts the private homogeneous double-description kernel. Before
the kernel expands any generators, admission checks the H-row count, the
theorem-backed ray and candidate-pair bounds, coefficient growth, weighted
exact-arithmetic work, affine-dimension rank work, and a conservative upper
bound on canonical result bytes against Jacobian's 10 MiB canonical JSON
egress limit. Requests whose exact result can exceed the serializable rational
envelope or the transport limit are rejected as resource refusals.

The value permits zero-dimensional ambient space. Constant inequalities are
classified exactly: `0 <= b` is redundant when `b >= 0`, and `0 <= b` makes the
polyhedron empty when `b < 0`.

The unit-square example in the operation catalog shows the request shape. For
an unbounded ray, `-x <= 0` on axis `[x]` returns the finite point `0` and the
oriented recession ray `+1`. With no inequalities on `[x]`, the result is the
whole line represented by point `0` and lineality direction `1`.
