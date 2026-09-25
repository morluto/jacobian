# Fundamental holes of a positive affine semigroup

[`affine_semigroup.fundamental_holes.compute`](../tools.md) returns the
`Q`-minimal elements of the saturation difference

```text
Qsat minus Q,    Qsat = cone(Q) intersect gp(Q).
```

A hole `h` is fundamental when no nonzero `s` in `Q` has `h-s` in `Qsat`;
equivalently, `Qsat intersect (h-Q) = {h}`. Every hole is a fundamental hole
plus an element of `Q`, and the fundamental-hole set is finite. This is the
definition and finiteness result of Takemura and Yoshida, [“A generalization of
the integer linear infeasibility problem”](https://arxiv.org/abs/math/0603108).

The current operation admits full-rank pointed cones in two ambient dimensions.
Choose actual source generators `u,v` on the two extreme rays. In generated-
lattice coordinates, each fundamental hole has unique coordinates
`h = lambda*u + mu*v` with `0 <= lambda,mu < 1`: if either coefficient is at
least one, subtracting that ray generator leaves a saturation point, which
must still be a hole and contradicts minimality. The operation bounds the
integer box containing this half-open parallelogram before lattice-point
enumeration. It then computes the exact semigroup closure there and removes any
hole whose subtraction by a source generator remains a hole.

Admission limits the exact number of lattice points in the parallelogram to at
most 50,000 and the combined candidate scan, semigroup closure, and minimality
work to 2,000,000 units. Row HNF for the ray lattice gives exactly `|det(u,v)|`
rectangular coset representatives. Exact Cramer numerators and floor
translations move each one into the unique half-open parallelogram; the work is
proportional to the lattice index, even when its ambient axis-aligned box is
large. With eight-digit source entries, the generated-lattice index is bounded
by the 2-by-2 minors; canonical HNF basis entries are bounded by that index,
and Cramer's rule bounds generator coordinates by 2 times the source scalar
limit. The HNF and translation intermediate heights are bounded before
enumeration. These bounds cap parallelogram determinants below 17 digits and
ambient output coordinates below 26 digits. The result has at most 100,000
hole-coordinate cells. Candidate, work, cell, and scalar bounds are admitted
before point enumeration; transport applies its own serialization limits.

For generators `(2,0)`, `(0,2)`, `(1,1)`, `(1,0)`, the generated lattice is
`Z^2`. The holes are `(0,1),(0,3),(0,5),...`; the only fundamental hole is
`(0,1)`, since adding `(0,2)` produces every later hole.

The operation does not enumerate all holes, return a degree prefix, or decide
normality by itself. It does not currently support rank-one or higher-rank
semigroups.
