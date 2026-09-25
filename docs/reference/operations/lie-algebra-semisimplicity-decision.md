# Lie algebra semisimplicity decision

`lie_algebra.is_semisimple.compute` accepts a finite-dimensional Lie algebra
over `QQ` and returns the direct boolean decision `is_semisimple`. For
characteristic zero, Cartan's criterion says this is true exactly when the
Killing form is nondegenerate. The operation admits the algebra, computes the
exact Killing-form radical, and returns only the decision.

The result contains no source algebra, Killing form, or radical, so it makes no
source-bound or serialized-witness claim. Consumers that receive a boolean
alongside an algebra from caller-authored data must recompute the decision for
that algebra before relying on it. The detailed Killing form and radical are
available separately from `lie_algebra.killing_form.compute` and
`lie_algebra.killing_form.radical.compute`. This operation does not identify
simple factors, compute a classification, or calculate the solvable radical.

The supported dimension is at most eight, as in the current
finite-dimensional Lie-algebra carrier.

For the theorem, see MIT OpenCourseWare's [18.745 Lecture 17: Proofs of the
Cartan Criteria and Properties of Semisimple Lie
Algebras](https://ocw.mit.edu/courses/18-745-lie-groups-and-lie-algebras-i-fall-2020/mit18_745_f20_lec17.pdf).
