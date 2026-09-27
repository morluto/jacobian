# Lie algebra semisimplicity decision

`lie_algebra_is_semisimple` is a native convenience projection over the
published `lie_algebra.killing_form.radical.compute` operation: over `QQ`,
Cartan's criterion says the algebra is semisimple exactly when the Killing form
has zero radical. It returns a source-bound result containing the canonical
algebra and the exact boolean decision. The helper is not a separate catalog
operation because the decision is already a cheap projection of the public
radical result.

This helper does not identify simple factors, compute a classification, or
calculate the solvable radical.

The supported dimension is at most eight, as in the current
finite-dimensional Lie-algebra carrier.

For the theorem, see MIT OpenCourseWare's [18.745 Lecture 17: Proofs of the
Cartan Criteria and Properties of Semisimple Lie
Algebras](https://ocw.mit.edu/courses/18-745-lie-groups-and-lie-algebras-i-fall-2020/mit18_745_f20_lec17.pdf).
