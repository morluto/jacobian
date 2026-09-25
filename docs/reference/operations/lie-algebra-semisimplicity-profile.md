# Lie algebra semisimplicity profile

`lie_algebra.semisimplicity.profile.compute` accepts a finite-dimensional Lie
algebra over `QQ` and returns its exact Killing form, the canonical RREF basis
of the Killing-form radical, and `is_semisimple`. By Cartan's criterion in
characteristic zero, the status is true exactly when this radical is zero.
The result retains its source algebra and the form/radical values, so callers
can inspect or pass along the exact data that determines the status.

The operation uses the existing Killing-form and exact nullspace kernels. Lie
algebra admission checks Jacobi and bounds the form, nullspace, and result
construction before they run. The supported dimension is at most eight, as in
the finite-dimensional Lie-algebra carrier. The profile does not identify
simple factors, compute a classification, or call the Killing-form radical the
solvable radical.

For the theorem, see MIT OpenCourseWare's [18.745 Lecture 17: Proofs of the
Cartan Criteria and Properties of Semisimple Lie
Algebras](https://ocw.mit.edu/courses/18-745-lie-groups-and-lie-algebras-i-fall-2020/mit18_745_f20_lec17.pdf).
