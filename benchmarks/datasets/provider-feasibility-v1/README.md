# Jacobian provider-feasibility-v1

This Oracle-only dataset reproduces five optional-backend feasibility spikes in
provider-specific pinned environments: cddlib, CGAL, GUDHI, nauty, and Regina.
Each task reports pin fidelity, the expected bounded provider
outcome, and explicit non-conclusions when its dedicated environment cannot be
constructed.

These tasks do not install providers into Jacobian's core environment, register
operations, or authorize provider implementations as independent checkers.
Provider absence here is a task-environment failure and has no bearing on
unrelated datasets or kernel startup.

Run one task with `make provider-eval PROVIDER=<name>`.
