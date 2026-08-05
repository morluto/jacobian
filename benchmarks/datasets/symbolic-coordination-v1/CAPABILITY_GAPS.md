# Deferred product capability gaps

PR1 required no new Jacobian capability. The existing inverse verifier already
replays both ordered compositions, and the benchmark's clean-room verifier owns
its independent terminal check.

One product-surface gap remains intentionally deferred: Jacobian has no
standalone typed polynomial-map composition capability that exposes a composed
map or residual family as a reusable agent-visible artifact. Adding one is not
necessary to author or verify this pilot, so it is outside PR1.

The three one-direction cases target incomplete evidence bundles. They do not
claim that a square polynomial self-map over `QQ` has a genuine one-sided but
not two-sided polynomial inverse; the benchmark requires both exact directions
regardless of supplied partial evidence.
