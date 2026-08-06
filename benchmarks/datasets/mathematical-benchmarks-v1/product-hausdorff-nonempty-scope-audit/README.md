# Product Hausdorff nonempty-factor scope audit

This Assurance benchmark comes from ProofNetVerif test row 59 at immutable revision `91183e5b12d64374827bf2782db629b5b0f8f319` (MIT). It audits an autoformalization that drops the nonemptiness assumptions needed to recover factor embeddings from a product.

The verifier checks an agent-chosen finite T0 non-Hausdorff topology, all topology closure laws, pairwise T0/Hausdorff predicates, and the empty product. Requiring at least four points, five open sets, three factors, and T0 separation blocks the indiscrete two-point micro-witness while preserving alternative valid topologies.

Family: **Assurance**. Primary objective: **missing-assumption diagnosis**. Difficulty: **Hard (provisional)** because the agent must coordinate a complete finite topology and product-scope countermodel; empirical calibration is pending. Quality score: **86/100**. Assurance is capped at `COMPUTED`; Lean elaboration and infinite topological spaces are outside scope.
