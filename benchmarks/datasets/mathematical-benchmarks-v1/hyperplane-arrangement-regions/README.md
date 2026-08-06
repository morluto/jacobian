# Hyperplane arrangement regions

This Regression benchmark transforms ByteDance-Seed/BeyondAIME train row 20 at immutable revision `c705198ae1043810b1e1693bd879250b51a7a523` (CC0-1.0).

The source answer is public, but the task requires a complete insertion certificate. The clean-room verifier derives the ten planes from frozen coordinates, detects the cube/tetrahedron duplicate, restricts prior planes to each new plane, canonicalizes the resulting affine lines, counts their exact intersections, and recomputes every region increment. Plane order and nonzero integer scaling are free.

Family: **Regression**. Primary objective: **exact hyperplane-arrangement certification**. Quality score: **90/100**. Difficulty: **Hard (provisional)** because the agent must handle degeneracy, duplicate geometry, restrictions, and insertion counts; baseline calibration is pending. Assurance is `COMPUTED`, not proof-assistant verified.

Shortcut audit: `64`, a generic-position formula, or the nine-unique-plane observation alone cannot pass. The verifier accepts alternative insertion orders, so it does not force a single trace.
