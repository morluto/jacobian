# Topology generation order audit

This **Assurance** benchmark is derived from `PAug/ProofNetVerif` validation row 0 at immutable revision `91183e5c11ad63a6a207b8193c1e42378151a3f5` (MIT).

The source statement asks for the unique smallest topology containing a family of topologies, while the proposed formalization merely asserts the existence of a common order bound. The task does not reveal a countermodel. An agent must construct an incomparable family of finite topologies, compute its generated topology, and provide a distinct common subtopology demonstrating that the proposed contract does not express the source objective.

The verifier accepts alternative finite models on three or four points. It exhaustively checks topology axioms, recomputes the least containing topology by enumerating all finite topologies, and validates both containment directions. It does not invoke Lean or claim that its finite set-family convention mechanically interprets Lean's `TopologicalSpace` order.

Primary reasoning objective: **semantic assurance calibration for an order-sensitive formalization**. Difficulty is **Medium–Hard (provisional)**: discovery of a nontrivial incomparable family and exact closure is required, but the finite verifier bounds the search. The shortcut audit excludes two-point and comparable-family witnesses and does not expose an Oracle model.

Assurance is capped at **COMPUTED**.
