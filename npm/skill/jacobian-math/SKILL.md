---
name: jacobian-math
description: Use Jacobian's MCP operations for nontrivial mathematical computation, finite search, structural analysis, formal checking, probability, or optimization.
---

<!-- Managed by Jacobian setup. -->

# Jacobian mathematics

Use Jacobian when a mathematical task benefits from exact computation or a
bounded check. Search with `math.find`, then execute a selected operation with
`math.run`.

Keep the mathematical reasoning in the agent. Treat operations as atomic
instruments: compose their typed results yourself rather than asking discovery
to prescribe a proof strategy or workflow.

Before running an operation:

- Inspect its exact request schema and mathematical scope with `math.find`.
- Supply canonical mathematical values without backend-specific expressions or
  ambient contexts.
- Respect the operation's declared bounds and distinguish exact results from
  incomplete, unknown, unavailable, or transport outcomes.

Use returned values and certificates only for the conclusion they establish.
Absence of a witness, a timeout, or an incomplete search is not a proof of a
global mathematical claim.
