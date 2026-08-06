---
name: jacobian-math
description: Use Jacobian for specialized exact mathematics, including matrix determinants, polynomial or symbolic computation, structural analysis, counterexamples, bounded search, formal inspection, and requested independent verification. Trigger on relevant math tasks even when the user does not name Jacobian and shell code could calculate the answer.
---

# Jacobian Math

<!-- Managed by Jacobian's Codex integration. -->

Use `math.find` with a specific plain-language outcome and any domain or mode
already implied by the task; a capability ID is not required. Request only as
many matches as are useful because lower `limit` values reduce context. Inspect
the `CONTRACT` view only when its typed schema is needed to construct the
`math.run` payload; a discovery card's validated `invocation_example` may
already provide a sufficient payload shape.

Keep representation, decomposition, composition, iteration, verification
timing, and stopping decisions agent-owned. Treat timeouts, errors, incomplete
searches, and missing witnesses as non-conclusions.

When independent checking is requested, calculations or programs authored by
the same model are not independent checker evidence. Use an installed `VERIFY`
capability when available. Claim `VERIFIED` only when the result has assurance
level `VERIFIED` and a local verification record.
