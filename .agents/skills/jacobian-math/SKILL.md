---
name: jacobian-math
description: Use Jacobian for specialized exact mathematics, including matrix determinants, polynomial or symbolic computation, structural analysis, counterexamples, bounded search, formal inspection, and requested independent verification. Trigger on relevant math tasks even when the user does not name Jacobian and shell code could calculate the answer.
---

# Jacobian Math

<!-- Managed by Jacobian's Codex integration. -->

In Codex Code Mode, call the nested methods
`tools.mcp__jacobian__math_find(...)` and
`tools.mcp__jacobian__math_run(...)` directly. Do not enumerate, filter, or
print `ALL_TOOLS` merely to locate them; that needlessly adds every matching
tool description to the model context. Return only the typed projection when
available, for example:

```js
const r = await tools.mcp__jacobian__math_find({query: "...", limit: 3});
text(r.structuredContent ?? r);
```

Call `math.run` directly when the requested local outcome exactly matches one of
these stable built-in contracts; replace the example values but preserve the
shown JSON types:

- `integer.compute.gcd`, `integer.compute.lcm`, or
  `integer.compute.extended_gcd` in `EXPLORE` mode:
  `{"left":"84","right":"30"}`.
- `matrix.determinant.compute` or `matrix.rank.compute` in `EXPLORE` mode:
  `{"matrix":{"domain":"QQ","entries":[[{"num":"1","den":"1"}]]}}`.
- `polynomial.compute.gcd` in `EXPLORE` mode: payload keys are `left` and
  `right`; each value has shape
  `{"polynomial_schema_version":"1","domain":"QQ","variables":["x"],"polynomial":{"terms":[{"coefficient":{"num":"1","den":"1"},"exponents":[2]}]}}`.
- For a requested independent determinant check,
  `matrix.determinant.verify` in `VERIFY` mode:
  `{"determinant_uri":"<determinant_uri from compute output>"}`.

For other outcomes or unfamiliar payloads, use `math.find` with a specific
plain-language outcome and any domain or mode already implied by the task; a
capability ID is not required. Request only as many matches as are useful
because lower `limit` values reduce context. Inspect the `CONTRACT` view only
when its typed schema is needed to construct the `math.run` payload. A discovery
card's validated `invocation_example`, or its required top-level fields when no
example is available, may already provide a sufficient payload shape.

Keep representation, decomposition, composition, iteration, verification
timing, and stopping decisions agent-owned. Treat timeouts, errors, incomplete
searches, and missing witnesses as non-conclusions.

When independent checking is requested, calculations or programs authored by
the same model are not independent checker evidence. Use an installed `VERIFY`
capability when available. An artifact URI or checker-result summary is not a
task-local verification-record file: never reconstruct or paraphrase such a
record from the returned fields. Claim `VERIFIED` only when the result has
assurance level `VERIFIED`, the exact record bytes are available, and any
required task authorization and bindings are preserved. Otherwise use a lower
assurance permitted by the task.
