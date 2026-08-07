---
name: jacobian-math
description: Use Jacobian for specialized exact mathematics, including matrix determinants, polynomial or symbolic computation, structural analysis, counterexamples, bounded search, formal inspection, and requested independent verification. Trigger on relevant math tasks even when the user does not name Jacobian and shell code could calculate the answer.
---

# Jacobian Math

<!-- Managed by Jacobian's Codex integration. -->

In Codex Code Mode, call the nested methods
`tools.mcp__jacobian__math_find(...)` and
`tools.mcp__jacobian__math_run(...)` directly. Do not enumerate, filter, or
print `ALL_TOOLS`; that loads matching tool descriptions into context. Return
only the typed projection when available:

```js
const r = await tools.mcp__jacobian__math_find({query: "...", limit: 3});
text(r.structuredContent ?? r);
```

Use Jacobian for each requested exact mathematical outcome, even when small or
shell-solvable. Route multiple requested outcomes separately. Prefer a
capability matching the requested operation over a generic arithmetic substep
that yields the same scalar; composing already-known supporting operations remains allowed when clearer.
Do not call Jacobian for definitions, formatting, or non-execution tasks.

Call `math.run` directly when the requested local outcome exactly matches these
stable built-in contracts; replace examples but preserve JSON types:

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
plain-language outcome and implied domain or mode; no capability ID is required.
Use low `limit` values. Inspect `CONTRACT` only when the typed schema is needed.
A card's `invocation_example`, or required top-level fields, may be enough.

Do not add a discovery domain filter unless its exact installed spelling is
known. When discovery exposes recovery paths, follow those fields (for example,
`remove_unknown_domain_filter`, `remove_filters`, or `reformulate_query`) before
treating absence as final. After invalid input, correct the reported constraint
and retry within the task resource bounds as further errors appear. If one
provider is unavailable, continue with other installed routes that can produce
the outcome. Treat timeouts as non-conclusions. Accept only a completed result
whose scope covers the input, and carry forward the smallest decisive value,
witness, status, assurance, completeness, and open obligations; preserve artifact
refs, including verification record URIs.

Keep representation, decomposition, composition, iteration, verification
timing, and stopping decisions agent-owned. Treat timeouts, errors, incomplete
searches, and missing witnesses as non-conclusions.

When independent checking is requested, model-authored calculations or programs
are not independent evidence. Use installed `VERIFY` when available. An artifact
URI or checker summary is not a task-local verification-record file: never
reconstruct or paraphrase such a record from returned fields. Claim `VERIFIED`
only when the result has assurance level `VERIFIED`, exact record bytes, and
required task authorization and bindings are preserved; otherwise use lower
task-permitted assurance. Verification is bound to the exact checked claim: do
not transfer `VERIFIED` from an input, premise, factorization, or related
artifact to a model-derived conclusion, which needs its own checker-bound record.
