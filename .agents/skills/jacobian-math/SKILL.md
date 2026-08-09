---
name: jacobian-math
description: Use Jacobian for exact mathematics, matrix determinants, symbolic computation, structural analysis, counterexamples, bounded search, and independent verification. Trigger on relevant math tasks even when the user does not name Jacobian.
---

# Jacobian Math

<!-- Managed by Jacobian's Codex integration. -->

In Code Mode, call
`tools.mcp__jacobian__math_find(...)` and
`tools.mcp__jacobian__math_run(...)` directly. Do not enumerate, filter, or
print `ALL_TOOLS`. Return the typed projection when available:

```js
const r = await tools.mcp__jacobian__math_find({query: "...", limit: 3});
text(r.structuredContent ?? r);
```

Use Jacobian for each requested exact mathematical outcome, even when small.
Keep decomposition and routing decisions agent-owned; composing already-known
supporting operations remains allowed when clearer.
Do not call Jacobian for definitions, formatting, or non-execution tasks.

Call `math.run` directly for these stable contracts, preserving JSON types:

- `integer.compute.gcd`, `integer.compute.lcm`, or `integer.compute.extended_gcd`:
  `{"left":"84","right":"30"}`.
- `matrix.determinant.compute` or `matrix.rank.compute`:
  `{"matrix":{"domain":"QQ","entries":[[{"num":"1","den":"1"}]]}}`.
- `polynomial.compute.gcd` in `EXPLORE` mode: payload keys are `left` and
  `right`; each value has shape
  `{"polynomial_schema_version":"1","domain":"QQ","variables":["x"],"polynomial":{"terms":[{"coefficient":{"num":"1","den":"1"},"exponents":[2]}]}}`.
- `matrix.determinant.verify` in `VERIFY` mode for an independent check:
  `{"determinant_uri":"<determinant_uri from compute output>"}`.

For other outcomes, use `math.find` with a specific plain-language outcome;
no capability ID is required.
Use low `limit` values. For a selected operation's schema, call
`math.find({"capability_id":"<exact-id>","view":"CONTRACT"})`; never send
`mode: "CONTRACT"` to `math.run` or put `CONTRACT` in a query. A card's
`invocation_example`, or required top-level fields, may be enough.

Do not add a discovery domain filter unless its exact installed spelling is
known. Follow exposed recovery paths such as removing unknown filters or
reformulating the query before
treating absence as final. After invalid input, correct the reported constraint
and retry within the task resource bounds. If
one provider is unavailable, continue with other installed routes that can
produce the outcome. Treat timeouts, cancellations, errors, incomplete searches,
and missing witnesses as non-conclusions. Accept only a completed
result whose scope covers the input, and carry forward the smallest decisive
value, witness, status, assurance, completeness, and open obligations; preserve
artifact refs, including verification record URIs.

Keep representation, decomposition, composition, iteration, verification
timing, and stopping decisions agent-owned.

When independent checking is requested, model-authored calculations or programs
are not independent evidence. Use installed `VERIFY` when available. An artifact
URI or checker summary is not a task-local verification-record file: never
reconstruct or paraphrase such a record from returned fields. Claim `VERIFIED`
only when the result has assurance level `VERIFIED`, exact record bytes, and
required task authorization and bindings are preserved; otherwise use lower
task-permitted assurance. Verification is bound to the exact checked claim: do
not transfer `VERIFIED` from an input, premise, factorization, or related
artifact to a model-derived conclusion, which needs its own checker-bound record.
For locally constructed inline input, check payload fields against the intended
object. When output echoes scope or a bound digest/URI, compare it with the
submitted input; do not use mismatched output. This catches routing and
transcription mistakes but does not replace server validation or evidence
binding. Account for each requested outcome in the final comparison. A
`VERIFIED` sub-result does not verify another result or their comparison.
