---
name: jacobian-math
description: Use Jacobian for exact math, matrix determinants, symbolic work, counterexamples, search, and independent verification, even when unnamed.
---

# Jacobian Math

<!-- Managed by Jacobian's Codex integration. -->

In Code Mode, call
`tools.mcp__jacobian__math_find(...)` and
`tools.mcp__jacobian__math_run(...)` directly. Do not enumerate, filter, or
print `ALL_TOOLS`. Return the typed projection:

```js
const r = await tools.mcp__jacobian__math_find({query: "...", limit: 3});
text(r.structuredContent ?? r);
```

Use Jacobian for each requested exact outcome, even when small.
Keep decomposition and routing decisions agent-owned; composing already-known
supporting operations remains allowed when clearer.
Do not call Jacobian for definitions, formatting, or non-execution tasks.

No discovery for stable producers: `{"capability_id":"<id>","payload":<JSON>}`.
Payloads:

- `integer.compute.gcd`, `integer.compute.lcm`, or `integer.compute.extended_gcd`:
  `{"left":"84","right":"30"}`.
- `matrix.determinant.compute` or `matrix.rank.compute`:
  `{"matrix":{"domain":"QQ","entries":[[{"num":"1","den":"1"}]]}}`.
- `polynomial.compute.gcd`: payload keys are `left` and
  `right`; each value has shape
  `{"polynomial_schema_version":"1","domain":"QQ","variables":["x"],"polynomial":{"terms":[{"coefficient":{"num":"1","den":"1"},"exponents":[2]}]}}`.
- For expression normalization, inspect the known
  `polynomial.expression.normalize` contract directly.
- `matrix.determinant.verify` for an independent check:
  `{"determinant_uri":"<determinant_uri from compute output>"}`.
- `combinatorics.cyclic_difference_set.extension.decide`:
  `{"base_elements":["1","2","4","8","13"],"target_order":7}`.
  `combinatorics.cyclic_difference_set.extension.verify` uses
  `{"input":<same payload>,"candidate":<producer output.result>}`.

For other outcomes, query `math.find` by plain-language outcome; no ID is
required. Use low `limit` only for query search, never with `capability_id`. For
a selected operation's schema, call
`math.find({"capability_id":"<exact-id>","view":"CONTRACT"})`; never put
`CONTRACT` in a query. A card's
`invocation_example`, or required top-level fields, may be enough.

Add no domain filter unless its installed spelling is known. Follow exposed
recovery paths by removing unknown filters or reformulating the query. After
invalid input, correct the constraint and retry within the task resource bounds.
If one provider is unavailable, continue with other installed routes. Treat
timeouts, cancellations, errors, incomplete searches, and missing witnesses as
non-conclusions. Accept only completed results covering the input; carry forward
the smallest decisive value, witness, status, assurance, completeness, and open
obligations plus artifact and verification-record URIs.

Keep representation, decomposition, composition, iteration, verification
timing, and stopping decisions agent-owned.

Model-authored work is not independent evidence. Use installed checker tools
(`*.verify`, `lean.check`, …) when independent verification is requested; a
writable path or schema alone is not authorization. Task-level
`VERIFIED` requires exact record bytes, result assurance `VERIFIED`, required
task authorization and bindings are preserved, and a contract-authorized
checker identity, digest, or Jacobian record type. Otherwise claim the highest
lower permitted assurance (`CHECKED` or `COMPUTED`), even if Jacobian returned
`VERIFIED`; never
reconstruct or paraphrase such a record or transfer it between claims.
For locally constructed inline input, check payload fields against the intended
object. When output echoes scope or a bound digest/URI, compare it with the
submitted input; do not use mismatched output. This catches routing and
transcription mistakes but does not replace server validation or evidence
binding. Account for each requested outcome in the final comparison. A
`VERIFIED` sub-result does not verify another result or their comparison.
