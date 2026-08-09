---
name: jacobian-math
description: Use Jacobian for exact mathematics, symbolic computation, structural analysis, counterexamples, bounded search, formal inspection, and independent verification. Trigger on relevant math tasks even when the user does not name Jacobian.
---

# Jacobian Math

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
shell-solvable. Keep decomposition and routing decisions agent-owned;
composing already-known supporting operations remains allowed when clearer.
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
Use low `limit` values. For a selected operation's schema, call
`math.find({"capability_id":"<exact-id>","view":"CONTRACT"})`; never send
`mode: "CONTRACT"` to `math.run` or put `CONTRACT` in a query. A card's
`invocation_example`, or required top-level fields, may be enough.

Do not add a discovery domain filter unless its exact installed spelling is
known. When discovery exposes recovery paths, follow those fields before
treating absence as final. After invalid input, correct the reported constraint
and retry within the task resource bounds. If
one provider is unavailable, continue with other installed routes that can
produce the outcome. Treat timeouts as non-conclusions. Accept only a completed
result whose scope covers the input, and carry forward the smallest decisive
value, witness, status, assurance, completeness, and open obligations; preserve
artifact refs, including verification record URIs.

When independent checking is requested, model-authored calculations or programs
are not independent evidence. Use installed `VERIFY` when available. An artifact
URI or checker summary is not a task-local verification-record file: never
reconstruct or paraphrase such a record from returned fields. Claim `VERIFIED`
only when the result has assurance level `VERIFIED`, exact record bytes, and
required task authorization and bindings are preserved; otherwise use lower
task-permitted assurance. Verification is bound to the exact checked claim: do
not transfer `VERIFIED` from an input, premise, factorization, or related
artifact to a model-derived conclusion, which needs its own checker-bound record.
Before each run, compare every payload component with the intended object.
Afterward, compare echoed scope parameters or its bound digest/URI with that
same input. If either check is missing or mismatched, do not apply the output.
Before concluding, account for every exact outcome in the final comparison. A
`VERIFIED` sub-result does not validate another result or their comparison; use
suitable installed operations and compare returned values.
