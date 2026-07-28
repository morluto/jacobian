# Public research challenge corpus

This directory contains answer-visible, research-level diagnostic cases for
testing how an agent uses Jacobian. The cases are deliberately public and
must not be reported as held-out model evaluations.

`public_postdoc_v1.json` is the first curated suite.
`public_postdoc_frontier_v1.json` adds six cross-domain cases selected after the
portfolio-hardening runs:

- one current-portfolio closure candidate;
- two compositional stretches; and
- three deliberate capability-gap probes.

The frontier supplement includes parameterized identities, exact graded
syzygies, finite design obstructions, bounded formal certificates, exact
probabilistic LP evidence, and graph-formalization semantic auditing. It is a
separate versioned suite so the original twelve-case sample and its recorded
runs remain stable.

Each case contains:

- a self-contained mathematical statement and copy-ready prompt;
- primary-source provenance, including immutable revisions when available;
- the Jacobian catalog digest against which capability fit was assessed;
- an answer-visible oracle summary and required evidence;
- the nearby installed Jacobian capabilities;
- the expected tool fit and known capability boundary; and
- fail-closed conditions that must remain non-conclusions.

The suite uses three diagnostic tiers:

- `CLOSURE_CANDIDATE`: the current portfolio should plausibly produce the
  decisive finite evidence.
- `COMPOSITIONAL_STRETCH`: the ingredients overlap the current portfolio, but
  the agent must compose several operations or bridge a bounded scale gap.
- `CAPABILITY_GAP_PROBE`: the source problem is suitable precisely because a
  successful run should expose a missing domain operation, scale envelope, or
  independent checker.

Every prompt forbids web search and external knowledge retrieval so a run
measures mathematical composition rather than source retrieval. The source
URLs are evaluator metadata and must not be supplied to the model during a
run. Jacobian's own retrieval capabilities are also out of scope for these
diagnostics; mathematical compute and checker capabilities remain in scope.

## Evaluation policy

These cases are public reproductions:

- `scored` is always `false`;
- source answers and artifacts may be in model training data;
- a matching final answer is not evidence that Jacobian caused the result;
- tool traces, artifacts, and verification boundaries are the useful output;
- `TIMEOUT`, `ERROR`, incomplete search, and failure to find a witness remain
  non-conclusions; and
- only an operator-authorized independent checker may return `VERIFIED`.

Use a separate, access-controlled manifest and independent oracle for a
held-out comparative evaluation.

## Running a sample

Select cases from the JSON by `challenge_id` and pass the `prompt` field
unchanged to the model. Record:

- model and reasoning configuration;
- Jacobian catalog digest and deployment revision;
- wall time and termination reason;
- capability calls and materialized artifact URIs;
- mathematical conclusion, completeness, and assurance separately; and
- any unsupported contract, timeout, or scale limit.

Random sampling should record the seed and sampled IDs. Do not silently replace
a timed-out case with another case.
