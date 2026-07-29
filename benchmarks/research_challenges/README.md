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

`public_postdoc_frontier_status_v2.json` is the current status and evaluation
overlay for that immutable v1 input. In particular, the v1 suite's 256-capability
snapshot and its `MISSING` classification for `jcb-postdoc-014` are historical
facts, not live portfolio claims. The v2 overlay binds the exact v1 file digest,
records current default and no-retrieval catalog snapshots, names repository
regressions, and carries discovery/evaluation gates for unresolved candidates.
The catalog snapshots describe the pre-manifest environment at their named
repository commit; provider availability remains installation-specific. Future
status changes create a new overlay version; they do not rewrite v1.

`public_postdoc_status_v2.json` applies the same immutable-input rule to the
original twelve-case suite. It records the 269-capability portfolio at commit
`081834c979ec8f1c3b3995ebb86908bd82333a07`, corrects current boundaries such
as the later Hamiltonian-path capability without editing historical labels,
and freezes the probability, discrete-mathematics, computational-geometry, and
topology coverage matrix used by the four-domain capability roadmap. Its four
accepted discovery records are implementation handoffs, not claims that the
capabilities already exist.

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

The dedicated runner validates the suite, passes each `prompt` field unchanged,
records the selection seed and prompt digest, and starts a fresh local Jacobian
server under `COMPUTE_VERIFY_NO_RETRIEVAL`. It is plan-only unless model work and
an explicit process budget are both authorized:

```console
uv run python benchmarks/research_challenge.py \
  --challenge jcb-postdoc-014

uv run python benchmarks/research_challenge.py \
  --sample-size 3 \
  --seed 17 \
  --repetitions 2 \
  --model gpt-5.6 \
  --reasoning-effort xhigh \
  --execute \
  --max-model-runs 6
```

The same isolated capability profile can be exposed as an independent
streamable-HTTP evaluation endpoint:

```console
uv run jacobian-mcp \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8001 \
  --allow-anonymous \
  --anonymous-tenant-id public-frontier-evaluation \
  --capability-policy-profile COMPUTE_VERIFY_NO_RETRIEVAL
```

Keep that endpoint separate from a default production endpoint: its catalog and
policy digests are part of the evaluation intervention. The runner records:

- model and reasoning configuration;
- Jacobian catalog digest and deployment revision;
- wall time and termination reason;
- capability calls and materialized artifact URIs;
- mathematical conclusion, completeness, and assurance separately; and
- any unsupported contract, timeout, or scale limit.

Random sampling should record the seed and sampled IDs. Do not silently replace
a timed-out case with another case.
