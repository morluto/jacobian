# Observable trajectory-state value estimation

[Documentation home](../../index.md) · [Evaluation methods](evaluation-methods.md)

This research surface tests whether Jacobian's typed mathematical runtime plus
the model-authored external reasoning log can provide a cheap, interpretable
state abstraction for offline value estimation. It borrows the milestone idea
from [Numca and Hista](https://arxiv.org/abs/2605.29782), but does not use model
hidden states, train a critic, update model weights, or authorize mathematical
assurance.

The work is staged. The version 1 extractor, deterministic offline value
comparison, and observation-only replay scorer are defined. The repeated Codex
study is separately preregistered before its labels are collected.

## Version 1 state contract

[`trajectory-state-v1.schema.json`](schemas/trajectory-state-v1.schema.json)
is generated from the closed Pydantic contract in
`jacobian.eval.trajectory_state`. An extraction binds the raw Codex JSONL by
SHA-256, the task family, ordered state snapshots, optional clean-room terminal
evidence, and the fixed declaration `assurance_authority = false`.

Each snapshot contains:

- a hard state: typed inline objects and artifacts, candidate and checker
  state, open and discharged obligations, execution, completeness, assurance,
  scope, binding validity, scope-escalation errors, the latest meaningful
  transition, and reasoning-protocol state;
- an optional soft state containing only PLAN, latest AFTER_TOOL, and FINAL
  external summaries authored by the model; and
- its observation boundary, hard-state digest, changed fields, eligible
  milestone kinds, and an explicit eligibility reason.

The soft state is observable external text. It is not hidden chain-of-thought.
It is not used to decide milestone eligibility.

## Extraction and anti-hacking semantics

The deterministic extractor accepts completed Codex MCP events. A successful
`reasoning.write` creates PLAN, AFTER_TOOL, or FINAL observation boundaries.
A `math.run` completion creates a TOOL_RESULT boundary and reads only the typed
capability result: output, artifacts, obligations, diagnostics, scope,
completeness, and assurance. Rejected reasoning writes do not exist in the
durable reasoning log and are ignored.

A boundary is an eligible milestone only when at least one of these typed
changes occurs:

- a new mathematical object, artifact, candidate, or repaired candidate;
- checker acceptance or rejection;
- an obligation opens or is discharged;
- independently verified evidence makes a binding valid, or diagnostics make
  one invalid;
- scope changes or a scope escalation is rejected; or
- completeness or assurance changes.

The following invariants are enforced:

- a tool invocation is never a milestone by itself;
- repeated identical outputs, artifacts, candidates, and statuses add no
  milestone;
- longer or rewritten summaries add no milestone;
- TIMEOUT, CANCELLED, ERROR, missing results, and incomplete search do not
  materialize claimed output objects;
- checker acceptance and `VERIFIED` assurance remain separate; a candidate is
  recorded as verified only when the typed result has `VERIFIED` assurance and
  a verification-record binding;
- clean-room terminal acceptance is a separate label, never intermediate
  assurance and never a milestone reward; and
- the extractor cannot mutate Jacobian state or mathematical assurance.

Malformed JSONL fails closed. Closed models reject unknown fields. A terminal
label marked TIMEOUT, CANCELLED, or ERROR must be INCONCLUSIVE.

## PR1 real-trajectory observation

The immutable sample under
[`pr1-gcd-real-codex`](../../../tests/unit/tooling/fixtures/trajectory_state/pr1_gcd_real_codex/manifest.json)
was collected on source revision
`86666f6fc27564bbc32f6e652b64a5f4ca50940e` with Codex CLI 0.147.0,
`gpt-5.4-mini`, medium reasoning effort, a read-only workspace, and REQUIRED
reasoning logs. The manifest binds the exact prompt, catalog and policy
digests, raw Codex JSONL, exported durable reasoning log, stderr, and extracted
record.

Manual inspection found four states:

| Boundary | Eligible | Typed change |
| --- | --- | --- |
| PLAN | no | protocol and soft summary only |
| TOOL_RESULT | yes | exact gcd object, declared scope, COMPLETE coverage, COMPUTED assurance |
| AFTER_TOOL | no | protocol and soft summary only |
| FINAL | no | protocol and soft summary only |

The model first made a rejected PLAN call with an extra field, recovered, and
completed a valid reasoning cycle. That failure changed the design: only
successful durable reasoning writes may create states. The one real sample is
a parser and interpretability check, not evidence of predictive value. It has
no clean-room terminal verifier label, so it cannot enter the later value
comparison as a labelled evaluation row.

## Version 1 offline value comparison

[`trajectory-value-corpus-v1.schema.json`](schemas/trajectory-value-corpus-v1.schema.json)
and
[`trajectory-value-evaluation-v1.schema.json`](schemas/trajectory-value-evaluation-v1.schema.json)
define the closed input and output contracts in
`jacobian.eval.trajectory_value`. The corpus schema references the immutable
version 1 extraction schema rather than copying it.

A labelled trajectory is eligible only when its clean-room verifier completed,
returned ACCEPTED or REJECTED, and explicitly bound both the exact input and
artifact. INCONCLUSIVE, TIMEOUT, CANCELLED, ERROR, missing binding, duplicate
transcript digests, and task groups with only one rollout fail closed. The
terminal reward is derived by the evaluator and cannot be supplied by a
caller:

- `1` means clean-room ACCEPTED;
- `0` means clean-room REJECTED; and
- inconclusive evidence is excluded rather than coerced to failure.

Only PLAN, eligible typed milestones, and the AFTER_TOOL immediately following
an eligible TOOL_RESULT become comparison observations. A repeated
non-milestone result and its prose cannot add an observation. FINAL and
TERMINAL are excluded to prevent terminal-answer leakage. AFTER_TOOL remains
an observable state boundary, but is not itself a milestone or reward.

### Compared estimators

All five estimators cluster only within one task group, matching the
cross-rollout setting. Their features are deterministic:

1. `GROUP_ROLLOUT` uses the other rollouts' terminal mean and no intermediate
   feature.
2. `NUMCA_NUMERICAL` groups the cumulative, first-seen sequence of exact
   numerical spans in PLAN and AFTER_TOOL text. This diagnostic baseline does
   not make numerical text a Jacobian milestone.
3. `REASONING_TEXT` uses lowercase word unigrams and bigrams, corpus TF-IDF,
   cosine similarity, and deterministic average-link agglomeration at the
   exact threshold `350000 / 1000000`.
4. `JACOBIAN_TYPED` uses an exact qualitative signature containing boundary,
   task family, exact typed object and artifact identities, exact candidate
   digest, candidate and checker state, exact obligation identities, execution,
   completeness, assurance, exact scope digest and escalation errors, binding
   state, latest transition, and reasoning-protocol state.
5. `HYBRID_TYPED_TEXT` first partitions by the complete typed signature, then
   applies the same text clustering only within that compatible partition.

Clustering is unsupervised and transductive: a target state's observable
features can affect cluster geometry, but its terminal label cannot. Every
value uses leave-one-trajectory-out support, and each supporting trajectory
votes once even if it has several states in a cluster. A singleton cluster
falls back to the other rollouts in the same task group. Reported Brier score
and mean absolute error first average within each trajectory and then across
trajectories, so longer logs receive no extra metric weight. Assignments expose
the cluster, support trajectories, feature digests, estimated value, and
eventual result. Every output declares `assurance_authority = false`.

### PR2 controlled experiment

The frozen
[`comparison summary`](../../../tests/unit/tooling/fixtures/trajectory_value/pr2_controlled/comparison-summary.json)
is regenerated by the unit fixture and binds evaluator version, configuration,
corpus digest, and exact metrics. It has eight deliberately constructed
polynomial trajectories, two task groups, and 24 selected observations. One
group makes opposite checker states use identical language; the other gives
opposite reasoning branches the same typed state. All numerical spans are
uninformative.

| Estimator | Clusters | Brier | MAE |
| --- | ---: | ---: | ---: |
| Group/rollout | 2 | 0.444444444445 | 0.666666666667 |
| Numca-like numerical | 2 | 0.444444444445 | 0.666666666667 |
| Reasoning text only | 3 | 0.222222222222 | 0.333333333334 |
| Jacobian typed only | 8 | 0.296296296297 | 0.444444444445 |
| Hybrid typed plus text | 11 | 0.074074074074 | 0.111111111111 |

The designed result is useful as a mechanism check, not as a performance
estimate. Text-only clustering incorrectly aliases identical prose across
accepted and rejected checker states. Typed-only clustering correctly
separates those states, but aliases the deliberately identical typed states
whose external reasoning branches differ. Hybrid clustering separates both.
A repeated non-milestone tool result adds zero observations. This falsifies the
idea that either text or the current typed signature is universally sufficient
on its own, but only for these constructed aliasing cases.

## Version 1 observation-only replay scorer

[`trajectory-score-replay-v1.schema.json`](schemas/trajectory-score-replay-v1.schema.json)
defines the closed output from `jacobian.eval.trajectory_score`. The scorer
consumes one immutable PR2 comparison and selects one declared estimator and
trajectory. It does not refit clusters, inspect hidden model state, invoke
Codex, call Jacobian tools or verifiers, choose an action, alter a prompt, or
mutate runtime or mathematical assurance.

Each replay row exposes:

`observation/state digest → cluster and feature summary → support rollouts → estimated value → value delta → milestone credit → eventual terminal result`.

The comparison digest binds the complete sorted evaluator JSON, while the
source corpus digest preserves the PR2 label boundary. Stale cluster references,
duplicate state identities, missing trajectories, inconsistent terminal
bindings, and replays that do not begin with PLAN fail closed. All scorer and
row contracts declare `observation_only = true` and
`assurance_authority = false`; the root additionally fixes `chooses_tools`,
`changes_prompts`, and `mutates_runtime` to false.

Credit is deterministic and deliberately narrower than value inspection:

- the initial PLAN has no delta and receives credit `0`;
- every later row reports `value_delta = current_value - previous_selected_value`;
- a typed milestone receives exactly that delta as transition credit; and
- every non-milestone transition receives credit `0`, even when its estimated
  value changes sharply.

The frozen
[`PR3 replay summary`](../../../tests/unit/tooling/fixtures/trajectory_score/pr3_controlled/replay-summary.json)
demonstrates the distinction:

| Boundary | Milestone | Value | Delta | Credit | Cumulative credit |
| --- | --- | ---: | ---: | ---: | ---: |
| PLAN | no | 0.4 | — | 0.0 | 0.0 |
| TOOL_RESULT | yes | 0.7 | +0.3 | +0.3 | 0.3 |
| AFTER_TOOL | no | 0.2 | −0.5 | 0.0 | 0.3 |
| TOOL_RESULT | yes | 0.6 | +0.4 | +0.4 | 0.7 |

The controlled observation shows that a value drop can remain available for
analysis without turning prose, call count, or an observation boundary into
reward. It also shows that cumulative milestone credit need not equal the
last-minus-first value when intervening non-milestone value changes are
intentionally excluded. These are scorer semantics, not evidence that the
drop predicts real failure.

## PR4 preregistered real-Codex study

[`trajectory-value-study-v1.schema.json`](schemas/trajectory-value-study-v1.schema.json)
defines the closed study protocol. The frozen
[`trajectory-value-study-v1.json`](../../../benchmarks/config/trajectory-value-study-v1.json)
selects the exact locally catalogued `gpt-5.4-mini` model at medium reasoning
effort with Codex CLI 0.147.0. The runner refuses a missing model, unsupported
reasoning level, CLI-version drift, a dirty source tree, an existing result
directory, or model execution without an explicit `--execute` flag. It never
substitutes a model alias.

The study fixes four task groups and four independent rollouts per group:

| Task group | Exact terminal object |
| --- | --- |
| integer Bézout | gcd plus any valid pair of Bézout coefficients |
| matrix determinant | exact determinant of a fixed 5 by 5 integer matrix |
| polynomial gcd | monic gcd plus any exact polynomial Bézout pair over `QQ` |
| graph independent set | any maximum independent set and its optimum size |

Every rollout receives an isolated temporary workspace, a separate Jacobian
state directory, a REQUIRED reasoning-log server, and an ephemeral Codex
session. User configuration and repository rules are not loaded. The task owns
its input and submission schema; the agent may choose any mathematical method.
Tool count, tokens, latency, and log length are diagnostics only. The scorer is
not visible to the model and cannot intervene.

The terminal verifier in
`benchmarks.tooling.trajectory_value_study_verifier` uses only the Python
standard library and does not import Jacobian, the runner, the extractor, the
evaluator, or a mathematical backend. It independently recomputes each exact
relation or optimum and accepts alternate valid witnesses. It digest-binds the
exact task and regular submission before interpreting the answer. A bound wrong
or malformed answer is `REJECTED`; a missing artifact, substituted input,
non-completed Codex command, or verifier failure is `INCONCLUSIVE` and cannot
be coerced to failure.

The analysis is frozen before labels. It compares all five PR2 estimators,
replays every estimator without intervention, evaluates a strictly negative
preterminal value delta as a failure warning, measures same-typed/different-text
pairs, identifies text-cluster aliases separated by scope, assurance, checker,
obligation, completeness, or binding state, and computes leave-one-trajectory-
out univariate signal for each hard-state dimension. No threshold is tuned
after labels.

Codex CLI cannot resume an arbitrary intermediate tool boundary. The declared
Monte-Carlo-style surrogate is therefore the leave-one-trajectory-out terminal
success frequency of independently sampled compatible cross-rollout states.
It is not an exact continuation value and does not support a causal claim.

Before preregistration, a non-study gcd infrastructure smoke exercised the same
runner path. It completed one PLAN, two bound capability cycles, two AFTER_TOOL
entries, and one FINAL; the exact verifier accepted the artifact and the
extractor emitted two eligible typed milestones. That smoke is not part of the
study corpus and cannot affect metrics.

After committing the preregistration boundary, the operator command is:

```sh
uv run python -m benchmarks.tooling.trajectory_value_study run \
  --spec benchmarks/config/trajectory-value-study-v1.json \
  --output benchmarks/studies/trajectory-state-value-codex-v1 \
  --execute
```

### Frozen study result

The command was executed once from clean preregistration commit
`cd7e5d52abe3556a8ad0beb50cb82e9f4e42c86c`. The committed
[`manifest.json`](../../../benchmarks/studies/trajectory-state-value-codex-v1/manifest.json)
binds the exact source, spec, local model-catalog record, runner, verifier,
extractor, evaluator, scorer, and all 285 non-manifest artifacts. The corpus
contains the raw Codex JSONL, external reasoning logs, task-visible files,
submissions, exact verifier records, typed extractions, MCP surface and server
logs, replays, comparison, and summary for every rollout.

| Observation | Frozen result |
| --- | ---: |
| independent rollouts | 16 |
| exact verifier `ACCEPTED` | 15 |
| exact verifier `REJECTED` | 0 |
| `INCONCLUSIVE` after command timeout | 1 |
| labelled trajectories | 15 |
| selected preterminal observations | 27 |
| reasoning protocol `COMPLETE` | 9 |
| reasoning protocol `INCOMPLETE` | 7 |

The timeout occurred in `polynomial-gcd-bezout-01-r03` with one pending
`BEFORE_TOOL` record, no bound invocation result, and no final artifact. It is
excluded rather than converted into a failure label. Six other runs had an
incomplete external reasoning protocol but independently valid terminal
objects; their terminal labels remain `ACCEPTED`. This separation is expected:
reasoning-log compliance is observable workflow state, not mathematical
assurance.

All 15 labelled trajectories succeeded. Consequently every estimator has
zero Brier score and zero mean absolute error, and the preregistered A, B, C,
and F comparisons have no outcome variation with which to distinguish the
estimators, warning rule, or individual hard-state dimensions. Precision and
recall for a negative value-drop warning are undefined; no labelled trajectory
failed and no estimator emitted such a warning.

The descriptive diagnostics still expose representation behavior. There are
29 same-typed/different-text state pairs, of which the hybrid representation
separates 16, but none has mixed terminal outcomes. Text clustering also merges
12 pairs that differ in scope digest, assurance, and completeness. This
confirms that text alone can alias critical typed state in the observed corpus,
but it does not establish a predictive benefit for typed or hybrid features.

The pilot therefore falsifies the adequacy of this four-task suite for ranking
the five estimators; it does not falsify or support the estimators themselves.
No training, scorer intervention, post-label threshold tuning, or causal claim
is authorized. A future version needs a separately preregistered, more
difficult and outcome-diverse held-out suite. This v1 corpus remains immutable
instead of being repaired or supplemented post hoc.

## Current limitations

Version 1 deliberately uses conservative generic output interpretation. A
non-empty typed capability output becomes one content-addressed object, while
candidate-like fields receive a separate candidate identity. Domain-specific
semantic equivalence is not inferred. Evidence binding becomes valid only from
verified checker evidence or clean-room terminal evidence; ordinary
reasoning-call protocol binding is not mathematical progress.

The real v1 study has only one task per family, four unseeded repetitions per
task, one model and reasoning level, and no labelled failures. It does not show
that typed or hybrid states predict real continuation success, outperform
baselines, or generalize beyond the designed cases. TF-IDF vocabulary and
cluster geometry are fitted to the complete feature corpus, though labels
remain strictly leave-one-trajectory-out. The fixed threshold is a declared
first version, not a tuned optimum. Exact content-addressed compatibility fails
safe against merging distinct candidates, but may fragment semantically
equivalent objects or independently produced verification records; the
success-only corpus cannot measure that support-loss tradeoff.

The PR3 scorer replays a completed frozen comparison; it is not an online
critic and cannot score a previously unseen state without a new PR2 comparison.
That limitation prevents intervention and avoids inventing an out-of-sample
assignment rule, but it means "early" warnings are retrospective predictions
evaluated only after the rollout corpus is frozen. Codex CLI also exposes
neither an exact arbitrary intermediate-state resume operation nor a sampling
seed, so the study uses compatible independent rollouts as a non-causal
Monte-Carlo-style surrogate. Any label-informed change to task difficulty,
extraction, compatibility, clustering, threshold, or credit semantics requires
a new schema and experiment boundary.
