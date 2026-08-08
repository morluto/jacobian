# Observable trajectory-state value estimation

[Documentation home](../../index.md) · [Evaluation methods](evaluation-methods.md)

This research surface tests whether Jacobian's typed mathematical runtime plus
the model-authored external reasoning log can provide a cheap, interpretable
state abstraction for offline value estimation. It borrows the milestone idea
from [Numca and Hista](https://arxiv.org/abs/2605.29782), but does not use model
hidden states, train a critic, update model weights, or authorize mathematical
assurance.

The work is staged. The version 1 extractor, exact-state offline value
comparison, and observation-only replay scorer are defined. A separate semantic
value-state and six-estimator comparison are also frozen before the mixed-
difficulty Codex labels are collected.

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

## PR5 mixed-difficulty calibration protocol

[`trajectory-value-calibration-v1.schema.json`](schemas/trajectory-value-calibration-v1.schema.json)
defines a closed calibration protocol over existing Harbor tasks. The frozen
[`trajectory-value-calibration-v1.json`](../../../benchmarks/config/trajectory-value-calibration-v1.json)
keeps the PR4 model and reasoning settings (`gpt-5.4-mini`, medium), uses two
independent rollouts for each of eight candidates, and caps each rollout at 420
seconds. It covers candidate/checker repair, bounded-search non-conclusions,
scope and assurance traps, one-sided evidence, artifact binding, and meaningful
capability-routing choices.

Every rollout exposes only `instruction.md`, `input.json`, and
`submission_schema.json` from the existing task. The manifest binds the pinned
Harbor task digest, every public-file digest, and every task-owned verifier-file
digest. The existing verifier runs in a fresh child interpreter after Codex
exits. A completed verifier reward of one is `ACCEPTED`; a completed verifier
reward below one is `REJECTED`. Model timeout, model error, missing submission,
or verifier error is `INCONCLUSIVE`, never a negative label.

The task-selection rule is frozen before calibration labels:

- require two labelled rollouts;
- retain empirical acceptance between 20% and 80%, inclusive;
- order eligible tasks by their original candidate order and keep at most four;
- report 95% Wilson intervals without using them to change selection; and
- perform no retry of a rejected mathematical answer.

The calibration changes neither prompts nor runtime in response to an outcome.
Tool calls, tokens, reasoning-log length, and repeated actions have reward zero.
Its output is difficulty-screening evidence, not estimator evaluation. The
selected task set is frozen in a separate main-study contract only after this
calibration completes and before any H1--H3 labels are collected.

### Initial calibration result and bounded extension

The initial calibration was executed once from clean preregistration commit
`2ad56d98f265ad1cd04c55b0cd6d2e86c2704422`. Its immutable
[`manifest.json`](../../../benchmarks/studies/trajectory-value-calibration-codex-v1/manifest.json)
binds 243 non-manifest artifacts across all 16 planned rollouts. Every rollout
produced a completed clean-room label: 5 were `ACCEPTED`, 11 were `REJECTED`,
and none was `INCONCLUSIVE`.

The original metadata split sharply: the two symbolic audit tasks were each
2/2 accepted, while radical elimination, Metric TSP repair, exact Farkas,
polynomial collision, and the finite-scheme audit were each 0/2. Only
`graph-artifact-composition` was mixed at 1/2. The 95% Wilson interval for its
50% point estimate is approximately `[0.095, 0.905]`; the small calibration is
used for screening, not a precise success-rate claim.

One eligible task is insufficient for the cross-task H1--H3 study. Before
collecting any further labels, the separate
[`trajectory-value-calibration-extension-v1.json`](../../../benchmarks/config/trajectory-value-calibration-extension-v1.json)
freezes eight new candidates near the observed easy/hard boundary. It retains
the same model, prompt, budgets, terminal label, and selection rule. The
extension does not rerun or alter an initial candidate and cannot revise the
immutable first-batch result.

The extension was then executed once from clean preregistration commit
`ffe29ed6e4019ebf97ec498f62baf10465406a8d`. Its immutable
[`manifest.json`](../../../benchmarks/studies/trajectory-value-calibration-extension-codex-v1/manifest.json)
binds 246 non-manifest artifacts across all 16 planned rollouts. All verifier
executions completed: eight were accepted, eight were rejected, and none was
inconclusive. Three symbolic tasks were each 2/2 accepted; Hermite normal form,
polynomial normalization, and symmetric-polynomial divisibility were each 0/2.
The two remaining tasks were mixed:

| Selected extension task | Accepted | Rejected | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| Apollonius proof-gap repair | 1 | 1 | `[0.095, 0.905]` |
| Projective-plane homology lattice | 1 | 1 | `[0.095, 0.905]` |

The mixed failures are semantically useful rather than execution failures. In
the Apollonius task, the rejected trajectory supplied the correct points,
circle, distance polynomial, scope, evidence, limitations, and assurance, but
used the reciprocal multiplier `-1/3` instead of `-3`. In the projective-plane
task, the rejected trajectory supplied a correct spanning tree, boundary
matrix, determinant, and homology conclusion, but failed the exact evidence
contract. Both RP2 trajectories also left the external reasoning protocol
incomplete; one still had a valid terminal certificate. This preserves the
intended separation between workflow state and mathematical acceptance.

Across both bounded calibration batches there were 32 rollouts, 13 accepted
and 19 rejected. Exactly three tasks met the frozen 20--80% rule at 1/2 each:
`graph-artifact-composition`, `apollonius-gap-repair`, and
`rp2-homology-lattice`. No third calibration batch is needed.

## Frozen mixed-difficulty study

[`trajectory-value-mixed-study-v1.schema.json`](schemas/trajectory-value-mixed-study-v1.schema.json)
and
the [original frozen contract](../../../benchmarks/studies/trajectory-value-hypothesis-codex-v1/frozen-contracts/trajectory-value-mixed-study-v1.json)
freeze the main study before any main labels are collected. Those bytes were
the live PR5 contract at freeze time; the PR7 manifest retains them after later
calibration hardening refreshed the separately maintained
[`live contract`](../../../benchmarks/config/trajectory-value-mixed-study-v1.json).
The frozen contract binds both calibration manifests and summaries by digest,
recomputes the eligible population in source order and candidate order, and
rejects task substitution, evidence drift, or estimator reordering. Every
selected task also binds the exact Harbor public and verifier contract through
its calibration manifest; the loader recomputes the pinned Harbor, public-file,
and verifier-file digests and refuses execution after task drift.

The main matrix contains the three selected task groups with eight independent
rollouts each, for 24 planned rollouts. It retains the exact calibration model,
medium reasoning effort, prompt, 420-second timeout, isolated workspace and
state, no web search, no wrong-answer retries, and clean-room terminal reward.
The six estimators are frozen in this order: group/rollout, Numca-like numeric,
reasoning text, exact typed state, abstract value-state, and abstract
value-state plus text. The next section defines the two abstract
representations without changing this population. PR7 must preregister its
remaining analysis details before execution.

The H3 warning rule is already fixed as any strictly negative change between
selected preterminal value estimates. No threshold may be tuned on the main
labels. The scorer remains observation-only, and the contract records that
exact intermediate resume is unavailable; compatible independent rollouts are
the declared non-causal surrogate.

## Version 2 semantic value-state comparison

[`trajectory-value-state-abstraction-v1.schema.json`](schemas/trajectory-value-state-abstraction-v1.schema.json)
and
[`trajectory-value-evaluation-v2.schema.json`](schemas/trajectory-value-evaluation-v2.schema.json)
define the separate clustering representation and six-estimator output in
`jacobian.eval.trajectory_value_abstraction`. The version 1 hard state remains
unchanged and authoritative for replay and integrity. Each version 2 estimate
embeds both the complete abstract state and its digest while retaining the
exact typed-state digest. The result embeds and digest-binds its complete
version 1 source corpus. Deserialization independently rebinds evaluator
configuration, every observation and feature digest, cluster partition,
leave-one-out support set, terminal-success count, and aggregate metric;
stale or substituted comparison material is rejected.

The abstract signature keeps semantic state but removes exact identity:

- task family and observation boundary;
- counted object types, without content digests;
- counted artifact roles, without artifact URIs;
- candidate and checker state;
- counted open and discharged obligation classes, derived from the owning
  capability's domain name rather than obligation URI;
- whether scope is absent, declared, or rejected as an escalation, plus its
  relation to the previous exact snapshot;
- execution, completeness, completeness assurance, mathematical assurance,
  and binding validity;
- meaningful transition kinds; and
- external reasoning-protocol state.

It deliberately excludes candidate, object, scope, and artifact identities,
sets `exact_identity_fields_included = false`, and has no assurance authority.
Consequently it may cluster independently materialized states that have the
same observable mathematical role. It never replaces the exact state, changes
a verifier result, or licenses mathematical equivalence between the underlying
objects.

The version 2 comparison fixes all six estimators in the PR5 order:

1. `GROUP_ROLLOUT` uses only other rollouts from the task group.
2. `NUMCA_NUMERICAL` uses cumulative numerical spans from external summaries.
3. `REASONING_TEXT` uses the fixed version 1 deterministic TF-IDF clustering.
4. `JACOBIAN_TYPED_EXACT` uses the complete version 1 typed signature,
   including exact identities.
5. `ABSTRACT_VALUE_STATE` uses exact equality of the identity-free semantic
   signature.
6. `ABSTRACT_VALUE_STATE_TEXT` first requires the same abstract signature and
   then applies the fixed text clustering within that partition.

Clustering remains unsupervised and task-group local. Values remain
leave-one-trajectory-out terminal success frequencies; each supporting
trajectory votes once, and each trajectory has equal metric weight. Singleton
clusters fall back only to the other rollouts in the same task group. Every
estimate reports support IDs, cluster members, source, Brier/absolute error
inputs, and a 95% Wilson interval. No estimator is learned, no target label is
used for clustering or its own estimate, and none can alter prompts, tool
routing, retries, runtime, reward, or assurance.

### PR6 controlled semantic experiment

The immutable
[`comparison summary`](../../../tests/unit/tooling/fixtures/trajectory_value/pr6_semantic/comparison-summary.json)
binds an eight-trajectory, 24-observation mechanism check. Every produced
object, candidate, artifact, obligation, and scope has a distinct exact
identity. In the checker group, pairs nevertheless share the same semantic
state and terminal outcome. In the reasoning group, all four trajectories
share the same semantic state, while two exact-two-sided reasoning summaries
end accepted and two one-sided-shortcut summaries end rejected.

| Estimator | Clusters | Fallbacks | Mean support | Brier | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Group/rollout | 2 | 0 | 3.000 | 0.444444444445 | 0.666666666667 |
| Numca-like numerical | 2 | 0 | 3.000 | 0.444444444445 | 0.666666666667 |
| Reasoning text | 3 | 0 | 2.000 | 0.222222222222 | 0.333333333334 |
| Exact typed state | 18 | 16 | 3.000 | 0.444444444445 | 0.666666666667 |
| Abstract value-state | 8 | 0 | 2.333 | 0.296296296297 | 0.444444444445 |
| Abstract state plus text | 11 | 0 | 1.333 | 0.074074074074 | 0.111111111111 |

The controlled result demonstrates both intended mechanisms. Semantic
abstraction recovers cross-rollout support lost solely to exact identity; text
then separates useful and unsafe reasoning branches that have the same
abstract mathematical state. The wider Wilson intervals for the more specific
estimators expose their smaller support rather than hiding it. This fixture is
not real-model predictive evidence and does not answer H1, H2, or H3.

## PR7 label-free H1--H3 preregistration

[`trajectory-value-hypothesis-study-v1.schema.json`](schemas/trajectory-value-hypothesis-study-v1.schema.json)
and
[`trajectory-value-hypothesis-study-v1.json`](../../../benchmarks/config/trajectory-value-hypothesis-study-v1.json)
close the remaining analysis choices before any main-study label exists. The
contract digest-binds the then-live PR5 mixed-study bytes and revalidates its
two calibration sources, three selected Harbor tasks, current public and
verifier files, eight repetitions per task, model, prompt, timeout, and
no-retry policy. Historical replay now resolves those exact bytes through the
manifest-bound frozen snapshot. The 24-rollout population cannot be substituted
by this analysis layer.

H1 reports the complete version 2 metrics for all six estimators: Brier, MAE,
cluster observation and trajectory sizes, support range and mean, task-group
fallback count, and mean per-estimate Wilson 95% interval width. An exact,
abstract, or abstract-plus-text estimator receives directional support only if
both its Brier and MAE are strictly lower than both the group and Numca-like
baselines. Each trajectory retains equal metric weight.

H2 considers pairs from different trajectories in the same task group only
when they have exactly the same identity-free abstract-state digest, different
external reasoning-text digests, and opposite terminal rewards. It records
whether text-only and abstract-plus-text clustering separate each pair. H2
receives directional support only if at least one such pair is hybrid-separated
and the hybrid has strictly lower overall Brier and MAE than abstract state
alone.

H3 applies the already frozen rule without an epsilon or fitted threshold: the
first strictly negative change between consecutive selected preterminal values
warns that a trajectory will fail. For every estimator it reports the full
confusion matrix, precision, recall, failure prevalence, precision lift, false
alarms among accepted trajectories, and lead time measured as the number of
later selected preterminal observations. Directional support requires positive
recall, precision above prevalence, and positive mean true-positive lead time.
This stricter final condition distinguishes an earlier actionable signal from
a drop at the last selected state.

The task-owned verifier's aggregate reward remains the only terminal reward.
The study separately digest-binds the exact raw `submission.json` consumed by
the verifier and the model-visible public input bundle. Thus a completed
`REJECTED` result caused by an invalid evidence digest is still an exactly
bound negative label; the submission's failed evidence dimension is not
confused with failure to bind the terminal label itself. Input drift, missing
or changing submissions, model timeout or error, and verifier failure remain
`INCONCLUSIVE` and are excluded.

The runner requires local ChatGPT authentication, the exact catalogued
`gpt-5.4-mini` model at medium effort, Codex CLI 0.147.0, a clean source tree,
an unused output directory, and explicit operator opt-in. It forwards no API
key and preserves raw Codex JSONL, exported reasoning logs, task-visible
workspaces, verifier records and logs, exact extractions, catalog and policy
digests, both value representations, the six-estimator comparison, analysis,
and a digest-complete manifest. The scorer remains retrospective and
observation-only. No learned component, training, intervention, or causal
claim is authorized.

After committing this preregistration boundary, execute exactly once with:

```sh
uv run python -m benchmarks.tooling.trajectory_value_hypothesis_study run \
  --spec benchmarks/config/trajectory-value-hypothesis-study-v1.json \
  --output benchmarks/studies/trajectory-value-hypothesis-codex-v1 \
  --execute
```

## PR7 real H1--H3 pilot

The immutable
[`study manifest`](../../../benchmarks/studies/trajectory-value-hypothesis-codex-v1/manifest.json)
records all 24 locally authenticated Codex 0.147.0 rollouts with
`gpt-5.4-mini` at medium reasoning effort, no API-key forwarding, no
wrong-answer retry, and the frozen 420-second budget. The task-owned clean-room
verifiers observed 12 accepted and 12 rejected original submissions, with no
false certification. Retrospective analysis admitted 19 trajectories: 9
accepted and 10 rejected. Three runs lacked a successful `PLAN` boundary and
two runs were conservatively excluded after a trajectory-extractor failure.
Those two model calls were not repeated; their raw Codex JSONL, reasoning logs,
surfaces, and original verifier results remain preserved, while the manifest
records the lost ephemeral submission workspaces and `INCONCLUSIVE` analysis
disposition.

Later calibration hardening intentionally refreshed the live PR5 mixed-study
file and reduced its currently eligible population. That change cannot
retroactively substitute the 24-rollout contract used here. The exact original
bytes are therefore retained as a manifest-bound
[`frozen mixed-study contract`](../../../benchmarks/studies/trajectory-value-hypothesis-codex-v1/frozen-contracts/trajectory-value-mixed-study-v1.json).
Historical loading requires that path explicitly, rebinds the preregistration,
mixed contract, and all three task-contract records by digest, and refuses to
authorize current-task verification or new model execution. The strict live
loader continues to reject the drifted path.

After PR6 added mandatory terminal-to-transcript binding and corrected
estimator validation, the derived corpus, comparison, and analysis were replayed
once from the same raw trajectories. No model call or verifier outcome changed.
The manifest preserves the previous derived-artifact digests, records the
reanalysis source revision and evaluator digests, and binds every migrated
terminal record to its exact `codex.jsonl` digest.

The historical-only loader also derives and checks the later mandatory
soft-state digest from each already manifest-bound state payload. A supplied
but substituted digest fails closed; the adapter still cannot authorize a new
execution or mutate the frozen corpus.

| Estimator | Clusters | Fallbacks | Mean support | Brier | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Group/rollout | 30 | 0 | 5.413 | 0.238245614035 | 0.400000000000 |
| Numca-like numerical | 32 | 8 | 4.130 | 0.272358674464 | 0.429239766082 |
| Reasoning text | 40 | 13 | 2.196 | 0.345107212476 | 0.422222222222 |
| Exact typed state | 22 | 15 | 4.565 | 0.266686159844 | 0.425730994152 |
| Abstract value-state | 18 | 11 | 4.565 | 0.247010233918 | 0.411842105263 |
| Abstract state plus text | 34 | 25 | 3.543 | 0.331160540240 | 0.457038429407 |

H1 is not directionally supported under its preregistered conjunctive rule.
Exact typed state and abstract value-state each beat the Numca-like baseline on
both Brier and MAE, and abstraction improves on exact identity, but neither
beats the simpler task-group baseline. The hybrid is worse than both baselines.
Typed state therefore captures useful mathematical-stage regularity and
abstraction recovers some identity-fragmented support, but the pilot does not
show incremental predictive validity beyond knowing the task group.

H2 is also not directionally supported. There are 30 cross-trajectory,
same-task pairs with the same abstract state, different reasoning digests, and
opposite rewards. Reasoning text separates 29 and the hybrid separates 22, so
text visibly distinguishes many policy branches. That extra specificity does
not improve prediction: hybrid minus abstract is +0.084150306322 Brier and
+0.045196324144 MAE. The hybrid partition produces 34 clusters and 25 fallbacks,
showing that branch separation can destroy more support than it contributes.

H3 satisfies its preregistered directional rule for the reasoning-text
estimator only, but the scientific result is insufficient. It issues one
warning for rejected RP2 r08, one selected observation before the terminal
state, for precision 1.0, recall 0.1, and no accepted-run false alarm.
Exact and abstract state still issue three warnings, all on accepted graph
trajectories, for precision 0, recall 0, and a 33.3% false-alarm rate among
accepted trajectories. Group, Numca-like, and hybrid estimators issue no
warnings. The H3 result therefore rests on one true positive and does not show
broad failure coverage.

Representative trajectories explain the result. Graph r01 completes the
external reasoning protocol and is accepted, while compatible graph r05 stops
after its tool result and is rejected; text distinguishes this useful branch.
Apollonius r06 is nevertheless accepted with an incomplete external reasoning
protocol, so protocol completion is not a necessary proxy for mathematical
success. RP2 r06 and r02 both compute determinant -2, finalize the same
`Z/2Z` conclusion, and have nearly identical typed mathematical progress, but
r02 fails the task-owned evidence-content contract while r06 passes. The
reasoning summary does not retain the terminal evidence artifact's exact
contract-relevant wording, so neither typed state nor text predicts that
difference before verification.

The engineering closeout therefore records H1 and H2 as unsupported and H3 as
insufficient. The single H3 event is too sparse to justify value-guided runtime
intervention: it has 10% recall, comes from a text-only estimator, and has no
held-out task-family replication. No further estimator, clustering, threshold,
task, or label tuning is authorized on this public corpus. Any future research
would require a separately justified, preregistered, held-out population; it is
not part of this stack. This is a small public predictive-validity pilot, not a
statistically powered or causal claim.

## Engineering closeout and replay boundary

Terminal construction and exclusion state remain inspectable without changing
the frozen reports. Each run retains the original task-owned `verifier.json`,
the analysis terminal in `run.json`, and, for runner failures, an
`infrastructure-failure.json` that records the operational reason, preserved
and missing artifacts, original verifier outcome, and the fact that no model
rerun occurred. `analysis.json` lists every excluded trajectory and its exact
eligibility reason. A completed verifier result is usable as a label only when
its input, submission, verifier, and transcript bindings are exact; timeout,
error, extraction failure, missing extraction, or missing `PLAN` state remains
an exclusion rather than a negative label.

The exact version 1 state is the provenance and replay-integrity record. The
abstract version 1 value-state is derived from and digest-bound to that exact
state and is used for clustering only; it carries no assurance authority and
does not assert semantic equivalence. The committed artifact replay is covered
by the focused hypothesis-study tests, which validate all manifest digests,
historical source and soft-state bindings, deterministic gzip bytes, frozen
labels, and H1--H3 outputs without executing Codex or a task verifier:

```sh
uv run --locked pytest -q \
  tests/unit/tooling/test_trajectory_value_hypothesis_study.py \
  tests/unit/tooling/test_trajectory_value_abstraction.py \
  tests/unit/tooling/test_trajectory_value.py \
  tests/unit/tooling/test_trajectory_state.py
```

## Current limitations

Version 1 deliberately uses conservative generic output interpretation. A
non-empty typed capability output becomes one content-addressed object, while
candidate-like fields receive a separate candidate identity. Domain-specific
semantic equivalence is not inferred. Evidence binding becomes valid only from
verified checker evidence or clean-room terminal evidence; ordinary
reasoning-call protocol binding is not mathematical progress.

The real PR7 pilot has only one task per family, eight unseeded repetitions per
task, one model and reasoning level, and 19 labelled trajectories after five
fail-closed exclusions. It is too small to establish generalization or precise
uncertainty across task families. TF-IDF vocabulary and cluster geometry are
fitted to the complete feature corpus, though labels remain strictly
leave-one-trajectory-out. The fixed threshold is a declared first version, not
a tuned optimum. Exact content-addressed compatibility fails safe against
merging distinct candidates, but fragments semantically equivalent objects or
independently produced verification records; the real comparison exposes that
support-loss tradeoff.

Two interrupted runs exposed that the initial extractor rejected a
noncanonical candidate value after terminal verification and before publishing
the ephemeral workspace. The recovery path preserves their original verifier
outcomes as observation-only evidence but excludes both labels and performs no
rerun. The hardened runner now publishes the workspace first and confines a
future extraction failure to one `INCONCLUSIVE` record. Three additional runs
were excluded for lacking a successful `PLAN` boundary, so the reported
metrics apply only to the frozen eligible observation protocol, not every
terminal submission.

The abstract state trades that fragmentation for possible semantic aliasing.
It counts domain-level object, artifact, and obligation classes but does not
prove that two exact candidates or scopes are mathematically equivalent. Its
scope relation compares exact digests only within one trajectory, and its
capability-domain normalization is a fixed syntactic abstraction rather than a
domain theorem. These choices are suitable only for clustering and remain
visible beside the exact identities. The PR6 controlled fixture validates the
mechanism, not its predictive validity on real trajectories.

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
