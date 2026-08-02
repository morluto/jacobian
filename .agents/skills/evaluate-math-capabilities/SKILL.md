---
name: evaluate-math-capabilities
description: Consume evidence-backed capability candidates or implemented capability deltas and design, implement, run, and interpret Jacobian-specific model-in-the-loop evaluations that measure whether a mathematical operation, checker, or portfolio change improves autonomous agent outcomes. Use when asked to benchmark a Jacobian capability, compare baseline and treatment portfolios, create held-out mathematical cases or independent oracles, detect workflow leakage, measure correctness or false certification, evaluate retrieval or tool composition, or decide whether to keep, expand, split, consolidate, rank, stabilize, or retire a capability. Use discover-math-capabilities first when the question is what to build, implement-math-capability for producer construction, and implement-math-capability-checker for verification construction.
---

# Evaluate Math Capabilities

Determine whether a concrete Jacobian capability or portfolio change helps
agents on new mathematical work. Keep the strategy agent-owned and score
outcomes and evidence, not a preferred sequence of calls.

Use the [shared handoff format](../../../docs/reference/capability-development-handoffs.md)
for inputs, returns, and portfolio decisions.

## Classify the evidence before running

Give each case and result one primary evidence role:

- **task/verifier validation:** checks the benchmark harness, task contract,
  independent Oracle, or deliberate failure cases; it is harness evidence,
  not model capability evidence;
- **regression/public reproduction:** known or public case; useful for contract
  and breakage detection, never held-out causal evidence;
- **assurance/conformance:** tests calibration, scope, schemas, artifacts,
  discovery, or parameterization, and may not measure mathematical capability
  value;
- **Jacobian workflow observation:** records agent discovery, tool use,
  artifacts, and assurance behavior on a configured Harbor workflow; it is
  workflow evidence, not comparative performance evidence;
- **held-out causal comparison:** an unseen or transformed case with a frozen
  control/treatment intervention and an independently judgeable outcome.

Do not promote a public regression, a single observation, or a reward change
caused only by assurance calibration into a claim that Jacobian improved
mathematical problem solving. A causal comparison is not ready until a control
pilot is below the success ceiling, treatment has a plausible capability-to-
outcome hypothesis, and the plan specifies multiple repetitions and uncertainty
reporting.

## Confirm evaluation readiness

Start from a concrete intervention:

- a capability or portfolio delta;
- a discovery candidate record with supporting move episodes;
- an implementation handoff identifying exact contracts, artifacts, provider,
  failure semantics, validation, and capability-availability delta;
- a checker handoff and obligation ledger when verification is the treatment;
- one expected counterfactual benefit;
- a runnable baseline and treatment;
- public reproduction evidence or a stable contract;
- an independently judgeable mathematical outcome; and
- at least one plausible wrong or falsely certified path.

If the operation, contract, backend, or intended mathematical outcome is still
unclear, return the question to `discover-math-capabilities`. Evaluation should
not invent the product hypothesis it is meant to test.

If the intervention is not runnable or its artifacts and failures do not match
the accepted candidate, return it to `implement-math-capability`. If a
verification treatment lacks an independent checker, exact evidence bindings,
or fail-closed outcomes, return it to
`implement-math-capability-checker`. Evaluation must not repair product or
trust-boundary defects inside the benchmark.

## Audit the discovery handoff

Confirm that the handoff identifies:

- exact source rows, traces, proofs, or counterexamples that motivated the move;
- the before-state, operation, output artifact, downstream use, and verification
  boundary for each supporting episode;
- current portfolio overlap and why composition does not already close the gap;
- public reproduction cases separated from held-out scored cases;
- candidate backend, contract, failure semantics, and checker boundary;
- contamination risks and plausible falsely persuasive paths; and
- the predicted autonomous behavior that should change under treatment.

For an implemented intervention, also confirm that the runnable catalog
snapshot matches the candidate; public reproductions pass; invalid,
inapplicable, timeout, and incomplete states remain non-conclusions; and no
producer or checker artifact leaks into the held-out visible bundle.

Use missing handoff evidence as a reason to return the candidate to discovery,
not as an invitation to fill the gap with evaluation fixtures.

## Inspect the repository harness

Read `AGENTS.md` and inspect:

- `docs/reference/capability-workflow-evaluations.md`
- `benchmarks/datasets/agent-workflow-v1/`
- `src/jacobian/eval_telemetry.py`
- the Harbor task and verifier validation checks
- the relevant capability descriptors, checkers, and contracts.

Use the repository-local `harbor-benchmarks` skill for Harbor packaging,
verifier validation, task-digest updates, Oracle runs, and Jacobian observation
jobs. Extend the committed Harbor task contracts when practical. Keep any
future control/treatment configuration outside task bundles; do not create a
custom task runner for one experiment.

Treat the Harbor boundary as a handoff: this skill owns the evaluation
question, intervention, oracle independence, scoring interpretation, and
evidence report; `harbor-benchmarks` owns the runnable task/job implementation.

## State the comparison

Freeze the evaluation question before running either condition. Define:

- **control**: current portfolio or a targeted ablation;
- **treatment**: the same portfolio plus the intervention;
- the same model, reasoning effort, prompt, budget, environment, and visible
  artifacts;
- randomized condition order and recorded fixture/order seeds;
- multiple repetitions when making comparative claims; and
- the exact capability-availability snapshot for each condition.

Prescribed-tool cases measure contract usability and conformance. Autonomous
portfolio cases let the agent choose operations and measure portfolio value.
Do not confuse the two. If the agent is told the successful sequence mined from
the source cases, report contract replay rather than autonomous portfolio value.

Change one intervention dimension at a time. Evaluate a capability delta,
descriptor change, example, ranking policy, or reusable strategy skill as
separate treatments when the question is which one caused the improvement.

## Build the held-out case

Give the agent the smallest natural task and source bundle available before the
answer. Keep the answer, scoring discriminator, and checker-only data outside
the evaluated workspace.

A hidden oracle means inaccessible to the agent during the run, not necessarily
secret forever. It may be an independently generated object, a checker,
withheld dataset fields, or a public answer used with an explicit contamination
warning.

For every case, freeze:

- visible input bundle and output schema;
- oracle version and provenance;
- accepted conclusions and alternative valid strategies;
- exact evidence bindings required for stronger assurance;
- one tempting incomplete, misbound, or semantically mismatched route;
- facts that must remain uncertain; and
- source lineage, cutoff date, and contamination risk.

Prefer generated or transformed variants for scored claims. Public solved
problems are valuable regressions but weak evidence of generalization.
Prefer held-out source families when practical. If a scored case descends from
the discovery corpus, record the lineage and transform it enough that success
requires using the mathematical operation rather than recalling the mined
answer or workflow.

## Keep the oracle independent

Use an implementation independent of the capability being evaluated. The
oracle must bind its judgment to the exact claim, semantics, candidate, scope,
and certificate where applicable. It must reject:

- provider output treated as self-verification;
- bounded failure to find a witness treated as proof;
- incomplete enumeration reported as exhaustive;
- an isomorphic or altered object substituted without provenance;
- a proof of a mismatched formal statement; and
- timeout, cancellation, or execution failure converted into a conclusion.

Place hidden oracle material where the evaluated agent cannot read it. Avoid
answer leakage through filenames, comments, caches, neighboring artifacts, or
transcript instructions.

## Score outcomes before efficiency

Make these primary:

- oracle correctness;
- false certification;
- execution and completion state;
- scope and completeness accuracy;
- evidence and verification-record bindings; and
- acceptance of mathematically valid alternative strategies.

Keep mathematical correctness, evidence validity, scope/completeness, false
certification, and assurance calibration as separate reported metrics. An
aggregate reward may summarize a workflow contract, but it must not be the
primary evidence for a capability-benefit claim when it combines these
dimensions.

Among semantically correct runs, measure:

- wall time;
- input and output tokens;
- total and successful tool calls;
- tool and parameter errors;
- repeated or irrelevant calls; and
- infrastructure work reported by the agent.

A faster wrong answer loses. A verifier-backed bounded result must not be
scored as a proof outside its checked scope.

## Validate the evaluation

Before trusting a comparison:

1. Confirm a capable agent has enough visible information to solve the task.
2. Run or reason through one known-good report.
3. Run or reason through at least one tempting bad report.
4. Confirm the bad path fails for the intended mathematical reason.
5. Confirm alternate correct strategies can pass.
6. Test the case loader, scorer, telemetry parsing, and checker replay.
7. Version the case whenever its source, oracle, contract, or environment
   changes.

Do not weaken a case to make the treatment pass. Record an unavailable backend,
tool failure, or insufficient prompt as its own failure class.

## Run and preserve evidence

Use the repository runner and keep generated results under its ignored results
directory. Record:

- the reproducibility fingerprint: git tree, visible case-bundle digest,
  catalog-availability digest, provider/runtime identity, model/settings,
  prompt, oracle, and scorer identities;
- order and fixture seeds;
- raw transcripts and structured reports;
- oracle and scorer outcomes; and
- validation actually run.

Never expose hidden oracle contents in the treatment prompt or model workspace.
Do not report provider sampling as deterministic when no generation seed is
available.

## Interpret the result

Separate:

- capability value;
- discovery or descriptor quality;
- parameterization and contract usability;
- workflow or answer leakage;
- backend reliability;
- verification and provenance behavior; and
- mathematical reasoning limits.

Choose a portfolio action supported by the transcripts:

- keep or expand;
- improve discovery, examples, defaults, or errors;
- split useful intermediate outcomes;
- consolidate redundant outcomes;
- add independent verification;
- rerun with more repetitions;
- defer; or
- retire.

Experimental availability need not wait for evaluation. Use results to guide
recommendations, defaults, consolidation, and retirement rather than turning
evaluation into an access gate.

Feed failures back to discovery with the failed candidate gate. A capability
that is unused because its outcome has no leverage is different from one that
is valuable but undiscoverable, difficult to parameterize, unavailable in the
environment, or missing an independent checker. Do not respond to a strategy,
descriptor, or evaluation-design failure by manufacturing another operation.

## Report

Return a `stage=evaluation,status=complete` handoff using the shared format.
Include the frozen question and intervention, case/oracle provenance, baseline
and treatment definitions, correctness and false-certification results,
transcript-derived failure classes, contamination and proof gaps, the justified
portfolio action, and exact reproduction commands.

State whether the result is a harness validation, public regression, pilot, or
comparative performance claim. Do not promote a single pair to a statistically
powered conclusion.
