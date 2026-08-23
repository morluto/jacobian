---
name: decompose-mathematical-solution-corpora
description: Decompose a bounded corpus of mathematical proofs, formalizations, scripts, and certificates into recurring solution techniques and the smallest reusable Jacobian postconditions. Use for repository- or corpus-level “what can Jacobian learn?” audits; do not use for one operation contract or one agent trajectory.
---

# Decompose Mathematical Solution Corpora

Extract reusable mathematical moves from a bounded solution corpus without
turning the result into a file-by-file summary or a catalog of theorem wrappers.
The unit of analysis is a method family: a recurring local mathematical move,
its exact carrier, and the global reasoning that lifts it into a proof.

Use `audit-mathematical-vocabulary` for one bounded mathematical slice,
`learn-from-math-agent-trajectories` for one completed or paused investigation,
and `audit-public-operation-contracts` when a specific operation needs a deep
contract review. This skill owns the repository- or corpus-level decomposition
that precedes those focused audits.

## Freeze the scope

Record the immutable revision of every source repository and the Jacobian
revision and catalog used for comparison. Define the included directories,
campaigns, or certificate families and any exclusions. Do not claim corpus
closure from a partial clone, truncated artifact, generated summary, or
unreadable dependency.

Inventory artifact types before reading deeply: papers and notes, Lean or other
formal developments, exact Python or native programs, numerical experiments,
solver encodings and proofs, prompts and trajectories, certificate bundles,
test fixtures, and replay or publication metadata. Use the inventory to find
method families, not to produce a chronological summary of every file.

## Group by solution technique

Cluster sources by mathematical move rather than conjecture name or language.
Examples include exact finite enumeration, dynamic programming, local-lemma
witnesses, linear or semidefinite duality, algebraic elimination, interval
enclosure, canonicalization, dependent rounding, coding-theoretic profiles,
and finite-state transfer arguments.

Select representative sources from each family. Prefer sources that expose the
local move, exact hypotheses, boundary behavior, and an independently replayable
fixture. Continue sampling within a family until another representative no
longer reveals a new carrier, postcondition, representation regime, or failure
mode.

## Decompose each representative

Write a compact method record:

```text
target theorem:
exact finite carrier:
local mathematical move:
theorem-specific lift:
source representation:
established technique:
stable reusable postcondition:
discovery vocabulary:
technique disposition:
evidence and fixture:
```

The local move is not automatically an operation. Reject boundaries that
merely expose one loop iteration, solver control, callback, proof bookkeeping,
or temporary data structure. Also reject the opposite boundary when it bundles
the motivating theorem, search strategy, interpretation, and stopping rule.
Look for one postcondition that remains meaningful if the surrounding paper or
conjecture disappears.

Classify an established technique as a public-operation candidate, native-only
function, private kernel, invariant or fixture, or caller reasoning. Technique
names may be discovery vocabulary for a public operation without becoming
separate operation IDs. Require an independently consumable postcondition
before making an intermediate technique separately runnable.

Treat representation as mathematical execution evidence. State whether the
carrier is materialized, succinct, generated, or oracle-backed; what expansion
the implementation performs; whether that expansion is predictable before
execution; and whether a compact representation changes the complexity class
or output obligation.

## Research the method

Trace the local move to primary literature or an authoritative formal/library
source. Verify the exact hypotheses, conclusion, conventions, algorithmic
regime, and representation-sensitive complexity. Distinguish neighboring
methods that share vocabulary but prove different guarantees. Use secondary
surveys only to discover sources or terminology, then verify the conclusion
against the primary source.

Research maintained exact backends and standard algorithms in proportion to
the candidate. The question is whether a bounded, typed Jacobian contract is
feasible—not whether the corpus's handwritten implementation should be copied.
Record uncertainty when the literature supports the theorem but not an
admissible exact kernel at the required scale.

## Compare with Jacobian

Inspect the current catalog, native API, canonical values, request and result
models, tests, admission decisions, and narrowly related issues. Attempt the
smallest exact composition before declaring a gap. A manual coordinate change,
cheap projection, or theorem-specific assembly normally remains caller work;
incompatible values, detached certificates, or hidden expansion may instead
identify an interoperability or contract problem.

Give every method family one disposition:

- existing operation or exact composition;
- representation or interoperability repair;
- discovery repair;
- request/result contract repair;
- scale or backend improvement;
- new bounded operation candidate;
- defining, convention, adversarial, metamorphic, producer-consumer, or stress
  fixture;
- reasoning, theorem-specific workflow, or certificate infrastructure; or
- no supported Jacobian action.

For operation candidates, state the semantic domain, stable postcondition,
source representation, controlling work and output quantities, reconstruction
or defining invariant, typed incomplete states, and at least one discriminating
fixture. Separate the existence of a reusable gap from public-catalog admission.

## Route actions without overclaiming

Verify issue ownership narrowly before proposing a new issue. Reinforce the
canonical owner when the operation, contract, or scale question is already in
scope. Keep distinct semantics separate even when they share a backend. Do not
file, comment, edit external systems, or make repository changes without user
authorization.

Prefer compact in-thread findings and focused repository actions. Do not create
large durable reports, copied source archives, or generated inventories unless
the user requests them. Preserve only the small fixtures and evidence needed to
replay a conclusion.

## Establish closure

Stop when every inventoried method family has a disposition, every proposed
operation has been compared with exact current composition and issue ownership,
and additional representative sources yield no new local move, representation
regime, postcondition, or fixture role. Report the frozen revisions, coverage,
important exclusions, and unresolved uncertainties. “No gaps remain” means no
unclassified reusable move within that declared scope, not that the corpus or
mathematical literature contains nothing else.

Lead the final result with a compact technique-to-disposition matrix, followed
by the few highest-value operation, contract, scale, and fixture actions. Keep
the proof workflow separate from the atomic mathematical move throughout.
