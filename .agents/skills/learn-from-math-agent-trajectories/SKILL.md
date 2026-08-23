---
name: learn-from-math-agent-trajectories
description: Review completed or paused mathematical agent transcripts, visible reasoning, code, searches, tool calls, corrections, and final claims to extract evidence-backed lessons for Jacobian operations, discovery, contracts, skills, evaluations, and documentation. Use for mathematical workflow retrospectives and "what should we learn from this trajectory?" requests. Do not use to continue solving the problem or infer hidden chain-of-thought.
---

# Learn from Math Agent Trajectories

Turn an observed mathematical investigation into reusable Jacobian learning.
Evaluate both the mathematical evidence and the workflow that produced it. A
correct final answer can still expose a weak process, and a failed proof search
can still reveal a valuable operation or evaluation gap.

Use only observable evidence: the transcript, visible reasoning summaries,
tool invocations and results, code, generated artifacts, cited sources, user
corrections, and final answer. Do not claim access to hidden chain-of-thought.
Progress narration such as “searching” or “checking a theorem” is not evidence
unless the corresponding source, call, or result is available.

## Establish the record

Identify the mathematical task, intended outcome, actual outcome, stopping
condition, and whether the trajectory is complete, paused, or truncated. Bind
claims to the available repository revision, operation catalog, environment,
source date, and transcript coverage. A capability present on current main was
not necessarily visible or callable in the audited session.

Reconstruct only decisions that affected correctness, cost, progress, or
confidence. Preserve later corrections instead of silently replacing earlier
claims; record what was first claimed, what new evidence changed it, and what
remains unresolved.

## Build a mathematical claim ledger

For every decisive claim, record its scope separately from its evidence kind.
Useful evidence kinds include:

- a source-backed theorem with its exact hypotheses and publication status;
- an exact symbolic identity or derivation, including domains and excluded
  branches;
- an exact bounded computation or exhaustive finite search, including the
  complete input range;
- a numerical or heuristic candidate with no certification;
- a checked witness, refutation, or certificate for one bounded proposition;
  and
- a general or asymptotic proof.

Never promote a claim across evidence kinds or scopes without new evidence. An
exact identity for one family is not a theorem for arbitrary configurations;
an exhaustive check through one bound says nothing beyond that bound; a
numerical residual is not an exact solution; and a bound for one overlap type
does not control uncounted conflict types.

Treat assumptions as part of the claim. Check nonzero denominators,
nondegeneracy, ordering, distinctness, general-position conditions, excluded
algebraic branches, and source hypotheses. When a later exact replay rejects a
rounded numerical point, distinguish “this reported point is not a witness”
from “no nearby exact witness exists.”

When a symbolic derivation clears denominators or assumes polynomial factors
are nonzero, record the equation ideal and the excluded locus separately. A
Gröbner basis of the cleared numerators describes a closure that may retain
components supported entirely on forbidden factors; do not treat it as the
original rational system without saturation or equivalent exclusion evidence.
Likewise, a rank computed over a rational-function field with algebraically
independent parameters does not establish the rank after imposing polynomial
relations among those parameters.

For numerical searches, require enough retained state to reproduce and certify
the candidate: equations, variable order and normalization, side conditions,
solver and status, random seeds, full-precision values, residual definition,
tolerances, and search domain. Missing this record is a handoff defect even
when the numerical experiment was useful.

Interpret solver outputs by their actual contract. `UNKNOWN`, timeout,
incompleteness, cancellation, or failure to find a witness is not a
mathematical conclusion. `SAT`, `UNSAT`, feasibility, and optimality are useful
only when the encoding, domain, objective, and certificate support the claim;
for example, optimality with a constant objective may establish only
feasibility.

## Audit the work, not just the answer

Inspect scratch code as mathematical evidence. Look for vacuous tested ranges,
incorrect quantifiers, incomplete enumeration, floating-point equality,
rounded witnesses, ignored tolerance parameters, unseeded randomness, broad
exception handling, sentinel values that masquerade as mathematics, and
relaxations whose combinatorial feasibility does not imply geometric or
algebraic realizability. Preserve useful code and exact outputs when they are
needed to reproduce a finding.

At every bespoke-code escape, state the desired mathematical postcondition.
Audit the mathematical move even when the agent never mentioned or used
Jacobian. Tool non-use is not itself the finding; handwritten code is evidence
of a demanded operation, which must then be classified as existing, missing,
or unsuitable for public admission. Then inspect the session-visible Jacobian
surface when possible:

1. Was a matching operation available?
2. Could natural `math.find` language discover it?
3. Did the agent inspect the schema, bounds, examples, and result semantics?
4. Was it selected and called with a valid payload?
5. Was its typed result used within its stated scope?

Select operations against the downstream evidence the trajectory needed, not
only the operation title or primary scalar value. When neighboring operations
differ by transformations, bases, representatives, reconstruction data, or
certificates, determine whether later work needed that stronger result. If a
certificate-bearing operation was available, rebuilding its missing evidence
from a lossy sibling is a selection failure rather than an operation gap.

Do not infer use from a generic “called tool” marker. Compare repeated scalar
calls, manual all-pairs or all-subsets loops, and custom symbolic or solver code
against available aggregate operations. An N+1 trace can suggest a missing
profile operation, but verify the catalog before proposing one.

At a SAT, SMT, optimization, or other solver escape, separate the solver engine
from the mathematical result being sought. A generic bounded solver operation
may be the correct tool for one investigation, while a repository gap may
instead be a typed domain result such as a solution family, embedding, or
factorization with the solver kept private. Do not propose public blocking
clauses, search sessions, or model-enumeration mechanics merely because the
scratch implementation used them; also do not reject a solver-backed operation
when its public postcondition is stable, bounded, and mathematical.

Audit literature work by source quality and theorem fit, not by the number of
searches or websites. Verify precise hypotheses and dates against primary
sources. Mark discussion-thread claims, preprints, peer-reviewed results, and
agent conjectures distinctly.

## Attribute the lesson

Classify each important event as one of:

- an existing capability that worked;
- an availability or environment limitation;
- a discovery or selection failure;
- execution or result-use friction in an existing contract;
- a missing bounded operation;
- a representation, interoperability, scale, or backend gap;
- a validation or handoff failure; or
- mathematical reasoning that a tool could support but not replace.

Use `evaluate-mcp-tool-adoption` for a controlled availability, discovery, or
selection question. Use `audit-mcp-tool-friction` when the agent selected an
operation but struggled to call it or consume its result. Use
`recent-conjecture-evaluations` for a new source-grounded held-out probe, and
use the evaluation skills only after there is a frozen task and an independent
oracle.

Before calling something an operation gap, verify current source and catalog
membership. A candidate operation must expose one reusable bounded
postcondition with typed inputs and an exact, incomplete, or unknown result. It
must not encode the motivating conjecture, prescribe a proof strategy, or
claim an asymptotic theorem that still requires model reasoning.

When an existing operation has the required postcondition but rejects a large
input through a coarse cap, attribute the lesson to admission or scale/backend
rather than inventing a second operation. Check whether the result and derived
work are compact, whether an exact sparse, factored, modular, symbolic, or
implicit representation avoids expansion, and whether a maintained backend or
different algorithm regime would widen the envelope. A timeout alone neither
proves that the cap is necessary nor establishes a mathematical negative.

Separate gap diagnosis from public-operation admission. First decide whether
the trajectory establishes a reusable missing mathematical postcondition on
the inspected surface. Recording that gap does not assert that the result
belongs in the agent-visible catalog. Then report the admission evidence
separately: it may support public consideration, suggest a native-only helper,
or leave the disposition unresolved. Public-admission concerns such as weak
leverage over ordinary Python or a cheap projection can change the eventual
disposition without erasing an otherwise well-evidenced gap. Conversely, an
absent convenience is not an operation gap unless the missing postcondition is
independently canonical or reusable.

## Turn observations into repository learning

Choose the smallest durable action supported by the evidence:

- update a repo-local skill for a reusable agent decision rule;
- improve discovery metadata when an existing operation was hard to find;
- repair an operation contract when a selected tool was hard to call or use;
- record an operation gap for a repeated, stable bounded postcondition, and
  propose public admission only when the separate admission evidence supports
  it;
- update product documentation only for a public contract or durable product
  behavior that users need outside agent instructions;
- preserve a trajectory as an evaluation when it has a frozen input,
  independent oracle, discriminating failure, and contamination controls; or
- take no repository action when the evidence shows an isolated agent slip.

An unresolved conjecture usually cannot be the evaluation oracle. Extract
bounded checkpoints with known answers—identity verification, exact finite
instances, certificate checking, assumption validation, or correct
non-conclusion handling—instead of scoring whether the agent solved the open
problem.

For every proposed action, state the triggering event, evidence, expected
input and output, insertion point, counterfactual benefit, owner, success
criterion, and what the change still cannot prove. Search narrowly for an
existing issue before proposing a new one. Do not file, comment, or otherwise
mutate external systems without user authorization.

## Report the retrospective

Lead with what the trajectory teaches, not a chronological replay. Include:

- the audit scope, evidence coverage, and important limitations;
- a compact claim-and-correction ledger;
- mathematical strengths and failures, separated from final-answer quality;
- Jacobian capabilities that helped or would have helped;
- findings with attribution, counterfactuals, and confidence;
- focused skill, operation, discovery, contract, documentation, or evaluation
  actions; and
- unresolved mathematical work that still requires proof or human judgment.

For each operation-gap finding, state the gap verdict and the admission posture
separately. Do not make a reader infer one from the other.

Keep confirmed facts, plausible hypotheses, and open questions visibly
separate. The retrospective succeeds when another agent or maintainer can
verify the lesson from the original artifacts and decide on a bounded next
action without inheriting the original agent’s overclaims.
