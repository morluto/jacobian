# Mathematical evidence in trajectories

## Build a mathematical claim ledger

For every decisive claim, record its scope separately from its evidence kind.
Useful evidence kinds include:

- a source-backed theorem with its exact hypotheses and publication status;
- a discussion-thread claim, unpublished manuscript, preprint, or
  peer-reviewed result, each recorded with its distinct status;
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

When a trajectory embeds useful mathematics inside a specialized proof,
record the method boundary before classifying it:

```text
mathematical goal:
local exact move:
global proof lift:
source representation:
stable reusable postcondition:
literature identity:
Jacobian disposition:
```

This record prevents opposite errors: promoting the whole proof workflow into
one operation, or dismissing a stable operation because its implementation
appears inside a theorem-specific script. For example, one nullspace update is
usually too low-level and an entire allocation theorem is too high-level;
exact rounding that preserves named rows and bounds the monitored-column error
can be the reusable boundary. Base every field on observable artifacts and
primary sources rather than inferred hidden reasoning.
