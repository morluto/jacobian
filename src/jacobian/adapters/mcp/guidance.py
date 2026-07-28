"""Agent-facing guidance for Jacobian's MCP surface."""

from __future__ import annotations

SERVER_DESCRIPTION = (
    "Discover and execute installed mathematical operations with explicit scope, "
    "completeness, assurance, and optional independent verification."
)

SERVER_INSTRUCTIONS = (
    "When a task involves exact computation, structure discovery, transformation, "
    "bounded search, counterexample generation, formal-environment inspection, or "
    "independent checking, begin with capability.describe. You do not need to know a "
    "capability ID. Start discovery with limit 5, inspect only the strongest one or "
    "two relevant summaries, then request one CONTRACT view before invoking. Compose "
    "capability.invoke calls as you judge useful. "
    "Jacobian does not choose the task decomposition or research strategy. "
    "Direct workspace.* tools publish their own input shape and do not use capability "
    "discovery. EXPLORE returns proposed, heuristic, or computed evidence; use VERIFY "
    "only with an installed independent checker. Execution status, completeness, "
    "mathematical conclusion, and assurance are separate. Failure to find a witness "
    "and bounded or exhausted search are not mathematical conclusions. Synchronous "
    "SAT/SMT calls are capped at 150 seconds; partition larger searches. Only assurance "
    "level VERIFIED with a local verification record is verified. Repeating finite or "
    "bounded instances does not establish an all-orders or all-parameters theorem. A "
    "workspace entry never promotes mathematical assurance. Read "
    "jacobian://instructions for the complete operating guide."
)

CAPABILITY_DESCRIBE_DESCRIPTION = """\
Discover installed mathematical outcomes or inspect one exact capability contract.

Use this first when a task may benefit from exact computation, structural analysis,
transformation, bounded search, counterexample generation, formal-environment
inspection, or independent checking. You do not need to know a capability ID.

Discovery:
- Pass `query` to rank compact installed outcomes by mathematical intent.
- Optionally filter with `domain` and `mode`; `limit` is between 1 and 20 and
  defaults to 5.
- Omit all arguments to browse a compact installed catalog.
- When `next_cursor` is present, pass it back with the same filters and limit to
  continue without loading the complete catalog.
- Ranking is deterministic retrieval over published IDs, titles, descriptions, and
  tags. Match fields and terms are returned; results are candidates, not
  recommendations.
- Once a domain-owned match fits the required outcome, inspect only the strongest one
  or two candidates. Search again when none fits. Invoke a producer before looking
  for its checker, then follow the checker, certificate, and verification fields in
  the producer result instead of guessing a generic verifier.
- Compose capabilities as the investigation requires; Jacobian does not prescribe a
  mathematical workflow.

Exact inspection:
- Pass `capability_id` for the small default SUMMARY projection when judging fit.
- Request `view="CONTRACT"` once a capability fits and before invoking it. CONTRACT
  adds the validation-equivalent input schema, runtime identity, related operations,
  and validated invocation examples.
- `view="COMPACT"` remains a compatibility alias for the previous contract projection.
- Add `view="FULL"` only when complete output schema, provider configuration, licensing,
  or audit metadata is needed.
- Every exact response states the scope rule: repeated bounded invocations remain
  finite evidence and do not establish an all-orders or all-parameters conclusion.

Examples:
- `{"query":"find a counterexample to associativity","domain":"universal_algebra","mode":"EXPLORE","limit":5}`
- `{"query":"verify a polynomial identity","mode":"VERIFY","limit":5}`
- `{"capability_id":"polynomial.compute.gcd"}`
- `{"capability_id":"polynomial.compute.gcd","view":"CONTRACT"}`
- `{"capability_id":"polynomial.compute.gcd","view":"FULL"}`
"""

CAPABILITY_INVOKE_DESCRIPTION = """\
Execute one installed mathematical capability after inspecting its exact descriptor.

Call capability.describe first; do not guess payload fields or aliases. Use EXPLORE
for proposed, heuristic, or computed evidence and VERIFY only for an installed
checker-backed contract. After invocation, inspect execution, scope, completeness,
relationships, open obligations, assurance, diagnostics, and artifact URIs
independently. COMPLETED does not by itself establish a mathematical conclusion.
One invocation covers only its exact supplied input or claim. Repeating finite or
bounded instances does not widen that scope to an all-orders or all-parameters claim.
Synchronous SAT and SMT requests are capped at 150 seconds so structured TIMEOUT or
CANCELLED diagnostics can arrive before remote transport deadlines. Partition larger
searches; retrying a cancelled call starts a new computation.

Examples:
- `{"capability_id":"integer.compute.gcd","mode":"EXPLORE","payload":{"left":"84","right":"30"}}`
- `{"capability_id":"polynomial.identity.verify","mode":"VERIFY","payload":{"variables":["x"],"left":{"terms":[]},"right":{"terms":[]}}}`

These demonstrate valid envelopes, not a required sequence or research strategy.
"""

OPERATING_GUIDE = """\
# Jacobian MCP operating guide

Jacobian exposes a broad installed portfolio of atomic mathematical capabilities
through two MCP tools. Mathematical operations remain namespaced capability IDs
rather than becoming top-level MCP tools.

The agent owns decomposition, mathematical strategy, capability composition,
iteration, stopping criteria, and the decision to pursue independent checking.
Jacobian exposes operations and feedback; it does not prescribe a research workflow.

## When to use Jacobian

Begin with `capability.describe` when a task involves exact computation, structure
discovery, transformation, bounded search, counterexample generation,
formal-environment inspection, or independent checking. You do not need to know an
installed capability ID.

## Discover, inspect, and invoke

Search with `capability.describe(query=...)`, optionally filtered by `domain` and
`mode`. Results are compact candidates ranked by deterministic matches against
published descriptor metadata; `matched_on` and `matched_terms` make that retrieval
visible. Ranking is not a recommendation. Follow `next_cursor` with unchanged
filters and limit when a discovery result is truncated. Start with `limit=5`; inspect
only the strongest one or two relevant candidates before broadening or paging.

Search as many outcomes, concepts, or domains as useful. For any candidate, call
`capability.describe(capability_id=...)` for the small exact SUMMARY. Once the
outcome fits, request `view="CONTRACT"` and follow its validation-equivalent input
schema or validated invocation example. `view="COMPACT"` is the compatibility alias
for the previous contract projection. Request `view="FULL"` only when complete output
schema or provider audit metadata is needed. Once one domain-owned outcome fits,
invoke it before searching for a checker; follow the checker, certificate, and
verification fields returned by the producer instead of guessing a generic verifier.
Compose `capability.invoke` calls in whatever sequence the mathematical investigation
requires. Inspect execution, scope, completeness, relationships, obligations,
assurance, diagnostics, and artifacts as separate result dimensions.

`capability://catalog` is the complete machine-readable installed inventory.
`capability.describe` is the agent-oriented discovery and exact-inspection surface.

## Exploration and verification

`EXPLORE` returns proposed, heuristic, or computed evidence. Search, generation,
evaluation, solver output, retrieved memory, and workspace entries are not proof.

`VERIFY` may return `VERIFIED` only when an operator-authorized independent checker
accepts evidence bound to the exact claim, semantics, candidate, scope, certificate
format, and checker identity. Only assurance level `VERIFIED` with a local
verification record is verified.

Execution status is not a mathematical conclusion. `COMPLETED` bounded execution may
still have partial or unknown completeness and open obligations. A timeout,
cancellation, error, incomplete enumeration, or failure to find a witness is a
non-conclusion.

A finite collection of concrete parameter checks remains finite evidence and cannot
establish an all-parameters or all-orders claim. A hard expression-growth diagnostic
also remains a non-conclusion; increasing the size of the same full-expansion family
does not turn that bounded operation into a symbolic proof.

## Artifacts and workspaces

Follow returned `artifact://` and `experiment://` resources instead of requesting
large payloads inline. Workspace findings, attempts, lifecycle marks, focus, and
retrieval remain agent-authored operational state. Writing, closing, retracting,
superseding, archiving, or pinning a workspace entry never promotes mathematical
assurance. In particular, a workspace card titled as an exact value remains
UNVERIFIED unless a separate checker-backed capability returns VERIFIED with its
bound verification record.
"""


def discovery_prompt(task: str) -> str:
    """Render optional protocol guidance without choosing a research strategy."""

    return f"""\
Use Jacobian's capability protocol for this mathematical task:

<task>
{task}
</task>

1. Keep task decomposition, research strategy, iteration, and stopping criteria
   under your control.
2. Search any outcomes or concepts that may help with `capability.describe(query=...)`.
   Add `domain` or `mode` only when those filters are useful.
3. Treat compact matches as retrieval candidates, not recommendations. Search again
   across other concepts or domains whenever useful.
4. Inspect any candidate with `capability.describe(capability_id=...)`.
5. Once it fits, request `view="CONTRACT"` and construct `capability.invoke` from the
   exact input schema or a returned validated invocation example. Do not guess fields.
6. Compose operations as the investigation demands. Interpret execution,
   completeness, assurance, obligations, and artifacts separately. If independent
   checking is useful, discover an installed VERIFY capability rather than treating
   computed evidence as verified.
"""


def evidence_check_prompt(claim: str, artifact_uri: str | None = None) -> str:
    """Render evidence-checking guidance without claiming checker availability."""

    artifact_context = (
        f"\nCandidate evidence artifact: `{artifact_uri}`\n" if artifact_uri else "\n"
    )
    return f"""\
Use Jacobian to look for an independent checking path for this claim:

<claim>
{claim}
</claim>
{artifact_context}
1. Search with `capability.describe(query=..., mode="VERIFY")`.
2. Treat an empty result as checker unavailability, not evidence for or against the
   claim.
3. Describe the selected exact capability. Confirm that its semantics, scope,
   candidate representation, evidence format, and fixed checker identity match the
   claim and artifact.
4. Invoke only with the exact advertised schema. Do not translate a producer,
   evaluator, solver, or search result directly into `VERIFIED`.
5. Accept verification only when the result reports assurance level `VERIFIED` and
   includes the bound local verification record. Report any remaining obligations or
   scope mismatch.
"""
