"""Agent-facing guidance for Jacobian's MCP surface."""

from __future__ import annotations

SERVER_DESCRIPTION = (
    "Use specialized exact and symbolic mathematics with explicit scope, "
    "completeness, assurance, and optional independent verification."
)

SERVER_INSTRUCTIONS = (
    "Use Jacobian whenever a task may benefit from a specialized exact mathematical "
    "operation, including matrix or polynomial computation, symbolic transformation, "
    "structural analysis, examples or counterexamples, bounded search, formal-environment "
    "inspection, or requested independent verification. This applies even when the "
    "user does not name Jacobian and shell code could also calculate the result. Unless "
    "an exact installed capability ID and its typed contract are already available, call "
    "math.find with a plain-language desired local mathematical outcome; no capability "
    "ID is required. math.run may execute a known contract directly. "
    "Do not report that no specialized mathematical operation is available without "
    "checking math.find. When independent checking is requested, multiple calculations "
    "or programs authored by the same model are not independent checker evidence. "
    "Search or browse again whenever the objective or available evidence changes. "
    "The model owns representation, decomposition, composition, iteration, verification "
    "timing, and stopping. Results keep execution status, scope, completeness, "
    "mathematical conclusion, and assurance separate. No descriptor match, timeout, "
    "bounded or exhausted search, or failure to find a witness is a mathematical "
    "conclusion. Only assurance level VERIFIED with a local verification record is "
    "verified. A verification record for an input, premise, factorization, or related "
    "artifact does not verify a model-derived conclusion; the record must be bound to "
    "the exact final claim. Read jacobian://instructions for the complete operating "
    "guide."
)

MATH_FIND_DESCRIPTION = """\
Search or inspect installed math tools by desired outcome or exact ID. Use when a
task may benefit from exact computation, search, structural analysis, or a separate
checker tool—even if shell code could also calculate the answer.

Forms:
- `query`: plain-language mathematical outcome (compact tool cards).
- Optional `domain` filter; `limit` 1-20 (default 5).
- Omit arguments to browse; follow `next_cursor` with the same filters to continue.
- Ranking is deterministic lexical retrieval; matches are not recommendations.
- `capability_id`: exact inspect (SUMMARY / CONTRACT / FULL views).

Checker tools are separate IDs (often `*.verify`), not a switch on producers.

Examples:
- `{"query":"compute an exact matrix determinant","domain":"matrix","limit":3}`
- `{"query":"find a counterexample to associativity","domain":"universal_algebra"}`
- `{"capability_id":"polynomial.compute.gcd","view":"CONTRACT"}`
"""

MATH_RUN_DESCRIPTION = """\
Run one installed math tool by ID with its typed `payload`. Read the mathematical
value in `output` first, then execution status. If the payload shape is unknown,
use math.find with view CONTRACT.

Ordinary tools return calculations. Independent checking uses a separate checker
tool ID (for example `polynomial.identity.verify` or `case.partition.finite.verify`),
not a switch on the producer. Failed or incomplete runs are not mathematical conclusions.

Examples:
- `{"capability_id":"integer.compute.gcd","payload":{"left":"84","right":"30"}}`
- `{"capability_id":"polynomial.identity.verify","payload":{"variables":["x"],"left":{"terms":[]},"right":{"terms":[]}}}`
"""

OPERATING_GUIDE = """\
# Jacobian MCP operating guide

Jacobian exposes a broad installed portfolio of atomic mathematical capabilities
through two MCP tools. Mathematical operations remain namespaced capability IDs
rather than becoming top-level MCP tools.

The agent owns decomposition, mathematical strategy, capability composition,
iteration, stopping criteria, and the decision to pursue independent checking.
Jacobian exposes operations and feedback; it does not prescribe a research workflow.

## Mathematical affordances

Jacobian may be useful for exact computation, structure discovery, transformation,
examples and counterexamples, bounded search, formal-environment inspection, or
independent checking. An installed capability ID is not required for search or browse.

## Search, browse, inspect, and run

Search with `math.find(query=...)`, optionally filtered by `domain`. Results are
compact candidates ranked by deterministic matches against published descriptor
metadata; `matched_on` and `matched_terms` make that retrieval visible. Ranking is
not a recommendation. Follow `next_cursor` with unchanged filters and limit when a
discovery result is truncated. Omit all arguments to browse.

The same tool accepts `capability_id` for exact inspection. SUMMARY is the compact
projection, CONTRACT adds the validation-equivalent input schema and examples, and
FULL adds complete provider and audit metadata. An agent that already has an exact
contract may invoke the operation directly. Related operations describe compatibility
and evidence flow; they do not prescribe what to do next.

`math.run` returns the canonical complete `CapabilityResult` directly as MCP
structured content. Calls may be composed in any sequence the mathematical
investigation requires. Inspect execution, scope, completeness, relationships,
obligations, assurance, diagnostics, intermediates, and artifacts as separate result
dimensions. Follow returned `artifact://` references to read durable results.

`capability://catalog` is the complete machine-readable installed inventory.
`math.find` is the agent-oriented discovery and exact-inspection surface.
Weak or empty discovery results expose query reformulation, filter removal, browse,
and catalog-inspection paths. They do not establish operation absence or mathematical
impossibility.

## Producers and checkers

Ordinary producer tools return proposed, heuristic, or computed evidence. Search,
generation, evaluation, solver output, and retrieved memory are not proof.

A separate checker tool (often a `*.verify` ID) may return `VERIFIED` only when an
operator-authorized independent checker accepts evidence bound to the exact claim,
semantics, candidate, scope, certificate format, and checker identity. Only assurance
level `VERIFIED` with a local verification record is verified.

Verification does not transfer across model-authored deductions. A record accepting
premises, inputs, factorizations, or related artifacts does not verify a derived
conclusion unless a checker record is bound to that exact conclusion.

Execution status is not a mathematical conclusion. `COMPLETED` bounded execution may
still have partial or unknown completeness and open obligations. A timeout,
cancellation, error, incomplete enumeration, or failure to find a witness is a
non-conclusion.

A finite collection of concrete parameter checks remains finite evidence and cannot
establish an all-parameters or all-orders claim. A hard expression-growth diagnostic
also remains a non-conclusion; increasing the size of the same full-expansion family
does not turn that bounded operation into a symbolic proof.

## Artifacts

Follow returned `artifact://` and `experiment://` resources instead of requesting
large payloads inline.

Reading an `artifact://sha256/<digest>` resource returns a JSON envelope with keys
`artifact_uri`, `manifest`, and `payload`. The `payload` field holds the bare
artifact content (for a verification record, the `VerificationRecord` JSON). When
persisting a verification record for a clean-room verifier, extract the `payload`
field rather than saving the full envelope. Follow the agent-visible
`verification_record_schema.json` when the task provides one: some schemas accept
the bare `VerificationRecord` payload, while others require a task-specific
wrapper containing additional fields (e.g. task_id, input_sha256, assignment).
Do not claim VERIFIED when no agent-visible record content contract is available.
"""


def discovery_prompt(task: str) -> str:
    """Render optional protocol guidance without choosing a research strategy."""

    return f"""\
Use Jacobian's capability protocol for this mathematical task:

<task>
{task}
</task>

Keep task decomposition, representation, research strategy, iteration, verification
timing, and stopping criteria under your control.

Available affordances:
- `math.find(query=...)` searches by a desired local mathematical outcome.
- `math.find()` browses when the installed vocabulary is unknown.
- `math.find(capability_id=...)` inspects an exact operation and its typed
  contract.
- `math.run(...)` runs one selected operation.

Compact matches are retrieval candidates, not recommendations. Compose operations as
the investigation demands, and interpret execution, scope, completeness, assurance,
obligations, intermediates, relationships, and artifacts separately. Computed evidence
is not independently verified evidence.
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
1. Search with `math.find(query=...)` for a checker tool (often a `*.verify` ID).
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
