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
    "user does not name Jacobian and shell code could also calculate the result. Call "
    "math.find with a plain-language desired local mathematical outcome; no capability "
    "ID is required. math.run executes a selected operation using its typed input "
    "contract. "
    "Do not report that no specialized mathematical operation is available without "
    "checking math.find. When independent checking is requested, multiple calculations "
    "or programs authored by the same model are not independent checker evidence. "
    "Search or browse again whenever the objective or available evidence changes. "
    "The model owns representation, decomposition, composition, iteration, verification "
    "timing, and stopping. Results keep execution status, scope, completeness, "
    "mathematical conclusion, and assurance separate. No descriptor match, timeout, "
    "bounded or exhausted search, or failure to find a witness is a mathematical "
    "conclusion. Only assurance level VERIFIED with a local verification record is "
    "verified. Read jacobian://instructions for the complete operating guide."
)

MATH_FIND_DESCRIPTION = """\
Use this when a task may benefit from a specialized exact mathematical operation, even
if shell code could also calculate the answer. Relevant outcomes include matrix
determinants, polynomial or symbolic computation, structural analysis, examples or
counterexamples, bounded search, formal-environment inspection, and requested
independent verification. Search Jacobian by desired local mathematical outcome, or
inspect one exact operation contract. A capability ID is not required for search or
browse.

Available forms:
- Pass `query` as a plain-language description of the desired local mathematical
  outcome. The response contains compact operation cards with accepted inputs, output
  summary, availability, scope, assurance ceiling, factual relationships, and one
  size-bounded validated invocation example when available.
- Optionally filter with `domain` and `mode`. `limit` is between 1 and 20 and defaults
  to 5; a smaller requested limit returns less model context.
- Omit all arguments to browse a compact installed catalog.
- When `next_cursor` is present, pass it back with the same filters and limit to
  continue without loading the complete catalog.
- Ranking is deterministic retrieval over published IDs, titles, descriptions, and
  tags. Match fields and terms are returned; candidates are not recommendations.
- Pass `capability_id` to inspect the exact operation. SUMMARY is compact, CONTRACT
  adds the validation-equivalent input schema and validated invocation examples, and
  FULL adds complete provider and audit metadata.

Weak or empty results do not imply impossibility. They include unranked recovery paths
for query reformulation, filter removal, browsing, and catalog inspection. Every exact
response states the operation's scope rule.

Examples:
- `{"query":"compute an exact matrix determinant","domain":"matrix","mode":"EXPLORE","limit":3}`
- `{"query":"find a counterexample to associativity","domain":"universal_algebra","mode":"EXPLORE","limit":3}`
- `{"query":"eliminate this denominator using the defining relations"}`
- `{}`
- `{"capability_id":"polynomial.compute.gcd"}`
- `{"capability_id":"polynomial.compute.gcd","view":"CONTRACT"}`
"""

MATH_RUN_DESCRIPTION = """\
Use this to run one selected Jacobian operation with its typed payload. If the payload
shape is unfamiliar, math.find can return the exact CONTRACT. EXPLORE returns proposed,
heuristic, or computed evidence; VERIFY is valid only for an installed checker-backed
contract.

The typed `CapabilityResult` keeps execution, scope, completeness, mathematical
conclusion, assurance, obligations, diagnostics, relationships, and artifacts distinct.

COMPLETED does not by itself establish a mathematical conclusion. One invocation
covers only its exact supplied input or claim, and repeated finite or bounded calls do
not widen that scope. Follow returned `artifact://` references when durable evidence
or a size-separated result is provided.

Examples:
- `{"capability_id":"integer.compute.gcd","mode":"EXPLORE","payload":{"left":"84","right":"30"}}`
- `{"capability_id":"polynomial.identity.verify","mode":"VERIFY","payload":{"variables":["x"],"left":{"terms":[]},"right":{"terms":[]}}}`

These are valid envelopes, not a required research strategy.
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

Search with `math.find(query=...)`, optionally filtered by `domain` and
`mode`. Results are compact candidates ranked by deterministic matches against
published descriptor metadata; `matched_on` and `matched_terms` make that retrieval
visible. Ranking is not a recommendation. Follow `next_cursor` with unchanged filters
and limit when a discovery result is truncated. Omit all arguments to browse.

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

## Exploration and verification

`EXPLORE` returns proposed, heuristic, or computed evidence. Search, generation,
evaluation, solver output, and retrieved memory are not proof.

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

REASONING_WRITE_DESCRIPTION = """\
Append one concise external reasoning summary. This records model-authored PLAN,
BEFORE_TOOL, AFTER_TOOL, or FINAL entries; it does not request or expose hidden
chain-of-thought and never establishes mathematical assurance. PLAN creates a run.
BEFORE_TOOL reserves one capability call. AFTER_TOOL interprets the bound actual
result or explicitly records RESULT_UNAVAILABLE after a lost response or runtime
restart. FINAL audits the completed run. Do not copy prompts, secrets, payloads, or
raw tool output into summary.
"""

_REASONING_PREFIX = (
    "Before using Jacobian, call reasoning.write with phase PLAN and a concise "
    "external work summary. Before each math.run call, write BEFORE_TOOL "
    "with the selected capability ID and mode, then pass the returned run_id and "
    "call_id to math.run. Write AFTER_TOOL after every result with the "
    "reported status, assurance, and completeness; use RESULT_UNAVAILABLE only when "
    "result content was lost. Write FINAL before completing the task. These are summaries, "
    "not hidden chain-of-thought. Do not copy raw tool output. "
)


def server_instructions(*, reasoning_enabled: bool) -> str:
    return (_REASONING_PREFIX if reasoning_enabled else "") + SERVER_INSTRUCTIONS


def operating_guide(*, reasoning_enabled: bool) -> str:
    if not reasoning_enabled:
        return OPERATING_GUIDE
    return OPERATING_GUIDE.replace(
        "through two MCP tools.",
        "through two capability tools plus the operational `reasoning.write` log tool.",
    ).replace(
        "## Mathematical affordances",
        "## External reasoning log\n\n"
        "Start with one concise `PLAN`. Surround every `math.run` with "
        "`BEFORE_TOOL` and `AFTER_TOOL`, then write one `FINAL` audit. The server "
        "binds actual status, assurance, completeness, and artifact URIs and records "
        "whether the model's structured interpretation matches them, without "
        "storing raw payload or output. This log is operational and unverified; it "
        "is not hidden chain-of-thought or mathematical evidence.\n\n"
        "## Mathematical affordances",
    )


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
1. Search with `math.find(query=..., mode="VERIFY")`.
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
