"""Agent-facing guidance for Jacobian's MCP surface."""

from __future__ import annotations

SERVER_DESCRIPTION = (
    "Search and run installed Jacobian exact-math operations, with separate checker "
    "operations for independent verification."
)

SERVER_INSTRUCTIONS = (
    "Search and run the locally installed Jacobian toolbox whenever a task may benefit "
    "from exact computation, symbolic transformation, finite search, formal inspection, "
    "or independent checking, even when the user does not name Jacobian or shell code "
    "could also calculate the result. math.find is the authoritative local inventory; "
    "internet search cannot show which operations are available in this runtime. Unless "
    "an exact installed capability ID and its typed contract are already available, call "
    "math.find with a plain-language desired local mathematical outcome. math.run may "
    "execute a known contract directly. "
    "For declaration queries explicitly targeting Jacobian's pinned CORE or MATHLIB "
    "environment, use the pinned mathematical operation; do not substitute repository "
    "search, cached Mathlib files, or a local Lean process because they may not match "
    "that server environment. Project-local Lean declarations are outside the server "
    "catalog and may require project-local tools. Repeating the same lookup against an "
    "unchanged catalog returns the same operation facts; math.find is operation lookup, "
    "not confirmation. Merely restating an accepted value without new evidence is not a "
    "mathematical-tool use case. "
    "The model owns representation, decomposition, composition, iteration, verification "
    "timing, and stopping. An operation match, timeout, incomplete search, or failure to "
    "find a witness is not a mathematical conclusion. Independent checking uses a "
    "separate checker operation; model-authored duplicate calculations are not independent "
    "evidence. Only a result with a local verification record URI is verified. A record "
    "for an input, premise, factorization, or related artifact does not verify a model-"
    "derived conclusion; the record must be bound to the exact final claim."
)

MATH_FIND_DESCRIPTION = """\
Search or inspect locally installed Jacobian math tools by desired outcome or exact ID.
This is the authoritative runtime inventory; internet search is not. Use when a task
may benefit from exact computation, search, structural analysis, or a separate checker
tool—even if shell code could also calculate the answer.

Forms:
- `request.op="search"`: plain-language mathematical outcome (compact cards).
- Optional `domain` filter; `limit` 1-20 (default 5).
- Follow `next_cursor` with the same query and filters to continue.
- Ranking is deterministic lexical retrieval; matches are not recommendations.
- `request.op="inspect"`: exact ID with authoritative schemas and examples.

Checker tools are separate IDs (often `*.verify`), not a switch on producers.

Examples:
- `{"request":{"op":"search","query":"exact matrix determinant","domain":"matrix","limit":3}}`
- `{"request":{"op":"search","query":"counterexample to associativity"}}`
- `{"request":{"op":"search","query":"find a theorem declaration in pinned Mathlib","domain":"lean"}}`
- `{"request":{"op":"inspect","capability_id":"polynomial.compute.gcd"}}`
"""

MATH_RUN_DESCRIPTION = """\
Run one installed math tool by ID with its typed `payload`. Check execution status
before treating `output` as mathematical evidence. For a completed run, interpret the
operation-owned result fields and any `verification_record_uri` to determine exactly
what the run established. If the payload shape is unknown, inspect the exact operation
with math.find and copy its `invocation_examples`. Do not call math.run with an empty
`payload` merely to discover required fields; the inspect result is the authoritative
contract.

Ordinary tools return calculations. Independent checking uses a separate checker
tool ID (for example `polynomial.identity.verify`), not a switch on the producer.
Failed, cancelled, timed-out, or incomplete runs are not mathematical conclusions.

Examples:
- `{"capability_id":"integer.compute.gcd","payload":{"left":"84","right":"30"}}`
- `{"capability_id":"polynomial.identity.verify","payload":{"variables":["x"],"left":{"terms":[]},"right":{"terms":[]}}}`
"""
