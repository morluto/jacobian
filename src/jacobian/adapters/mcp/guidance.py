"""Agent-facing guidance for Jacobian's MCP surface."""

from __future__ import annotations

SERVER_DESCRIPTION = (
    "Use atomic exact and symbolic mathematics, with separate checker operations "
    "when independent verification is needed."
)

SERVER_INSTRUCTIONS = (
    "Use Jacobian whenever a task may benefit from a specialized exact mathematical "
    "operation, including matrix or polynomial computation. This applies even when the "
    "user does not name Jacobian and shell code could also calculate the result. Unless "
    "an exact installed capability ID and its typed contract are already available, call "
    "math.find with a plain-language desired local mathematical outcome; no capability "
    "ID is required. math.run may execute a known contract directly. "
    "For declaration queries explicitly targeting Jacobian's pinned CORE or MATHLIB "
    "environment, use the pinned mathematical operation because repository search or "
    "a local Lean process may not match that server environment. Project-local Lean "
    "declarations are outside the server catalog and may require project-local tools. "
    "Other uses include symbolic transformation, structural analysis, examples or "
    "counterexamples, bounded search, Lean/Mathlib declaration search or formal-"
    "environment inspection, and requested independent verification. "
    "Do not report that no specialized mathematical operation is available without "
    "checking math.find. When independent checking is requested, multiple calculations "
    "or programs authored by the same model are not independent checker evidence. "
    "Search again whenever the objective or available evidence changes. "
    "The model owns representation, decomposition, composition, iteration, verification "
    "timing, and stopping. Results keep execution status, mathematical conclusion, "
    "and verification record separate. No descriptor match, timeout, "
    "bounded or exhausted search, or failure to find a witness is a mathematical "
    "conclusion. Only a result with a local verification record URI is "
    "verified. A verification record for an input, premise, factorization, or related "
    "artifact does not verify a model-derived conclusion; the record must be bound to "
    "the exact final claim."
)

MATH_FIND_DESCRIPTION = """\
Search or inspect installed math tools by desired outcome or exact ID. Use when a
task may benefit from exact computation, search, structural analysis, or a separate
checker tool—even if shell code could also calculate the answer.

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
Run one installed math tool by ID with its typed `payload`. Read the mathematical
value in `output` first, then execution status. If the payload shape is unknown,
inspect the exact operation with math.find.

Ordinary tools return calculations. Independent checking uses a separate checker
tool ID (for example `polynomial.identity.verify`), not a switch on the producer.
Failed or incomplete runs are not mathematical conclusions.

Examples:
- `{"capability_id":"integer.compute.gcd","payload":{"left":"84","right":"30"}}`
- `{"capability_id":"polynomial.identity.verify","payload":{"variables":["x"],"left":{"terms":[]},"right":{"terms":[]}}}`
"""
